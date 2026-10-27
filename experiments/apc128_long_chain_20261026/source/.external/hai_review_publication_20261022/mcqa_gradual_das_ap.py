"""Train AP DAS and optionally enforce the downstream AT mediator/readout."""

import os, torch
import torch.nn.functional as F
from mcqa_gradual_das import set_seed, answer_label_ids, answer_logits, output_label_ids, output_targets, output_iia, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention

AT_TARGET = "AT_after_AP_intervene"
Y_TARGET = "Y_after_AP_intervene"


def run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    """AP <- source AP subspace; return output Y and the AT state naturally produced downstream."""
    device = next(model.parameters()).device; all_y, all_at = [], []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"])); ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_ap[start:end].to(device); rows = torch.arange(len(ids), device=device); pos = (mask == 0).sum(1) + bank["base_position_by_id"][token_position][start:end].to(device); captured = {}

        def ap_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); base = h[rows, pos].float()
            h2[rows, pos] = (base + ((donor.float() - base) @ Q_ap) @ Q_ap.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2

        def at_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; captured["at"] = h[rows, pos].float()

        h1 = model.model.layers[ap_layer].register_forward_hook(ap_hook); h2 = model.model.layers[at_layer].register_forward_hook(at_hook)
        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            y = answer_logits(model, out, mask, output_ids); at = captured["at"]
            all_y.append(y if require_grad else y.detach().cpu()); all_at.append(at if require_grad else at.detach().cpu())
        finally: h1.remove(); h2.remove()
    return torch.cat(all_y), torch.cat(all_at)


def run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    """Replay the AT state produced by AP into a fresh clean base run; return Y."""
    device = next(model.parameters()).device; all_y = []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"])); ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated_at[start:end].to(device); rows = torch.arange(len(ids), device=device); pos = (mask == 0).sum(1) + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); base = h[rows, pos].float()
            h2[rows, pos] = (base + ((donor.float() - base) @ Q_at) @ Q_at.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2

        handle = model.model.layers[at_layer].register_forward_hook(hook)
        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            y = answer_logits(model, out, mask, output_ids); all_y.append(y if require_grad else y.detach().cpu())
        finally: handle.remove()
    return torch.cat(all_y)


@torch.no_grad()
def evaluate_ap(model, tokenizer, bank, source_ap, Q_ap, ap_layer, Q_at, at_layer, W_at, b_at, token_position, batch_size, output_space):
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space)
    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, token_position, batch_size, False)
    mediator_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, token_position, batch_size, False)
    at_target = bank["intervention_targets"][AT_TARGET]; y_target = bank["intervention_targets"][Y_TARGET]
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    return output_iia(direct_y, y_target, tokenizer, output_space), output_iia(mediator_y, y_target, tokenizer, output_space), float((at_logits.argmax(-1).cpu() == at_target.cpu()).float().mean())


def train_ap(model, tokenizer, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=None, cal_size=None, te_size=None, epochs=50, lr=1e-2, lambda_med=0.0, lambda_readout=0.0, train_batch_size=32, eval_batch_size=32, token_position=None, seed=0, save_path="results/das_ap.pt", banks=None, output_space="full"):
    set_seed(seed); device = next(model.parameters()).device; model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    at = torch.load(at_checkpoint, map_location="cpu", weights_only=False); Q_at = at["basis"].float().to(device); at_layer = int(at["layer"])
    readout = torch.load(at_readout_checkpoint, map_location="cpu", weights_only=False); W_at = readout["W_AT"].float().to(device); b_at = readout["b_AT"].float().to(device)
    token_position = at.get("token_position", "last_token") if token_position is None else token_position
    train_bank, cal_banks, test_banks = banks; cal_bank = cal_banks["AP"]; test_bank = test_banks["AP"]
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space); layers = list(range(at_layer)) if layers is None else [x for x in layers if x < at_layer]

    # Pick AP layer using full-state interchange and the correct Y-after-AP target.
    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_scores = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        iia = output_iia(logits, cal_bank["intervention_targets"][Y_TARGET], tokenizer, output_space); layer_scores.append((layer, iia)); print(f"[AP layer {layer}] iia={iia:.4f}")
    ap_layer = max(layer_scores, key=lambda x: x[1])[0]

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[ap_layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)
    y_train = train_bank["intervention_targets"][Y_TARGET]; at_train = train_bank["intervention_targets"][AT_TARGET]
    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device); optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    best_score, best_basis, best_epoch, best_metrics = -1.0, None, None, None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_bank["base_input_ids"])); totals = [0.0, 0.0, 0.0]
        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini = {"base_input_ids": train_bank["base_input_ids"][idx], "base_attention_mask": train_bank["base_attention_mask"][idx], "base_position_by_id": {k: v[idx] for k, v in train_bank["base_position_by_id"].items()}}
            optimizer.zero_grad(); Q_ap = alignment.basis()
            direct_y, generated_at = run_ap_intervention_and_capture_at(model, mini, train_source[idx], Q_ap, ap_layer, at_layer, output_ids, token_position, len(idx), True)
            mediator_y = run_frozen_at_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, token_position, len(idx), True)
            y = output_targets(y_train[idx], labels, output_space, device); at = at_train[idx].to(device)
            direct_loss = F.cross_entropy(direct_y, y); mediator_loss = F.cross_entropy(mediator_y, y); readout_loss = F.cross_entropy((generated_at @ Q_at) @ W_at.T + b_at, at)
            loss = direct_loss + lambda_med * mediator_loss + lambda_readout * readout_loss; loss.backward(); optimizer.step()
            for i, value in enumerate((direct_loss, mediator_loss, readout_loss)): totals[i] += value.item() * len(idx)

        metrics = evaluate_ap(model, tokenizer, cal_bank, cal_source, alignment.basis(), ap_layer, Q_at, at_layer, W_at, b_at, token_position, eval_batch_size, output_space)
        direct_iia, mediator_iia, readout_acc = metrics; active = [direct_iia] + ([mediator_iia] if lambda_med > 0 else []) + ([readout_acc] if lambda_readout > 0 else []); score = min(active)
        print(f"[AP epoch {epoch:02d}] direct={totals[0]/len(perm):.4f} med={totals[1]/len(perm):.4f} readout={totals[2]/len(perm):.4f} | cal={metrics}")
        if score > best_score: best_score, best_basis, best_epoch, best_metrics = score, alignment.basis().detach().cpu().clone(), epoch, metrics

    test_metrics = evaluate_ap(model, tokenizer, test_bank, test_source, best_basis.to(device), ap_layer, Q_at, at_layer, W_at, b_at, token_position, eval_batch_size, output_space)
    result = {"variable": "AP", "layer": ap_layer, "basis": best_basis, "token_position": token_position, "subspace_dim": subspace_dim, "cal": best_metrics, "test": test_metrics, "best_epoch": best_epoch}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True); torch.save(result, save_path)
    print(f"[AP BEST] layer={ap_layer} epoch={best_epoch} cal={best_metrics} test={test_metrics}")
    return result
