"""Train C DAS and test whether C naturally produces AP, AT, and Y."""

import os, torch
import torch.nn.functional as F
from mcqa_gradual_das import set_seed, answer_label_ids, answer_logits, output_label_ids, output_targets, output_iia, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention

AP_TARGET = "AP_after_C_intervene"
AT_TARGET = "AT_after_C_intervene"
Y_TARGET = "Y_after_C_intervene"


def run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_pos="color", ap_pos="last_token", at_pos="last_token", batch_size=8, require_grad=True):
    """C <- source C subspace; return Y plus AP/AT states naturally generated downstream."""
    device = next(model.parameters()).device; all_y, all_ap, all_at = [], [], []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"])); ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_c[start:end].to(device); rows = torch.arange(len(ids), device=device); pad = (mask == 0).sum(1)
        c_idx = pad + bank["base_position_by_id"][c_pos][start:end].to(device); ap_idx = pad + bank["base_position_by_id"][ap_pos][start:end].to(device); at_idx = pad + bank["base_position_by_id"][at_pos][start:end].to(device); captured = {}

        def c_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); base = h[rows, c_idx].float()
            h2[rows, c_idx] = (base + ((donor.float() - base) @ Q_c) @ Q_c.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2

        def ap_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; captured["ap"] = h[rows, ap_idx].float()

        def at_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; captured["at"] = h[rows, at_idx].float()

        hc = model.model.layers[c_layer].register_forward_hook(c_hook)
        if ap_layer == at_layer:
            def both_hook(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output; captured["ap"] = h[rows, ap_idx].float(); captured["at"] = h[rows, at_idx].float()
            ha = model.model.layers[ap_layer].register_forward_hook(both_hook); ht = None
        else:
            ha = model.model.layers[ap_layer].register_forward_hook(ap_hook); ht = model.model.layers[at_layer].register_forward_hook(at_hook)
        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            y = answer_logits(model, out, mask, output_ids); ap, at = captured["ap"], captured["at"]
            all_y.append(y if require_grad else y.detach().cpu()); all_ap.append(ap if require_grad else ap.detach().cpu()); all_at.append(at if require_grad else at.detach().cpu())
        finally:
            hc.remove(); ha.remove()
            if ht is not None: ht.remove()
    return torch.cat(all_y), torch.cat(all_ap), torch.cat(all_at)


def run_frozen_mediator(model, bank, generated, Q, layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    """Replay a generated AP/AT state into a fresh clean base run and return Y."""
    device = next(model.parameters()).device; all_y = []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"])); ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated[start:end].to(device); rows = torch.arange(len(ids), device=device); pos = (mask == 0).sum(1) + bank["base_position_by_id"][token_position][start:end].to(device)
        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); base = h[rows, pos].float(); h2[rows, pos] = (base + ((donor.float() - base) @ Q) @ Q.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2
        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            y = answer_logits(model, out, mask, output_ids); all_y.append(y if require_grad else y.detach().cpu())
        finally: handle.remove()
    return torch.cat(all_y)


@torch.no_grad()
def evaluate_c(model, tokenizer, bank, source_c, Q_c, c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, c_pos, batch_size, output_space):
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space)
    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_pos, ap_pos, at_pos, batch_size, False)
    ap_mediator_y = run_frozen_mediator(model, bank, generated_ap, Q_ap, ap_layer, output_ids, ap_pos, batch_size, False)
    at_mediator_y = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, at_pos, batch_size, False)
    t = bank["intervention_targets"]; ap_logits = (generated_ap.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap; at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at
    return (
        output_iia(direct_y, t[Y_TARGET], tokenizer, output_space),
        output_iia(ap_mediator_y, t[Y_TARGET], tokenizer, output_space),
        float((ap_logits.argmax(-1).cpu() == t[AP_TARGET].cpu()).float().mean()),
        output_iia(at_mediator_y, t[Y_TARGET], tokenizer, output_space),
        float((at_logits.argmax(-1).cpu() == t[AT_TARGET].cpu()).float().mean()),
    )


def train_c(model, tokenizer, ap_checkpoint, ap_readout_checkpoint, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, lambda_ap_med=0.0, lambda_ap_readout=0.0, lambda_at_med=0.0, lambda_at_readout=0.0, train_batch_size=32, eval_batch_size=32, token_position="color", seed=0, save_path="results/das_c.pt", banks=None, output_space="full"):
    set_seed(seed); device = next(model.parameters()).device; model.eval()
    for p in model.parameters(): 
        p.requires_grad_(False)

    ap = torch.load(ap_checkpoint, map_location="cpu", weights_only=False); Q_ap = ap["basis"].float().to(device); ap_layer = int(ap["layer"]); ap_pos = ap.get("token_position", "last_token")
    at = torch.load(at_checkpoint, map_location="cpu", weights_only=False); Q_at = at["basis"].float().to(device); at_layer = int(at["layer"]); at_pos = at.get("token_position", "last_token")

    ap_ro = torch.load(ap_readout_checkpoint, map_location="cpu", weights_only=False); W_ap = ap_ro["W_AP"].float().to(device); b_ap = ap_ro["b_AP"].float().to(device)
    at_ro = torch.load(at_readout_checkpoint, map_location="cpu", weights_only=False); W_at = at_ro["W_AT"].float().to(device); b_at = at_ro["b_AT"].float().to(device)
    
    train_bank, cal_banks, test_banks = banks; cal_bank = cal_banks["C"]; test_bank = test_banks["C"]
    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space); max_layer = min(ap_layer, at_layer); layers = list(range(max_layer)) if layers is None else [x for x in layers if x < max_layer]

    # Pick C layer using direct Y_after_C_intervene.
    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_scores = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        iia = output_iia(logits, cal_bank["intervention_targets"][Y_TARGET], tokenizer, output_space); layer_scores.append((layer, iia)); print(f"[C layer {layer}] iia={iia:.4f}")
    c_layer = max(layer_scores, key=lambda x: x[1])[0]

    train_source = collect_layer_states(model, train_bank["source_input_ids"], train_bank["source_attention_mask"], train_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[c_layer]
    test_source = collect_layer_states(model, test_bank["source_input_ids"], test_bank["source_attention_mask"], test_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)
    t = train_bank["intervention_targets"]; alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device); optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    best_score, best_basis, best_epoch, best_metrics = -1.0, None, None, None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_bank["base_input_ids"]))
        totals = [0.0] * 5
        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini = {"base_input_ids": train_bank["base_input_ids"][idx], "base_attention_mask": train_bank["base_attention_mask"][idx], "base_position_by_id": {k: v[idx] for k, v in train_bank["base_position_by_id"].items()}}

            optimizer.zero_grad()
            Q_c = alignment.basis()

            direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, mini, train_source[idx], Q_c, c_layer, ap_layer, at_layer, output_ids, token_position, ap_pos, at_pos, len(idx), True)
            ap_mediator_y = run_frozen_mediator(model, mini, generated_ap, Q_ap, ap_layer, output_ids, ap_pos, len(idx), True); at_mediator_y = run_frozen_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, at_pos, len(idx), True)
            y = output_targets(t[Y_TARGET][idx], labels, output_space, device); ap_target = t[AP_TARGET][idx].to(device); at_target = t[AT_TARGET][idx].to(device)

            losses = [F.cross_entropy(direct_y, y), F.cross_entropy(ap_mediator_y, y), F.cross_entropy((generated_ap @ Q_ap) @ W_ap.T + b_ap, ap_target), F.cross_entropy(at_mediator_y, y), F.cross_entropy((generated_at @ Q_at) @ W_at.T + b_at, at_target)]
            loss = losses[0] + lambda_ap_med * losses[1] + lambda_ap_readout * losses[2] + lambda_at_med * losses[3] + lambda_at_readout * losses[4]
            loss.backward()
            optimizer.step()

            for i, value in enumerate(losses): 
                totals[i] += value.item() * len(idx)

        metrics = evaluate_c(model, tokenizer, cal_bank, cal_source, alignment.basis(), c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, token_position, eval_batch_size, output_space)
        direct, ap_med, ap_ro_acc, at_med, at_ro_acc = metrics; active = [direct] + ([ap_med] if lambda_ap_med > 0 else []) + ([ap_ro_acc] if lambda_ap_readout > 0 else []) + ([at_med] if lambda_at_med > 0 else []) + ([at_ro_acc] if lambda_at_readout > 0 else []); score = min(active)
        print(f"[C epoch {epoch:02d}] losses={[round(x/len(perm),4) for x in totals]} | cal={metrics}")
        if score > best_score: best_score, best_basis, best_epoch, best_metrics = score, alignment.basis().detach().cpu().clone(), epoch, metrics

    test_metrics = evaluate_c(model, tokenizer, test_bank, test_source, best_basis.to(device), c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, token_position, eval_batch_size, output_space)
    result = {"variable": "C", "layer": c_layer, "basis": best_basis, "token_position": token_position, "subspace_dim": subspace_dim, "cal": best_metrics, "test": test_metrics, "best_epoch": best_epoch}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True); torch.save(result, save_path)
    print(f"[C BEST] layer={c_layer} epoch={best_epoch} cal={best_metrics} test={test_metrics}")
    return result
