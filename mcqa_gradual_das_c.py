import argparse, os, random, torch
import torch.nn.functional as F

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import load_generated_dataset, rows_to_pairs, predict_correct, create_bank
from mcqa_gradual_das import set_seed, answer_label_ids, answer_logits, output_label_ids, output_targets, output_iia, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention


def run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_pos="color", ap_pos="last_token", at_pos="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device; all_logits, all_ap, all_at = [], [], []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device); source = source_c[start:end].to(device)
        rows = torch.arange(len(ids), device=device); pad = (mask == 0).sum(1)
        c_idx = pad + bank["base_position_by_id"][c_pos][start:end].to(device)
        ap_idx = pad + bank["base_position_by_id"][ap_pos][start:end].to(device)
        at_idx = pad + bank["base_position_by_id"][at_pos][start:end].to(device)
        captured = {}

        def c_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); base = h[rows, c_idx, :].float()
            h2[rows, c_idx, :] = (base + ((source.float() - base) @ Q_c) @ Q_c.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2

        def ap_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; captured["ap"] = h[rows, ap_idx, :].float()

        def at_hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; captured["at"] = h[rows, at_idx, :].float()

        hc = model.model.layers[c_layer].register_forward_hook(c_hook)
        if ap_layer == at_layer:
            def shared_hook(_m, _i, output):
                h = output[0] if isinstance(output, tuple) else output
                captured["ap"] = h[rows, ap_idx, :].float(); captured["at"] = h[rows, at_idx, :].float()
            ha = model.model.layers[ap_layer].register_forward_hook(shared_hook); ht = None
        else:
            ha = model.model.layers[ap_layer].register_forward_hook(ap_hook)
            ht = model.model.layers[at_layer].register_forward_hook(at_hook)

        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            logits = answer_logits(model, out, mask, output_ids)
            if require_grad:
                all_logits.append(logits); all_ap.append(captured["ap"]); all_at.append(captured["at"])
            else:
                all_logits.append(logits.detach().cpu()); all_ap.append(captured["ap"].detach().cpu()); all_at.append(captured["at"].detach().cpu())
        finally:
            hc.remove(); ha.remove()
            if ht is not None: ht.remove()

    return torch.cat(all_logits), torch.cat(all_ap), torch.cat(all_at)


def run_frozen_mediator(model, bank, generated, Q, layer, output_ids, token_position="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device; all_logits = []
    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device); mask = bank["base_attention_mask"][start:end].to(device); donor = generated[start:end].to(device)
        rows = torch.arange(len(ids), device=device); pos = (mask == 0).sum(1) + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_m, _i, output):
            h = output[0] if isinstance(output, tuple) else output; h2 = h.clone(); clean = h[rows, pos, :].float()
            h2[rows, pos, :] = (clean + ((donor.float() - clean) @ Q) @ Q.T).to(h.dtype)
            return (h2,) + output[1:] if isinstance(output, tuple) else h2

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            out = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            logits = answer_logits(model, out, mask, output_ids); all_logits.append(logits if require_grad else logits.detach().cpu())
        finally:
            handle.remove()
    return torch.cat(all_logits)


@torch.no_grad()
def evaluate_c(model, bank, source_c, Q_c, c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, labels, c_pos, batch_size, tokenizer, output_space):
    output_ids = output_label_ids(labels, output_space)
    direct, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_pos, ap_pos, at_pos, batch_size, False)
    ap_med = run_frozen_mediator(model, bank, generated_ap, Q_ap, ap_layer, output_ids, ap_pos, batch_size, False)
    at_med = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, at_pos, batch_size, False)

    direct_target = bank["counterfactual_label_ids"]["color"].cpu()
    ap_target = bank["counterfactual_label_ids"]["answer_pointer"].to(Q_ap.device)
    at_target = bank["counterfactual_label_ids"]["answer_token"].to(Q_at.device)
    ap_y_target = bank["counterfactual_label_ids"]["answer_token_after_pointer_interchange"].cpu()
    at_y_target = bank["counterfactual_label_ids"]["answer_token"].cpu()

    ap_logits = (generated_ap.to(Q_ap.device) @ Q_ap) @ W_ap.T + b_ap
    at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    direct_iia = output_iia(direct, direct_target, tokenizer, output_space)
    ap_med_iia = output_iia(ap_med, ap_y_target, tokenizer, output_space)
    at_med_iia = output_iia(at_med, at_y_target, tokenizer, output_space)
    ap_readout = float((ap_logits.argmax(-1) == ap_target).float().mean())
    at_readout = float((at_logits.argmax(-1) == at_target).float().mean())
    return direct_iia, ap_med_iia, ap_readout, at_med_iia, at_readout


def train_c(model, tokenizer, ap_checkpoint, ap_readout_checkpoint, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, lambda_ap_med=0.0, lambda_ap_readout=0.0, lambda_at_med=0.0, lambda_at_readout=0.0, train_batch_size=32, eval_batch_size=32, token_position="color", seed=0, save_path="results/das_c.pt", banks=None, output_space="full"):
    set_seed(seed); device = next(model.parameters()).device
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    labels = answer_label_ids(tokenizer); output_ids = output_label_ids(labels, output_space)
    ap_result = torch.load(ap_checkpoint, map_location="cpu", weights_only=False); Q_ap = ap_result["basis"].float().to(device); ap_layer = int(ap_result["layer"]); ap_pos = ap_result.get("token_position", "last_token")
    at_result = torch.load(at_checkpoint, map_location="cpu", weights_only=False); Q_at = at_result["basis"].float().to(device); at_layer = int(at_result["layer"]); at_pos = at_result.get("token_position", "last_token")
    ap_ro = torch.load(ap_readout_checkpoint, map_location="cpu", weights_only=False); W_ap = ap_ro["W_AP"].float().to(device); b_ap = ap_ro["b_AP"].float().to(device)
    at_ro = torch.load(at_readout_checkpoint, map_location="cpu", weights_only=False); W_at = at_ro["W_AT"].float().to(device); b_at = at_ro["b_AT"].float().to(device)

    if banks is None: raise ValueError("train_c requires C_AP_AT banks via banks=...")
    fit_bank, cal_banks, te_banks = banks; cal_bank = cal_banks["color"]; te_bank = te_banks["color"]

    required = {"color", "answer_pointer", "answer_token", "answer_token_after_pointer_interchange"}
    missing = required - set(fit_bank["counterfactual_label_ids"])
    if missing: raise ValueError(f"C bank missing targets: {sorted(missing)}")

    max_c_layer = min(ap_layer, at_layer)
    layers = list(range(max_c_layer)) if layers is None else [x for x in layers if x < max_c_layer]
    if not layers: raise ValueError(f"No valid C layers before AP=L{ap_layer}, AT=L{at_layer}")

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)
    layer_results = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, output_ids, token_position, eval_batch_size)
        iia = output_iia(logits, cal_bank["counterfactual_label_ids"]["color"], tokenizer, output_space)
        layer_results.append({"layer": int(layer), "cal_iia": iia}); print(f"[C layer] layer={layer} iia={iia:.4f}")

    best_layer_row = max(layer_results, key=lambda x: x["cal_iia"]); c_layer = int(best_layer_row["layer"])
    print(f"[C layer selected] layer={c_layer} iia={best_layer_row['cal_iia']:.4f}")

    ft_source = collect_layer_states(model, fit_bank["source_input_ids"], fit_bank["source_attention_mask"], fit_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[c_layer]
    te_source = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], c_layer, token_position, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device); optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    direct_ft = fit_bank["counterfactual_label_ids"]["color"]
    ap_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]
    at_ft = fit_bank["counterfactual_label_ids"]["answer_token"]
    ap_y_ft = fit_bank["counterfactual_label_ids"]["answer_token_after_pointer_interchange"]
    at_y_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_score, best_basis, best_epoch, best_metrics = -1.0, None, None, None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"])); totals = [0.0] * 5; total_n = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini = {"base_input_ids": fit_bank["base_input_ids"][idx], "base_attention_mask": fit_bank["base_attention_mask"][idx], "base_position_by_id": {k: v[idx] for k, v in fit_bank["base_position_by_id"].items()}}
            optimizer.zero_grad(); Q_c = alignment.basis()

            direct, generated_ap, generated_at = run_c_intervention_and_capture(model, mini, ft_source[idx], Q_c, c_layer, ap_layer, at_layer, output_ids, token_position, ap_pos, at_pos, len(idx), True)
            ap_med = run_frozen_mediator(model, mini, generated_ap, Q_ap, ap_layer, output_ids, ap_pos, len(idx), True)
            at_med = run_frozen_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, at_pos, len(idx), True)

            direct_target = output_targets(direct_ft[idx], labels, output_space, device)
            ap_med_target = output_targets(ap_y_ft[idx], labels, output_space, device)
            at_med_target = output_targets(at_y_ft[idx], labels, output_space, device)
            ap_target = ap_ft[idx].to(device); at_target = at_ft[idx].to(device)

            direct_loss = F.cross_entropy(direct, direct_target)
            ap_med_loss = F.cross_entropy(ap_med, ap_med_target)
            ap_readout_loss = F.cross_entropy((generated_ap @ Q_ap) @ W_ap.T + b_ap, ap_target)
            at_med_loss = F.cross_entropy(at_med, at_med_target)
            at_readout_loss = F.cross_entropy((generated_at @ Q_at) @ W_at.T + b_at, at_target)

            loss = direct_loss + lambda_ap_med * ap_med_loss + lambda_ap_readout * ap_readout_loss + lambda_at_med * at_med_loss + lambda_at_readout * at_readout_loss
            loss.backward(); optimizer.step()

            n = len(idx); vals = [direct_loss, ap_med_loss, ap_readout_loss, at_med_loss, at_readout_loss]
            totals = [t + float(v.detach()) * n for t, v in zip(totals, vals)]; total_n += n

        with torch.no_grad():
            Q_eval = alignment.basis()
            metrics = evaluate_c(model, cal_bank, cal_source, Q_eval, c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, labels, token_position, eval_batch_size, tokenizer, output_space)

        direct_cal, ap_med_cal, ap_readout_cal, at_med_cal, at_readout_cal = metrics
        active = [direct_cal]
        if lambda_ap_med > 0: active.append(ap_med_cal)
        if lambda_ap_readout > 0: active.append(ap_readout_cal)
        if lambda_at_med > 0: active.append(at_med_cal)
        if lambda_at_readout > 0: active.append(at_readout_cal)
        score = min(active)

        dl, am, ar, tm, tr = [x / total_n for x in totals]
        print(f"[epoch {epoch:02d}] direct={dl:.4f} ap_med={am:.4f} ap_ro={ar:.4f} at_med={tm:.4f} at_ro={tr:.4f} | cal direct={direct_cal:.4f} ap_med={ap_med_cal:.4f} ap_ro={ap_readout_cal:.4f} at_med={at_med_cal:.4f} at_ro={at_readout_cal:.4f} score={score:.4f}")

        if score >= best_score:
            best_score = score; best_basis = Q_eval.detach().cpu().clone(); best_epoch = epoch; best_metrics = metrics

    test_metrics = evaluate_c(model, te_bank, te_source, best_basis.to(device), c_layer, Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, labels, token_position, eval_batch_size, tokenizer, output_space)
    direct_test, ap_med_test, ap_readout_test, at_med_test, at_readout_test = test_metrics
    direct_cal, ap_med_cal, ap_readout_cal, at_med_cal, at_readout_cal = best_metrics

    result = {
        "variable": "color", "method": "Gradual DAS: direct C + AP/AT mediator + AP/AT readout",
        "layer": c_layer, "subspace_dim": subspace_dim, "basis": best_basis, "token_position": token_position, "output_space": output_space,
        "layer_selection": {"method": "full_layer_interchange_cal_iia", "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]), "all_layers": layer_results},
        "cal": {"best_epoch": best_epoch, "direct_iia": direct_cal, "ap_mediator_iia": ap_med_cal, "ap_readout_accuracy": ap_readout_cal, "at_mediator_iia": at_med_cal, "at_readout_accuracy": at_readout_cal, "score": best_score},
        "test": {"direct_iia": direct_test, "ap_mediator_iia": ap_med_test, "ap_readout_accuracy": ap_readout_test, "at_mediator_iia": at_med_test, "at_readout_accuracy": at_readout_test},
        "config": {"ft_size": ft_size, "cal_size": cal_size, "te_size": te_size, "epochs": epochs, "lr": lr, "lambda_ap_med": lambda_ap_med, "lambda_ap_readout": lambda_ap_readout, "lambda_at_med": lambda_at_med, "lambda_at_readout": lambda_at_readout, "train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size, "seed": seed, "output_space": output_space},
        "downstream": {
            "AP": {"layer": ap_layer, "subspace_dim": int(Q_ap.shape[1]), "checkpoint": ap_checkpoint, "readout_checkpoint": ap_readout_checkpoint, "mediator_target": "answer_token_after_pointer_interchange"},
            "AT": {"layer": at_layer, "subspace_dim": int(Q_at.shape[1]), "checkpoint": at_checkpoint, "readout_checkpoint": at_readout_checkpoint, "mediator_target": "answer_token"},
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True); torch.save(result, save_path)
    print(f"[C BEST] layer={c_layer} epoch={best_epoch} direct={direct_cal:.4f} ap_med={ap_med_cal:.4f} ap_ro={ap_readout_cal:.4f} at_med={at_med_cal:.4f} at_ro={at_readout_cal:.4f}")
    print(f"[C TEST] direct={direct_test:.4f} ap_med={ap_med_test:.4f} ap_ro={ap_readout_test:.4f} at_med={at_med_test:.4f} at_ro={at_readout_test:.4f}")
    print(f"saved to: {save_path}")
    return result


def build_c_chain_banks(model, tokenizer, data_path, ft_size, cal_size, te_size, batch_size, seed):
    ds = load_generated_dataset(data_path); sizes = {"train": ft_size, "validation": cal_size, "test": te_size}; banks = {}; cache = {}
    for offset, split in enumerate(("train", "validation", "test")):
        ids = list(range(len(ds[split]))); random.Random(seed + offset).shuffle(ids); pairs = []
        for start in range(0, len(ids), 64):
            cand = rows_to_pairs(ds[split], ids[start:start+64], split)
            cand = [p for p in cand if p["pair_type"] == "C_AP_AT"]
            pairs += predict_correct(model, tokenizer, cand, batch_size, cache)
            if len(pairs) >= sizes[split]: break
        banks[split] = create_bank(pairs, tokenizer, sizes[split], seed + offset)
        print(f"[C chain bank] {split}: {len(banks[split]['pairs'])} C_AP_AT pairs")
    return banks["train"], {"color": banks["validation"]}, {"color": banks["test"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="randomized_copycolors")
    p.add_argument("--ap-checkpoint", default="results/das_ap.pt")
    p.add_argument("--ap-readout-checkpoint", default="results/ap_readout_LogisticRegression.pt")
    p.add_argument("--at-checkpoint", default="results/das_at.pt")
    p.add_argument("--at-readout-checkpoint", default="results/at_readout_LogisticRegression.pt")
    p.add_argument("--layers", default="all")
    p.add_argument("--subspace-dim", type=int, default=128)
    p.add_argument("--ft-size", type=int, default=800)
    p.add_argument("--cal-size", type=int, default=400)
    p.add_argument("--te-size", type=int, default=400)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda-ap-med", type=float, default=0.0)
    p.add_argument("--lambda-ap-readout", type=float, default=0.0)
    p.add_argument("--lambda-at-med", type=float, default=0.0)
    p.add_argument("--lambda-at-readout", type=float, default=0.0)
    p.add_argument("--train-batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--token-position", default="color")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-path", default="results/das_c.pt")
    p.add_argument("--output-space", choices=("full", "az"), default="full")
    a = p.parse_args()

    model, tokenizer = load_gemma_model()
    ls = None if a.layers == "all" else [int(x) for x in a.layers.split(",") if x.strip()]
    banks = build_c_chain_banks(model, tokenizer, a.data_path, a.ft_size, a.cal_size, a.te_size, a.eval_batch_size, a.seed)
    train_c(model, tokenizer, a.ap_checkpoint, a.ap_readout_checkpoint, a.at_checkpoint, a.at_readout_checkpoint, ls, a.subspace_dim, a.ft_size, a.cal_size, a.te_size, a.epochs, a.lr, a.lambda_ap_med, a.lambda_ap_readout, a.lambda_at_med, a.lambda_at_readout, a.train_batch_size, a.eval_batch_size, a.token_position, a.seed, a.save_path, banks, a.output_space)


if __name__ == "__main__": main()
