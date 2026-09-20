import argparse
import os

import torch
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import set_seed, answer_label_ids, answer_logits, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention


def run_ap_intervention_and_capture_at(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, label_ids, token_position="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device
    all_logits = []
    all_generated_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_ap_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
        captured = {}

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base = hidden[rows, pos, :].float()
            delta = source.float() - base
            hidden_new[rows, pos, :] = (base + ((delta @ Q_ap) @ Q_ap.T)).to(hidden.dtype)

            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["at"] = hidden[rows, pos, :].float()

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, label_ids)
            generated_at = captured["at"]

            if require_grad:
                all_logits.append(logits)
                all_generated_at.append(generated_at)
            else:
                all_logits.append(logits.detach().cpu())
                all_generated_at.append(generated_at.detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_logits, dim=0), torch.cat(all_generated_at, dim=0)


def run_frozen_at_mediator(model, bank, generated_at_states, Q_at, at_layer, label_ids, token_position="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        generated = generated_at_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            clean = hidden[rows, pos, :].float()
            delta = generated.float() - clean
            hidden_new[rows, pos, :] = (clean + ((delta @ Q_at) @ Q_at.T)).to(hidden.dtype)

            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        handle = model.model.layers[at_layer].register_forward_hook(at_hook)
        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, label_ids)
            all_logits.append(logits if require_grad else logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def evaluate_ap(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, Q_at, W_at, b_at, label_ids, token_position, batch_size):
    direct_logits, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap_states, Q_ap, ap_layer, at_layer, label_ids, token_position, batch_size, require_grad=False)
    mediator_logits = run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, label_ids, token_position, batch_size, require_grad=False)

    ap_target = bank["counterfactual_label_ids"]["answer_pointer"].cpu()
    direct_iia = float((direct_logits[:, :4].argmax(dim=-1) == ap_target).float().mean())
    mediator_iia = float((mediator_logits[:, :4].argmax(dim=-1) == ap_target).float().mean())

    generated_at = generated_at.to(Q_at.device)
    readout_logits = (generated_at @ Q_at) @ W_at.T + b_at
    at_target = bank["counterfactual_label_ids"]["answer_token"].to(readout_logits.device)
    readout_accuracy = float((readout_logits.argmax(dim=-1) == at_target).float().mean())

    return direct_iia, mediator_iia, readout_accuracy


def train_ap(model, tokenizer, at_checkpoint, at_readout_checkpoint, layers=None, subspace_dim=128, ft_size=None, cal_size=None, te_size=None, epochs=50, lr=1e-2, lambda_med=1.0, lambda_readout=1.0, train_batch_size=32, eval_batch_size=32, token_position=None, seed=0, save_path="results/gradual_das_ap_readout.pt"):
    set_seed(seed)
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)

    at_result = torch.load(at_checkpoint, map_location="cpu")
    at_layer = int(at_result["layer"])
    Q_at = at_result["basis"].float().to(device)

    readout_result = torch.load(at_readout_checkpoint, map_location="cpu")
    W_at = readout_result["W_AT"].float().to(device)
    b_at = readout_result["b_AT"].float().to(device)

    if token_position is None:
        token_position = at_result["token_position"]

    config = at_result["config"]
    ft_size = config["ft_size"] if ft_size is None else ft_size
    cal_size = config["cal_size"] if cal_size is None else cal_size
    te_size = config["te_size"] if te_size is None else te_size

    fit_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=None, split="train", device=device, batch_size=eval_batch_size, seed=seed)
    cal_bank = cal_banks["answer_pointer"]
    te_bank = te_banks["answer_pointer"]

    if layers is None:
        layers = list(range(at_layer))
    else:
        layers = [layer for layer in layers if layer < at_layer]

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)

    layer_results = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, labels, token_position, eval_batch_size)
        pred = logits[:, :4].argmax(dim=-1)
        target = cal_bank["counterfactual_label_ids"]["answer_pointer"].cpu()
        iia = float((pred == target).float().mean())

        layer_results.append({"layer": int(layer), "cal_iia": iia})
        print(f"[AP layer] layer={layer} iia={iia:.4f}")

    best_layer_row = max(layer_results, key=lambda row: row["cal_iia"])
    ap_layer = int(best_layer_row["layer"])
    print(f"[AP layer selected] layer={ap_layer} iia={best_layer_row['cal_iia']:.4f}")

    ft_source_ap = collect_layer_states(model, fit_bank["source_input_ids"], fit_bank["source_attention_mask"], fit_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)
    cal_source_ap = cal_source_by_layer[ap_layer]
    te_source_ap = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], ap_layer, token_position, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ap_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]
    target_at_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_score = -1.0
    best_basis = None
    best_epoch = None
    best_direct_cal = None
    best_mediator_cal = None
    best_readout_cal = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"]))
        total_direct_loss = 0.0
        total_med_loss = 0.0
        total_readout_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini_bank = {
                "base_input_ids": fit_bank["base_input_ids"][idx],
                "base_attention_mask": fit_bank["base_attention_mask"][idx],
                "base_position_by_id": {k: v[idx] for k, v in fit_bank["base_position_by_id"].items()}
            }

            optimizer.zero_grad()
            Q_ap = alignment.basis()

            direct_logits, generated_at = run_ap_intervention_and_capture_at(model, mini_bank, ft_source_ap[idx], Q_ap, ap_layer, at_layer, labels, token_position, len(idx), require_grad=True)
            mediator_logits = run_frozen_at_mediator(model, mini_bank, generated_at, Q_at, at_layer, labels, token_position, len(idx), require_grad=True)

            target_ap = target_ap_ft[idx].to(device)
            target_at = target_at_ft[idx].to(device)

            direct_loss = F.cross_entropy(direct_logits[:, :4], target_ap)
            mediator_loss = F.cross_entropy(mediator_logits[:, :4], target_ap)
            readout_logits = (generated_at @ Q_at) @ W_at.T + b_at
            readout_loss = F.cross_entropy(readout_logits, target_at)
            loss = direct_loss + lambda_med * mediator_loss + lambda_readout * readout_loss

            loss.backward()
            optimizer.step()

            n = len(idx)
            total_direct_loss += direct_loss.item() * n
            total_med_loss += mediator_loss.item() * n
            total_readout_loss += readout_loss.item() * n
            total_count += n

        with torch.no_grad():
            Q_eval = alignment.basis()
            direct_cal, mediator_cal, readout_cal = evaluate_ap(model, cal_bank, cal_source_ap, Q_eval, ap_layer, at_layer, Q_at, W_at, b_at, labels, token_position, eval_batch_size)

        score = direct_cal
        if lambda_med > 0:
            score = min(score, mediator_cal)
        if lambda_readout > 0:
            score = min(score, readout_cal)

        print(f"[epoch {epoch:02d}] direct_loss={total_direct_loss / total_count:.4f} med_loss={total_med_loss / total_count:.4f} readout_loss={total_readout_loss / total_count:.4f} direct_cal={direct_cal:.4f} mediator_cal={mediator_cal:.4f} readout_cal={readout_cal:.4f} score={score:.4f}")

        if score > best_score:
            best_score = score
            best_basis = Q_eval.detach().cpu().clone()
            best_epoch = epoch
            best_direct_cal = direct_cal
            best_mediator_cal = mediator_cal
            best_readout_cal = readout_cal

    Q_ap = best_basis.to(device)
    direct_test, mediator_test, readout_test = evaluate_ap(model, te_bank, te_source_ap, Q_ap, ap_layer, at_layer, Q_at, W_at, b_at, labels, token_position, eval_batch_size)

    result = {
        "variable": "answer_pointer",
        "method": "Gradual DAS: direct AP + AT mediator + AT readout",
        "layer": ap_layer,
        "subspace_dim": subspace_dim,
        "basis": best_basis,
        "token_position": token_position,
        "layer_selection": {
            "method": "full_layer_interchange_cal_iia",
            "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]),
            "all_layers": layer_results
        },
        "cal": {
            "best_epoch": best_epoch,
            "direct_iia": best_direct_cal,
            "mediator_iia": best_mediator_cal,
            "readout_accuracy": best_readout_cal,
            "score": best_score
        },
        "test": {
            "direct_iia": direct_test,
            "mediator_iia": mediator_test,
            "readout_accuracy": readout_test
        },
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "epochs": epochs,
            "lr": lr,
            "lambda_med": lambda_med,
            "lambda_readout": lambda_readout,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "seed": seed
        },

        "mediator": {
            "variable": "answer_token",
            "layer": at_layer,
            "subspace_dim": int(Q_at.shape[1]),
            "checkpoint": at_checkpoint,
            "readout_checkpoint": at_readout_checkpoint
        }
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)

    print(f"[AP BEST] layer={ap_layer} epoch={best_epoch} direct_cal={best_direct_cal:.4f} mediator_cal={best_mediator_cal:.4f} readout_cal={best_readout_cal:.4f}")
    print(f"[AP TEST] direct_iia={direct_test:.4f} mediator_iia={mediator_test:.4f} readout_accuracy={readout_test:.4f}")
    print(f"saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-checkpoint", default="results/das_at.pt")
    parser.add_argument("--at-readout-checkpoint", default="results/at_readout_LogisticRegression.pt")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--subspace-dim", type=int, default=128)
    parser.add_argument("--ft-size", type=int, default=None)
    parser.add_argument("--cal-size", type=int, default=None)
    parser.add_argument("--te-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=1.0)
    parser.add_argument("--lambda-readout", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--token-position", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="results/gradual_das_ap_readout.pt")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",") if x.strip()]
    train_ap(model, tokenizer, args.at_checkpoint, args.at_readout_checkpoint, layers, args.subspace_dim, args.ft_size, args.cal_size, args.te_size, args.epochs, args.lr, args.lambda_med, args.lambda_readout, args.train_batch_size, args.eval_batch_size, args.token_position, args.seed, args.save_path)


if __name__ == "__main__":
    main()
