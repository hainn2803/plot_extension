import argparse
import os

import torch
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import set_seed, answer_label_ids, collect_layer_states, collect_all_layer_states, LearnedSubspace, run_full_layer_intervention, run_das_intervention


@torch.no_grad()
def evaluate_at_iia(model, bank, source_states, Q, layer, label_ids, token_position, batch_size):
    logits = run_das_intervention(model, bank, source_states, Q, layer, label_ids, token_position, batch_size, require_grad=False)
    pred = logits.argmax(dim=-1)
    target = bank["counterfactual_label_ids"]["answer_token"].cpu()
    return float((pred == target).float().mean())


def train_at(model, tokenizer, layers=None, subspace_dim=128, ft_size=400, cal_size=200, te_size=200, epochs=50, lr=1e-2, train_batch_size=32, eval_batch_size=32, token_position="last_token", seed=0, save_path="results/gradual_das_at_layer_selected.pt"):
    set_seed(seed)
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)

    fit_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=None, split="train", device=device, batch_size=eval_batch_size, seed=seed)
    cal_bank = cal_banks["answer_token"]
    te_bank = te_banks["answer_token"]

    if layers is None:
        layers = list(range(model.config.num_hidden_layers))

    cal_source_by_layer = collect_all_layer_states(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], layers, token_position, eval_batch_size)

    layer_results = []
    for layer in layers:
        logits = run_full_layer_intervention(model, cal_bank, cal_source_by_layer[layer], layer, labels, token_position, eval_batch_size)
        pred = logits.argmax(dim=-1)
        target = cal_bank["counterfactual_label_ids"]["answer_token"].cpu()
        iia = float((pred == target).float().mean())

        layer_results.append({"layer": int(layer), "cal_iia": iia})
        print(f"[AT layer] layer={layer} iia={iia:.4f}")

    best_layer_row = max(layer_results, key=lambda row: row["cal_iia"])
    layer = int(best_layer_row["layer"])
    print(f"[AT layer selected] layer={layer} iia={best_layer_row['cal_iia']:.4f}")

    ft_source = collect_layer_states(model, fit_bank["source_input_ids"], fit_bank["source_attention_mask"], fit_bank["source_position_by_id"], layer, token_position, eval_batch_size)
    cal_source = cal_source_by_layer[layer]
    te_source = collect_layer_states(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], layer, token_position, eval_batch_size)

    alignment = LearnedSubspace(model.config.hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)
    target_ft = fit_bank["counterfactual_label_ids"]["answer_token"]

    best_cal = -1.0
    best_basis = None
    best_epoch = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"]))
        total_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini_bank = {
                "base_input_ids": fit_bank["base_input_ids"][idx],
                "base_attention_mask": fit_bank["base_attention_mask"][idx],
                "base_position_by_id": {k: v[idx] for k, v in fit_bank["base_position_by_id"].items()}
            }

            optimizer.zero_grad()
            Q = alignment.basis()
            logits = run_das_intervention(model, mini_bank, ft_source[idx], Q, layer, labels, token_position, len(idx), require_grad=True)
            loss = F.cross_entropy(logits, target_ft[idx].to(device))

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            total_count += len(idx)

        with torch.no_grad():
            Q_eval = alignment.basis()
            cal_iia = evaluate_at_iia(model, cal_bank, cal_source, Q_eval, layer, labels, token_position, eval_batch_size)

        print(f"[epoch {epoch:02d}] loss={total_loss / total_count:.4f} cal_iia={cal_iia:.4f}")

        if cal_iia > best_cal:
            best_cal = cal_iia
            best_basis = Q_eval.detach().cpu().clone()
            best_epoch = epoch

    test_iia = evaluate_at_iia(model, te_bank, te_source, best_basis.to(device), layer, labels, token_position, eval_batch_size)

    result = {
        "variable": "answer_token",
        "method": "DAS learned orthonormal subspace",
        "layer": layer,
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
            "direct_iia": best_cal
        },
        "test": {
            "direct_iia": test_iia
        },
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "epochs": epochs,
            "lr": lr,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "seed": seed
        }
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)

    print(f"[AT BEST] layer={layer} k={subspace_dim} epoch={best_epoch} cal_iia={best_cal:.4f} test_iia={test_iia:.4f}")
    print(f"saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="all")
    parser.add_argument("--subspace-dim", type=int, default=128)
    parser.add_argument("--ft-size", type=int, default=400)
    parser.add_argument("--cal-size", type=int, default=200)
    parser.add_argument("--te-size", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--token-position", default="last_token")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", default="results/das_at.pt")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",") if x.strip()]
    train_at(model, tokenizer, layers, args.subspace_dim, args.ft_size, args.cal_size, args.te_size, args.epochs, args.lr, args.train_batch_size, args.eval_batch_size, args.token_position, args.seed, args.save_path)


if __name__ == "__main__":
    main()
