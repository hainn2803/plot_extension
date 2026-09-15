import argparse
import os
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks, letter_token_id

from mcqa_neural_net import load_gemma_model


ANSWER_LETTERS = tuple(string.ascii_uppercase)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def answer_label_ids(tokenizer):
    return [letter_token_id(tokenizer, letter) for letter in ANSWER_LETTERS]


def answer_logits(model, outputs, attention_mask, label_ids):
    device = outputs.last_hidden_state.device
    rows = torch.arange(attention_mask.shape[0], device=device)
    idx = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask * idx.unsqueeze(0)).max(dim=1).values

    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, device=device, dtype=torch.long)
    W = model.lm_head.weight[ids].to(hidden.dtype)
    logits = hidden @ W.T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[ids]

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits.float()


@torch.no_grad()
def collect_layer_states(
    model,
    input_ids,
    attention_mask,
    position_by_id,
    layer,
    token_position="last_token",
    batch_size=32,
):
    device = next(model.parameters()).device
    states = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = position_by_id[token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["x"] = hidden[rows, pos, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
        finally:
            handle.remove()

        states.append(captured["x"])

    return torch.cat(states, dim=0)


@torch.no_grad()
def collect_all_layer_states(
    model,
    input_ids,
    attention_mask,
    position_by_id,
    layers,
    token_position="last_token",
    batch_size=32,
):
    device = next(model.parameters()).device
    layers = [int(layer) for layer in layers]
    collected = {layer: [] for layer in layers}

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = position_by_id[token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        handles = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                collected[layer].append(
                    hidden[rows, pos, :].detach().float().cpu()
                )
            return hook

        for layer in layers:
            handles.append(
                model.model.layers[layer].register_forward_hook(make_hook(layer))
            )

        try:
            model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

    return {layer: torch.cat(parts, dim=0) for layer, parts in collected.items()}


class LearnedSubspace(nn.Module):
    """
    Learn a k-dimensional orthonormal basis Q in R^hidden_size.

    DAS interchange in this subspace is
        h' = h + ((h_source - h) @ Q) @ Q^T

    This is equivalent to learning a rotation, swapping the k aligned
    coordinates, and rotating back, but avoids storing an H x H matrix.
    """

    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        init = torch.randn(hidden_size, subspace_dim, dtype=torch.float32)
        init = torch.linalg.qr(init, mode="reduced").Q
        self.raw = nn.Parameter(init)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


@torch.no_grad()
def run_full_layer_intervention(
    model,
    bank,
    source_states,
    layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    """Replace the entire hidden vector at one layer/token with source."""
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device=device)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            hidden_new[rows, pos, :] = source.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def evaluate_full_layer_iia(
    model,
    bank,
    source_states,
    layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    logits = run_full_layer_intervention(
        model=model,
        bank=bank,
        source_states=source_states,
        layer=layer,
        label_ids=label_ids,
        token_position=token_position,
        batch_size=batch_size,
    )
    pred = logits.argmax(dim=-1)
    target = bank["counterfactual_label_ids"]["answer_token"].cpu()
    return float((pred == target).float().mean().item())


def run_das_intervention(
    model,
    bank,
    source_states,
    Q,
    layer,
    label_ids,
    token_position="last_token",
    batch_size=8,
    require_grad=True,
):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base = hidden[rows, pos, :].float()
            delta = source - base
            delta_subspace = (delta @ Q) @ Q.T
            patched = base + delta_subspace

            hidden_new[rows, pos, :] = patched.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            logits = answer_logits(model, outputs, mask, label_ids)
            all_logits.append(logits if require_grad else logits.detach())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def evaluate_iia(
    model,
    bank,
    source_states,
    Q,
    layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    logits = run_das_intervention(
        model=model,
        bank=bank,
        source_states=source_states,
        Q=Q,
        layer=layer,
        label_ids=label_ids,
        token_position=token_position,
        batch_size=batch_size,
        require_grad=False,
    )
    pred = logits.argmax(dim=-1).cpu()
    target = bank["counterfactual_label_ids"]["answer_token"].cpu()
    return float((pred == target).float().mean().item())


def train_phase1_at(
    model,
    tokenizer,
    layers=None,
    subspace_dim=128,
    ft_size=400,
    cal_size=200,
    te_size=200,
    epochs=10,
    lr=1e-2,
    train_batch_size=8,
    eval_batch_size=16,
    token_position="last_token",
    seed=0,
    save_path="results/gradual_das_at.pt",
):
    set_seed(seed)
    device = next(model.parameters()).device

    # Frozen language model. Gradients only update the learned AT subspace.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)


    fit_bank, cal_banks, te_banks = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        dataset_size=None,
        split="train",
        device=device,
        batch_size=eval_batch_size,
        seed=seed,
    )

    cal_bank = cal_banks["answer_token"]
    te_bank = te_banks["answer_token"]

    print(f"[banks] FT={len(fit_bank['base_input_ids'])} CAL={len(cal_bank['base_input_ids'])} TEST={len(te_bank['base_input_ids'])}")

    # ------------------------------------------------------------
    # Stage 0: select the AT layer by full-layer interchange.
    # ------------------------------------------------------------
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))
    else:
        layers = [int(layer) for layer in layers]

    print(f"[AT layer search] scanning {len(layers)} layers on CAL")
    cal_source_by_layer = collect_all_layer_states(
        model,
        cal_bank["source_input_ids"],
        cal_bank["source_attention_mask"],
        cal_bank["source_position_by_id"],
        layers,
        token_position=token_position,
        batch_size=eval_batch_size,
    )

    layer_results = []
    for candidate_layer in layers:
        iia = evaluate_full_layer_iia(
            model=model,
            bank=cal_bank,
            source_states=cal_source_by_layer[candidate_layer],
            layer=candidate_layer,
            label_ids=labels,
            token_position=token_position,
            batch_size=eval_batch_size,
        )
        layer_results.append({"layer": int(candidate_layer), "cal_iia": float(iia)})
        print(f"[AT full-layer CAL] layer={candidate_layer} iia={iia:.4f}")

    best_layer_row = max(layer_results, key=lambda row: row["cal_iia"])
    layer = int(best_layer_row["layer"])
    print(f"[AT LAYER SELECTED] layer={layer} full_layer_cal_iia={best_layer_row['cal_iia']:.4f}")

    # freeze that layer choice and collect source states only there
    ft_source = collect_layer_states(
        model,
        fit_bank["source_input_ids"],
        fit_bank["source_attention_mask"],
        fit_bank["source_position_by_id"],
        layer,
        token_position,
        batch_size=eval_batch_size,
    )
    cal_source = cal_source_by_layer[layer]
    te_source = collect_layer_states(
        model,
        te_bank["source_input_ids"],
        te_bank["source_attention_mask"],
        te_bank["source_position_by_id"],
        layer,
        token_position,
        batch_size=eval_batch_size,
    )

    del cal_source_by_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hidden_size = model.config.hidden_size
    alignment = LearnedSubspace(hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ft = fit_bank["counterfactual_label_ids"]["answer_token"]
    best_cal = -1.0
    best_basis = None

    # DAS training, optimize only Q using counterfactual AT output loss
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"]))
        total_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]

            mini_bank = {
                "base_input_ids": fit_bank["base_input_ids"][idx],
                "base_attention_mask": fit_bank["base_attention_mask"][idx],
                "base_position_by_id": {
                    k: v[idx] for k, v in fit_bank["base_position_by_id"].items()
                },
                "counterfactual_label_ids": {
                    "answer_token": target_ft[idx]
                },
            }
            mini_source = ft_source[idx]

            optimizer.zero_grad()
            Q = alignment.basis()

            logits = run_das_intervention(
                model=model,
                bank=mini_bank,
                source_states=mini_source,
                Q=Q,
                layer=layer,
                label_ids=labels,
                token_position=token_position,
                batch_size=len(idx),
                require_grad=True,
            )

            target = target_ft[idx].to(device)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(idx)
            total_count += len(idx)

        with torch.no_grad():
            Q_eval = alignment.basis()
            cal_iia = evaluate_iia(
                model,
                cal_bank,
                cal_source,
                Q_eval,
                layer,
                labels,
                token_position,
                eval_batch_size,
            )

        mean_loss = total_loss / max(total_count, 1)
        print(f"[epoch {epoch:02d}] loss={mean_loss:.4f} cal_iia={cal_iia:.4f}")

        if cal_iia > best_cal:
            best_cal = cal_iia
            best_basis = Q_eval.detach().cpu().clone()

    # freeze the CAL-selected AT alignment, then touch TEST once
    best_basis_device = best_basis.to(device)
    test_iia = evaluate_iia(
        model,
        te_bank,
        te_source,
        best_basis_device,
        layer,
        labels,
        token_position,
        eval_batch_size,
    )

    result = {
        "variable": "answer_token",
        "method": "DAS learned orthonormal subspace",
        "layer": int(layer),
        "layer_selection": {
            "method": "full_layer_interchange_cal_iia",
            "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]),
            "all_layers": layer_results,
        },
        "token_position": token_position,
        "hidden_size": int(hidden_size),
        "subspace_dim": int(subspace_dim),
        "basis": best_basis,
        "cal_iia": float(best_cal),
        "test_iia": float(test_iia),
        "answer_label_ids": labels,
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "epochs": epochs,
            "lr": lr,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "seed": seed,
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)

    print()
    print(
        f"[AT BEST] layer={layer} k={subspace_dim} "
        f"cal_iia={best_cal:.4f} test_iia={test_iia:.4f}"
    )
    print(f"saved to: {save_path}")

    return result


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
    parser.add_argument("--save-path", default="results/gradual_das_at_layer_selected.pt")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    print(next(model.parameters()).device)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    train_phase1_at(
        model=model,
        tokenizer=tokenizer,
        layers=(
            None
            if args.layers == "all"
            else [int(x) for x in args.layers.split(",") if x.strip()]
        ),
        subspace_dim=args.subspace_dim,
        ft_size=args.ft_size,
        cal_size=args.cal_size,
        te_size=args.te_size,
        epochs=args.epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.token_position,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()