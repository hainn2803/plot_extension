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


def logits_for_variable(logits, var_name):
    if var_name == "answer_pointer":
        return logits[:, :4]
    if var_name == "answer_token":
        return logits
    raise ValueError(var_name)


def iia_from_logits(logits, target, var_name):
    scores = logits_for_variable(logits, var_name)
    pred = scores.argmax(dim=-1).cpu()
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    return float((pred == target).float().mean().item())


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
def evaluate_full_layer_ap_iia(
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
    return iia_from_logits(
        logits,
        bank["counterfactual_label_ids"]["answer_pointer"],
        "answer_pointer",
    )


def run_ap_intervention_and_capture_at(
    model,
    bank,
    source_ap_states,
    Q_ap,
    ap_layer,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=8,
    require_grad=True,
):
    """
    Patch the learned AP subspace at L_AP, continue the model, and capture the
    resulting state at L_AT. No detach is used on the captured AT state when
    require_grad=True, so mediator gradients can flow back into Q_AP.
    """
    device = next(model.parameters()).device
    all_logits = []
    all_generated_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_ap_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        captured = {}

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base = hidden[rows, pos, :].float()
            delta = source - base
            patched = base + ((delta @ Q_ap) @ Q_ap.T)

            hidden_new[rows, pos, :] = patched.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        def at_capture_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["at"] = hidden[rows, pos, :].float()

        h_ap = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        h_at = model.model.layers[at_layer].register_forward_hook(at_capture_hook)

        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            logits = answer_logits(model, outputs, mask, label_ids)
            generated_at = captured["at"]

            if require_grad:
                all_logits.append(logits)
                all_generated_at.append(generated_at)
            else:
                all_logits.append(logits.detach().cpu())
                all_generated_at.append(generated_at.detach().cpu())
        finally:
            h_ap.remove()
            h_at.remove()

    return torch.cat(all_logits, dim=0), torch.cat(all_generated_at, dim=0)


def run_frozen_at_mediator(
    model,
    bank,
    generated_at_states,
    Q_at,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=8,
    require_grad=True,
):
    """
    Run the clean base trajectory, but at L_AT replace only the frozen AT
    coordinates with the coordinates produced by the AP-intervened trajectory.

        h_clean -> h_clean + ((h_generated - h_clean) Q_AT) Q_AT^T

    Q_AT is frozen, but generated_at_states must NOT be detached during AP
    training, so gradients flow back through the AP trajectory into Q_AP.
    """
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        generated = generated_at_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            clean = hidden[rows, pos, :].float()
            delta = generated - clean
            patched = clean + ((delta @ Q_at) @ Q_at.T)

            hidden_new[rows, pos, :] = patched.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        handle = model.model.layers[at_layer].register_forward_hook(at_hook)
        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            logits = answer_logits(model, outputs, mask, label_ids)
            all_logits.append(logits if require_grad else logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def evaluate_ap(
    model,
    bank,
    source_ap_states,
    Q_ap,
    ap_layer,
    Q_at,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    direct_logits, generated_at = run_ap_intervention_and_capture_at(
        model=model,
        bank=bank,
        source_ap_states=source_ap_states,
        Q_ap=Q_ap,
        ap_layer=ap_layer,
        at_layer=at_layer,
        label_ids=label_ids,
        token_position=token_position,
        batch_size=batch_size,
        require_grad=False,
    )

    mediator_logits = run_frozen_at_mediator(
        model=model,
        bank=bank,
        generated_at_states=generated_at,
        Q_at=Q_at,
        at_layer=at_layer,
        label_ids=label_ids,
        token_position=token_position,
        batch_size=batch_size,
        require_grad=False,
    )

    target = bank["counterfactual_label_ids"]["answer_pointer"]
    direct_iia = iia_from_logits(direct_logits, target, "answer_pointer")
    mediator_iia = iia_from_logits(mediator_logits, target, "answer_pointer")
    return direct_iia, mediator_iia


def make_mini_bank(bank, idx):
    return {
        "base_input_ids": bank["base_input_ids"][idx],
        "base_attention_mask": bank["base_attention_mask"][idx],
        "base_position_by_id": {
            k: v[idx] for k, v in bank["base_position_by_id"].items()
        },
        "counterfactual_label_ids": {
            "answer_pointer": bank["counterfactual_label_ids"]["answer_pointer"][idx],
        },
    }


def train_phase2_ap(
    model,
    tokenizer,
    at_checkpoint,
    layers=None,
    subspace_dim=128,
    ft_size=None,
    cal_size=None,
    te_size=None,
    epochs=10,
    lr=1e-2,
    lambda_med=1.0,
    train_batch_size=8,
    eval_batch_size=16,
    token_position=None,
    seed=0,
    save_path="results/gradual_das_ap.pt",
):
    set_seed(seed)
    device = next(model.parameters()).device

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    labels = answer_label_ids(tokenizer)

    # load and freeze the validated AT alignment from phase 1
    at_result = torch.load(at_checkpoint, map_location="cpu")
    at_layer = int(at_result["layer"])
    Q_at = at_result["basis"].float().to(device)
    Q_at.requires_grad_(False)

    if token_position is None:
        token_position = at_result.get("token_position", "last_token")

    phase1_config = at_result.get("config", {})
    if ft_size is None:
        ft_size = int(phase1_config.get("ft_size", 400))
    if cal_size is None:
        cal_size = int(phase1_config.get("cal_size", 200))
    if te_size is None:
        te_size = int(phase1_config.get("te_size", 200))

    print(
        f"[frozen AT] layer={at_layer} k={Q_at.shape[1]} "
        f"cal_iia={at_result.get('cal_iia', float('nan')):.4f} "
        f"test_iia={at_result.get('test_iia', float('nan')):.4f}"
    )

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

    cal_bank = cal_banks["answer_pointer"]
    te_bank = te_banks["answer_pointer"]

    print(
        f"[banks] FT={len(fit_bank['base_input_ids'])} "
        f"CAL_AP={len(cal_bank['base_input_ids'])} "
        f"TEST_AP={len(te_bank['base_input_ids'])}"
    )

    # select L_AP by full-layer AP interchange IIA
    if layers is None:
        layers = list(range(model.config.num_hidden_layers))
    else:
        layers = [int(layer) for layer in layers]

    print(f"[AP layer search] scanning {len(layers)} layers on AP CAL")

    cal_source_by_layer = collect_all_layer_states(
        model=model,
        input_ids=cal_bank["source_input_ids"],
        attention_mask=cal_bank["source_attention_mask"],
        position_by_id=cal_bank["source_position_by_id"],
        layers=layers,
        token_position=token_position,
        batch_size=eval_batch_size,
    )

    layer_results = []
    for candidate_layer in layers:
        iia = evaluate_full_layer_ap_iia(
            model=model,
            bank=cal_bank,
            source_states=cal_source_by_layer[candidate_layer],
            layer=candidate_layer,
            label_ids=labels,
            token_position=token_position,
            batch_size=eval_batch_size,
        )
        row = {"layer": int(candidate_layer), "cal_iia": float(iia)}
        layer_results.append(row)
        print(f"[AP full-layer CAL] layer={candidate_layer} iia={iia:.4f}")

    best_layer_row = max(layer_results, key=lambda row: row["cal_iia"])
    ap_layer = int(best_layer_row["layer"])

    print(
        f"[AP LAYER SELECTED] layer={ap_layer} "
        f"full_layer_cal_iia={best_layer_row['cal_iia']:.4f}"
    )

    if ap_layer >= at_layer:
        raise ValueError(
            f"Selected AP layer {ap_layer} is not upstream of frozen AT layer "
            f"{at_layer}. The gradual AP->AT objective requires L_AP < L_AT. "
            f"This result is evidence against the assumed layer ordering; "
            f"inspect the full-layer AP scan instead of forcing the mediator."
        )

    # freeze L_AP and collect source states only at that layer
    ft_source_ap = collect_layer_states(
        model,
        fit_bank["source_input_ids"],
        fit_bank["source_attention_mask"],
        fit_bank["source_position_by_id"],
        ap_layer,
        token_position,
        batch_size=eval_batch_size,
    )
    cal_source_ap = cal_source_by_layer[ap_layer]
    te_source_ap = collect_layer_states(
        model,
        te_bank["source_input_ids"],
        te_bank["source_attention_mask"],
        te_bank["source_position_by_id"],
        ap_layer,
        token_position,
        batch_size=eval_batch_size,
    )

    del cal_source_by_layer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    hidden_size = int(model.config.hidden_size)
    alignment = LearnedSubspace(hidden_size, subspace_dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=lr)

    target_ft = fit_bank["counterfactual_label_ids"]["answer_pointer"]

    best_score = -1.0
    best_direct_cal = -1.0
    best_mediator_cal = -1.0
    best_basis = None
    best_epoch = None

    # Gradual DAS training
    # only Q_AP is learned. Model and Q_AT stay frozen
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(fit_bank["base_input_ids"]))
        total_loss = 0.0
        total_direct_loss = 0.0
        total_med_loss = 0.0
        total_count = 0

        for start in range(0, len(perm), train_batch_size):
            idx = perm[start:start + train_batch_size]
            mini_bank = make_mini_bank(fit_bank, idx)
            mini_source_ap = ft_source_ap[idx]
            target = target_ft[idx].to(device)

            optimizer.zero_grad()
            Q_ap = alignment.basis()

            direct_logits, generated_at = run_ap_intervention_and_capture_at(
                model=model,
                bank=mini_bank,
                source_ap_states=mini_source_ap,
                Q_ap=Q_ap,
                ap_layer=ap_layer,
                at_layer=at_layer,
                label_ids=labels,
                token_position=token_position,
                batch_size=len(idx),
                require_grad=True,
            )

            mediator_logits = run_frozen_at_mediator(
                model=model,
                bank=mini_bank,
                generated_at_states=generated_at,
                Q_at=Q_at,
                at_layer=at_layer,
                label_ids=labels,
                token_position=token_position,
                batch_size=len(idx),
                require_grad=True,
            )

            direct_loss = F.cross_entropy(
                logits_for_variable(direct_logits, "answer_pointer"),
                target,
            )
            mediator_loss = F.cross_entropy(
                logits_for_variable(mediator_logits, "answer_pointer"),
                target,
            )
            loss = direct_loss + float(lambda_med) * mediator_loss

            loss.backward()
            optimizer.step()

            n = len(idx)
            total_loss += float(loss.item()) * n
            total_direct_loss += float(direct_loss.item()) * n
            total_med_loss += float(mediator_loss.item()) * n
            total_count += n

        with torch.no_grad():
            Q_eval = alignment.basis()
            direct_cal_iia, mediator_cal_iia = evaluate_ap(
                model=model,
                bank=cal_bank,
                source_ap_states=cal_source_ap,
                Q_ap=Q_eval,
                ap_layer=ap_layer,
                Q_at=Q_at,
                at_layer=at_layer,
                label_ids=labels,
                token_position=token_position,
                batch_size=eval_batch_size,
            )

        if float(lambda_med) == 0.0:
            selection_score = direct_cal_iia
        else:
            selection_score = min(direct_cal_iia, mediator_cal_iia)
        denom = max(total_count, 1)

        print(
            f"[epoch {epoch:02d}] "
            f"loss={total_loss / denom:.4f} "
            f"direct_loss={total_direct_loss / denom:.4f} "
            f"med_loss={total_med_loss / denom:.4f} "
            f"direct_cal_iia={direct_cal_iia:.4f} "
            f"mediator_cal_iia={mediator_cal_iia:.4f} "
            f"score={selection_score:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            best_direct_cal = direct_cal_iia
            best_mediator_cal = mediator_cal_iia
            best_basis = Q_eval.detach().cpu().clone()
            best_epoch = int(epoch)

    # freeze CAL-selected Q_AP and touch AP TEST once.
    Q_ap_best = best_basis.to(device)
    direct_test_iia, mediator_test_iia = evaluate_ap(
        model=model,
        bank=te_bank,
        source_ap_states=te_source_ap,
        Q_ap=Q_ap_best,
        ap_layer=ap_layer,
        Q_at=Q_at,
        at_layer=at_layer,
        label_ids=labels,
        token_position=token_position,
        batch_size=eval_batch_size,
    )

    result = {
        "variable": "answer_pointer",
        "method": (
            "Standard DAS: direct AP only"
            if float(lambda_med) == 0.0
            else "Gradual DAS: direct AP + frozen AT mediator"
        ),
        "ap_layer": int(ap_layer),
        "at_layer": int(at_layer),
        "ap_subspace_dim": int(subspace_dim),
        "at_subspace_dim": int(Q_at.shape[1]),
        "ap_basis": best_basis,
        "at_checkpoint": at_checkpoint,
        "lambda_med": float(lambda_med),
        "layer_selection": {
            "method": "full_layer_interchange_ap_cal_iia",
            "selected_full_layer_cal_iia": float(best_layer_row["cal_iia"]),
            "all_layers": layer_results,
        },
        "cal": {
            "best_epoch": best_epoch,
            "direct_iia": float(best_direct_cal),
            "mediator_iia": float(best_mediator_cal),
            "selection_score_min": float(best_score),
        },
        "test": {
            "direct_iia": float(direct_test_iia),
            "mediator_iia": float(mediator_test_iia),
        },
        "token_position": token_position,
        "config": {
            "ft_size": int(ft_size),
            "cal_size": int(cal_size),
            "te_size": int(te_size),
            "epochs": int(epochs),
            "lr": float(lr),
            "lambda_med": float(lambda_med),
            "train_batch_size": int(train_batch_size),
            "eval_batch_size": int(eval_batch_size),
            "seed": int(seed),
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)

    print()
    print(
        f"[AP BEST] L_AP={ap_layer} L_AT={at_layer} k_AP={subspace_dim} "
        f"epoch={best_epoch} direct_cal={best_direct_cal:.4f} "
        f"mediator_cal={best_mediator_cal:.4f} score={best_score:.4f}"
    )
    print(
        f"[AP TEST] direct_iia={direct_test_iia:.4f} "
        f"mediator_iia={mediator_test_iia:.4f}"
    )
    print(f"saved to: {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--at-checkpoint",
        default="results/gradual_das_at_layer_selected.pt",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--subspace-dim", type=int, default=128)
    parser.add_argument("--ft-size", type=int, default=None)
    parser.add_argument("--cal-size", type=int, default=None)
    parser.add_argument("--te-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda-med", type=float, default=0.00)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--token-position", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--save-path",
        default="results/standard_das_ap.pt",
    )
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    print(next(model.parameters()).device)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    train_phase2_ap(
        model=model,
        tokenizer=tokenizer,
        at_checkpoint=args.at_checkpoint,
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
        lambda_med=args.lambda_med,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.token_position,
        seed=args.seed,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()