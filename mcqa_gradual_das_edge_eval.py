import argparse
import os
import random
import string

import numpy as np
import torch

from mcqa_data_load_all import build_mcqa_banks, letter_token_id
from mcqa_neural_net import load_gemma_model


ANSWER_LETTERS = tuple(string.ascii_uppercase)
EPS = 1e-8


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


def answer_probs(logits):
    return torch.softmax(logits.float(), dim=-1)


def ap_iia(logits, target):
    # AP is a 4-way variable: position 0/1/2/3 corresponds to A/B/C/D.
    pred = logits[:, :4].argmax(dim=-1).cpu()
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    return float((pred == target).float().mean().item())


def full_answer_accuracy(logits, target):
    pred = logits.argmax(dim=-1).cpu()
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
    batch_size=16,
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
def run_clean(model, bank, label_ids, batch_size=16):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        outputs = model.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
            use_cache=False,
            return_dict=True,
        )
        all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def run_ap_patch_capture_at(
    model,
    bank,
    source_ap_states,
    Q_ap,
    ap_layer,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    """
    AP intervention at L_AP, then capture the resulting state at L_AT.

        h_AP' = h_AP + ((h_AP_source - h_AP) Q_AP) Q_AP^T
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
            patched = base + (((source - base) @ Q_ap) @ Q_ap.T)
            hidden_new[rows, pos, :] = patched.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        def at_capture_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["at"] = hidden[rows, pos, :].detach().float().cpu()

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
            all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())
            all_generated_at.append(captured["at"])
        finally:
            h_at.remove()
            h_ap.remove()

    return torch.cat(all_logits, dim=0), torch.cat(all_generated_at, dim=0)


@torch.no_grad()
def run_recovery(
    model,
    bank,
    generated_at_states,
    Q_at,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    """
    RECOVERY TEST.

    Take only the frozen AT coordinates produced downstream of the AP patch and
    insert them into an otherwise clean base trajectory:

        h_recovery = h_clean
                   + ((h_AT_generated - h_clean) Q_AT) Q_AT^T

    If AP really produces the appropriate AT state, this isolated AT transfer
    should recover the AP counterfactual output.
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
            recovered = clean + (((generated - clean) @ Q_at) @ Q_at.T)
            hidden_new[rows, pos, :] = recovered.to(hidden.dtype)
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
            all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def run_restoration(
    model,
    bank,
    source_ap_states,
    clean_base_at_states,
    Q_ap,
    Q_at,
    ap_layer,
    at_layer,
    label_ids,
    token_position="last_token",
    batch_size=16,
):
    """
    RESTORATION TEST.

    1) Patch AP at L_AP.
    2) Let the intervention propagate to L_AT.
    3) At L_AT, restore ONLY the frozen AT coordinates to their clean-base
       values, while leaving the orthogonal remainder of the AP-patched
       trajectory untouched:

        h_restored = h_current
                   + ((h_AT_base - h_current) Q_AT) Q_AT^T

    If AP's output effect is mediated by AT, restoring AT toward its base state
    should remove most of the AP counterfactual effect and return Y toward base.
    """
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source_ap = source_ap_states[start:end].to(device=device, dtype=torch.float32)
        base_at = clean_base_at_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        raw_pos = bank["base_position_by_id"][token_position][start:end].to(device)
        pos = pad_offset + raw_pos

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            base = hidden[rows, pos, :].float()
            patched = base + (((source_ap - base) @ Q_ap) @ Q_ap.T)
            hidden_new[rows, pos, :] = patched.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        def at_restore_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            current = hidden[rows, pos, :].float()
            restored = current + (((base_at - current) @ Q_at) @ Q_at.T)
            hidden_new[rows, pos, :] = restored.to(hidden.dtype)
            if isinstance(output, tuple):
                return (hidden_new,) + output[1:]
            return hidden_new

        h_ap = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        h_at = model.model.layers[at_layer].register_forward_hook(at_restore_hook)

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
            h_at.remove()
            h_ap.remove()

    return torch.cat(all_logits, dim=0)


def row_l2(x):
    return torch.linalg.vector_norm(x.float(), dim=-1)


def mean_cosine(x, y, eps=EPS):
    x = x.float()
    y = y.float()
    denom = row_l2(x) * row_l2(y)
    cos = (x * y).sum(dim=-1) / denom.clamp_min(eps)
    valid = denom > eps
    if not bool(valid.any()):
        return float("nan")
    return float(cos[valid].mean().item())


def effect_recovery_metrics(base_logits, direct_logits, recovery_logits):
    p_base = answer_probs(base_logits)
    p_direct = answer_probs(direct_logits)
    p_recovery = answer_probs(recovery_logits)

    direct_shift = p_direct - p_base
    recovery_shift = p_recovery - p_base

    direct_norm = row_l2(direct_shift)
    error_norm = row_l2(p_recovery - p_direct)
    recovered_fraction = 1.0 - error_norm / direct_norm.clamp_min(EPS)
    valid = direct_norm > EPS

    return {
        "mean_output_effect_recovered_fraction": (
            float(recovered_fraction[valid].mean().item()) if bool(valid.any()) else float("nan")
        ),
        "median_output_effect_recovered_fraction": (
            float(recovered_fraction[valid].median().item()) if bool(valid.any()) else float("nan")
        ),
        "shift_cosine": mean_cosine(direct_shift, recovery_shift),
    }


def effect_restoration_metrics(base_logits, direct_logits, restored_logits):
    p_base = answer_probs(base_logits)
    p_direct = answer_probs(direct_logits)
    p_restored = answer_probs(restored_logits)

    direct_shift = p_direct - p_base
    residual_shift = p_restored - p_base

    direct_norm = row_l2(direct_shift)
    residual_norm = row_l2(residual_shift)
    removed_fraction = 1.0 - residual_norm / direct_norm.clamp_min(EPS)
    valid = direct_norm > EPS

    return {
        "mean_output_effect_removed_fraction": (
            float(removed_fraction[valid].mean().item()) if bool(valid.any()) else float("nan")
        ),
        "median_output_effect_removed_fraction": (
            float(removed_fraction[valid].median().item()) if bool(valid.any()) else float("nan")
        ),
        "residual_effect_ratio": (
            float((residual_norm[valid] / direct_norm[valid].clamp_min(EPS)).mean().item())
            if bool(valid.any()) else float("nan")
        ),
    }


def evaluate_edge(
    model,
    tokenizer,
    at_checkpoint,
    ap_checkpoint,
    batch_size=16,
    save_path="results/gradual_das_ap_at_edge_eval.pt",
):
    device = next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    at_result = torch.load(at_checkpoint, map_location="cpu")
    ap_result = torch.load(ap_checkpoint, map_location="cpu")

    at_layer = int(at_result["layer"])
    ap_layer = int(ap_result["ap_layer"])
    Q_at = at_result["basis"].float().to(device)
    Q_ap = ap_result["ap_basis"].float().to(device)

    if int(ap_result["at_layer"]) != at_layer:
        raise ValueError(
            f"Checkpoint mismatch: AP checkpoint says L_AT={ap_result['at_layer']} "
            f"but AT checkpoint has L_AT={at_layer}"
        )
    if ap_layer >= at_layer:
        raise ValueError(f"Need L_AP < L_AT for AP->AT edge test, got {ap_layer} >= {at_layer}")
    if Q_at.shape[0] != Q_ap.shape[0]:
        raise ValueError(f"Hidden-size mismatch: Q_AP={tuple(Q_ap.shape)}, Q_AT={tuple(Q_at.shape)}")

    ap_cfg = ap_result.get("config", {})
    at_cfg = at_result.get("config", {})
    ft_size = int(ap_cfg.get("ft_size", at_cfg.get("ft_size", 400)))
    cal_size = int(ap_cfg.get("cal_size", at_cfg.get("cal_size", 200)))
    te_size = int(ap_cfg.get("te_size", at_cfg.get("te_size", 200)))
    seed = int(ap_cfg.get("seed", at_cfg.get("seed", 0)))
    token_position = ap_result.get(
        "token_position",
        at_result.get("token_position", "last_token"),
    )

    set_seed(seed)
    labels = answer_label_ids(tokenizer)

    print(
        f"[alignment] L_AP={ap_layer} k_AP={Q_ap.shape[1]} "
        f"L_AT={at_layer} k_AT={Q_at.shape[1]} token={token_position}"
    )
    print(
        f"[split] ft={ft_size} cal={cal_size} test={te_size} seed={seed}"
    )

    # Rebuild exactly the repo-style split used during training, then evaluate
    # only the AP-sensitive TEST bank. No checkpoint selection is done here.
    _, _, te_banks = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        dataset_size=None,
        split="train",
        device=device,
        batch_size=batch_size,
        seed=seed,
    )
    test_bank = te_banks["answer_pointer"]

    source_ap_states = collect_layer_states(
        model=model,
        input_ids=test_bank["source_input_ids"],
        attention_mask=test_bank["source_attention_mask"],
        position_by_id=test_bank["source_position_by_id"],
        layer=ap_layer,
        token_position=token_position,
        batch_size=batch_size,
    )
    clean_base_at_states = collect_layer_states(
        model=model,
        input_ids=test_bank["base_input_ids"],
        attention_mask=test_bank["base_attention_mask"],
        position_by_id=test_bank["base_position_by_id"],
        layer=at_layer,
        token_position=token_position,
        batch_size=batch_size,
    )

    base_logits = run_clean(
        model=model,
        bank=test_bank,
        label_ids=labels,
        batch_size=batch_size,
    )

    direct_logits, generated_at_states = run_ap_patch_capture_at(
        model=model,
        bank=test_bank,
        source_ap_states=source_ap_states,
        Q_ap=Q_ap,
        ap_layer=ap_layer,
        at_layer=at_layer,
        label_ids=labels,
        token_position=token_position,
        batch_size=batch_size,
    )

    recovery_logits = run_recovery(
        model=model,
        bank=test_bank,
        generated_at_states=generated_at_states,
        Q_at=Q_at,
        at_layer=at_layer,
        label_ids=labels,
        token_position=token_position,
        batch_size=batch_size,
    )

    restored_logits = run_restoration(
        model=model,
        bank=test_bank,
        source_ap_states=source_ap_states,
        clean_base_at_states=clean_base_at_states,
        Q_ap=Q_ap,
        Q_at=Q_at,
        ap_layer=ap_layer,
        at_layer=at_layer,
        label_ids=labels,
        token_position=token_position,
        batch_size=batch_size,
    )

    ap_target = test_bank["counterfactual_label_ids"]["answer_pointer"]
    base_target = test_bank["base_answer_label_ids"]

    direct_iia = ap_iia(direct_logits, ap_target)
    recovery_iia = ap_iia(recovery_logits, ap_target)
    restored_cf_iia = ap_iia(restored_logits, ap_target)

    base_accuracy = full_answer_accuracy(base_logits, base_target)
    restored_base_accuracy = full_answer_accuracy(restored_logits, base_target)

    recovery_effect = effect_recovery_metrics(
        base_logits=base_logits,
        direct_logits=direct_logits,
        recovery_logits=recovery_logits,
    )
    restoration_effect = effect_restoration_metrics(
        base_logits=base_logits,
        direct_logits=direct_logits,
        restored_logits=restored_logits,
    )

    result = {
        "at_checkpoint": at_checkpoint,
        "ap_checkpoint": ap_checkpoint,
        "ap_layer": ap_layer,
        "at_layer": at_layer,
        "ap_subspace_dim": int(Q_ap.shape[1]),
        "at_subspace_dim": int(Q_at.shape[1]),
        "token_position": token_position,
        "num_test": int(len(test_bank["base_input_ids"])),
        "clean": {
            "base_accuracy": base_accuracy,
        },
        "direct_ap": {
            "counterfactual_iia": direct_iia,
        },
        "recovery": {
            "counterfactual_iia": recovery_iia,
            **recovery_effect,
        },
        "restoration": {
            "restored_base_accuracy": restored_base_accuracy,
            "counterfactual_iia_after_restore": restored_cf_iia,
            **restoration_effect,
        },
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "seed": seed,
            "batch_size": batch_size,
        },
    }

    print()
    print("===== AP -> AT EDGE TEST ON AP-SENSITIVE TEST SET =====")
    print(f"[CLEAN]       base_accuracy={base_accuracy:.4f}")
    print(f"[DIRECT AP]   cf_iia={direct_iia:.4f}")
    print(
        f"[RECOVERY]    cf_iia={recovery_iia:.4f} "
        f"recovered_fraction={recovery_effect['mean_output_effect_recovered_fraction']:.4f} "
        f"shift_cosine={recovery_effect['shift_cosine']:.4f}"
    )
    print(
        f"[RESTORATION] base_accuracy={restored_base_accuracy:.4f} "
        f"cf_iia_after_restore={restored_cf_iia:.4f} "
        f"removed_fraction={restoration_effect['mean_output_effect_removed_fraction']:.4f} "
        f"residual_ratio={restoration_effect['residual_effect_ratio']:.4f}"
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)
    print(f"saved to: {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--at-checkpoint",
        default="results/gradual_das_at_layer_selected.pt",
    )
    parser.add_argument(
        "--ap-checkpoint",
        default="results/gradual_das_ap.pt",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--save-path",
        default="results/gradual_das_ap_at_edge_eval.pt",
    )
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    print(next(model.parameters()).device)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    evaluate_edge(
        model=model,
        tokenizer=tokenizer,
        at_checkpoint=args.at_checkpoint,
        ap_checkpoint=args.ap_checkpoint,
        batch_size=args.batch_size,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()