import argparse
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from arc_data_load_all import build_arc_banks
from mcqa_gradual_das import (
    answer_label_ids,
    answer_logits,
    collect_layer_states,
    run_das_intervention,
    set_seed,
)
from mcqa_gradual_das_ap import (
    run_ap_intervention_and_capture_at,
    run_frozen_at_mediator,
)


MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOKEN_POSITION = "last_token"
EPS = 1e-8


def load_model():
    token = os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=token,
        dtype=dtype,
    ).to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return model, tokenizer


def xorder_iia(logits, target):
    pred = logits[:, :4].argmax(dim=-1).cpu()
    # pred = logits[:, :].argmax(dim=-1).cpu()
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    return float((pred == target).float().mean().item())


def full_accuracy(logits, target):
    pred = logits.argmax(dim=-1).cpu()
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    return float((pred == target).float().mean().item())


def answer_probs(logits):
    return torch.softmax(logits.float(), dim=-1)


def row_l2(x):
    return torch.linalg.vector_norm(x.float(), dim=-1)


def mean_cosine(x, y, eps=EPS):
    x = x.float()
    y = y.float()
    denom = row_l2(x) * row_l2(y)
    valid = denom > eps
    if not bool(valid.any()):
        return float("nan")
    cosine = (x * y).sum(dim=-1) / denom.clamp_min(eps)
    return float(cosine[valid].mean().item())


def recovery_effect_metrics(base_logits, direct_logits, recovery_logits):
    p_base = answer_probs(base_logits)
    p_direct = answer_probs(direct_logits)
    p_recovery = answer_probs(recovery_logits)

    direct_shift = p_direct - p_base
    recovery_shift = p_recovery - p_base
    direct_norm = row_l2(direct_shift)
    error_norm = row_l2(p_recovery - p_direct)
    valid = direct_norm > EPS
    recovered_fraction = 1.0 - error_norm / direct_norm.clamp_min(EPS)

    return {
        "mean_output_effect_recovered_fraction": float(recovered_fraction[valid].mean().item()) if bool(valid.any()) else float("nan"),
        "shift_cosine": mean_cosine(direct_shift, recovery_shift),
    }


def restoration_effect_metrics(base_logits, direct_logits, restored_logits):
    p_base = answer_probs(base_logits)
    p_direct = answer_probs(direct_logits)
    p_restored = answer_probs(restored_logits)

    direct_shift = p_direct - p_base
    residual_shift = p_restored - p_base
    direct_norm = row_l2(direct_shift)
    residual_norm = row_l2(residual_shift)
    valid = direct_norm > EPS
    removed_fraction = 1.0 - residual_norm / direct_norm.clamp_min(EPS)

    return {
        "mean_output_effect_removed_fraction": float(removed_fraction[valid].mean().item()) if bool(valid.any()) else float("nan"),
        "residual_effect_ratio": float((residual_norm[valid] / direct_norm[valid].clamp_min(EPS)).mean().item()) if bool(valid.any()) else float("nan"),
    }


@torch.no_grad()
def run_clean(model, bank, label_ids, batch_size=8):
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
def run_restoration(
    model, bank, source_xorder_states, clean_base_oanswer_states,
    Q_xorder, Q_oanswer, xorder_layer, oanswer_layer, label_ids,
    batch_size=8,
):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)

        source_xorder = source_xorder_states[start:end].to(device=device, dtype=torch.float32)
        clean_oanswer = clean_base_oanswer_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][TOKEN_POSITION][start:end].to(device)

        def xorder_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            base = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (base + ((source_xorder - base) @ Q_xorder) @ Q_xorder.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def oanswer_restore_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            current = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (current + ((clean_oanswer - current) @ Q_oanswer) @ Q_oanswer.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        xorder_handle = model.model.layers[xorder_layer].register_forward_hook(xorder_hook)
        oanswer_handle = model.model.layers[oanswer_layer].register_forward_hook(oanswer_restore_hook)

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
            oanswer_handle.remove()
            xorder_handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def run_conflict(
    model, xorder_bank, source_xorder_states, donor_oanswer_states,
    Q_xorder, Q_oanswer, xorder_layer, oanswer_layer, label_ids,
    batch_size=8,
):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(xorder_bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(xorder_bank["base_input_ids"]))
        ids = xorder_bank["base_input_ids"][start:end].to(device)
        mask = xorder_bank["base_attention_mask"][start:end].to(device)

        source_xorder = source_xorder_states[start:end].to(device=device, dtype=torch.float32)
        donor_oanswer = donor_oanswer_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + xorder_bank["base_position_by_id"][TOKEN_POSITION][start:end].to(device)

        def xorder_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            base = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (base + ((source_xorder - base) @ Q_xorder) @ Q_xorder.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def oanswer_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            current = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (current + ((donor_oanswer - current) @ Q_oanswer) @ Q_oanswer.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        xorder_handle = model.model.layers[xorder_layer].register_forward_hook(xorder_hook)
        oanswer_handle = model.model.layers[oanswer_layer].register_forward_hook(oanswer_hook)

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
            oanswer_handle.remove()
            xorder_handle.remove()

    return torch.cat(all_logits, dim=0)


def make_conflict_donors(xorder_target, base_target, oanswer_pool_target, seed=0):
    rng = random.Random(seed + 12345)
    donors = []

    for i in range(len(xorder_target)):
        candidates = [
            j for j in range(len(oanswer_pool_target))
            if int(oanswer_pool_target[j]) != int(xorder_target[i])
            and int(oanswer_pool_target[j]) != int(base_target[i])
        ]
        if not candidates:
            raise RuntimeError(f"No valid conflict donor for example {i}.")
        donors.append(rng.choice(candidates))

    return torch.tensor(donors, dtype=torch.long)


def random_orthonormal_basis(hidden_size, subspace_dim, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    raw = torch.randn(hidden_size, subspace_dim, generator=generator, dtype=torch.float32)
    return torch.linalg.qr(raw, mode="reduced").Q.to(device)


def load_oanswer_readout(checkpoint_path, device):
    result = torch.load(checkpoint_path, map_location="cpu")
    W = result["W_AT"].float().to(device)
    b = result["b_AT"].float().to(device)
    return W, b, result


def evaluate_all_tests(
    model,
    tokenizer,
    oanswer_checkpoint,
    xorder_checkpoint,
    oanswer_readout_checkpoint=None,
    batch_size=8,
    test_size=None,
    num_random_controls=5,
    save_path="results_arc/arc_all_tests.pt",
):
    device = next(model.parameters()).device

    oanswer_result = torch.load(oanswer_checkpoint, map_location="cpu")
    xorder_result = torch.load(xorder_checkpoint, map_location="cpu")

    oanswer_layer = int(oanswer_result["layer"])
    xorder_layer = int(xorder_result.get("ap_layer", xorder_result["layer"]))
    Q_oanswer = oanswer_result["basis"].float().to(device)
    Q_xorder = xorder_result.get("ap_basis", xorder_result["basis"]).float().to(device)

    if xorder_layer >= oanswer_layer:
        raise ValueError(
            "Edge tests require XOrder upstream of OAnswer, "
            f"got XOrder=L{xorder_layer}, OAnswer=L{oanswer_layer}."
        )

    if oanswer_readout_checkpoint is None:
        oanswer_readout_checkpoint = xorder_result.get("at_readout_checkpoint")
    if oanswer_readout_checkpoint is None:
        raise ValueError(
            "Need --oanswer-readout-checkpoint, or an XOrder checkpoint containing 'at_readout_checkpoint'."
        )

    W_oanswer, b_oanswer, _ = load_oanswer_readout(oanswer_readout_checkpoint, device)

    xorder_cfg = xorder_result.get("config", {})
    oanswer_cfg = oanswer_result.get("config", {})
    ft_size = int(xorder_cfg.get("ft_size", oanswer_cfg.get("ft_size", 400)))
    cal_size = int(xorder_cfg.get("cal_size", oanswer_cfg.get("cal_size", 200)))
    # te_size = int(xorder_cfg.get("te_size", oanswer_cfg.get("te_size", 200)))
    te_size = (
        test_size
        if test_size is not None
        else int(xorder_cfg.get("te_size", oanswer_cfg.get("te_size", 200)))
    )
    seed = int(xorder_cfg.get("seed", oanswer_cfg.get("seed", 0)))

    set_seed(seed)
    labels = answer_label_ids(tokenizer)

    print(f"[alignment] XOrder=L{xorder_layer}/k{Q_xorder.shape[1]} OAnswer=L{oanswer_layer}/k{Q_oanswer.shape[1]}")
    print(f"[split] ft={ft_size} cal={cal_size} test={te_size} seed={seed}")

    banks = build_arc_banks(
        model,
        tokenizer,
        fit_size=ft_size,
        cal_size=cal_size,
        test_size=te_size,
        batch_size=batch_size,
        seed=seed,
    )

    xorder_bank = banks["answer_pointer"]["test"]
    oanswer_bank = banks["answer_token"]["test"]

    xorder_clean_logits = run_clean(model, xorder_bank, labels, batch_size)
    oanswer_clean_logits = run_clean(model, oanswer_bank, labels, batch_size)

    # 1) DIRECT XOrder + capture downstream OAnswer.
    xorder_source_states = collect_layer_states(
        model, xorder_bank["source_input_ids"], xorder_bank["source_attention_mask"],
        xorder_bank["source_position_by_id"], xorder_layer, TOKEN_POSITION, batch_size,
    )

    xorder_direct_logits, generated_oanswer_states = run_ap_intervention_and_capture_at(
        model, xorder_bank, xorder_source_states, Q_xorder, xorder_layer, oanswer_layer,
        labels, TOKEN_POSITION, batch_size, require_grad=False,
    )

    xorder_target = xorder_bank["counterfactual_label_ids"]["answer_pointer"]
    oanswer_target_after_xorder = xorder_bank["counterfactual_label_ids"]["answer_token"]
    xorder_direct_iia = xorder_iia(xorder_direct_logits, xorder_target)

    # 2) XOrder -> OAnswer readout consistency.
    oanswer_readout_logits_after_xorder = (
        (generated_oanswer_states.to(device) @ Q_oanswer) @ W_oanswer.T + b_oanswer
    )
    xorder_to_oanswer_readout_accuracy = full_accuracy(
        oanswer_readout_logits_after_xorder.cpu(), oanswer_target_after_xorder
    )

    clean_base_oanswer_states = collect_layer_states(
        model, xorder_bank["base_input_ids"], xorder_bank["base_attention_mask"],
        xorder_bank["base_position_by_id"], oanswer_layer, TOKEN_POSITION, batch_size,
    )
    clean_readout_logits = (
        (clean_base_oanswer_states.to(device) @ Q_oanswer) @ W_oanswer.T + b_oanswer
    )
    clean_oanswer_readout_accuracy = full_accuracy(
        clean_readout_logits.cpu(), xorder_bank["base_answer_label_ids"]
    )

    # 3) XOrder -> OAnswer recovery / mediator replay.
    recovery_logits = run_frozen_at_mediator(
        model, xorder_bank, generated_oanswer_states, Q_oanswer, oanswer_layer,
        labels, TOKEN_POSITION, batch_size, require_grad=False,
    )
    recovery_iia = xorder_iia(recovery_logits, xorder_target)
    recovery_effect = recovery_effect_metrics(
        xorder_clean_logits, xorder_direct_logits, recovery_logits
    )

    # 4) XOrder -> OAnswer restoration.
    restored_logits = run_restoration(
        model, xorder_bank, xorder_source_states, clean_base_oanswer_states,
        Q_xorder, Q_oanswer, xorder_layer, oanswer_layer, labels, batch_size,
    )
    restoration_base_accuracy = full_accuracy(
        restored_logits, xorder_bank["base_answer_label_ids"]
    )
    restoration_cf_iia = xorder_iia(restored_logits, xorder_target)
    restoration_effect = restoration_effect_metrics(
        xorder_clean_logits, xorder_direct_logits, restored_logits
    )

    # 5) DIRECT OAnswer.
    oanswer_source_states = collect_layer_states(
        model, oanswer_bank["source_input_ids"], oanswer_bank["source_attention_mask"],
        oanswer_bank["source_position_by_id"], oanswer_layer, TOKEN_POSITION, batch_size,
    )
    oanswer_direct_logits = run_das_intervention(
        model, oanswer_bank, oanswer_source_states, Q_oanswer, oanswer_layer,
        labels, TOKEN_POSITION, batch_size, require_grad=False,
    )
    oanswer_target = oanswer_bank["counterfactual_label_ids"]["answer_token"]
    oanswer_direct_iia = full_accuracy(oanswer_direct_logits, oanswer_target)

    # 6) Conflict: XOrder says one answer, OAnswer is forced to another.
    conflict_donor_idx = make_conflict_donors(
        oanswer_target_after_xorder.cpu(),
        xorder_bank["base_answer_label_ids"].cpu(),
        oanswer_bank["source_answer_label_ids"].cpu(),
        seed,
    )
    conflict_donor_states = oanswer_source_states[conflict_donor_idx]
    conflict_target = oanswer_bank["source_answer_label_ids"][conflict_donor_idx]

    conflict_logits = run_conflict(
        model, xorder_bank, xorder_source_states, conflict_donor_states,
        Q_xorder, Q_oanswer, xorder_layer, oanswer_layer, labels, batch_size,
    )
    conflict_follows_oanswer = full_accuracy(conflict_logits, conflict_target)
    conflict_follows_xorder = full_accuracy(conflict_logits, oanswer_target_after_xorder)
    conflict_follows_base = full_accuracy(conflict_logits, xorder_bank["base_answer_label_ids"])

    # 7) Random-subspace controls.
    random_xorder_rows = []
    random_oanswer_rows = []

    for random_idx in range(num_random_controls):
        Q_xorder_random = random_orthonormal_basis(
            Q_xorder.shape[0], Q_xorder.shape[1], seed + 1000 + random_idx, device
        )
        Q_oanswer_random = random_orthonormal_basis(
            Q_oanswer.shape[0], Q_oanswer.shape[1], seed + 2000 + random_idx, device
        )

        random_xorder_logits, random_generated_oanswer = run_ap_intervention_and_capture_at(
            model, xorder_bank, xorder_source_states, Q_xorder_random,
            xorder_layer, oanswer_layer, labels, TOKEN_POSITION, batch_size,
            require_grad=False,
        )
        random_xorder_recovery_logits = run_frozen_at_mediator(
            model, xorder_bank, random_generated_oanswer, Q_oanswer, oanswer_layer,
            labels, TOKEN_POSITION, batch_size, require_grad=False,
        )
        random_oanswer_readout_logits = (
            (random_generated_oanswer.to(device) @ Q_oanswer) @ W_oanswer.T + b_oanswer
        )

        random_xorder_rows.append({
            "seed": seed + 1000 + random_idx,
            "direct_iia": xorder_iia(random_xorder_logits, xorder_target),
            "recovery_iia": xorder_iia(random_xorder_recovery_logits, xorder_target),
            "readout_accuracy": full_accuracy(
                random_oanswer_readout_logits.cpu(), oanswer_target_after_xorder
            ),
        })

        random_oanswer_logits = run_das_intervention(
            model, oanswer_bank, oanswer_source_states, Q_oanswer_random,
            oanswer_layer, labels, TOKEN_POSITION, batch_size, require_grad=False,
        )
        random_oanswer_rows.append({
            "seed": seed + 2000 + random_idx,
            "direct_iia": full_accuracy(random_oanswer_logits, oanswer_target),
        })

    mean_random_xorder_direct = sum(row["direct_iia"] for row in random_xorder_rows) / len(random_xorder_rows)
    mean_random_xorder_recovery = sum(row["recovery_iia"] for row in random_xorder_rows) / len(random_xorder_rows)
    mean_random_xorder_readout = sum(row["readout_accuracy"] for row in random_xorder_rows) / len(random_xorder_rows)
    mean_random_oanswer_direct = sum(row["direct_iia"] for row in random_oanswer_rows) / len(random_oanswer_rows)

    result = {
        "checkpoints": {
            "xorder": xorder_checkpoint,
            "oanswer": oanswer_checkpoint,
            "oanswer_readout": oanswer_readout_checkpoint,
        },
        "alignment": {
            "xorder_layer": xorder_layer,
            "oanswer_layer": oanswer_layer,
            "xorder_subspace_dim": int(Q_xorder.shape[1]),
            "oanswer_subspace_dim": int(Q_oanswer.shape[1]),
            "token_position": TOKEN_POSITION,
        },
        "clean": {
            "xorder_bank_base_accuracy": full_accuracy(
                xorder_clean_logits, xorder_bank["base_answer_label_ids"]
            ),
            "oanswer_bank_base_accuracy": full_accuracy(
                oanswer_clean_logits, oanswer_bank["base_answer_label_ids"]
            ),
        },
        "xorder_direct": {"counterfactual_iia": xorder_direct_iia},
        "oanswer_direct": {"counterfactual_iia": oanswer_direct_iia},
        "xorder_to_oanswer_readout": {
            "clean_base_readout_accuracy": clean_oanswer_readout_accuracy,
            "after_xorder_intervention_readout_accuracy": xorder_to_oanswer_readout_accuracy,
        },
        "xorder_to_oanswer_recovery": {
            "counterfactual_iia": recovery_iia,
            **recovery_effect,
        },
        "xorder_to_oanswer_restoration": {
            "restored_base_accuracy": restoration_base_accuracy,
            "counterfactual_iia_after_restore": restoration_cf_iia,
            **restoration_effect,
        },
        "conflict": {
            "follows_conflicting_oanswer": conflict_follows_oanswer,
            "follows_xorder_implied_answer": conflict_follows_xorder,
            "follows_base": conflict_follows_base,
            "donor_indices": conflict_donor_idx,
            "conflict_targets": conflict_target,
        },
        "random_controls": {
            "xorder": {
                "runs": random_xorder_rows,
                "mean_direct_iia": mean_random_xorder_direct,
                "mean_recovery_iia": mean_random_xorder_recovery,
                "mean_readout_accuracy": mean_random_xorder_readout,
            },
            "oanswer": {
                "runs": random_oanswer_rows,
                "mean_direct_iia": mean_random_oanswer_direct,
            },
        },
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "num_random_controls": num_random_controls,
            "batch_size": batch_size,
            "seed": seed,
        },
    }

    print()
    print("===== ARC XOrder / OAnswer CAUSAL TESTS =====")
    print(f"[XOrder DIRECT]      IIA={xorder_direct_iia:.4f}")
    print(f"[OAnswer DIRECT]     IIA={oanswer_direct_iia:.4f}")
    print(
        f"[OAnswer READOUT]    clean={clean_oanswer_readout_accuracy:.4f} "
        f"after_XOrder={xorder_to_oanswer_readout_accuracy:.4f}"
    )
    print(
        f"[RECOVERY]           IIA={recovery_iia:.4f} "
        f"recovered={recovery_effect['mean_output_effect_recovered_fraction']:.4f} "
        f"cos={recovery_effect['shift_cosine']:.4f}"
    )
    print(
        f"[RESTORATION]        base_acc={restoration_base_accuracy:.4f} "
        f"cf_iia={restoration_cf_iia:.4f} "
        f"removed={restoration_effect['mean_output_effect_removed_fraction']:.4f}"
    )
    print(
        f"[CONFLICT]           follows_OAnswer={conflict_follows_oanswer:.4f} "
        f"follows_XOrder={conflict_follows_xorder:.4f} "
        f"follows_base={conflict_follows_base:.4f}"
    )
    print(
        f"[RANDOM XOrder]      direct={mean_random_xorder_direct:.4f} "
        f"recovery={mean_random_xorder_recovery:.4f} "
        f"readout={mean_random_xorder_readout:.4f}"
    )
    print(f"[RANDOM OAnswer]     direct={mean_random_oanswer_direct:.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)
    print(f"saved to: {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oanswer-checkpoint", required=True)
    parser.add_argument("--xorder-checkpoint", required=True)
    parser.add_argument("--oanswer-readout-checkpoint", default="results_arc/gradual_das_ap_lambda_med_0.0_lambda_readout_1.0/oanswer_readout.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-random-controls", type=int, default=5)
    parser.add_argument("--save-path", default="results_arc/arc_all_tests.pt")
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    model, tokenizer = load_model()
    print(next(model.parameters()).device)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    evaluate_all_tests(
        model=model,
        tokenizer=tokenizer,
        oanswer_checkpoint=args.oanswer_checkpoint,
        xorder_checkpoint=args.xorder_checkpoint,
        oanswer_readout_checkpoint=args.oanswer_readout_checkpoint,
        batch_size=args.batch_size,
        test_size=args.test_size,
        num_random_controls=args.num_random_controls,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()
