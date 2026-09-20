import argparse
import os
import random

import torch

from mcqa_data_load_all import FAMILIES, build_bank, factual_filter, load_mcqa_pairs
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import answer_label_ids, answer_logits, collect_layer_states, run_das_intervention, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at, run_frozen_at_mediator


EPS = 1e-8


def ap_iia(logits, target):
    """4-way AP accuracy: AP values 0/1/2/3 correspond to A/B/C/D."""
    pred = logits[:, :4].argmax(dim=-1).cpu()
    target = torch.as_tensor(target, dtype=torch.long).cpu()
    return float((pred == target).float().mean().item())


def full_accuracy(logits, target):
    """Accuracy in the full A-Z answer-token space."""
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


def probability_drift_metrics(base_logits, intervened_logits):
    p_base = answer_probs(base_logits)
    p_int = answer_probs(intervened_logits)
    delta = p_int - p_base

    return {
        "mean_l1_probability_drift": float(delta.abs().sum(dim=-1).mean().item()),
        "mean_l2_probability_drift": float(row_l2(delta).mean().item()),
    }


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
        "median_output_effect_recovered_fraction": float(recovered_fraction[valid].median().item()) if bool(valid.any()) else float("nan"),
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
        "median_output_effect_removed_fraction": float(removed_fraction[valid].median().item()) if bool(valid.any()) else float("nan"),
        "residual_effect_ratio": float((residual_norm[valid] / direct_norm[valid].clamp_min(EPS)).mean().item()) if bool(valid.any()) else float("nan"),
    }


@torch.no_grad()
def run_clean(model, bank, label_ids, batch_size=32):
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
def run_restoration(model, bank, source_ap_states, clean_base_at_states, Q_ap, Q_at, ap_layer, at_layer, label_ids, token_position="last_token", batch_size=32):
    """
    AP -> AT restoration:
      1) intervene on AP;
      2) let the effect propagate to AT;
      3) restore only the learned AT coordinates to the clean-base AT values.

    This test is NOT directly optimized by Gradual DAS.
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
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            base = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (base + ((source_ap - base) @ Q_ap) @ Q_ap.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def at_restore_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            current = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (current + ((base_at - current) @ Q_at) @ Q_at.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_restore_hook)

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
            at_handle.remove()
            ap_handle.remove()

    return torch.cat(all_logits, dim=0)


@torch.no_grad()
def run_conflict(model, bank, source_ap_states, donor_at_states, Q_ap, Q_at, ap_layer, at_layer, label_ids, token_position="last_token", batch_size=32):
    """
    Force AP to one value and then force AT to a conflicting downstream value.
    If AP -> AT -> Y is the correct ordering, final behavior should follow AT.
    """
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source_ap = source_ap_states[start:end].to(device=device, dtype=torch.float32)
        donor_at = donor_at_states[start:end].to(device=device, dtype=torch.float32)

        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            base = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (base + ((source_ap - base) @ Q_ap) @ Q_ap.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        def at_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            current = hidden[rows, pos, :].float()
            hidden_new[rows, pos, :] = (current + ((donor_at - current) @ Q_at) @ Q_at.T).to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_hook)

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
            at_handle.remove()
            ap_handle.remove()

    return torch.cat(all_logits, dim=0)


def make_conflict_donors(ap_target, base_target, at_target, seed=0):
    """Choose AT donors whose target differs from both AP-implied and clean-base answers."""
    rng = random.Random(seed + 12345)
    donors = []

    for i in range(len(at_target)):
        candidates = [
            j
            for j in range(len(at_target))
            if j != i and int(at_target[j]) != int(ap_target[i]) and int(at_target[j]) != int(base_target[i])
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


def load_linear_readout(checkpoint_path, variable, device):
    result = torch.load(checkpoint_path, map_location="cpu")

    if variable == "AP":
        weight_keys = ("W_AP", "W_ap", "W")
        bias_keys = ("b_AP", "b_ap", "b")
    elif variable == "AT":
        weight_keys = ("W_AT", "W_at", "W")
        bias_keys = ("b_AT", "b_at", "b")
    else:
        raise ValueError(variable)

    W = next((result[key] for key in weight_keys if key in result), None)
    b = next((result[key] for key in bias_keys if key in result), None)

    if W is None or b is None:
        raise KeyError(
            f"Could not find {variable} readout weights/bias in {checkpoint_path}. "
            f"Tried weights={weight_keys}, bias={bias_keys}. Keys={list(result.keys())}"
        )

    return W.float().to(device), b.float().to(device), result


def prepare_eval_rows(model, tokenizer, train_pool_size, dataset_size=None, split="train", batch_size=32, seed=0):
    """
    Reproduce the same pair-level train/holdout split used by build_mcqa_banks.

    Important: this does NOT change the training bank. It is only used inside
    the diagnostic evaluator so that direct tests and invariance tests use
    states from the same held-out pool.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    pairs_by_family = load_mcqa_pairs(dataset_size=dataset_size, split=split)
    filtered = factual_filter(model, tokenizer, pairs_by_family, device=device, batch_size=batch_size)

    pooled_rows = []
    for family in FAMILIES:
        pooled_rows.extend(filtered[family])

    random.Random(int(seed)).shuffle(pooled_rows)
    if train_pool_size > len(pooled_rows):
        raise ValueError(
            f"train_pool_size={train_pool_size}, but only "
            f"{len(pooled_rows)} filtered rows are available"
        )

    return pooled_rows[train_pool_size:]


def build_direct_test_banks(tokenizer, holdout_rows, cal_size, te_size, seed=0):
    """
    Build the same AP-sensitive and AT-sensitive TEST banks as build_mcqa_banks.

    These are still the original predefined dataset counterfactual pairs.
    """
    direct_test_banks = {}

    for target in ("answer_pointer", "answer_token"):
        positive_rows = []

        for row in holdout_rows:
            if target == "answer_pointer":
                changed = row["base_answer_pointer"] != row["source_answer_pointer"]
            else:
                changed = row["base_answer_letter"] != row["source_answer_letter"]

            if changed:
                positive_rows.append(row)

        random.Random(f"{int(seed)}:holdout:{target}").shuffle(positive_rows)

        required = cal_size + te_size
        if len(positive_rows) < required:
            raise ValueError(
                f"target={target} needs {required} changed holdout rows, "
                f"found {len(positive_rows)}"
            )

        direct_test_banks[target] = build_bank(
            tokenizer,
            positive_rows[cal_size:required],
        )

    return direct_test_banks


def collect_unique_factual_states(rows):
    """
    Convert held-out predefined pairs into a pool of factual states.

    Each state is one prompt together with its high-level values:
        AP = answer position
        AT = answer symbol/token

    Both the base and source side of every factually-correct held-out pair are
    included. Duplicate prompts are removed.
    """
    states_by_prompt = {}

    for row in rows:
        for side in ("base", "source"):
            prompt = row[f"{side}_prompt"]
            state = {
                "prompt": prompt,
                "answer_letter": row[f"{side}_answer_letter"],
                "answer_pointer": int(row[f"{side}_answer_pointer"]),
            }

            if prompt in states_by_prompt:
                old = states_by_prompt[prompt]
                if (
                    old["answer_letter"] != state["answer_letter"]
                    or old["answer_pointer"] != state["answer_pointer"]
                ):
                    raise ValueError(
                        "Same prompt appeared with inconsistent AP/AT labels."
                    )
            else:
                states_by_prompt[prompt] = state

    return list(states_by_prompt.values())


def make_invariance_pairs(states, target, max_pairs=None, seed=0):
    """
    Re-pair factual states according to the value of the variable being tested.

    AP invariance:
        AP_base == AP_source
        AT_base != AT_source

    AT invariance:
        AT_base == AT_source
        AP_base != AP_source

    Unlike the original dataset banks, these pairs are allowed to come from
    different examples. This is necessary for same-AT / different-AP pairs.
    """
    if target not in {"answer_pointer", "answer_token"}:
        raise ValueError(target)

    if target == "answer_pointer":
        target_key = "answer_pointer"
        other_key = "answer_letter"
        family_name = "cross_example_ap_invariance"
        rng = random.Random(int(seed) + 71001)
    else:
        target_key = "answer_letter"
        other_key = "answer_pointer"
        family_name = "cross_example_at_invariance"
        rng = random.Random(int(seed) + 72001)

    # target_value -> other_value -> list[state]
    grouped = {}
    for state in states:
        target_value = state[target_key]
        other_value = state[other_key]
        grouped.setdefault(target_value, {}).setdefault(other_value, []).append(state)

    candidate_pairs = []

    for base in states:
        target_value = base[target_key]
        base_other = base[other_key]
        by_other = grouped[target_value]

        alternative_values = [
            value
            for value, bucket in by_other.items()
            if value != base_other and len(bucket) > 0
        ]
        if not alternative_values:
            continue

        donor_other = rng.choice(alternative_values)
        donor_candidates = [
            state
            for state in by_other[donor_other]
            if state["prompt"] != base["prompt"]
        ]
        if not donor_candidates:
            continue

        source = rng.choice(donor_candidates)

        candidate_pairs.append({
            "base_prompt": base["prompt"],
            "base_answer_letter": base["answer_letter"],
            "base_answer_pointer": base["answer_pointer"],
            "source_prompt": source["prompt"],
            "source_answer_letter": source["answer_letter"],
            "source_answer_pointer": source["answer_pointer"],
            "source_family": family_name,
        })

    if not candidate_pairs:
        relation = (
            "same AP / different AT"
            if target == "answer_pointer"
            else "same AT / different AP"
        )
        raise ValueError(f"No valid {relation} invariance pairs could be constructed.")

    # We build at most one donor per base state. If a cap is requested, sample
    # from those valid pairs reproducibly. Set max_pairs <= 0 to use all pairs.
    if max_pairs is not None and int(max_pairs) > 0 and len(candidate_pairs) > int(max_pairs):
        candidate_pairs = rng.sample(candidate_pairs, int(max_pairs))

    return candidate_pairs


def build_invariance_banks(tokenizer, holdout_rows, invariance_size=200, seed=0):
    """
    Build diagnostic invariance banks by cross-example re-pairing.

    This function is evaluation-only. It does not modify the DAS training bank.
    """
    states = collect_unique_factual_states(holdout_rows)

    ap_rows = make_invariance_pairs(
        states,
        target="answer_pointer",
        max_pairs=invariance_size,
        seed=seed,
    )
    at_rows = make_invariance_pairs(
        states,
        target="answer_token",
        max_pairs=invariance_size,
        seed=seed,
    )

    # Defensive checks: make sure the constructed banks satisfy exactly the
    # high-level invariance relations we intended.
    assert all(
        row["base_answer_pointer"] == row["source_answer_pointer"]
        and row["base_answer_letter"] != row["source_answer_letter"]
        for row in ap_rows
    )
    assert all(
        row["base_answer_letter"] == row["source_answer_letter"]
        and row["base_answer_pointer"] != row["source_answer_pointer"]
        for row in at_rows
    )

    invariance_banks = {
        "answer_pointer": build_bank(tokenizer, ap_rows),
        "answer_token": build_bank(tokenizer, at_rows),
    }

    print(
        f"[invariance banks] factual_states={len(states)} "
        f"AP_same/diff_AT={len(ap_rows)} "
        f"AT_same/diff_AP={len(at_rows)}"
    )

    return invariance_banks


def build_eval_banks(
    model,
    tokenizer,
    train_pool_size,
    cal_size,
    te_size,
    invariance_size,
    dataset_size=None,
    split="train",
    batch_size=32,
    seed=0,
):
    """
    Evaluation-only bank builder.

    Direct TEST banks:
        use the original predefined dataset counterfactual pairs.

    Invariance banks:
        re-pair held-out factual states across examples so that the tested
        variable is identical while the other variable differs.
    """
    holdout_rows = prepare_eval_rows(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=train_pool_size,
        dataset_size=dataset_size,
        split=split,
        batch_size=batch_size,
        seed=seed,
    )

    direct_test_banks = build_direct_test_banks(
        tokenizer=tokenizer,
        holdout_rows=holdout_rows,
        cal_size=cal_size,
        te_size=te_size,
        seed=seed,
    )

    invariance_banks = build_invariance_banks(
        tokenizer=tokenizer,
        holdout_rows=holdout_rows,
        invariance_size=invariance_size,
        seed=seed,
    )

    print(
        f"[eval banks] AP_direct={len(direct_test_banks['answer_pointer']['base_input_ids'])} "
        f"AT_direct={len(direct_test_banks['answer_token']['base_input_ids'])} "
        f"AP_invariance={len(invariance_banks['answer_pointer']['base_input_ids'])} "
        f"AT_invariance={len(invariance_banks['answer_token']['base_input_ids'])}"
    )

    return direct_test_banks, invariance_banks


def evaluate_all_tests(model, tokenizer, at_checkpoint, ap_checkpoint, at_readout_checkpoint=None, ap_readout_checkpoint=None, batch_size=32, invariance_size=200, num_random_controls=5, save_path="results/gradual_das_all_tests.pt"):
    """
    Tests applicable to the current AP -> AT -> Y causal model.

    Node-level:
      - AP direct IIA
      - AP readout node validation
      - AT direct IIA
      - AP invariance
      - AT invariance
      - random-subspace controls for AP and AT

    Edge/path-level:
      - AP -> AT readout consistency
      - AP -> AT recovery / mediator replay
      - AP -> AT restoration
      - AP / AT conflict ordering

    AT restoration is intentionally omitted because AT is terminal immediately
    before Y in the current causal model; patching AT and immediately undoing it
    would be trivial rather than a meaningful mediation test.
    """
    device = next(model.parameters()).device
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    at_result = torch.load(at_checkpoint, map_location="cpu")
    ap_result = torch.load(ap_checkpoint, map_location="cpu")

    at_layer = int(at_result["layer"])
    ap_layer = int(ap_result["ap_layer"])
    Q_at = at_result["basis"].float().to(device)
    Q_ap = ap_result["ap_basis"].float().to(device)

    if ap_layer >= at_layer:
        raise ValueError(f"Expected AP upstream of AT, got L_AP={ap_layer}, L_AT={at_layer}.")
    if Q_ap.shape[0] != Q_at.shape[0]:
        raise ValueError(f"Hidden-size mismatch: Q_AP={tuple(Q_ap.shape)}, Q_AT={tuple(Q_at.shape)}")

    if at_readout_checkpoint is None:
        at_readout_checkpoint = ap_result.get("at_readout_checkpoint")
    if at_readout_checkpoint is None:
        raise ValueError("Need --at-readout-checkpoint, or an AP checkpoint containing 'at_readout_checkpoint'.")

    if ap_readout_checkpoint is None:
        ap_readout_checkpoint = ap_result.get("ap_readout_checkpoint")
    if ap_readout_checkpoint is None:
        raise ValueError("Need --ap-readout-checkpoint, or an AP checkpoint containing 'ap_readout_checkpoint'.")

    W_at, b_at, _ = load_linear_readout(at_readout_checkpoint, "AT", device)
    W_ap, b_ap, _ = load_linear_readout(ap_readout_checkpoint, "AP", device)

    ap_cfg = ap_result.get("config", {})
    at_cfg = at_result.get("config", {})
    ft_size = int(ap_cfg.get("ft_size", at_cfg.get("ft_size", 400)))
    cal_size = int(ap_cfg.get("cal_size", at_cfg.get("cal_size", 200)))
    te_size = int(ap_cfg.get("te_size", at_cfg.get("te_size", 200)))
    seed = int(ap_cfg.get("seed", at_cfg.get("seed", 0)))
    token_position = ap_result.get("token_position", at_result.get("token_position", "last_token"))

    set_seed(seed)
    labels = answer_label_ids(tokenizer)

    print(f"[alignment] AP=L{ap_layer}/k{Q_ap.shape[1]} AT=L{at_layer}/k{Q_at.shape[1]} token={token_position}")
    print(f"[split] ft={ft_size} cal={cal_size} test={te_size} seed={seed}")

    direct_banks, invariance_banks = build_eval_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        invariance_size=invariance_size,
        dataset_size=None,
        split="train",
        batch_size=batch_size,
        seed=seed,
    )

    ap_bank = direct_banks["answer_pointer"]
    at_bank = direct_banks["answer_token"]
    ap_inv_bank = invariance_banks["answer_pointer"]
    at_inv_bank = invariance_banks["answer_token"]

    # ------------------------------------------------------------------
    # Clean trajectories.
    # ------------------------------------------------------------------
    ap_clean_logits = run_clean(model, ap_bank, labels, batch_size)
    at_clean_logits = run_clean(model, at_bank, labels, batch_size)
    ap_inv_clean_logits = run_clean(model, ap_inv_bank, labels, batch_size)
    at_inv_clean_logits = run_clean(model, at_inv_bank, labels, batch_size)

    # ------------------------------------------------------------------
    # 1) DIRECT AP + capture the AT state produced downstream.
    # ------------------------------------------------------------------
    ap_source_states = collect_layer_states(
        model, ap_bank["source_input_ids"], ap_bank["source_attention_mask"],
        ap_bank["source_position_by_id"], ap_layer, token_position, batch_size,
    )
    ap_direct_logits, generated_at_states = run_ap_intervention_and_capture_at(
        model, ap_bank, ap_source_states, Q_ap, ap_layer, at_layer, labels,
        token_position, batch_size, require_grad=False,
    )

    ap_target = ap_bank["counterfactual_label_ids"]["answer_pointer"]
    at_target_after_ap = ap_bank["counterfactual_label_ids"]["answer_token"]
    ap_direct_iia = ap_iia(ap_direct_logits, ap_target)

    # ------------------------------------------------------------------
    # 2) AP READOUT NODE VALIDATION.
    # This checks whether the learned AP coordinates themselves are decodable.
    # We report clean-base and source accuracy. We do NOT count "AP readout
    # after AP patch" as a separate causal test: with an orthonormal Q_AP,
    # the patched Q_AP coordinates are algebraically the source coordinates.
    # ------------------------------------------------------------------
    clean_base_ap_states = collect_layer_states(
        model, ap_bank["base_input_ids"], ap_bank["base_attention_mask"],
        ap_bank["base_position_by_id"], ap_layer, token_position, batch_size,
    )
    ap_clean_readout_logits = (clean_base_ap_states.to(device) @ Q_ap) @ W_ap.T + b_ap
    ap_source_readout_logits = (ap_source_states.to(device) @ Q_ap) @ W_ap.T + b_ap

    ap_clean_readout_accuracy = full_accuracy(ap_clean_readout_logits.cpu(), ap_bank["base_answer_pointer_ids"])
    ap_source_readout_accuracy = full_accuracy(ap_source_readout_logits.cpu(), ap_bank["source_answer_pointer_ids"])

    # ------------------------------------------------------------------
    # 3) AP -> AT READOUT CONSISTENCY.
    # This is directly optimized by the Gradual DAS readout loss.
    # ------------------------------------------------------------------
    generated_at_device = generated_at_states.to(device)
    at_readout_logits_after_ap = (generated_at_device @ Q_at) @ W_at.T + b_at
    ap_to_at_readout_accuracy = full_accuracy(at_readout_logits_after_ap.cpu(), at_target_after_ap)

    clean_base_at_states = collect_layer_states(
        model, ap_bank["base_input_ids"], ap_bank["base_attention_mask"],
        ap_bank["base_position_by_id"], at_layer, token_position, batch_size,
    )
    source_at_states = collect_layer_states(
        model, ap_bank["source_input_ids"], ap_bank["source_attention_mask"],
        ap_bank["source_position_by_id"], at_layer, token_position, batch_size,
    )

    clean_readout_logits = (clean_base_at_states.to(device) @ Q_at) @ W_at.T + b_at
    source_readout_logits = (source_at_states.to(device) @ Q_at) @ W_at.T + b_at
    clean_readout_accuracy = full_accuracy(clean_readout_logits.cpu(), ap_bank["base_answer_label_ids"])
    source_readout_accuracy = full_accuracy(source_readout_logits.cpu(), at_target_after_ap)

    # ------------------------------------------------------------------
    # 4) AP -> AT RECOVERY / MEDIATOR REPLAY.
    # This is directly optimized by the Gradual DAS mediator loss.
    # ------------------------------------------------------------------
    recovery_logits = run_frozen_at_mediator(
        model, ap_bank, generated_at_states, Q_at, at_layer, labels,
        token_position, batch_size, require_grad=False,
    )
    recovery_iia = ap_iia(recovery_logits, ap_target)
    recovery_effect = recovery_effect_metrics(ap_clean_logits, ap_direct_logits, recovery_logits)

    # ------------------------------------------------------------------
    # 5) AP -> AT RESTORATION.
    # Out-of-objective test: Gradual DAS does not directly optimize this.
    # ------------------------------------------------------------------
    restored_logits = run_restoration(
        model, ap_bank, ap_source_states, clean_base_at_states, Q_ap, Q_at,
        ap_layer, at_layer, labels, token_position, batch_size,
    )
    restoration_base_accuracy = full_accuracy(restored_logits, ap_bank["base_answer_label_ids"])
    restoration_cf_iia = ap_iia(restored_logits, ap_target)
    restoration_effect = restoration_effect_metrics(ap_clean_logits, ap_direct_logits, restored_logits)

    # ------------------------------------------------------------------
    # 6) DIRECT AT.
    # ------------------------------------------------------------------
    at_source_states = collect_layer_states(
        model, at_bank["source_input_ids"], at_bank["source_attention_mask"],
        at_bank["source_position_by_id"], at_layer, token_position, batch_size,
    )
    at_direct_logits = run_das_intervention(
        model, at_bank, at_source_states, Q_at, at_layer, labels,
        token_position, batch_size, require_grad=False,
    )
    at_target = at_bank["counterfactual_label_ids"]["answer_token"]
    at_direct_iia = full_accuracy(at_direct_logits, at_target)

    # ------------------------------------------------------------------
    # 7) AP INVARIANCE: same AP, different AT, when such pairs exist.
    # ------------------------------------------------------------------
    ap_inv_source_states = collect_layer_states(
        model, ap_inv_bank["source_input_ids"], ap_inv_bank["source_attention_mask"],
        ap_inv_bank["source_position_by_id"], ap_layer, token_position, batch_size,
    )
    ap_inv_logits = run_das_intervention(
        model, ap_inv_bank, ap_inv_source_states, Q_ap, ap_layer, labels,
        token_position, batch_size, require_grad=False,
    )
    ap_invariance_accuracy = full_accuracy(ap_inv_logits, ap_inv_bank["base_answer_label_ids"])
    ap_invariance_drift = probability_drift_metrics(ap_inv_clean_logits, ap_inv_logits)

    # ------------------------------------------------------------------
    # 8) AT INVARIANCE: same AT, different AP, only when the dataset
    # actually contains such counterfactual pairs.
    # ------------------------------------------------------------------
    at_inv_source_states = collect_layer_states(
        model, at_inv_bank["source_input_ids"], at_inv_bank["source_attention_mask"],
        at_inv_bank["source_position_by_id"], at_layer, token_position, batch_size,
    )
    at_inv_logits = run_das_intervention(
        model, at_inv_bank, at_inv_source_states, Q_at, at_layer, labels,
        token_position, batch_size, require_grad=False,
    )
    at_invariance_accuracy = full_accuracy(at_inv_logits, at_inv_bank["base_answer_label_ids"])
    at_invariance_drift = probability_drift_metrics(at_inv_clean_logits, at_inv_logits)

    # ------------------------------------------------------------------
    # 9) AP / AT CONFLICT.
    # ------------------------------------------------------------------
    conflict_donor_idx = make_conflict_donors(
        ap_target.cpu(),
        ap_bank["base_answer_label_ids"].cpu(),
        at_target_after_ap.cpu(),
        seed,
    )
    conflict_donor_states = source_at_states[conflict_donor_idx]
    conflict_target = at_target_after_ap.cpu()[conflict_donor_idx]

    conflict_logits = run_conflict(
        model, ap_bank, ap_source_states, conflict_donor_states, Q_ap, Q_at,
        ap_layer, at_layer, labels, token_position, batch_size,
    )
    conflict_follows_at = full_accuracy(conflict_logits, conflict_target)
    conflict_follows_ap_full = full_accuracy(conflict_logits, ap_target)
    conflict_follows_ap_4way = ap_iia(conflict_logits, ap_target)
    conflict_follows_base = full_accuracy(conflict_logits, ap_bank["base_answer_label_ids"])

    # ------------------------------------------------------------------
    # 10) RANDOM-SUBSPACE CONTROLS.
    # Same layer and same rank as the learned alignment.
    # ------------------------------------------------------------------
    random_ap_rows = []
    random_at_rows = []

    for random_idx in range(num_random_controls):
        Q_ap_random = random_orthonormal_basis(Q_ap.shape[0], Q_ap.shape[1], seed + 1000 + random_idx, device)
        Q_at_random = random_orthonormal_basis(Q_at.shape[0], Q_at.shape[1], seed + 2000 + random_idx, device)

        random_ap_logits, random_generated_at = run_ap_intervention_and_capture_at(
            model, ap_bank, ap_source_states, Q_ap_random, ap_layer, at_layer, labels,
            token_position, batch_size, require_grad=False,
        )
        random_ap_recovery_logits = run_frozen_at_mediator(
            model, ap_bank, random_generated_at, Q_at, at_layer, labels,
            token_position, batch_size, require_grad=False,
        )
        random_generated_at_device = random_generated_at.to(device)
        random_ap_readout_logits = (random_generated_at_device @ Q_at) @ W_at.T + b_at

        random_ap_rows.append({
            "seed": seed + 1000 + random_idx,
            "direct_iia": ap_iia(random_ap_logits, ap_target),
            "recovery_iia": ap_iia(random_ap_recovery_logits, ap_target),
            "readout_accuracy": full_accuracy(random_ap_readout_logits.cpu(), at_target_after_ap),
        })

        random_at_logits = run_das_intervention(
            model, at_bank, at_source_states, Q_at_random, at_layer, labels,
            token_position, batch_size, require_grad=False,
        )
        random_at_rows.append({
            "seed": seed + 2000 + random_idx,
            "direct_iia": full_accuracy(random_at_logits, at_target),
        })

    mean_random_ap_direct = sum(row["direct_iia"] for row in random_ap_rows) / len(random_ap_rows)
    mean_random_ap_recovery = sum(row["recovery_iia"] for row in random_ap_rows) / len(random_ap_rows)
    mean_random_ap_readout = sum(row["readout_accuracy"] for row in random_ap_rows) / len(random_ap_rows)
    mean_random_at_direct = sum(row["direct_iia"] for row in random_at_rows) / len(random_at_rows)

    result = {
        "checkpoints": {
            "ap": ap_checkpoint,
            "at": at_checkpoint,
            "ap_readout": ap_readout_checkpoint,
            "at_readout": at_readout_checkpoint,
        },
        "alignment": {
            "ap_layer": ap_layer,
            "at_layer": at_layer,
            "ap_subspace_dim": int(Q_ap.shape[1]),
            "at_subspace_dim": int(Q_at.shape[1]),
            "token_position": token_position,
        },
        "clean": {
            "ap_bank_base_accuracy": full_accuracy(ap_clean_logits, ap_bank["base_answer_label_ids"]),
            "at_bank_base_accuracy": full_accuracy(at_clean_logits, at_bank["base_answer_label_ids"]),
        },
        "ap_direct": {
            "counterfactual_iia": ap_direct_iia,
        },
        "ap_readout": {
            "clean_base_accuracy": ap_clean_readout_accuracy,
            "source_accuracy": ap_source_readout_accuracy,
        },
        "at_direct": {
            "counterfactual_iia": at_direct_iia,
        },
        "ap_to_at_readout": {
            "clean_base_readout_accuracy": clean_readout_accuracy,
            "source_readout_accuracy": source_readout_accuracy,
            "after_ap_intervention_readout_accuracy": ap_to_at_readout_accuracy,
        },
        "ap_to_at_recovery": {
            "counterfactual_iia": recovery_iia,
            **recovery_effect,
        },
        "ap_to_at_restoration": {
            "restored_base_accuracy": restoration_base_accuracy,
            "counterfactual_iia_after_restore": restoration_cf_iia,
            **restoration_effect,
        },
        "ap_invariance": {
            "available": True,
            "num_examples": int(len(ap_inv_bank["base_input_ids"])),
            "same_ap_different_at_accuracy": ap_invariance_accuracy,
            **ap_invariance_drift,
        },
        "at_invariance": {
            "available": True,
            "num_examples": int(len(at_inv_bank["base_input_ids"])),
            "same_at_different_ap_accuracy": at_invariance_accuracy,
            **at_invariance_drift,
        },
        "conflict": {
            "follows_conflicting_at": conflict_follows_at,
            "follows_ap_full_space": conflict_follows_ap_full,
            "follows_ap_4way": conflict_follows_ap_4way,
            "follows_base": conflict_follows_base,
            "donor_indices": conflict_donor_idx,
            "conflict_targets": conflict_target,
        },
        "random_controls": {
            "ap": {
                "runs": random_ap_rows,
                "mean_direct_iia": mean_random_ap_direct,
                "mean_recovery_iia": mean_random_ap_recovery,
                "mean_readout_accuracy": mean_random_ap_readout,
            },
            "at": {
                "runs": random_at_rows,
                "mean_direct_iia": mean_random_at_direct,
            },
        },
        "config": {
            "ft_size": ft_size,
            "cal_size": cal_size,
            "te_size": te_size,
            "invariance_size": invariance_size,
            "num_random_controls": num_random_controls,
            "batch_size": batch_size,
            "seed": seed,
        },
    }

    print()
    print("===== ALL AP / AT CAUSAL TESTS =====")
    print(f"[AP DIRECT]       IIA={ap_direct_iia:.4f}")
    print(
        f"[AP READOUT]      clean={ap_clean_readout_accuracy:.4f} "
        f"source={ap_source_readout_accuracy:.4f}"
    )
    print(f"[AT DIRECT]       IIA={at_direct_iia:.4f}")
    print(
        f"[AT READOUT]      clean={clean_readout_accuracy:.4f} "
        f"source={source_readout_accuracy:.4f} after_AP={ap_to_at_readout_accuracy:.4f}"
    )
    print(
        f"[RECOVERY]        IIA={recovery_iia:.4f} "
        f"recovered={recovery_effect['mean_output_effect_recovered_fraction']:.4f} "
        f"cos={recovery_effect['shift_cosine']:.4f}"
    )
    print(
        f"[RESTORATION]     base_acc={restoration_base_accuracy:.4f} "
        f"cf_iia={restoration_cf_iia:.4f} "
        f"removed={restoration_effect['mean_output_effect_removed_fraction']:.4f}"
    )
    print(
        f"[AP INVARIANCE]   acc={ap_invariance_accuracy:.4f} "
        f"L1_drift={ap_invariance_drift['mean_l1_probability_drift']:.4f}"
    )
    print(
        f"[AT INVARIANCE]   acc={at_invariance_accuracy:.4f} "
        f"L1_drift={at_invariance_drift['mean_l1_probability_drift']:.4f}"
    )
    print(
        f"[CONFLICT]        follows_AT={conflict_follows_at:.4f} "
        f"follows_AP_full={conflict_follows_ap_full:.4f} "
        f"follows_base={conflict_follows_base:.4f}"
    )
    print(
        f"[RANDOM AP]       direct={mean_random_ap_direct:.4f} "
        f"recovery={mean_random_ap_recovery:.4f} readout={mean_random_ap_readout:.4f}"
    )
    print(f"[RANDOM AT]       direct={mean_random_at_direct:.4f}")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(result, save_path)
    print(f"saved to: {save_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--at-checkpoint", default="results/das_at.pt")
    parser.add_argument("--ap-checkpoint", default="results/standard_das_ap.pt")
    parser.add_argument("--at-readout-checkpoint", default="results/at_readout_LogisticRegression.pt")
    parser.add_argument("--ap-readout-checkpoint", default="results/standard_das_ap_readout_LogisticRegression.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--invariance-size", type=int, default=200)
    parser.add_argument("--num-random-controls", type=int, default=5)
    parser.add_argument("--save-path", default="results/standard_das_all_tests.pt")
    args = parser.parse_args()

    model, tokenizer = load_gemma_model()
    print(next(model.parameters()).device)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu")

    evaluate_all_tests(
        model=model,
        tokenizer=tokenizer,
        at_checkpoint=args.at_checkpoint,
        ap_checkpoint=args.ap_checkpoint,
        at_readout_checkpoint=args.at_readout_checkpoint,
        ap_readout_checkpoint=args.ap_readout_checkpoint,
        batch_size=args.batch_size,
        invariance_size=args.invariance_size,
        num_random_controls=args.num_random_controls,
        save_path=args.save_path,
    )


if __name__ == "__main__":
    main()