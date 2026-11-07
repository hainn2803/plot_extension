"""Core faithfulness evaluation for C -> AP -> AT -> Y.

Measures only three families:
1. Correctness/readout: joint node correctness after intervention, plus marginal node correctness.
2. Mediator replay: replay the immediate generated mediator into a clean base run.
3. Restoration: keep the upstream intervention, restore the immediate mediator DAS
   projection to its clean-base value, and test whether Y returns to the base answer.

This file is designed for the gradual-das-mcqa branch and cross-token AP checkpoints.
"""

import argparse
import csv
from pathlib import Path

import torch

from mcqa_gradual_das import answer_label_ids, answer_logits, collect_joint_token_states_in_single_layer, collect_layer_states, output_iia, output_label_ids, run_das_intervention, set_seed
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank
from mcqa_neural_net import load_gemma_model


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    Q = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    token_position = checkpoint["token_position"]
    return Q, layer, token_position


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    W = checkpoint[f"W_{name}"].float().to(device)
    b = checkpoint[f"b_{name}"].float().to(device)
    return W, b, checkpoint


def output_correct(logits, target, tokenizer, output_space):
    prediction = logits.argmax(dim=-1).cpu()

    if output_space == "az":
        return prediction == target.cpu()

    correct = []
    for predicted_id, target_id in zip(prediction, target):
        predicted_text = tokenizer.decode([int(predicted_id)]).strip().upper()
        expected_text = chr(ord("A") + int(target_id))
        correct.append(predicted_text == expected_text)

    return torch.tensor(correct, dtype=torch.bool)


def accuracy(correct):
    return float(correct.float().mean())


def projected_swap_state(base_state, source_state, Q):
    original_shape = base_state.shape
    base_flat = base_state.reshape(base_state.shape[0], -1).to(Q.device).float()
    source_flat = source_state.reshape(source_state.shape[0], -1).to(Q.device).float()
    swapped_flat = base_flat + ((source_flat - base_flat) @ Q) @ Q.T
    return swapped_flat.reshape(original_shape)


def readout_prediction(states, Q, W, b):
    flat = states.reshape(states.shape[0], -1).to(Q.device).float()
    logits = (flat @ Q) @ W.T + b
    return logits.argmax(dim=-1).cpu()


def color_targets(bank, color_to_class, side):
    labels = []
    for pair in bank["pairs"]:
        color = str(pair[f"{side}_color"]).strip().casefold()
        labels.append(color_to_class[color])
    return torch.tensor(labels, dtype=torch.long)


@torch.no_grad()
def clean_readout_accuracy(model, bank, Q, layer, token_position, W, b, target, batch_size, side):
    input_ids = bank[f"{side}_input_ids"]
    attention_mask = bank[f"{side}_attention_mask"]
    position_by_id = bank[f"{side}_position_by_id"]

    if isinstance(token_position, list):
        states = collect_joint_token_states_in_single_layer(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)
    else:
        states = collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)

    prediction = readout_prediction(states, Q, W, b)
    return float((prediction == target.cpu()).float().mean())


@torch.no_grad()
def run_ap_with_at_restoration(model, bank, source_ap, base_at, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, output_ids, batch_size):
    """Do AP <- source AP, then restore only the AT DAS projection to clean base."""
    if ap_layer >= at_layer:
        raise ValueError(f"Restoration requires AP layer < AT layer, got AP={ap_layer}, AT={at_layer}")

    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_ap = source_ap[start:end].to(device)
        clean_base_at = base_at[start:end].to(device)

        num_examples = ids.shape[0]
        rows = torch.arange(num_examples, device=device)
        pad_offset = (mask == 0).sum(dim=1)

        ap_positions = []
        for token_position in list_ap_token_positions:
            position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
            ap_positions.append(position)
        ap_positions = torch.stack(ap_positions, dim=1)

        at_position = pad_offset + bank["base_position_by_id"][at_token_position][start:end].to(device)

        def ap_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current_ap = hidden[rows[:, None], ap_positions, :].float()
            current_flat = current_ap.reshape(num_examples, -1)
            donor_flat = donor_ap.reshape(num_examples, -1).float()
            new_flat = current_flat + ((donor_flat - current_flat) @ Q_ap) @ Q_ap.T
            new_ap = new_flat.reshape_as(current_ap)
            new_hidden[rows[:, None], ap_positions, :] = new_ap.to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        def at_restore_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current_at = hidden[rows, at_position, :].float()
            restored_at = current_at + ((clean_base_at.float() - current_at) @ Q_at) @ Q_at.T
            new_hidden[rows, at_position, :] = restored_at.to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_restore_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)
            all_y.append(y.detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y, dim=0)


@torch.no_grad()
def run_c_with_ap_restoration(model, bank, source_c, base_ap, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, output_ids, batch_size):
    """Do C <- source C, then restore only the AP DAS projection to clean base."""
    if c_layer >= ap_layer:
        raise ValueError(f"Restoration requires C layer < AP layer, got C={c_layer}, AP={ap_layer}")

    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_c = source_c[start:end].to(device)
        clean_base_ap = base_ap[start:end].to(device)

        num_examples = ids.shape[0]
        rows = torch.arange(num_examples, device=device)
        pad_offset = (mask == 0).sum(dim=1)

        c_position = pad_offset + bank["base_position_by_id"][c_token_position][start:end].to(device)

        ap_positions = []
        for token_position in list_ap_token_positions:
            position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)
            ap_positions.append(position)
        ap_positions = torch.stack(ap_positions, dim=1)

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current_c = hidden[rows, c_position, :].float()
            new_c = current_c + ((donor_c.float() - current_c) @ Q_c) @ Q_c.T
            new_hidden[rows, c_position, :] = new_c.to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        def ap_restore_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current_ap = hidden[rows[:, None], ap_positions, :].float()
            current_flat = current_ap.reshape(num_examples, -1)
            base_flat = clean_base_ap.reshape(num_examples, -1).float()
            restored_flat = current_flat + ((base_flat - current_flat) @ Q_ap) @ Q_ap.T
            restored_ap = restored_flat.reshape_as(current_ap)
            new_hidden[rows[:, None], ap_positions, :] = restored_ap.to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_restore_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            y = answer_logits(model, outputs, mask, output_ids)
            all_y.append(y.detach().cpu())
        finally:
            c_handle.remove()
            ap_handle.remove()

    return torch.cat(all_y, dim=0)


@torch.no_grad()
def eval_at(model, tokenizer, bank, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    base_at = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], at_layer, at_token_position, batch_size)
    source_at = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], at_layer, at_token_position, batch_size)

    post_at = projected_swap_state(base_at, source_at, Q_at)
    direct_y = run_das_intervention(model, bank, source_at, Q_at, at_layer, output_ids, at_token_position, batch_size, require_grad=False)

    at_target = bank["intervention_targets"]["AT_after_AT_intervene"]
    y_target = bank["intervention_targets"]["Y_after_AT_intervene"]

    at_ok = readout_prediction(post_at, Q_at, W_at, b_at) == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)

    return {
        "Joint_AT_Y_after_AT": accuracy(at_ok & y_ok),
        "AT_after_AT": accuracy(at_ok),
        "Y_after_AT": accuracy(y_ok),
    }


@torch.no_grad()
def eval_ap(model, tokenizer, bank, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    base_ap = collect_joint_token_states_in_single_layer(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], ap_layer, list_ap_token_positions, batch_size)
    source_ap = collect_joint_token_states_in_single_layer(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], ap_layer, list_ap_token_positions, batch_size)
    base_at = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], at_layer, at_token_position, batch_size)

    post_ap = projected_swap_state(base_ap, source_ap, Q_ap)
    direct_y, generated_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, ap_layer, at_layer, output_ids, list_ap_token_positions, at_token_position, batch_size, False)
    replay_y = run_frozen_at_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, at_token_position, batch_size, False)
    restored_y = run_ap_with_at_restoration(model, bank, source_ap, base_at, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, output_ids, batch_size)

    ap_target = bank["intervention_targets"]["AP_after_AP_intervene"]
    at_target = bank["intervention_targets"]["AT_after_AP_intervene"]
    y_target = bank["intervention_targets"]["Y_after_AP_intervene"]
    y_base_target = bank["base_answer_label_ids"]

    ap_ok = readout_prediction(post_ap, Q_ap, W_ap, b_ap) == ap_target.cpu()
    at_ok = readout_prediction(generated_at, Q_at, W_at, b_at) == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)
    replay_ok = output_correct(replay_y, y_target, tokenizer, output_space)
    restore_ok = output_correct(restored_y, y_base_target, tokenizer, output_space)

    return {
        "Joint_AP_AT_Y_after_AP": accuracy(ap_ok & at_ok & y_ok),
        "AP_after_AP": accuracy(ap_ok),
        "AT_after_AP": accuracy(at_ok),
        "Y_after_AP": accuracy(y_ok),
        "Y_via_AT_replay": accuracy(replay_ok),
        "Y_restored_via_AT": accuracy(restore_ok),
    }


@torch.no_grad()
def eval_c(model, tokenizer, bank, Q_c, c_layer, c_token_position, W_c, b_c, color_to_class, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    base_c = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], c_layer, c_token_position, batch_size)
    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], c_layer, c_token_position, batch_size)
    base_ap = collect_joint_token_states_in_single_layer(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], ap_layer, list_ap_token_positions, batch_size)

    post_c = projected_swap_state(base_c, source_c, Q_c)
    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size, False)
    replay_y = run_frozen_mediator(model, bank, generated_ap, Q_ap, ap_layer, output_ids, list_ap_token_positions, batch_size, False)
    restored_y = run_c_with_ap_restoration(model, bank, source_c, base_ap, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, output_ids, batch_size)

    c_target = color_targets(bank, color_to_class, "source")
    ap_target = bank["intervention_targets"]["AP_after_C_intervene"]
    at_target = bank["intervention_targets"]["AT_after_C_intervene"]
    y_target = bank["intervention_targets"]["Y_after_C_intervene"]
    y_base_target = bank["base_answer_label_ids"]

    c_ok = readout_prediction(post_c, Q_c, W_c, b_c) == c_target.cpu()
    ap_ok = readout_prediction(generated_ap, Q_ap, W_ap, b_ap) == ap_target.cpu()
    at_ok = readout_prediction(generated_at, Q_at, W_at, b_at) == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)
    replay_ok = output_correct(replay_y, y_target, tokenizer, output_space)
    restore_ok = output_correct(restored_y, y_base_target, tokenizer, output_space)

    return {
        "Joint_C_AP_AT_Y_after_C": accuracy(c_ok & ap_ok & at_ok & y_ok),
        "C_after_C": accuracy(c_ok),
        "AP_after_C": accuracy(ap_ok),
        "AT_after_C": accuracy(at_ok),
        "Y_after_C": accuracy(y_ok),
        "Y_via_AP_replay": accuracy(replay_ok),
        "Y_restored_via_AP": accuracy(restore_ok),
    }


def add_metric(rows, family, intervention, dataset, metric, value):
    rows.append({
        "family": family,
        "intervention": intervention,
        "dataset": dataset,
        "metric": metric,
        "value": float(value),
    })


def format_block(lines, title, metrics, names):
    lines.append(title)
    for name in names:
        lines.append(f"{name} = {metrics[name]:.4f}")
    lines.append("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default="results_joint_randomized_standard_cross_tokens_seed0")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    result_dir = Path(args.result_dir)

    Q_at, at_layer, at_token_position = load_ckpt(result_dir / "das_at.pt", device)
    Q_ap, ap_layer, list_ap_token_positions = load_ckpt(result_dir / "das_ap.pt", device)
    Q_c, c_layer, c_token_position = load_ckpt(result_dir / "das_c.pt", device)

    W_at, b_at, _ = load_readout(result_dir / "at_readout_LogisticRegression.pt", "AT", device)
    W_ap, b_ap, _ = load_readout(result_dir / "ap_readout_LogisticRegression.pt", "AP", device)
    W_c, b_c, c_info = load_readout(result_dir / "c_readout_LogisticRegression.pt", "C", device)

    if not isinstance(list_ap_token_positions, list):
        raise ValueError(f"AP token_position must be a list, got {type(list_ap_token_positions)}")

    expected_ap_width = len(list_ap_token_positions) * model.config.hidden_size
    if Q_ap.shape[0] != expected_ap_width:
        raise ValueError(f"AP basis width {Q_ap.shape[0]} != {len(list_ap_token_positions)} x {model.config.hidden_size} = {expected_ap_width}")
    if Q_at.shape[0] != model.config.hidden_size:
        raise ValueError(f"AT basis width {Q_at.shape[0]} != hidden size {model.config.hidden_size}")
    if Q_c.shape[0] != model.config.hidden_size:
        raise ValueError(f"C basis width {Q_c.shape[0]} != hidden size {model.config.hidden_size}")

    print("=== BUILD EVAL BANKS ===")
    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.test_size, args.batch_size, args.seed + 1)
    at_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AT_only", args.test_size, args.batch_size, args.seed + 2)
    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.test_size, args.batch_size, args.seed + 3)
    c_ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 4)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 5)

    color_to_class = c_info["color_to_class"]

    print("\n=== CLEAN READOUT SANITY ===")
    ap_base = clean_readout_accuracy(model, ap_only, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_only["base_answer_pointer_ids"], args.batch_size, "base")
    ap_source = clean_readout_accuracy(model, ap_only, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, ap_only["source_answer_pointer_ids"], args.batch_size, "source")
    at_base = clean_readout_accuracy(model, at_only, Q_at, at_layer, at_token_position, W_at, b_at, at_only["base_answer_label_ids"], args.batch_size, "base")
    at_source = clean_readout_accuracy(model, at_only, Q_at, at_layer, at_token_position, W_at, b_at, at_only["source_answer_label_ids"], args.batch_size, "source")
    c_base_target = color_targets(c_only, color_to_class, "base")
    c_source_target = color_targets(c_only, color_to_class, "source")
    c_base = clean_readout_accuracy(model, c_only, Q_c, c_layer, c_token_position, W_c, b_c, c_base_target, args.batch_size, "base")
    c_source = clean_readout_accuracy(model, c_only, Q_c, c_layer, c_token_position, W_c, b_c, c_source_target, args.batch_size, "source")
    print(f"AP base/source: {ap_base:.4f} / {ap_source:.4f}")
    print(f"AT base/source: {at_base:.4f} / {at_source:.4f}")
    print(f"C  base/source: {c_base:.4f} / {c_source:.4f}")

    at_results = {"AT_only": eval_at(model, tokenizer, at_only, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)}

    ap_results = {}
    for name, bank in (("AP_only", ap_only), ("AP_AT", ap_at)):
        ap_results[name] = eval_ap(model, tokenizer, bank, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)

    c_results = {}
    for name, bank in (("C_only", c_only), ("C_AP_AT", c_ap_at)):
        c_results[name] = eval_c(model, tokenizer, bank, Q_c, c_layer, c_token_position, W_c, b_c, color_to_class, Q_ap, ap_layer, list_ap_token_positions, W_ap, b_ap, Q_at, at_layer, at_token_position, W_at, b_at, args.batch_size, args.output_space)

    lines = []
    rows = []

    lines.append("=== CLEAN READOUT SANITY ===")
    lines.append(f"AP base/source: {ap_base:.4f} / {ap_source:.4f}")
    lines.append(f"AT base/source: {at_base:.4f} / {at_source:.4f}")
    lines.append(f"C  base/source: {c_base:.4f} / {c_source:.4f}")
    lines.append("")

    lines.append("=== 1. CORRECTNESS / READOUT ===")
    lines.append("")

    for name, metrics in at_results.items():
        format_block(lines, f"NODE CORRECTNESS — AT intervention [{name}]", metrics, ["Joint_AT_Y_after_AT", "AT_after_AT", "Y_after_AT"])
        for metric in ("Joint_AT_Y_after_AT", "AT_after_AT", "Y_after_AT"):
            add_metric(rows, "correctness", "AT", name, metric, metrics[metric])

    for name, metrics in ap_results.items():
        format_block(lines, f"NODE CORRECTNESS — AP intervention [{name}]", metrics, ["Joint_AP_AT_Y_after_AP", "AP_after_AP", "AT_after_AP", "Y_after_AP"])
        for metric in ("Joint_AP_AT_Y_after_AP", "AP_after_AP", "AT_after_AP", "Y_after_AP"):
            add_metric(rows, "correctness", "AP", name, metric, metrics[metric])

    for name, metrics in c_results.items():
        format_block(lines, f"NODE CORRECTNESS — C intervention [{name}]", metrics, ["Joint_C_AP_AT_Y_after_C", "C_after_C", "AP_after_C", "AT_after_C", "Y_after_C"])
        for metric in ("Joint_C_AP_AT_Y_after_C", "C_after_C", "AP_after_C", "AT_after_C", "Y_after_C"):
            add_metric(rows, "correctness", "C", name, metric, metrics[metric])

    lines.append("=== 2. MEDIATOR REPLAY ===")
    lines.append("")

    for name, metrics in ap_results.items():
        format_block(lines, f"MEDIATOR — AP intervention [{name}]", metrics, ["Y_via_AT_replay"])
        add_metric(rows, "mediator", "AP", name, "Y_via_AT_replay", metrics["Y_via_AT_replay"])

    for name, metrics in c_results.items():
        format_block(lines, f"MEDIATOR — C intervention [{name}]", metrics, ["Y_via_AP_replay"])
        add_metric(rows, "mediator", "C", name, "Y_via_AP_replay", metrics["Y_via_AP_replay"])

    lines.append("=== 3. RESTORATION ===")
    lines.append("")

    for name, metrics in ap_results.items():
        format_block(lines, f"RESTORATION — AP intervention [{name}]", metrics, ["Y_restored_via_AT"])
        add_metric(rows, "restoration", "AP", name, "Y_restored_via_AT", metrics["Y_restored_via_AT"])

    for name, metrics in c_results.items():
        format_block(lines, f"RESTORATION — C intervention [{name}]", metrics, ["Y_restored_via_AP"])
        add_metric(rows, "restoration", "C", name, "Y_restored_via_AP", metrics["Y_restored_via_AP"])

    text = "\n".join(lines).rstrip() + "\n"
    print("\n" + text)

    txt_path = result_dir / "eval_core_faithfulness.txt"
    csv_path = result_dir / "eval_core_faithfulness.csv"

    with open(txt_path, "w") as file:
        file.write(text)

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["family", "intervention", "dataset", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"TXT: {txt_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
