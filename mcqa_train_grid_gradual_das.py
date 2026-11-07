"""Train a fixed-layer Gradual DAS grid and evaluate full-chain causal faithfulness.

Default grid:
    AT: 23,24,25
    AP: 14,13
    C:  1,0

Training:
    AT: direct Y loss (same as Standard DAS because there is no downstream latent)
    AP: direct Y + AT mediator replay + AT readout
    C:  direct Y + AP mediator replay + AP readout + AT mediator replay + AT readout

Training banks are the same mixed banks used by the Standard grid:
    AT: AT_only + AP_AT + C_AP_AT
    AP: AP_only + AP_AT + C_AP_AT
    C:  C_only + C_AP_AT

Evaluation uses fixed test banks only:
    AT_only, AP_only, C_only

For each of the 12 (AT, AP, C) configurations, report:
    correctness/readout
    mediator replay
    restoration

C intervention includes both AP and AT replay/restoration.
"""

import argparse
import csv
import shutil
from pathlib import Path

import torch

import mcqa_gradual_das_ap_cross_tokens as ap_module
from mcqa_gradual_das import (
    answer_label_ids,
    answer_logits,
    collect_joint_token_states_in_single_layer,
    collect_layer_states,
    fit_readout,
    output_label_ids,
    set_seed,
)
from mcqa_gradual_das_at import train_at
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import train_c, run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import (
    build_ap_readout_datasets,
    build_at_readout_datasets,
    build_c_readout_datasets,
    build_pair_bank,
    build_shared_readout_banks,
    create_bank,
)
from mcqa_neural_net import load_gemma_model


PAIR_TYPE_OFFSETS = {
    "AT_only": 11,
    "AP_only": 23,
    "C_only": 37,
    "AP_AT": 51,
    "C_AP_AT": 67,
}


def parse_int_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def parse_str_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


def balanced_counts(total, names):
    base = total // len(names)
    remainder = total % len(names)
    counts = {}
    for i, name in enumerate(names):
        counts[name] = base + int(i < remainder)
    return counts


def make_specs(at_token_position, ap_token_positions, c_token_position):
    return {
        "AT": {
            "pair_types": ["AT_only"],
            "token_position": at_token_position,
            "target": "Y_after_AT_intervene",
        },
        "AP": {
            "pair_types": ["AP_only"],
            "token_position": ap_token_positions,
            "target": "Y_after_AP_intervene",
        },
        "C": {
            "pair_types": ["C_only"],
            "token_position": c_token_position,
            "target": "Y_after_C_intervene",
        },
    }


def make_variable_banks(model, tokenizer, data_path, ft_size, cal_size, te_size, batch_size, seed, specs):
    split_sizes = {"train": ft_size, "validation": cal_size, "test": te_size}
    split_offsets = {"train": 0, "validation": 1000, "test": 2000}

    variable_banks = {}
    variable_components = {}
    for variable in specs:
        variable_banks[variable] = {}
        variable_components[variable] = {}

    for split, total_size in split_sizes.items():
        needed = {}
        counts_by_variable = {}

        for variable, spec in specs.items():
            counts = balanced_counts(total_size, spec["pair_types"])
            counts_by_variable[variable] = counts
            for pair_type, count in counts.items():
                needed[pair_type] = max(needed.get(pair_type, 0), count)

        cached = {}
        for pair_type, count in needed.items():
            pair_seed = seed + split_offsets[split] + PAIR_TYPE_OFFSETS[pair_type]
            cached[pair_type] = build_pair_bank(model, tokenizer, data_path, split, pair_type, count, batch_size, pair_seed)
            print(f"[bank] split={split} pair_type={pair_type} n={count}")

        for variable, spec in specs.items():
            mixed_pairs = []
            variable_components[variable][split] = {}

            for pair_type in spec["pair_types"]:
                count = counts_by_variable[variable][pair_type]
                pairs = list(cached[pair_type]["pairs"][:count])
                component_seed = seed + split_offsets[split] + PAIR_TYPE_OFFSETS[pair_type] + 100
                component_bank = create_bank(pairs, tokenizer, size=None, seed=component_seed)
                variable_components[variable][split][pair_type] = component_bank
                mixed_pairs.extend(pairs)

            mixed_seed = seed + split_offsets[split] + {"AT": 101, "AP": 202, "C": 303}[variable]
            variable_banks[variable][split] = create_bank(mixed_pairs, tokenizer, size=None, seed=mixed_seed)

            counts_text = []
            for pair_type in spec["pair_types"]:
                counts_text.append(f"{pair_type}={counts_by_variable[variable][pair_type]}")
            print(f"[{variable} {split}] total={len(mixed_pairs)} | " + " ".join(counts_text))

    banks = {}
    for variable in specs:
        banks[variable] = (
            variable_banks[variable]["train"],
            {variable: variable_banks[variable]["validation"]},
            {variable: variable_banks[variable]["test"]},
        )

    return banks, variable_components


# -----------------------------------------------------------------------------
# Pure Standard DAS training at a fixed layer
# -----------------------------------------------------------------------------



def subset_base_bank(bank, idx):
    positions = {}
    for name, values in bank["base_position_by_id"].items():
        positions[name] = values[idx]

    return {
        "base_input_ids": bank["base_input_ids"][idx],
        "base_attention_mask": bank["base_attention_mask"][idx],
        "base_position_by_id": positions,
    }


def collect_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size):
    if isinstance(token_position, list):
        return collect_joint_token_states_in_single_layer(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)
    return collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)


def flatten_states(states):
    if states.ndim == 2:
        return states
    return states.reshape(states.shape[0], -1)


def batch_positions(bank, token_position, start, end, mask, device):
    pad_offset = (mask == 0).sum(dim=1)

    if isinstance(token_position, list):
        positions = []
        for name in token_position:
            position = pad_offset + bank["base_position_by_id"][name][start:end].to(device)
            positions.append(position)
        return torch.stack(positions, dim=1)

    return pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)


def run_das_intervention_generic(model, bank, source_states, Q, layer, output_ids, token_position, batch_size, require_grad):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = source_states[start:end].to(device=device, dtype=torch.float32)
        rows = torch.arange(len(ids), device=device)
        positions = batch_positions(bank, token_position, start, end, mask, device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()

            if positions.ndim == 1:
                base_states = hidden[rows, positions, :].float()
            else:
                base_states = hidden[rows[:, None], positions, :].float()

            original_shape = base_states.shape
            base_flat = flatten_states(base_states)
            donor_flat = flatten_states(donor)
            new_flat = base_flat + ((donor_flat - base_flat) @ Q) @ Q.T

            if positions.ndim == 1:
                new_hidden[rows, positions, :] = new_flat.to(hidden.dtype)
            else:
                new_hidden[rows[:, None], positions, :] = new_flat.reshape(original_shape).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, output_ids)
            if require_grad:
                all_logits.append(logits)
            else:
                all_logits.append(logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


# -----------------------------------------------------------------------------
# Fixed-layer Gradual DAS training
# -----------------------------------------------------------------------------


def train_at_fixed(model, tokenizer, banks, layer, args, checkpoint_path):
    if checkpoint_path.exists() and not args.retrain:
        print(f"[reuse AT checkpoint] {checkpoint_path}")
        return

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n================ TRAIN GRADUAL AT L{layer} ================")
    train_at(
        model=model,
        tokenizer=tokenizer,
        layers=[layer],
        subspace_dim=args.at_dim,
        epochs=args.at_epochs,
        lr=args.lr,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.at_token_position,
        seed=args.seed,
        save_path=str(checkpoint_path),
        banks=banks["AT"],
        output_space=args.output_space,
    )


def train_ap_fixed(model, tokenizer, banks, at_checkpoint, at_readout, at_layer, ap_layer, ap_token_positions, args, checkpoint_path):
    if checkpoint_path.exists() and not args.retrain:
        print(f"[reuse AP checkpoint] {checkpoint_path}")
        return

    if ap_layer >= at_layer:
        raise ValueError(f"AP layer {ap_layer} must be before AT layer {at_layer}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ap_module.AP_LAYER = ap_layer

    print(f"\n================ TRAIN GRADUAL AP L{ap_layer} | AT L{at_layer} ================")
    ap_module.train_ap(
        model=model,
        tokenizer=tokenizer,
        at_checkpoint=str(at_checkpoint),
        at_readout_checkpoint=str(at_readout),
        layers=[ap_layer],
        subspace_dim=args.ap_dim,
        epochs=args.ap_epochs,
        lr=args.lr,
        lambda_med=args.ap_lambda_med,
        lambda_readout=args.ap_lambda_readout,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        list_ap_token_positions=ap_token_positions,
        at_token_position=args.at_token_position,
        seed=args.seed,
        save_path=str(checkpoint_path),
        banks=banks["AP"],
        output_space=args.output_space,
    )


def train_c_fixed(model, tokenizer, banks, ap_checkpoint, ap_readout, at_checkpoint, at_readout, at_layer, ap_layer, c_layer, args, checkpoint_path):
    if checkpoint_path.exists() and not args.retrain:
        print(f"[reuse C checkpoint] {checkpoint_path}")
        return

    if c_layer >= min(ap_layer, at_layer):
        raise ValueError(f"C layer {c_layer} must be before AP={ap_layer} and AT={at_layer}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n================ TRAIN GRADUAL C L{c_layer} | AP L{ap_layer} | AT L{at_layer} ================")
    train_c(
        model=model,
        tokenizer=tokenizer,
        ap_checkpoint=str(ap_checkpoint),
        ap_readout_checkpoint=str(ap_readout),
        at_checkpoint=str(at_checkpoint),
        at_readout_checkpoint=str(at_readout),
        layers=[c_layer],
        subspace_dim=args.c_dim,
        epochs=args.c_epochs,
        lr=args.lr,
        lambda_ap_med=args.c_lambda_ap_med,
        lambda_ap_readout=args.c_lambda_ap_readout,
        lambda_at_med=args.c_lambda_at_med,
        lambda_at_readout=args.c_lambda_at_readout,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        token_position=args.c_token_position,
        seed=args.seed,
        save_path=str(checkpoint_path),
        banks=banks["C"],
        output_space=args.output_space,
    )


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint["basis"].float().to(device), int(checkpoint["layer"]), checkpoint["token_position"]


def load_readout(path, name, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    W = checkpoint[f"W_{name}"].float().to(device)
    b = checkpoint[f"b_{name}"].float().to(device)
    return W, b, checkpoint


def fit_one_readout(model, checkpoint_path, readout_path, variable, shared_readout, token_position, batch_size, seed):
    if readout_path.exists():
        print(f"[reuse readout] {readout_path}")
        return

    positions = token_position if isinstance(token_position, list) else [token_position]

    if variable == "AT":
        bank_pair = build_at_readout_datasets(shared_readout)
        target = "AT"
        num_classes = 26
        extra = None
    elif variable == "AP":
        bank_pair = build_ap_readout_datasets(shared_readout)
        target = "AP"
        num_classes = 4
        extra = None
    elif variable == "C":
        bank_pair = build_c_readout_datasets(shared_readout)
        target = "C"
        colors = shared_readout[0]["color_to_class"]
        num_classes = len(colors)
        extra = {"color_to_class": colors}
    else:
        raise ValueError(variable)

    fit_readout(model, checkpoint_path, readout_path, bank_pair, target, positions, batch_size, seed, num_classes, extra)


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
    states = collect_states(model, bank[f"{side}_input_ids"], bank[f"{side}_attention_mask"], bank[f"{side}_position_by_id"], layer, token_position, batch_size)
    prediction = readout_prediction(states, Q, W, b)
    return float((prediction == target.cpu()).float().mean())


# -----------------------------------------------------------------------------
# Restoration interventions
# -----------------------------------------------------------------------------


@torch.no_grad()
def run_ap_with_at_restoration(model, bank, source_ap, base_at, Q_ap, ap_layer, list_ap_token_positions, Q_at, at_layer, at_token_position, output_ids, batch_size):
    """AP <- source AP, then restore only the AT DAS projection to clean base."""
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
            new_hidden[rows[:, None], ap_positions, :] = new_flat.reshape_as(current_ap).to(hidden.dtype)
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
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y, dim=0)


@torch.no_grad()
def run_c_with_ap_restoration(model, bank, source_c, base_ap, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, output_ids, batch_size):
    """C <- source C, then restore only the AP DAS projection to clean base."""
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
            new_hidden[rows[:, None], ap_positions, :] = restored_flat.reshape_as(current_ap).to(hidden.dtype)
            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        ap_handle = model.model.layers[ap_layer].register_forward_hook(ap_restore_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            c_handle.remove()
            ap_handle.remove()

    return torch.cat(all_y, dim=0)


@torch.no_grad()
def run_c_with_at_restoration(model, bank, source_c, base_at, Q_c, c_layer, c_token_position, Q_at, at_layer, at_token_position, output_ids, batch_size):
    """C <- source C, then restore only the AT DAS projection to clean base."""
    if c_layer >= at_layer:
        raise ValueError(f"Restoration requires C layer < AT layer, got C={c_layer}, AT={at_layer}")

    device = next(model.parameters()).device
    all_y = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor_c = source_c[start:end].to(device)
        clean_base_at = base_at[start:end].to(device)
        rows = torch.arange(len(ids), device=device)
        pad_offset = (mask == 0).sum(dim=1)

        c_position = pad_offset + bank["base_position_by_id"][c_token_position][start:end].to(device)
        at_position = pad_offset + bank["base_position_by_id"][at_token_position][start:end].to(device)

        def c_hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            new_hidden = hidden.clone()
            current_c = hidden[rows, c_position, :].float()
            new_c = current_c + ((donor_c.float() - current_c) @ Q_c) @ Q_c.T
            new_hidden[rows, c_position, :] = new_c.to(hidden.dtype)
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

        c_handle = model.model.layers[c_layer].register_forward_hook(c_hook)
        at_handle = model.model.layers[at_layer].register_forward_hook(at_restore_hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            all_y.append(answer_logits(model, outputs, mask, output_ids).detach().cpu())
        finally:
            c_handle.remove()
            at_handle.remove()

    return torch.cat(all_y, dim=0)


# -----------------------------------------------------------------------------
# Core faithfulness evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def eval_at(model, tokenizer, bank, Q_at, at_layer, at_token_position, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    base_at = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], at_layer, at_token_position, batch_size)
    source_at = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], at_layer, at_token_position, batch_size)

    post_at = projected_swap_state(base_at, source_at, Q_at)
    direct_y = run_das_intervention_generic(model, bank, source_at, Q_at, at_layer, output_ids, at_token_position, batch_size, False)

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
    base_at = collect_layer_states(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], at_layer, at_token_position, batch_size)

    post_c = projected_swap_state(base_c, source_c, Q_c)
    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, c_layer, ap_layer, at_layer, output_ids, c_token_position, list_ap_token_positions, at_token_position, batch_size, False)
    ap_replay_y = run_frozen_mediator(model, bank, generated_ap, Q_ap, ap_layer, output_ids, list_ap_token_positions, batch_size, False)
    at_replay_y = run_frozen_mediator(model, bank, generated_at, Q_at, at_layer, output_ids, [at_token_position], batch_size, False)
    restored_ap_y = run_c_with_ap_restoration(model, bank, source_c, base_ap, Q_c, c_layer, c_token_position, Q_ap, ap_layer, list_ap_token_positions, output_ids, batch_size)
    restored_at_y = run_c_with_at_restoration(model, bank, source_c, base_at, Q_c, c_layer, c_token_position, Q_at, at_layer, at_token_position, output_ids, batch_size)

    c_target = color_targets(bank, color_to_class, "source")
    ap_target = bank["intervention_targets"]["AP_after_C_intervene"]
    at_target = bank["intervention_targets"]["AT_after_C_intervene"]
    y_target = bank["intervention_targets"]["Y_after_C_intervene"]
    y_base_target = bank["base_answer_label_ids"]

    c_ok = readout_prediction(post_c, Q_c, W_c, b_c) == c_target.cpu()
    ap_ok = readout_prediction(generated_ap, Q_ap, W_ap, b_ap) == ap_target.cpu()
    at_ok = readout_prediction(generated_at, Q_at, W_at, b_at) == at_target.cpu()
    y_ok = output_correct(direct_y, y_target, tokenizer, output_space)
    ap_replay_ok = output_correct(ap_replay_y, y_target, tokenizer, output_space)
    at_replay_ok = output_correct(at_replay_y, y_target, tokenizer, output_space)
    restore_ap_ok = output_correct(restored_ap_y, y_base_target, tokenizer, output_space)
    restore_at_ok = output_correct(restored_at_y, y_base_target, tokenizer, output_space)

    return {
        "Joint_C_AP_AT_Y_after_C": accuracy(c_ok & ap_ok & at_ok & y_ok),
        "C_after_C": accuracy(c_ok),
        "AP_after_C": accuracy(ap_ok),
        "AT_after_C": accuracy(at_ok),
        "Y_after_C": accuracy(y_ok),
        "Y_via_AP_replay": accuracy(ap_replay_ok),
        "Y_via_AT_replay": accuracy(at_replay_ok),
        "Y_restored_via_AP": accuracy(restore_ap_ok),
        "Y_restored_via_AT": accuracy(restore_at_ok),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def add_metric(rows, at_layer, ap_layer, c_layer, family, intervention, dataset, metric, value):
    rows.append({
        "at_layer": int(at_layer),
        "ap_layer": int(ap_layer),
        "c_layer": int(c_layer),
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


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_config_report(config_dir, at_layer, ap_layer, c_layer, clean, at_metrics, ap_metrics, c_metrics):
    lines = []
    lines.append(f"CONFIG: AT=L{at_layer} AP=L{ap_layer} C=L{c_layer}")
    lines.append("")
    lines.append("=== CLEAN READOUT SANITY ===")
    lines.append(f"AT base/source: {clean['AT_base']:.4f} / {clean['AT_source']:.4f}")
    lines.append(f"AP base/source: {clean['AP_base']:.4f} / {clean['AP_source']:.4f}")
    lines.append(f"C  base/source: {clean['C_base']:.4f} / {clean['C_source']:.4f}")
    lines.append("")

    lines.append("=== 1. CORRECTNESS / READOUT ===")
    lines.append("")
    format_block(lines, "NODE CORRECTNESS — AT intervention [AT_only]", at_metrics, ["Joint_AT_Y_after_AT", "AT_after_AT", "Y_after_AT"])

    format_block(lines, "NODE CORRECTNESS — AP intervention [AP_only]", ap_metrics["AP_only"], ["Joint_AP_AT_Y_after_AP", "AP_after_AP", "AT_after_AP", "Y_after_AP"])

    format_block(lines, "NODE CORRECTNESS — C intervention [C_only]", c_metrics["C_only"], ["Joint_C_AP_AT_Y_after_C", "C_after_C", "AP_after_C", "AT_after_C", "Y_after_C"])

    lines.append("=== 2. MEDIATOR REPLAY ===")
    lines.append("")
    format_block(lines, "MEDIATOR — AP intervention [AP_only]", ap_metrics["AP_only"], ["Y_via_AT_replay"])
    format_block(lines, "MEDIATOR — C intervention [C_only]", c_metrics["C_only"], ["Y_via_AP_replay", "Y_via_AT_replay"])

    lines.append("=== 3. RESTORATION ===")
    lines.append("")
    format_block(lines, "RESTORATION — AP intervention [AP_only]", ap_metrics["AP_only"], ["Y_restored_via_AT"])
    format_block(lines, "RESTORATION — C intervention [C_only]", c_metrics["C_only"], ["Y_restored_via_AP", "Y_restored_via_AT"])

    text = "\n".join(lines).rstrip() + "\n"
    config_dir.mkdir(parents=True, exist_ok=True)
    with open(config_dir / "eval_faithfulness.txt", "w") as file:
        file.write(text)
    return text


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default="results_gradual_das_layer_grid_seed0_AP_only")

    parser.add_argument("--at-layers", default="23,24,25")
    parser.add_argument("--ap-layers", default="14,13")
    parser.add_argument("--c-layers", default="1,0")

    parser.add_argument("--at-token-position", default="last_token")
    parser.add_argument("--ap-token-positions", default="color_0,color_1,color_2,color_3")
    parser.add_argument("--c-token-position", default="color")

    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--eval-size", type=int, default=400)

    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)

    parser.add_argument("--at-epochs", type=int, default=10)
    parser.add_argument("--ap-epochs", type=int, default=10)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=32)

    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)

    parser.add_argument("--ap-lambda-med", type=float, default=2.0)
    parser.add_argument("--ap-lambda-readout", type=float, default=2.0)
    parser.add_argument("--c-lambda-ap-med", type=float, default=4.0)
    parser.add_argument("--c-lambda-ap-readout", type=float, default=4.0)
    parser.add_argument("--c-lambda-at-med", type=float, default=0.0)
    parser.add_argument("--c-lambda-at-readout", type=float, default=0.0)

    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    at_layers = parse_int_list(args.at_layers)
    ap_layers = parse_int_list(args.ap_layers)
    c_layers = parse_int_list(args.c_layers)
    ap_token_positions = parse_str_list(args.ap_token_positions)

    if not at_layers or not ap_layers or not c_layers:
        raise ValueError("Layer lists must be non-empty")
    if not ap_token_positions:
        raise ValueError("AP token positions must be non-empty")

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            if ap_layer >= at_layer:
                raise ValueError(f"Need AP < AT, got AP={ap_layer}, AT={at_layer}")
    for ap_layer in ap_layers:
        for c_layer in c_layers:
            if c_layer >= ap_layer:
                raise ValueError(f"Need C < AP, got C={c_layer}, AP={ap_layer}")

    specs = make_specs(args.at_token_position, ap_token_positions, args.c_token_position)

    result_dir = Path(args.result_dir)
    trained_dir = result_dir / "trained_handles"
    configs_dir = result_dir / "configs"
    trained_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    print("=== GRADUAL DAS GRID ===")
    print(f"AT layers: {at_layers}")
    print(f"AP layers: {ap_layers}")
    print(f"C layers : {c_layers}")
    print(f"Total configs: {len(at_layers) * len(ap_layers) * len(c_layers)}")
    print(f"AP objective: direct + {args.ap_lambda_med}*AT_mediator + {args.ap_lambda_readout}*AT_readout")
    print(
        "C objective: direct "
        f"+ {args.c_lambda_ap_med}*AP_mediator "
        f"+ {args.c_lambda_ap_readout}*AP_readout "
        f"+ {args.c_lambda_at_med}*AT_mediator "
        f"+ {args.c_lambda_at_readout}*AT_readout"
    )

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    print("\n=== BUILD CONTROLLED-ONLY TRAIN/CAL/TEST BANKS ===")
    banks, _ = make_variable_banks(
        model,
        tokenizer,
        args.data_path,
        args.ft_size,
        args.cal_size,
        args.te_size,
        args.eval_batch_size,
        args.seed,
        specs,
    )

    print("\n=== BUILD SHARED READOUT DATA ===")
    shared_readout = build_shared_readout_banks(
        model,
        tokenizer,
        args.data_path,
        args.readout_train_size,
        args.readout_val_size,
        args.eval_batch_size,
        args.seed,
    )

    at_checkpoint_paths = {}
    at_readout_paths = {}
    ap_checkpoint_paths = {}
    ap_readout_paths = {}
    c_checkpoint_paths = {}
    c_readout_paths = {}

    print("\n=== STAGE 1: TRAIN AT HANDLES ===")
    for at_layer in at_layers:
        handle_dir = trained_dir / f"AT_L{at_layer}"
        checkpoint_path = handle_dir / "das_at.pt"
        readout_path = handle_dir / "at_readout_LogisticRegression.pt"

        train_at_fixed(model, tokenizer, banks, at_layer, args, checkpoint_path)

        if args.retrain and readout_path.exists():
            readout_path.unlink()

        fit_one_readout(
            model=model,
            checkpoint_path=checkpoint_path,
            readout_path=readout_path,
            variable="AT",
            shared_readout=shared_readout,
            token_position=args.at_token_position,
            batch_size=args.eval_batch_size,
            seed=args.seed,
        )

        at_checkpoint_paths[at_layer] = checkpoint_path
        at_readout_paths[at_layer] = readout_path

    print("\n=== STAGE 2: TRAIN AP HANDLES CONDITIONED ON AT ===")
    for at_layer in at_layers:
        for ap_layer in ap_layers:
            key = (at_layer, ap_layer)
            handle_dir = trained_dir / f"AT_L{at_layer}__AP_L{ap_layer}"
            checkpoint_path = handle_dir / "das_ap.pt"
            readout_path = handle_dir / "ap_readout_LogisticRegression.pt"

            train_ap_fixed(
                model,
                tokenizer,
                banks,
                at_checkpoint_paths[at_layer],
                at_readout_paths[at_layer],
                at_layer,
                ap_layer,
                ap_token_positions,
                args,
                checkpoint_path,
            )

            if args.retrain and readout_path.exists():
                readout_path.unlink()

            fit_one_readout(
                model=model,
                checkpoint_path=checkpoint_path,
                readout_path=readout_path,
                variable="AP",
                shared_readout=shared_readout,
                token_position=ap_token_positions,
                batch_size=args.eval_batch_size,
                seed=args.seed,
            )

            ap_checkpoint_paths[key] = checkpoint_path
            ap_readout_paths[key] = readout_path

    print("\n=== STAGE 3: TRAIN C HANDLES CONDITIONED ON AP + AT ===")
    for at_layer in at_layers:
        for ap_layer in ap_layers:
            ap_key = (at_layer, ap_layer)

            for c_layer in c_layers:
                key = (at_layer, ap_layer, c_layer)
                handle_dir = trained_dir / f"AT_L{at_layer}__AP_L{ap_layer}__C_L{c_layer}"
                checkpoint_path = handle_dir / "das_c.pt"
                readout_path = handle_dir / "c_readout_LogisticRegression.pt"

                train_c_fixed(
                    model,
                    tokenizer,
                    banks,
                    ap_checkpoint_paths[ap_key],
                    ap_readout_paths[ap_key],
                    at_checkpoint_paths[at_layer],
                    at_readout_paths[at_layer],
                    at_layer,
                    ap_layer,
                    c_layer,
                    args,
                    checkpoint_path,
                )

                if args.retrain and readout_path.exists():
                    readout_path.unlink()

                fit_one_readout(
                    model=model,
                    checkpoint_path=checkpoint_path,
                    readout_path=readout_path,
                    variable="C",
                    shared_readout=shared_readout,
                    token_position=args.c_token_position,
                    batch_size=args.eval_batch_size,
                    seed=args.seed,
                )

                c_checkpoint_paths[key] = checkpoint_path
                c_readout_paths[key] = readout_path

    print("\n=== BUILD FIXED ONLY EVAL BANKS ===")
    at_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AT_only", args.eval_size, args.eval_batch_size, args.seed + 3001)
    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.eval_size, args.eval_batch_size, args.seed + 3002)
    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.eval_size, args.eval_batch_size, args.seed + 3004)

    at_handles = {}
    ap_handles = {}
    c_handles = {}

    for at_layer in at_layers:
        Q, ckpt_layer, token_position = load_ckpt(at_checkpoint_paths[at_layer], device)
        W, b, _ = load_readout(at_readout_paths[at_layer], "AT", device)
        at_handles[at_layer] = (Q, ckpt_layer, token_position, W, b)

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            key = (at_layer, ap_layer)
            Q, ckpt_layer, token_position = load_ckpt(ap_checkpoint_paths[key], device)
            W, b, _ = load_readout(ap_readout_paths[key], "AP", device)
            ap_handles[key] = (Q, ckpt_layer, token_position, W, b)

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            for c_layer in c_layers:
                key = (at_layer, ap_layer, c_layer)
                Q, ckpt_layer, token_position = load_ckpt(c_checkpoint_paths[key], device)
                W, b, info = load_readout(c_readout_paths[key], "C", device)
                c_handles[key] = (Q, ckpt_layer, token_position, W, b, info["color_to_class"])

    print("\n=== CACHE CLEAN READOUTS ===")
    clean_at = {}
    clean_ap = {}
    clean_c = {}

    for at_layer in at_layers:
        Q_at, _, at_pos, W_at, b_at = at_handles[at_layer]
        clean_at[at_layer] = {
            "base": clean_readout_accuracy(model, at_only, Q_at, at_layer, at_pos, W_at, b_at, at_only["base_answer_label_ids"], args.eval_batch_size, "base"),
            "source": clean_readout_accuracy(model, at_only, Q_at, at_layer, at_pos, W_at, b_at, at_only["source_answer_label_ids"], args.eval_batch_size, "source"),
        }

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            key = (at_layer, ap_layer)
            Q_ap, _, ap_pos, W_ap, b_ap = ap_handles[key]
            clean_ap[key] = {
                "base": clean_readout_accuracy(model, ap_only, Q_ap, ap_layer, ap_pos, W_ap, b_ap, ap_only["base_answer_pointer_ids"], args.eval_batch_size, "base"),
                "source": clean_readout_accuracy(model, ap_only, Q_ap, ap_layer, ap_pos, W_ap, b_ap, ap_only["source_answer_pointer_ids"], args.eval_batch_size, "source"),
            }

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            for c_layer in c_layers:
                key = (at_layer, ap_layer, c_layer)
                Q_c, _, c_pos, W_c, b_c, colors = c_handles[key]
                clean_c[key] = {
                    "base": clean_readout_accuracy(model, c_only, Q_c, c_layer, c_pos, W_c, b_c, color_targets(c_only, colors, "base"), args.eval_batch_size, "base"),
                    "source": clean_readout_accuracy(model, c_only, Q_c, c_layer, c_pos, W_c, b_c, color_targets(c_only, colors, "source"), args.eval_batch_size, "source"),
                }

    print("\n=== EVALUATE GRADUAL GRID ===")
    at_eval_cache = {}
    ap_eval_cache = {}
    c_eval_cache = {}

    for at_layer in at_layers:
        Q_at, _, at_pos, W_at, b_at = at_handles[at_layer]
        at_eval_cache[at_layer] = eval_at(
            model,
            tokenizer,
            at_only,
            Q_at,
            at_layer,
            at_pos,
            W_at,
            b_at,
            args.eval_batch_size,
            args.output_space,
        )

    for at_layer in at_layers:
        Q_at, _, at_pos, W_at, b_at = at_handles[at_layer]

        for ap_layer in ap_layers:
            key = (at_layer, ap_layer)
            Q_ap, _, ap_pos, W_ap, b_ap = ap_handles[key]
            ap_eval_cache[key] = {
                "AP_only": eval_ap(
                    model,
                    tokenizer,
                    ap_only,
                    Q_ap,
                    ap_layer,
                    ap_pos,
                    W_ap,
                    b_ap,
                    Q_at,
                    at_layer,
                    at_pos,
                    W_at,
                    b_at,
                    args.eval_batch_size,
                    args.output_space,
                )
            }

    for at_layer in at_layers:
        Q_at, _, at_pos, W_at, b_at = at_handles[at_layer]

        for ap_layer in ap_layers:
            ap_key = (at_layer, ap_layer)
            Q_ap, _, ap_pos, W_ap, b_ap = ap_handles[ap_key]

            for c_layer in c_layers:
                key = (at_layer, ap_layer, c_layer)
                Q_c, _, c_pos, W_c, b_c, colors = c_handles[key]

                c_eval_cache[key] = {
                    "C_only": eval_c(
                        model,
                        tokenizer,
                        c_only,
                        Q_c,
                        c_layer,
                        c_pos,
                        W_c,
                        b_c,
                        colors,
                        Q_ap,
                        ap_layer,
                        ap_pos,
                        W_ap,
                        b_ap,
                        Q_at,
                        at_layer,
                        at_pos,
                        W_at,
                        b_at,
                        args.eval_batch_size,
                        args.output_space,
                    )
                }

    all_rows = []
    summary_rows = []

    for at_layer in at_layers:
        for ap_layer in ap_layers:
            for c_layer in c_layers:
                ap_key = (at_layer, ap_layer)
                key = (at_layer, ap_layer, c_layer)

                config_name = f"AT_L{at_layer}__AP_L{ap_layer}__C_L{c_layer}"
                config_dir = configs_dir / config_name
                config_dir.mkdir(parents=True, exist_ok=True)

                shutil.copy2(at_checkpoint_paths[at_layer], config_dir / "das_at.pt")
                shutil.copy2(ap_checkpoint_paths[ap_key], config_dir / "das_ap.pt")
                shutil.copy2(c_checkpoint_paths[key], config_dir / "das_c.pt")
                shutil.copy2(at_readout_paths[at_layer], config_dir / "at_readout_LogisticRegression.pt")
                shutil.copy2(ap_readout_paths[ap_key], config_dir / "ap_readout_LogisticRegression.pt")
                shutil.copy2(c_readout_paths[key], config_dir / "c_readout_LogisticRegression.pt")

                clean = {
                    "AT_base": clean_at[at_layer]["base"],
                    "AT_source": clean_at[at_layer]["source"],
                    "AP_base": clean_ap[ap_key]["base"],
                    "AP_source": clean_ap[ap_key]["source"],
                    "C_base": clean_c[key]["base"],
                    "C_source": clean_c[key]["source"],
                }

                at_metrics = at_eval_cache[at_layer]
                ap_metrics = ap_eval_cache[ap_key]
                c_metrics = c_eval_cache[key]

                text = write_config_report(
                    config_dir,
                    at_layer,
                    ap_layer,
                    c_layer,
                    clean,
                    at_metrics,
                    ap_metrics,
                    c_metrics,
                )

                print("\n" + "=" * 80)
                print(text.rstrip())

                config_rows = []

                for metric in ("Joint_AT_Y_after_AT", "AT_after_AT", "Y_after_AT"):
                    add_metric(config_rows, at_layer, ap_layer, c_layer, "correctness", "AT", "AT_only", metric, at_metrics[metric])

                ap_only_metrics = ap_metrics["AP_only"]

                for metric in ("Joint_AP_AT_Y_after_AP", "AP_after_AP", "AT_after_AP", "Y_after_AP"):
                    add_metric(config_rows, at_layer, ap_layer, c_layer, "correctness", "AP", "AP_only", metric, ap_only_metrics[metric])

                add_metric(config_rows, at_layer, ap_layer, c_layer, "mediator", "AP", "AP_only", "Y_via_AT_replay", ap_only_metrics["Y_via_AT_replay"])
                add_metric(config_rows, at_layer, ap_layer, c_layer, "restoration", "AP", "AP_only", "Y_restored_via_AT", ap_only_metrics["Y_restored_via_AT"])

                c_only_metrics = c_metrics["C_only"]

                for metric in ("Joint_C_AP_AT_Y_after_C", "C_after_C", "AP_after_C", "AT_after_C", "Y_after_C"):
                    add_metric(config_rows, at_layer, ap_layer, c_layer, "correctness", "C", "C_only", metric, c_only_metrics[metric])

                add_metric(config_rows, at_layer, ap_layer, c_layer, "mediator", "C", "C_only", "Y_via_AP_replay", c_only_metrics["Y_via_AP_replay"])
                add_metric(config_rows, at_layer, ap_layer, c_layer, "mediator", "C", "C_only", "Y_via_AT_replay", c_only_metrics["Y_via_AT_replay"])
                add_metric(config_rows, at_layer, ap_layer, c_layer, "restoration", "C", "C_only", "Y_restored_via_AP", c_only_metrics["Y_restored_via_AP"])
                add_metric(config_rows, at_layer, ap_layer, c_layer, "restoration", "C", "C_only", "Y_restored_via_AT", c_only_metrics["Y_restored_via_AT"])

                save_csv(config_rows, config_dir / "eval_faithfulness.csv")
                all_rows.extend(config_rows)

                summary_rows.append({
                    "at_layer": at_layer,
                    "ap_layer": ap_layer,
                    "c_layer": c_layer,
                    "AT_only_joint": at_metrics["Joint_AT_Y_after_AT"],
                    "AT_only_AT": at_metrics["AT_after_AT"],
                    "AT_only_Y": at_metrics["Y_after_AT"],
                    "AP_only_joint": ap_only_metrics["Joint_AP_AT_Y_after_AP"],
                    "AP_only_AP": ap_only_metrics["AP_after_AP"],
                    "AP_only_AT": ap_only_metrics["AT_after_AP"],
                    "AP_only_Y": ap_only_metrics["Y_after_AP"],
                    "AP_only_mediator": ap_only_metrics["Y_via_AT_replay"],
                    "AP_only_restoration": ap_only_metrics["Y_restored_via_AT"],
                    "C_only_joint": c_only_metrics["Joint_C_AP_AT_Y_after_C"],
                    "C_only_C": c_only_metrics["C_after_C"],
                    "C_only_AP": c_only_metrics["AP_after_C"],
                    "C_only_AT": c_only_metrics["AT_after_C"],
                    "C_only_Y": c_only_metrics["Y_after_C"],
                    "C_only_AP_mediator": c_only_metrics["Y_via_AP_replay"],
                    "C_only_AT_mediator": c_only_metrics["Y_via_AT_replay"],
                    "C_only_AP_restoration": c_only_metrics["Y_restored_via_AP"],
                    "C_only_AT_restoration": c_only_metrics["Y_restored_via_AT"],
                    "clean_AT_base": clean["AT_base"],
                    "clean_AT_source": clean["AT_source"],
                    "clean_AP_base": clean["AP_base"],
                    "clean_AP_source": clean["AP_source"],
                    "clean_C_base": clean["C_base"],
                    "clean_C_source": clean["C_source"],
                })

    save_csv(all_rows, result_dir / "all_config_metrics.csv")
    save_csv(summary_rows, result_dir / "config_summary.csv")

    print("\n=== DONE ===")
    print(f"Root: {result_dir}")
    print(f"Detailed metrics: {result_dir / 'all_config_metrics.csv'}")
    print(f"One-row-per-config summary: {result_dir / 'config_summary.csv'}")
    print(f"Config directories: {configs_dir}")


if __name__ == "__main__":
    main()
