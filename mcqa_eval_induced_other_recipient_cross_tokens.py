"""Evaluate cross-context transfer of states induced by C and AP interventions with cross-token AP."""

import argparse
import random
from pathlib import Path

import torch

from mcqa_constants import LETTER_TO_CLASS
from mcqa_gradual_das import answer_label_ids, answer_logits, collect_joint_token_states_in_single_layer, collect_layer_states, output_iia, output_label_ids, set_seed
from mcqa_gradual_das_ap_cross_tokens import run_ap_intervention_and_capture_at, run_frozen_at_mediator
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank, color_position, factual_rows_correct, load_generated_dataset, option_color_positions, symbol_position
from mcqa_neural_net import load_gemma_model


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    basis = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    position = checkpoint.get("token_position", "last_token")
    return basis, layer, position


def accuracy(logits, target):
    prediction = logits.argmax(-1).cpu()
    return float((prediction == target.cpu()).float().mean())


def letter_from_class(class_id):
    return chr(ord("A") + int(class_id))


def row_position(row, tokenizer, token_position, target_ap=None):
    if token_position == "color":
        return color_position(row["prompt"], tokenizer)

    if token_position.startswith("color_"):
        option_index = int(token_position.split("_")[1])
        positions = option_color_positions(row["prompt"], tokenizer)
        return positions[option_index]

    if token_position == "golden_letter":
        return symbol_position(row["prompt"], str(row["answer"]), tokenizer)

    if token_position == "source_ap_slot_label":
        if target_ap is None:
            raise ValueError("target_ap is required for source_ap_slot_label")

        letter = str(row["choices"]["label"][int(target_ap)])
        return symbol_position(row["prompt"], letter, tokenizer)

    raise ValueError(f"Unsupported token position: {token_position}")


def build_recipient_bank(rows, tokenizer, token_positions, ap_targets=None):
    tokenizer.padding_side = "left"

    prompts = []

    for row in rows:
        prompts.append(row["prompt"])

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    last_token_positions = inputs["attention_mask"].sum(1).long() - 1

    position_by_id = {}

    for token_position in token_positions:
        if token_position in position_by_id:
            continue

        if token_position == "last_token":
            position_by_id[token_position] = last_token_positions
            continue

        positions = []

        for i in range(len(rows)):
            target_ap = None

            if ap_targets is not None:
                target_ap = int(ap_targets[i])

            positions.append(row_position(rows[i], tokenizer, token_position, target_ap))

        position_by_id[token_position] = torch.tensor(positions)

    return {
        "base_input_ids": inputs["input_ids"].long(),
        "base_attention_mask": inputs["attention_mask"].long(),
        "base_position_by_id": position_by_id,
    }


def select_ap_recipients(donor_bank, candidate_rows, condition, seed):
    rng = random.Random(seed)
    target_aps = donor_bank["intervention_targets"]["AP_after_C_intervene"]
    selected = []

    for i in range(len(donor_bank["pairs"])):
        donor_pair = donor_bank["pairs"][i]
        target_ap = int(target_aps[i])
        candidates = []

        for row in candidate_rows:
            if row["question_key"] == donor_pair["question"]:
                continue

            natural_ap = int(row["answer_position"])

            if condition == "same":
                if natural_ap != target_ap:
                    continue
            elif condition == "different":
                if natural_ap == target_ap:
                    continue
            else:
                raise ValueError(f"Unknown condition: {condition}")

            candidates.append(row)

        if len(candidates) == 0:
            raise RuntimeError(f"No AP recipient found for donor {i}, target AP={target_ap}, condition={condition}")

        recipient = rng.choice(candidates)
        selected.append(recipient)

    return selected

def select_at_recipients(donor_bank, target_key, candidate_rows, condition, seed):
    rng = random.Random(seed)
    target_ats = donor_bank["intervention_targets"][target_key]
    selected = []

    for i in range(len(donor_bank["pairs"])):
        donor_pair = donor_bank["pairs"][i]
        target_letter = letter_from_class(target_ats[i])
        candidates = []

        for row in candidate_rows:
            if row["question_key"] == donor_pair["question"]:
                continue

            natural_at = str(row["answer"])
            labels = []

            for label in row["choices"]["label"]:
                labels.append(str(label))

            if condition == "same":
                if natural_at != target_letter:
                    continue
            elif condition == "different":
                if natural_at == target_letter:
                    continue

                if target_letter not in labels:
                    continue
            else:
                raise ValueError(f"Unknown condition: {condition}")

            candidates.append(row)

        if len(candidates) == 0:
            raise RuntimeError(f"No AT recipient found for donor {i}, target AT={target_letter}, condition={condition}")

        recipient = rng.choice(candidates)
        selected.append(recipient)

    return selected

def ap_recipient_at_targets(rows, donor_bank):
    target_aps = donor_bank["intervention_targets"]["AP_after_C_intervene"]
    target_ats = []

    for i in range(len(rows)):
        target_ap = int(target_aps[i])
        target_letter = str(rows[i]["choices"]["label"][target_ap])
        target_ats.append(LETTER_TO_CLASS[target_letter])

    return torch.tensor(target_ats)


@torch.no_grad()
def replay_ap_capture_at(model, bank, generated_ap, Q_ap, L_ap, list_pos_ap, L_at, pos_at, output_ids, batch_size):
    device = next(model.parameters()).device
    all_y = []
    all_at = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))

        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        donor = generated_ap[start:end].to(device)

        num_examples = len(ids)
        num_ap_tokens = len(list_pos_ap)
        rows = torch.arange(num_examples, device=device)
        pad = (mask == 0).sum(1)

        ap_positions = []

        for pos_ap in list_pos_ap:
            ap_position = pad + bank["base_position_by_id"][pos_ap][start:end].to(device)
            ap_positions.append(ap_position)

        ap_positions = torch.stack(ap_positions, dim=1)
        at_idx = pad + bank["base_position_by_id"][pos_at][start:end].to(device)

        captured = {}

        def ap_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base_state = hidden[rows[:, None], ap_positions, :].float()
            base_flat = base_state.reshape(num_examples, -1)
            donor_flat = donor.reshape(num_examples, -1).float()
            delta_flat = ((donor_flat - base_flat) @ Q_ap) @ Q_ap.T
            new_ap = (base_flat + delta_flat).reshape(num_examples, num_ap_tokens, -1)
            new_hidden[rows[:, None], ap_positions, :] = new_ap.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]

            return new_hidden

        def at_hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            captured["at"] = hidden[rows, at_idx].float()

        ap_handle = model.model.layers[L_ap].register_forward_hook(ap_hook)
        at_handle = model.model.layers[L_at].register_forward_hook(at_hook)

        try:
            position_ids = (mask.long().cumsum(-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)

            y = answer_logits(model, outputs, mask, output_ids)

            all_y.append(y.detach().cpu())
            all_at.append(captured["at"].detach().cpu())
        finally:
            ap_handle.remove()
            at_handle.remove()

    return torch.cat(all_y), torch.cat(all_at)


@torch.no_grad()
def eval_c_induced_transfer(model, tokenizer, donor_bank, ap_same_bank, ap_same_rows, ap_different_bank, ap_different_rows, at_same_bank, at_different_bank, Q_c, L_c, pos_c, Q_ap, L_ap, list_pos_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_c = collect_layer_states(model, donor_bank["source_input_ids"], donor_bank["source_attention_mask"], donor_bank["source_position_by_id"], L_c, pos_c, batch_size)

    direct_y, generated_ap, generated_at = run_c_intervention_and_capture(model, donor_bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, pos_c, list_pos_ap, pos_at, batch_size, False)

    donor_at_target = donor_bank["intervention_targets"]["AT_after_C_intervene"]
    donor_y_target = donor_bank["intervention_targets"]["Y_after_C_intervene"]

    base_ap_y, base_ap_at = replay_ap_capture_at(model, donor_bank, generated_ap, Q_ap, L_ap, list_pos_ap, L_at, pos_at, output_ids, batch_size)
    base_ap_at_logits = (base_ap_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    base_at_y = run_frozen_mediator(model, donor_bank, generated_at, Q_at, L_at, output_ids, [pos_at], batch_size, False)

    ap_same_target = ap_recipient_at_targets(ap_same_rows, donor_bank)
    ap_same_y, ap_same_at = replay_ap_capture_at(model, ap_same_bank, generated_ap, Q_ap, L_ap, list_pos_ap, L_at, pos_at, output_ids, batch_size)
    ap_same_at_logits = (ap_same_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    ap_different_target = ap_recipient_at_targets(ap_different_rows, donor_bank)
    ap_different_y, ap_different_at = replay_ap_capture_at(model, ap_different_bank, generated_ap, Q_ap, L_ap, list_pos_ap, L_at, pos_at, output_ids, batch_size)
    ap_different_at_logits = (ap_different_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    at_same_y = run_frozen_mediator(model, at_same_bank, generated_at, Q_at, L_at, output_ids, [pos_at], batch_size, False)
    at_different_y = run_frozen_mediator(model, at_different_bank, generated_at, Q_at, L_at, output_ids, [pos_at], batch_size, False)

    return {
        "Y_after_C": output_iia(direct_y, donor_y_target, tokenizer, output_space),
        "AT_via_AP_replay_base": accuracy(base_ap_at_logits, donor_at_target),
        "Y_via_AP_replay_base": output_iia(base_ap_y, donor_y_target, tokenizer, output_space),
        "Y_via_AT_replay_base": output_iia(base_at_y, donor_at_target, tokenizer, output_space),
        "AT_via_AP_replay_recipient_same": accuracy(ap_same_at_logits, ap_same_target),
        "Y_via_AP_replay_recipient_same": output_iia(ap_same_y, ap_same_target, tokenizer, output_space),
        "AT_via_AP_replay_recipient_different": accuracy(ap_different_at_logits, ap_different_target),
        "Y_via_AP_replay_recipient_different": output_iia(ap_different_y, ap_different_target, tokenizer, output_space),
        "Y_via_AT_replay_recipient_same": output_iia(at_same_y, donor_at_target, tokenizer, output_space),
        "Y_via_AT_replay_recipient_different": output_iia(at_different_y, donor_at_target, tokenizer, output_space),
    }


@torch.no_grad()
def eval_ap_induced_transfer(model, tokenizer, donor_bank, at_same_bank, at_different_bank, Q_ap, L_ap, list_pos_ap, Q_at, L_at, pos_at, W_at, b_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_ap = collect_joint_token_states_in_single_layer(model, donor_bank["source_input_ids"], donor_bank["source_attention_mask"], donor_bank["source_position_by_id"], L_ap, list_pos_ap, batch_size)

    direct_y, generated_at = run_ap_intervention_and_capture_at(model, donor_bank, source_ap, Q_ap, L_ap, L_at, output_ids, list_pos_ap, pos_at, batch_size, False)

    at_target = donor_bank["intervention_targets"]["AT_after_AP_intervene"]
    y_target = donor_bank["intervention_targets"]["Y_after_AP_intervene"]

    generated_at_logits = (generated_at.to(Q_at.device) @ Q_at) @ W_at.T + b_at

    base_at_y = run_frozen_at_mediator(model, donor_bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    same_at_y = run_frozen_at_mediator(model, at_same_bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    different_at_y = run_frozen_at_mediator(model, at_different_bank, generated_at, Q_at, L_at, output_ids, pos_at, batch_size, False)

    return {
        "Y_after_AP": output_iia(direct_y, y_target, tokenizer, output_space),
        "AT_after_AP": accuracy(generated_at_logits, at_target),
        "Y_via_AT_replay_base": output_iia(base_at_y, at_target, tokenizer, output_space),
        "Y_via_AT_replay_recipient_same": output_iia(same_at_y, at_target, tokenizer, output_space),
        "Y_via_AT_replay_recipient_different": output_iia(different_at_y, at_target, tokenizer, output_space),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_cross_tokens_seed0_reprcac001")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    result_dir = Path(args.result_dir)
    seed_dir = result_dir.parent
    # shared_at_dir = seed_dir / "shared_at"
    shared_at_dir = result_dir

    Q_at, L_at, pos_at = load_ckpt(shared_at_dir / "das_at.pt", device)
    Q_ap, L_ap, list_pos_ap = load_ckpt(result_dir / "das_ap.pt", device)
    Q_c, L_c, pos_c = load_ckpt(result_dir / "das_c.pt", device)

    if not isinstance(list_pos_ap, list):
        list_pos_ap = [list_pos_ap] # compatible for old result
        # raise ValueError(f"AP token_position must be a list, got {type(list_pos_ap)}")

    expected_ap_width = len(list_pos_ap) * model.config.hidden_size

    if Q_ap.shape[0] != expected_ap_width:
        raise ValueError(f"AP basis width {Q_ap.shape[0]} does not match {len(list_pos_ap)} tokens x hidden size {model.config.hidden_size} = {expected_ap_width}")

    at_readout = torch.load(shared_at_dir / "at_readout_LogisticRegression.pt", map_location="cpu", weights_only=False)
    # at_readout = torch.load(result_dir / "at_readout_LogisticRegression.pt", map_location="cpu", weights_only=False)
    W_at = at_readout["W_AT"].float().to(device)
    b_at = at_readout["b_AT"].float().to(device)

    dataset = load_generated_dataset(args.data_path)

    candidate_rows = []

    for i in range(len(dataset["test"])):
        candidate_rows.append(dataset["test"][i])

    candidate_rows = factual_rows_correct(model, tokenizer, candidate_rows, args.batch_size)

    print(f"[recipient pool] correct factual test rows: {len(candidate_rows)}")

    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.test_size, args.batch_size, args.seed + 1)
    c_joint = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 2)
    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.test_size, args.batch_size, args.seed + 3)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 4)

    c_results = {}

    for offset, item in enumerate((("C_only", c_only), ("C_AP_AT", c_joint))):
        name, donor_bank = item
        target_aps = donor_bank["intervention_targets"]["AP_after_C_intervene"]

        ap_same_rows = select_ap_recipients(donor_bank, candidate_rows, "same", args.seed + 100 + offset)
        ap_different_rows = select_ap_recipients(donor_bank, candidate_rows, "different", args.seed + 200 + offset)

        at_same_rows = select_at_recipients(donor_bank, "AT_after_C_intervene", candidate_rows, "same", args.seed + 300 + offset)
        at_different_rows = select_at_recipients(donor_bank, "AT_after_C_intervene", candidate_rows, "different", args.seed + 400 + offset)

        ap_same_bank = build_recipient_bank(ap_same_rows, tokenizer, list_pos_ap + [pos_at], target_aps)
        ap_different_bank = build_recipient_bank(ap_different_rows, tokenizer, list_pos_ap + [pos_at], target_aps)
        at_same_bank = build_recipient_bank(at_same_rows, tokenizer, [pos_at])
        at_different_bank = build_recipient_bank(at_different_rows, tokenizer, [pos_at])

        c_results[name] = eval_c_induced_transfer(model, tokenizer, donor_bank, ap_same_bank, ap_same_rows, ap_different_bank, ap_different_rows, at_same_bank, at_different_bank, Q_c, L_c, pos_c, Q_ap, L_ap, list_pos_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    ap_results = {}

    for offset, item in enumerate((("AP_only", ap_only), ("AP_AT", ap_at))):
        name, donor_bank = item

        at_same_rows = select_at_recipients(donor_bank, "AT_after_AP_intervene", candidate_rows, "same", args.seed + 500 + offset)
        at_different_rows = select_at_recipients(donor_bank, "AT_after_AP_intervene", candidate_rows, "different", args.seed + 600 + offset)

        at_same_bank = build_recipient_bank(at_same_rows, tokenizer, [pos_at])
        at_different_bank = build_recipient_bank(at_different_rows, tokenizer, [pos_at])

        ap_results[name] = eval_ap_induced_transfer(model, tokenizer, donor_bank, at_same_bank, at_different_bank, Q_ap, L_ap, list_pos_ap, Q_at, L_at, pos_at, W_at, b_at, args.batch_size, args.output_space)

    lines = []

    for name in ("C_only", "C_AP_AT"):
        metrics = c_results[name]

        lines.append(f"=== C INTERVENTION -> INDUCED AP/AT TRANSFER [{name}] ===")
        lines.append(f"Y_after_C: {metrics['Y_after_C']:.4f}")
        lines.append("C-INDUCED AP -> BASE")
        lines.append(f"AT_via_AP_replay_base: {metrics['AT_via_AP_replay_base']:.4f}")
        lines.append(f"Y_via_AP_replay_base: {metrics['Y_via_AP_replay_base']:.4f}")
        lines.append("C-INDUCED AP -> RECIPIENT")
        lines.append(f"AT_via_AP_replay_recipient_same: {metrics['AT_via_AP_replay_recipient_same']:.4f}")
        lines.append(f"Y_via_AP_replay_recipient_same: {metrics['Y_via_AP_replay_recipient_same']:.4f}")
        lines.append(f"AT_via_AP_replay_recipient_different: {metrics['AT_via_AP_replay_recipient_different']:.4f}")
        lines.append(f"Y_via_AP_replay_recipient_different: {metrics['Y_via_AP_replay_recipient_different']:.4f}")
        lines.append("C-INDUCED AT -> BASE / RECIPIENT")
        lines.append(f"Y_via_AT_replay_base: {metrics['Y_via_AT_replay_base']:.4f}")
        lines.append(f"Y_via_AT_replay_recipient_same: {metrics['Y_via_AT_replay_recipient_same']:.4f}")
        lines.append(f"Y_via_AT_replay_recipient_different: {metrics['Y_via_AT_replay_recipient_different']:.4f}")
        lines.append("")

    for name in ("AP_only", "AP_AT"):
        metrics = ap_results[name]

        lines.append(f"=== AP INTERVENTION -> INDUCED AT TRANSFER [{name}] ===")
        lines.append(f"Y_after_AP: {metrics['Y_after_AP']:.4f}")
        lines.append(f"AT_after_AP: {metrics['AT_after_AP']:.4f}")
        lines.append(f"Y_via_AT_replay_base: {metrics['Y_via_AT_replay_base']:.4f}")
        lines.append(f"Y_via_AT_replay_recipient_same: {metrics['Y_via_AT_replay_recipient_same']:.4f}")
        lines.append(f"Y_via_AT_replay_recipient_different: {metrics['Y_via_AT_replay_recipient_different']:.4f}")
        lines.append("")

    text = "\n".join(lines).strip()

    print("\n" + text)

    output_path = result_dir / "eval_induced_other_recipient.txt"
    output_path.write_text(text + "\n")

    print(f"\nsaved results = {output_path}")


if __name__ == "__main__":
    main()
