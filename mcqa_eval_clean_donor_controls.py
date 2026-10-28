"""Matched clean-donor controls for C and AP interventions."""

import argparse
import random
import unicodedata
from pathlib import Path

import torch

from mcqa_gradual_das import ANSWER_LETTERS, answer_label_ids, collect_layer_states, output_label_ids, set_seed
from mcqa_gradual_das_ap import run_ap_intervention_and_capture_at
from mcqa_gradual_das_c import run_c_intervention_and_capture, run_frozen_mediator
from mcqa_load_randomized_ap_at_c import build_pair_bank, color_position, factual_rows_correct, load_generated_dataset, symbol_position
from mcqa_neural_net import load_gemma_model


def load_ckpt(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    basis = checkpoint["basis"].float().to(device)
    layer = int(checkpoint["layer"])
    position = checkpoint.get("token_position", "last_token")
    return basis, layer, position


def output_correct(logits, target_class, tokenizer, output_space):
    prediction = logits.argmax(-1).detach().cpu()

    if output_space == "az":
        return prediction == target_class.detach().cpu()

    predicted_letters = []

    for token_id in prediction:
        text = tokenizer.decode([int(token_id)])
        text = unicodedata.normalize("NFKC", text).strip().upper()
        predicted_letters.append(text)

    expected_letters = []

    for target in target_class.detach().cpu():
        expected_letters.append(ANSWER_LETTERS[int(target)])

    correct = []

    for prediction_text, expected_text in zip(predicted_letters, expected_letters):
        correct.append(prediction_text == expected_text)

    return torch.tensor(correct)


def mean_bool(values):
    if values.numel() == 0:
        return float("nan")

    return float(values.float().mean())


def conditional_mean(values, condition):
    selected = values[condition]

    if selected.numel() == 0:
        return float("nan")

    return float(selected.float().mean())


def row_position(row, tokenizer, token_position):
    if token_position == "last_token":
        return None

    if token_position == "color":
        return color_position(row["prompt"], tokenizer)

    if token_position == "golden_letter":
        return symbol_position(row["prompt"], str(row["answer"]), tokenizer)

    if token_position == "source_ap_slot_label":
        return symbol_position(row["prompt"], str(row["answer"]), tokenizer)

    raise ValueError(f"Unsupported donor token position: {token_position}")


def build_clean_state_inputs(rows, tokenizer, token_position):
    tokenizer.padding_side = "left"

    prompts = []

    for row in rows:
        prompts.append(row["prompt"])

    inputs = tokenizer(prompts, padding=True, return_tensors="pt")

    if token_position == "last_token":
        positions = inputs["attention_mask"].sum(1).long() - 1
    else:
        position_values = []

        for row in rows:
            position_values.append(row_position(row, tokenizer, token_position))

        positions = torch.tensor(position_values)

    return inputs["input_ids"].long(), inputs["attention_mask"].long(), {token_position: positions}


def build_clean_pool(model, tokenizer, data_path, batch_size):
    dataset = load_generated_dataset(data_path)
    rows = []

    for i in range(len(dataset["test"])):
        rows.append(dataset["test"][i])

    rows = factual_rows_correct(model, tokenizer, rows, batch_size)

    by_ap = {}
    by_at = {}

    for row in rows:
        ap = int(row["answer_position"])
        at = str(row["answer"])

        if ap not in by_ap:
            by_ap[ap] = []

        if at not in by_at:
            by_at[at] = []

        by_ap[ap].append(row)
        by_at[at].append(row)

    return rows, by_ap, by_at


def choose_clean_ap_donors(bank, by_ap, target_key, seed):
    rng = random.Random(seed)
    target_aps = bank["intervention_targets"][target_key]
    donors = []

    for i in range(len(bank["pairs"])):
        pair = bank["pairs"][i]
        target_ap = int(target_aps[i])
        candidates = []

        for row in by_ap[target_ap]:
            if row["question_key"] != pair["question"]:
                candidates.append(row)

        if len(candidates) == 0:
            raise RuntimeError(f"No clean AP donor for example {i}, target AP={target_ap}")

        donors.append(rng.choice(candidates))

    return donors


def choose_clean_at_donors(bank, by_at, target_key, seed):
    rng = random.Random(seed)
    target_ats = bank["intervention_targets"][target_key]
    donors = []

    for i in range(len(bank["pairs"])):
        pair = bank["pairs"][i]
        target_letter = ANSWER_LETTERS[int(target_ats[i])]
        candidates = []

        for row in by_at[target_letter]:
            if row["question_key"] != pair["question"]:
                candidates.append(row)

        if len(candidates) == 0:
            raise RuntimeError(f"No clean AT donor for example {i}, target AT={target_letter}")

        donors.append(rng.choice(candidates))

    return donors


def collect_clean_states(model, tokenizer, rows, layer, token_position, batch_size):
    input_ids, attention_mask, position_by_id = build_clean_state_inputs(rows, tokenizer, token_position)
    return collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position, batch_size)


@torch.no_grad()
def eval_c_donor_control(model, tokenizer, bank, clean_ap_rows, clean_at_rows, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_c = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_c, pos_c, batch_size)
    direct_y, induced_ap, induced_at = run_c_intervention_and_capture(model, bank, source_c, Q_c, L_c, L_ap, L_at, output_ids, pos_c, pos_ap, pos_at, batch_size, False)

    clean_ap = collect_clean_states(model, tokenizer, clean_ap_rows, L_ap, pos_ap, batch_size)
    clean_at = collect_clean_states(model, tokenizer, clean_at_rows, L_at, pos_at, batch_size)

    induced_ap_y = run_frozen_mediator(model, bank, induced_ap, Q_ap, L_ap, output_ids, pos_ap, batch_size, False)
    clean_ap_y = run_frozen_mediator(model, bank, clean_ap, Q_ap, L_ap, output_ids, pos_ap, batch_size, False)

    induced_at_y = run_frozen_mediator(model, bank, induced_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    clean_at_y = run_frozen_mediator(model, bank, clean_at, Q_at, L_at, output_ids, pos_at, batch_size, False)

    target = bank["intervention_targets"]["Y_after_C_intervene"]

    direct_ok = output_correct(direct_y, target, tokenizer, output_space)
    induced_ap_ok = output_correct(induced_ap_y, target, tokenizer, output_space)
    clean_ap_ok = output_correct(clean_ap_y, target, tokenizer, output_space)
    induced_at_ok = output_correct(induced_at_y, target, tokenizer, output_space)
    clean_at_ok = output_correct(clean_at_y, target, tokenizer, output_space)

    return {
        "Direct_C": mean_bool(direct_ok),
        "C_induced_AP_replay": mean_bool(induced_ap_ok),
        "Clean_AP_replay": mean_bool(clean_ap_ok),
        "C_induced_AP_given_clean_AP": conditional_mean(induced_ap_ok, clean_ap_ok),
        "C_induced_AT_replay": mean_bool(induced_at_ok),
        "Clean_AT_replay": mean_bool(clean_at_ok),
        "C_induced_AT_given_clean_AT": conditional_mean(induced_at_ok, clean_at_ok),
    }


@torch.no_grad()
def eval_ap_donor_control(model, tokenizer, bank, clean_ap_rows, clean_at_rows, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, batch_size, output_space):
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    source_ap = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], L_ap, pos_ap, batch_size)
    direct_y, induced_at = run_ap_intervention_and_capture_at(model, bank, source_ap, Q_ap, L_ap, L_at, output_ids, pos_ap, batch_size, False)

    clean_ap = collect_clean_states(model, tokenizer, clean_ap_rows, L_ap, pos_ap, batch_size)
    clean_at = collect_clean_states(model, tokenizer, clean_at_rows, L_at, pos_at, batch_size)

    clean_ap_y = run_frozen_mediator(model, bank, clean_ap, Q_ap, L_ap, output_ids, pos_ap, batch_size, False)
    induced_at_y = run_frozen_mediator(model, bank, induced_at, Q_at, L_at, output_ids, pos_at, batch_size, False)
    clean_at_y = run_frozen_mediator(model, bank, clean_at, Q_at, L_at, output_ids, pos_at, batch_size, False)

    target = bank["intervention_targets"]["Y_after_AP_intervene"]

    direct_ok = output_correct(direct_y, target, tokenizer, output_space)
    clean_ap_ok = output_correct(clean_ap_y, target, tokenizer, output_space)
    induced_at_ok = output_correct(induced_at_y, target, tokenizer, output_space)
    clean_at_ok = output_correct(clean_at_y, target, tokenizer, output_space)

    return {
        "Direct_AP": mean_bool(direct_ok),
        "Clean_AP_replay": mean_bool(clean_ap_ok),
        "Direct_AP_given_clean_AP": conditional_mean(direct_ok, clean_ap_ok),
        "AP_induced_AT_replay": mean_bool(induced_at_ok),
        "Clean_AT_replay": mean_bool(clean_at_ok),
        "AP_induced_AT_given_clean_AT": conditional_mean(induced_at_ok, clean_at_ok),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--result-dir", default="results_joint_randomized_gradual_at2.0_newrun")
    parser.add_argument("--test-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    args = parser.parse_args()

    set_seed(args.seed)

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    Q_c, L_c, pos_c = load_ckpt(f"{args.result_dir}/das_c.pt", device)
    Q_ap, L_ap, pos_ap = load_ckpt(f"{args.result_dir}/das_ap.pt", device)
    Q_at, L_at, pos_at = load_ckpt(f"{args.result_dir}/das_at.pt", device)

    rows, by_ap, by_at = build_clean_pool(model, tokenizer, args.data_path, args.batch_size)

    print(f"[clean donor pool] correct factual test rows: {len(rows)}")

    c_only = build_pair_bank(model, tokenizer, args.data_path, "test", "C_only", args.test_size, args.batch_size, args.seed + 1)
    c_joint = build_pair_bank(model, tokenizer, args.data_path, "test", "C_AP_AT", args.test_size, args.batch_size, args.seed + 2)
    ap_only = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_only", args.test_size, args.batch_size, args.seed + 3)
    ap_at = build_pair_bank(model, tokenizer, args.data_path, "test", "AP_AT", args.test_size, args.batch_size, args.seed + 4)

    c_results = {}

    for offset, item in enumerate((("C_only", c_only), ("C_AP_AT", c_joint))):
        name, bank = item

        clean_ap_rows = choose_clean_ap_donors(bank, by_ap, "AP_after_C_intervene", args.seed + 100 + offset)
        clean_at_rows = choose_clean_at_donors(bank, by_at, "AT_after_C_intervene", args.seed + 200 + offset)

        c_results[name] = eval_c_donor_control(model, tokenizer, bank, clean_ap_rows, clean_at_rows, Q_c, L_c, pos_c, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, args.batch_size, args.output_space)

    ap_results = {}

    for offset, item in enumerate((("AP_only", ap_only), ("AP_AT", ap_at))):
        name, bank = item

        clean_ap_rows = choose_clean_ap_donors(bank, by_ap, "AP_after_AP_intervene", args.seed + 300 + offset)
        clean_at_rows = choose_clean_at_donors(bank, by_at, "AT_after_AP_intervene", args.seed + 400 + offset)

        ap_results[name] = eval_ap_donor_control(model, tokenizer, bank, clean_ap_rows, clean_at_rows, Q_ap, L_ap, pos_ap, Q_at, L_at, pos_at, args.batch_size, args.output_space)

    lines = []

    for name in ("C_only", "C_AP_AT"):
        metrics = c_results[name]

        lines.append(f"=== C INTERVENTION CLEAN-DONOR CONTROL [{name}] ===")
        lines.append(f"Direct C: {metrics['Direct_C']:.4f}")
        lines.append("AP mediator")
        lines.append(f"C-induced AP replay: {metrics['C_induced_AP_replay']:.4f}")
        lines.append(f"Clean AP replay: {metrics['Clean_AP_replay']:.4f}")
        lines.append(f"C-induced AP GIVEN clean AP succeeds: {metrics['C_induced_AP_given_clean_AP']:.4f}")
        lines.append("AT mediator")
        lines.append(f"C-induced AT replay: {metrics['C_induced_AT_replay']:.4f}")
        lines.append(f"Clean AT replay: {metrics['Clean_AT_replay']:.4f}")
        lines.append(f"C-induced AT GIVEN clean AT succeeds: {metrics['C_induced_AT_given_clean_AT']:.4f}")
        lines.append("")

    for name in ("AP_only", "AP_AT"):
        metrics = ap_results[name]

        lines.append(f"=== AP INTERVENTION CLEAN-DONOR CONTROL [{name}] ===")
        lines.append(f"Direct AP: {metrics['Direct_AP']:.4f}")
        lines.append("AP donor")
        lines.append(f"Clean AP replay: {metrics['Clean_AP_replay']:.4f}")
        lines.append(f"Direct AP GIVEN clean AP succeeds: {metrics['Direct_AP_given_clean_AP']:.4f}")
        lines.append("AT mediator")
        lines.append(f"AP-induced AT replay: {metrics['AP_induced_AT_replay']:.4f}")
        lines.append(f"Clean AT replay: {metrics['Clean_AT_replay']:.4f}")
        lines.append(f"AP-induced AT GIVEN clean AT succeeds: {metrics['AP_induced_AT_given_clean_AT']:.4f}")
        lines.append("")

    text = "\n".join(lines).strip()

    print("\n" + text)

    output_path = Path(args.result_dir) / "eval_clean_donor_controls.txt"
    output_path.write_text(text + "\n")

    print(f"\nsaved results = {output_path}")


if __name__ == "__main__":
    main()
