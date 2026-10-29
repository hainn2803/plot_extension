import argparse
import csv
import statistics
import string
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch

from mcqa_gradual_das import answer_label_ids, collect_layer_states, output_label_ids, output_targets, run_das_intervention, set_seed
from mcqa_load_randomized_ap_at_c_balanced_eval import build_balanced_eval_bank
from mcqa_neural_net import load_gemma_model


ANSWER_LETTERS = tuple(string.ascii_uppercase)


def parse_int_list(value):
    values = []

    for item in value.split(","):
        item = item.strip()

        if item != "":
            values.append(int(item))

    return values


def save_csv(rows, path):
    if len(rows) == 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def normalize_prediction(tokenizer, token_id):
    text = tokenizer.decode([int(token_id)])
    text = unicodedata.normalize("NFKC", text).strip().upper()
    return text


def correct_vector(logits, target_class, tokenizer, output_space):
    predictions = logits.argmax(dim=-1).detach().cpu()
    targets = target_class.detach().cpu()
    correct = []

    if output_space == "az":
        for prediction, target in zip(predictions, targets):
            correct.append(int(prediction) == int(target))

        return correct

    for prediction, target in zip(predictions, targets):
        prediction_text = normalize_prediction(tokenizer, int(prediction))
        target_text = ANSWER_LETTERS[int(target)]
        correct.append(prediction_text == target_text)

    return correct


def target_margin_vector(logits, target_class, tokenizer, output_space):
    label_ids = answer_label_ids(tokenizer)
    device = logits.device
    target_indices = output_targets(target_class, label_ids, output_space, device)

    rows = torch.arange(len(logits), device=device)
    target_logits = logits[rows, target_indices]

    masked = logits.clone()
    masked[rows, target_indices] = -float("inf")
    max_wrong = masked.max(dim=-1).values

    margins = target_logits - max_wrong
    return margins.detach().cpu().tolist()


def pattern_from_correct(correct_by_layer, index, layers):
    winners = []

    for layer in layers:
        if correct_by_layer[layer][index]:
            winners.append(str(layer))

    if len(winners) == 0:
        return "none"

    return "+".join(winners)


def evaluate_layer(model, tokenizer, bank, checkpoint_path, expected_layer, token_position, batch_size, output_space):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    layer = int(checkpoint["layer"])

    if layer != expected_layer:
        raise ValueError(f"{checkpoint_path} says layer={layer}, expected {expected_layer}")

    Q = checkpoint["basis"].float().to(next(model.parameters()).device)

    source_states = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], layer, token_position, batch_size)

    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, output_space)

    logits = run_das_intervention(model, bank, source_states, Q, layer, output_ids, token_position, batch_size, require_grad=False)

    target = bank["intervention_targets"]["Y_after_AT_intervene"]

    correct = correct_vector(logits, target, tokenizer, output_space)
    margins = target_margin_vector(logits, target, tokenizer, output_space)

    return correct, margins


def summarize_group(rows, group_name, group_value, layers):
    output = {
        "group": group_name,
        "value": group_value,
        "n": len(rows),
    }

    for layer in layers:
        correct_values = []
        margin_values = []

        for row in rows:
            correct_values.append(float(row[f"L{layer}_correct"]))
            margin_values.append(float(row[f"L{layer}_margin"]))

        output[f"L{layer}_iia"] = statistics.mean(correct_values)
        output[f"L{layer}_margin"] = statistics.mean(margin_values)

    if 25 in layers:
        other_layers = []

        for layer in layers:
            if layer != 25:
                other_layers.append(layer)

        if len(other_layers) > 0:
            best_other_iia = max(output[f"L{layer}_iia"] for layer in other_layers)
            best_other_margin = max(output[f"L{layer}_margin"] for layer in other_layers)
            output["delta25_iia"] = output["L25_iia"] - best_other_iia
            output["delta25_margin"] = output["L25_margin"] - best_other_margin

    return output


def grouped_summary(example_rows, key, layers):
    groups = defaultdict(list)

    for row in example_rows:
        groups[row[key]].append(row)

    output = []

    for value in sorted(groups, key=lambda x: str(x)):
        output.append(summarize_group(groups[value], key, value, layers))

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--results-root", default="results_all/results_standard_vs_gradual_5seeds_no_dims/standard")
    parser.add_argument("--output-dir", default="at_layer_analysis")

    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--layers", default="23,24,25")

    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--change-ratio", type=float, default=1.0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--token-position", default="last_token")

    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    layers = parse_int_list(args.layers)

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    model, tokenizer = load_gemma_model()

    all_example_rows = []
    all_layer_summary = []

    for seed in seeds:
        print("\n" + "#" * 100)
        print(f"SEED = {seed}")
        print("#" * 100)

        set_seed(seed)

        bank = build_balanced_eval_bank(model, tokenizer, args.data_path, "validation", "AT", args.cal_size, args.batch_size, seed + 1, args.change_ratio)

        correct_by_layer = {}
        margin_by_layer = {}

        for layer in layers:
            checkpoint_path = results_root / f"seed_{seed}" / "at_layers" / f"layer_{layer}" / "das_at.pt"

            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)

            print(f"[EVAL] seed={seed} layer={layer} checkpoint={checkpoint_path}")

            correct, margins = evaluate_layer(model, tokenizer, bank, checkpoint_path, layer, args.token_position, args.batch_size, args.output_space)

            correct_by_layer[layer] = correct
            margin_by_layer[layer] = margins

            iia = statistics.mean(float(x) for x in correct)
            mean_margin = statistics.mean(margins)

            all_layer_summary.append({
                "seed": seed,
                "layer": layer,
                "n": len(correct),
                "iia": iia,
                "mean_margin": mean_margin,
            })

            print(f"  IIA={iia:.4f} margin={mean_margin:.4f}")

        for i, pair in enumerate(bank["pairs"]):
            row = {
                "seed": seed,
                "example_index": i,
                "pair_type": pair["pair_type"],
                "base_AT": pair["base_AT"],
                "source_AT": pair["source_AT"],
                "transition": f"{pair['base_AT']}->{pair['source_AT']}",
                "base_AP": int(pair["base_AP"]),
                "source_AP": int(pair["source_AP"]),
                "target_AT": pair["AT_after_AT_intervene"],
                "pattern": pattern_from_correct(correct_by_layer, i, layers),
            }

            for layer in layers:
                row[f"L{layer}_correct"] = int(correct_by_layer[layer][i])
                row[f"L{layer}_margin"] = float(margin_by_layer[layer][i])

            all_example_rows.append(row)

    save_csv(all_example_rows, output_dir / "at_layer_per_example.csv")
    save_csv(all_layer_summary, output_dir / "at_layer_summary.csv")

    by_transition = grouped_summary(all_example_rows, "transition", layers)
    by_base_ap = grouped_summary(all_example_rows, "base_AP", layers)
    by_source_at = grouped_summary(all_example_rows, "source_AT", layers)
    by_pair_type = grouped_summary(all_example_rows, "pair_type", layers)

    save_csv(by_transition, output_dir / "at_layer_by_transition.csv")
    save_csv(by_base_ap, output_dir / "at_layer_by_base_ap.csv")
    save_csv(by_source_at, output_dir / "at_layer_by_source_at.csv")
    save_csv(by_pair_type, output_dir / "at_layer_by_pair_type.csv")

    pattern_counts = defaultdict(int)

    for row in all_example_rows:
        pattern_counts[row["pattern"]] += 1

    pattern_rows = []

    for pattern in sorted(pattern_counts):
        pattern_rows.append({
            "pattern": pattern,
            "count": pattern_counts[pattern],
            "fraction": pattern_counts[pattern] / len(all_example_rows),
        })

    save_csv(pattern_rows, output_dir / "at_layer_pattern_counts.csv")

    print("\n================ LAYER SUMMARY ================")

    for row in all_layer_summary:
        print(f"seed={row['seed']} L{row['layer']} IIA={row['iia']:.4f} margin={row['mean_margin']:.4f}")

    print("\n================ TOP TRANSITIONS FOR L25 ================")

    if 25 in layers:
        ranked = sorted(by_transition, key=lambda row: row.get("delta25_iia", -999.0), reverse=True)

        for row in ranked:
            print(f"{row['value']:>8s} n={row['n']:4d} | " + " ".join(f"L{layer}={row[f'L{layer}_iia']:.3f}" for layer in layers) + f" | delta25={row.get('delta25_iia', float('nan')):+.3f}")

    print(f"\nSaved analysis to {output_dir}")


if __name__ == "__main__":
    main()
