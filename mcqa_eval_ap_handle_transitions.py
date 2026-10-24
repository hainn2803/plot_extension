import argparse, csv, random
from pathlib import Path

import torch

from mcqa_neural_net import load_gemma_model
from mcqa_load_randomized_ap_at_c import create_bank, load_generated_dataset, predict_correct, rows_to_pairs
from mcqa_gradual_das import answer_label_ids, collect_layer_states, output_label_ids, output_targets, run_das_intervention, set_seed


def load_ap_handle(path, device):
    """Load the frozen AP DAS handle."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    Q = ckpt["basis"].to(device).float()
    layer = int(ckpt["layer"])
    token_position = ckpt.get("token_position", "last_token")
    return Q, layer, token_position


def build_balanced_ap_bank(model, tokenizer, data_path, batch_size, seed, per_transition=None):
    """Build an AP-only test bank with the same number of examples for all 12 AP transitions."""
    ds = load_generated_dataset(data_path)
    ids = list(range(len(ds["test"])))
    random.Random(seed).shuffle(ids)

    groups = {(base_ap, source_ap): [] for base_ap in range(4) for source_ap in range(4) if base_ap != source_ap}
    cache = {}

    for start in range(0, len(ids), 64):
        candidates = rows_to_pairs(ds["test"], ids[start:start + 64], "test")
        candidates = [p for p in candidates if p["pair_type"] == "AP_only"]
        candidates = predict_correct(model, tokenizer, candidates, batch_size, cache)

        for pair in candidates:
            groups[(pair["base_AP"], pair["source_AP"])].append(pair)

        if per_transition is not None and all(len(group) >= per_transition for group in groups.values()):
            break

    counts = {transition: len(group) for transition, group in groups.items()}
    print("\nAvailable correct AP-only pairs:")
    for transition in groups:
        print(f"{transition[0]}->{transition[1]}: {counts[transition]}")

    n = min(counts.values()) if per_transition is None else per_transition
    if n == 0:
        raise RuntimeError("At least one AP transition has no valid examples.")

    for transition, count in counts.items():
        if count < n:
            raise ValueError(f"Transition {transition[0]}->{transition[1]} has only {count} examples, but requested {n}")

    rng = random.Random(seed)
    selected = []

    for transition, group in groups.items():
        rng.shuffle(group)
        selected.extend(group[:n])

    rng.shuffle(selected)
    print(f"\nBalanced dataset: {n} examples per transition, {len(selected)} total")
    return create_bank(selected, tokenizer, size=None, seed=seed), n


def evaluate_frozen_ap_handle(model, tokenizer, bank, Q, layer, token_position, batch_size, output_space):
    """Run the frozen AP DAS handle and return per-example predictions."""
    label_ids = answer_label_ids(tokenizer)
    source_states = collect_layer_states(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], layer, token_position, batch_size)
    logits = run_das_intervention(model, bank, source_states, Q, layer, output_label_ids(label_ids, output_space), token_position, batch_size, require_grad=False)

    predictions = logits.argmax(dim=-1).cpu()
    cf_targets = output_targets(bank["intervention_targets"]["Y_after_AP_intervene"], label_ids, output_space, "cpu").cpu()
    base_targets = output_targets(bank["base_answer_label_ids"], label_ids, output_space, "cpu").cpu()
    return predictions, cf_targets, base_targets, label_ids


def prediction_to_letter(prediction, label_ids, output_space):
    if output_space == "az":
        class_id = int(prediction)
        return chr(ord("A") + class_id) if 0 <= class_id < 26 else None

    token_id = int(prediction)
    token_to_class = {int(token): i for i, token in enumerate(label_ids)}
    class_id = token_to_class.get(token_id)
    return chr(ord("A") + class_id) if class_id is not None else None


def compute_stats(bank, predictions, cf_targets, base_targets, label_ids, output_space):
    """Report where the model's output lands among the four base answer positions for each AP transition."""
    stats = {(base_ap, source_ap): {"n": 0, "cf_correct": 0, "base_retained": 0, "pred_ap": [0, 0, 0, 0], "other": 0} for base_ap in range(4) for source_ap in range(4) if base_ap != source_ap}
    rows = []

    for i, pair in enumerate(bank["pairs"]):
        transition = (pair["base_AP"], pair["source_AP"])
        cf_correct = predictions[i].item() == cf_targets[i].item()
        base_retained = predictions[i].item() == base_targets[i].item()
        predicted_letter = prediction_to_letter(predictions[i].item(), label_ids, output_space)
        base_labels = list(pair["base_choices"]["label"])
        predicted_ap = base_labels.index(predicted_letter) if predicted_letter in base_labels else None

        stats[transition]["n"] += 1
        stats[transition]["cf_correct"] += int(cf_correct)
        stats[transition]["base_retained"] += int(base_retained)
        if predicted_ap is None:
            stats[transition]["other"] += 1
        else:
            stats[transition]["pred_ap"][predicted_ap] += 1

        rows.append({"base_AP": pair["base_AP"], "source_AP": pair["source_AP"], "transition": f"{pair['base_AP']}->{pair['source_AP']}", "base_answer": pair["base_Y"], "counterfactual_target": pair["Y_after_AP_intervene"], "predicted_letter": predicted_letter if predicted_letter is not None else "OTHER", "predicted_AP": predicted_ap if predicted_ap is not None else "OTHER", "cf_correct": int(cf_correct), "base_retained": int(base_retained)})

    return stats, rows


def print_transition_table(stats):
    print("\n================ OUTPUT AP DISTRIBUTION BY TRANSITION ================")
    print(f"{'transition':>10} {'N':>6} {'out=0':>8} {'out=1':>8} {'out=2':>8} {'out=3':>8} {'other':>8} {'CF acc':>10} {'base retain':>12}")

    for base_ap in range(4):
        for source_ap in range(4):
            if base_ap == source_ap:
                continue

            s = stats[(base_ap, source_ap)]
            cf_acc = s["cf_correct"] / s["n"]
            base_ret = s["base_retained"] / s["n"]
            p = s["pred_ap"]
            print(f"{base_ap}->{source_ap:1d} {s['n']:>10d} {p[0]:>8d} {p[1]:>8d} {p[2]:>8d} {p[3]:>8d} {s['other']:>8d} {cf_acc:>10.4f} {base_ret:>12.4f}")


def print_from_to_summary(stats):
    print("\n================ GROUPED BY WHERE AP MOVES FROM ================")
    print(f"{'AP':>4} {'N':>6} {'CF acc':>10} {'base retain':>12}")

    for base_ap in range(4):
        selected = [stats[(base_ap, source_ap)] for source_ap in range(4) if source_ap != base_ap]
        n = sum(x["n"] for x in selected)
        cf = sum(x["cf_correct"] for x in selected) / n
        base = sum(x["base_retained"] for x in selected) / n
        print(f"{base_ap:>4d} {n:>6d} {cf:>10.4f} {base:>12.4f}")

    print("\n================ GROUPED BY WHERE AP MOVES TO ================")
    print(f"{'AP':>4} {'N':>6} {'CF acc':>10} {'base retain':>12}")

    for source_ap in range(4):
        selected = [stats[(base_ap, source_ap)] for base_ap in range(4) if base_ap != source_ap]
        n = sum(x["n"] for x in selected)
        cf = sum(x["cf_correct"] for x in selected) / n
        base = sum(x["base_retained"] for x in selected) / n
        print(f"{source_ap:>4d} {n:>6d} {cf:>10.4f} {base:>12.4f}")


def save_results(output_dir, per_transition, stats, rows):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"ap_handle_transitions_balanced{per_transition}_summary.csv"
    pairs_path = output_dir / f"ap_handle_transitions_balanced{per_transition}_pairs.csv"

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base_AP", "source_AP", "transition", "N", "output_AP_0", "output_AP_1", "output_AP_2", "output_AP_3", "other", "counterfactual_accuracy", "base_retention"])

        for base_ap in range(4):
            for source_ap in range(4):
                if base_ap == source_ap:
                    continue

                s = stats[(base_ap, source_ap)]
                writer.writerow([base_ap, source_ap, f"{base_ap}->{source_ap}", s["n"], s["pred_ap"][0], s["pred_ap"][1], s["pred_ap"][2], s["pred_ap"][3], s["other"], s["cf_correct"] / s["n"], s["base_retained"] / s["n"]])

    with open(pairs_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nsaved summary = {summary_path}")
    print(f"saved pairs   = {pairs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors")
    parser.add_argument("--ap-checkpoint", default="results_joint_randomized_standard/das_ap.pt")
    parser.add_argument("--per-transition", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=["full", "az"], default="full")
    parser.add_argument("--output-dir", default="results_ap_transitions")
    args = parser.parse_args()

    set_seed(args.seed)
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    Q, layer, token_position = load_ap_handle(args.ap_checkpoint, device)
    bank, per_transition = build_balanced_ap_bank(model, tokenizer, args.data_path, args.batch_size, args.seed, args.per_transition)

    print(f"\ncheckpoint     = {args.ap_checkpoint}")
    print(f"layer          = {layer}")
    print(f"subspace dim   = {Q.shape[1]}")
    print(f"token position = {token_position}")
    print(f"output space   = {args.output_space}")

    predictions, cf_targets, base_targets, label_ids = evaluate_frozen_ap_handle(model, tokenizer, bank, Q, layer, token_position, args.batch_size, args.output_space)
    stats, rows = compute_stats(bank, predictions, cf_targets, base_targets, label_ids, args.output_space)

    overall_cf = float((predictions == cf_targets).float().mean())
    overall_base = float((predictions == base_targets).float().mean())
    print(f"\noverall counterfactual accuracy = {overall_cf:.4f}")
    print(f"overall base-answer retention    = {overall_base:.4f}")

    print_transition_table(stats)
    print_from_to_summary(stats)
    save_results(args.output_dir, per_transition, stats, rows)


if __name__ == "__main__":
    main()
