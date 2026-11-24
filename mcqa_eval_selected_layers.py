"""Evaluate selected AT/AP/C Standard DAS layer combinations.

Pipeline:
  1. Train and save handles with mcqa_train_all_layers_no_eval.py.
  2. Specify exact AT/AP/C layers on the command line.
  3. Evaluate all valid C < AP < AT combinations; save each result immediately.

Evaluation pair selection and shuffling match mcqa_eval_completed_gradual.py.
No training or optimization happens in this script.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch

from mcqa_load_randomized_ap_at_c import (
    PAIR_FIELDS,
    LETTER_TO_CLASS,
    intervention_targets,
    create_bank,
    load_generated_dataset,
    predict_correct,
    row_to_pairs,
)
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import answer_label_ids, output_label_ids
from mcqa_gradual_das_ap_cross_tokens import (
    run_ap_intervention_and_capture_at,
    run_frozen_at_mediator,
)
from mcqa_gradual_das_c import run_c_intervention_and_capture
from mcqa_train_grid_standard_das import (
    add_metric,
    accuracy,
    collect_states,
    balanced_counts,
    clean_readout_accuracy,
    color_targets,
    eval_ap,
    eval_at,
    eval_c,
    load_ckpt,
    load_readout,
    output_correct,
    projected_swap_state,
    readout_prediction,
    save_csv,
)

# Use the same settings as mcqa_eval_completed_gradual.py.
# To evaluate mixed pairs later, edit BOTH scripts to the same settings.
EVAL_PAIR_TYPES = {
    "AT": ["AT_only"],
    "AP": ["AP_only"],
    "C": ["C_only"],
}
EVAL_CHANGING_ONLY = True

TARGET_BY_VARIABLE = {
    "AT": "Y_after_AT_intervene",
    "AP": "Y_after_AP_intervene",
    "C": "Y_after_C_intervene",
}
PAIR_SEEDS = {
    "AT_only": 3001,
    "AP_only": 3002,
    "C_only": 3004,
    "AP_AT": 3005,
    "C_AP_AT": 3006,
}
BANK_SHUFFLE_OFFSETS = {"AT": 100, "AP": 200, "C": 300}
# Explicitly invalidates cached results computed with the old Standard eval seeds.
EVAL_CODE_VERSION = 3


def validate_eval_settings():
    for variable in ("AT", "AP", "C"):
        names = EVAL_PAIR_TYPES[variable]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{variable}: pair types must be nonempty and unique")
        for name in names:
            if name not in PAIR_FIELDS:
                raise ValueError(f"Unknown pair type: {name}")


def eval_profile_name(selected):
    # Each explicit layer selection gets an isolated evaluation folder.
    # This prevents summaries from including configs from earlier selections.
    settings = {
        "pairs": EVAL_PAIR_TYPES,
        "changing_only": EVAL_CHANGING_ONLY,
        "selected_layers": selected,
        "recipient_version": RECIPIENT_VERSION,
    }
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return f"eval_configs_{digest}"


def build_eval_bank(model, tokenizer, dataset_split, variable,
                    pair_types, total_size, batch_size, seed):
    """Match GDAS eval's example-selection and bank-shuffle seeds."""
    counts = balanced_counts(total_size, pair_types)
    mixed = []
    prediction_cache = {}

    for pair_type in pair_types:
        needed = counts[pair_type]
        indices = list(range(len(dataset_split)))
        random.Random(seed + PAIR_SEEDS[pair_type]).shuffle(indices)
        accepted = []

        for start in range(0, len(indices), 64):
            candidates = []
            for index in indices[start:start + 64]:
                row = dataset_split[int(index)]
                for pair in row_to_pairs(row, "test"):
                    if pair["pair_type"] != pair_type:
                        continue
                    if EVAL_CHANGING_ONLY and pair[TARGET_BY_VARIABLE[variable]] == pair["base_Y"]:
                        continue
                    candidates.append(pair)

            correct = (
                predict_correct(model, tokenizer, candidates, batch_size, prediction_cache)
                if candidates else []
            )
            accepted.extend(correct)
            if len(accepted) >= needed:
                break

        if len(accepted) < needed:
            raise ValueError(
                f"{variable}/{pair_type}: found {len(accepted)} qualifying examples; "
                f"need {needed}."
            )
        mixed.extend(accepted[:needed])
        print(f"[eval bank] {variable}/{pair_type}: {needed} examples", flush=True)

    # This matches GDAS evaluation exactly, including AT seed+100.
    shuffle_seed = seed + BANK_SHUFFLE_OFFSETS[variable]
    return create_bank(mixed, tokenizer, size=None, seed=shuffle_seed)


def parse_layers(values, variable):
    """Accept `23 24 25` or `23,24,25`; sort and remove duplicates."""
    layers = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                layer = int(part)
            except ValueError as exc:
                raise ValueError(f"{variable}: invalid layer {part!r}") from exc
            if layer < 0:
                raise ValueError(f"{variable}: layer must be >= 0, got {layer}")
            layers.append(layer)

    if not layers:
        raise ValueError(f"{variable}: provide at least one layer")
    return sorted(set(layers))


# -----------------------------------------------------------------------------
# Cross-question transfer of INDUCED AP/AT states (not direct C transfer).
# -----------------------------------------------------------------------------

RECIPIENT_VERSION = 1
RECIPIENT_OFFSETS = {
    ("AP_only", "AT", "same"): 4101,
    ("AP_only", "AT", "different"): 4102,
    ("C_only", "AP", "same"): 4201,
    ("C_only", "AP", "different"): 4202,
    ("C_only", "AT", "same"): 4301,
    ("C_only", "AT", "different"): 4302,
}


def build_recipient_pool(banks):
    """Only model-correct TEST questions, since build_eval_bank filters both sides.

    We use BOTH sides of the original controlled pairs as possible recipients.
    The pool is built once and is identical for every layer configuration.
    """
    by_prompt = {}
    for variable in ("AT", "AP", "C"):
        for pair in banks[variable]["pairs"]:
            for side in ("base", "source"):
                prompt = pair[f"{side}_prompt"]
                candidate = {
                    "prompt": prompt,
                    "choices": pair[f"{side}_choices"],
                    "color": pair[f"{side}_color"],
                    "AP": int(pair[f"{side}_AP"]),
                    "AT": str(pair[f"{side}_AT"]),
                    "question": pair["question"],
                }
                previous = by_prompt.get(prompt)
                if previous is not None and previous != candidate:
                    raise ValueError(f"Inconsistent recipient facts for {prompt!r}")
                by_prompt[prompt] = candidate
    return list(by_prompt.values())


def make_recipient_schedule(donor_pairs, pool, mediator, target_values, condition, seed):
    """One independent question per donor. same/different refers to AP or AT."""
    if mediator not in ("AP", "AT") or condition not in ("same", "different"):
        raise ValueError("Invalid recipient schedule setting")
    if len(donor_pairs) != len(target_values):
        raise ValueError("Donor and target lengths differ")

    rng = random.Random(seed)
    schedule = []
    for donor, target in zip(donor_pairs, target_values):
        target = int(target) if mediator == "AP" else str(target)
        eligible = []
        for candidate in pool:
            if candidate["question"] == donor["question"]:
                continue
            matches = candidate[mediator] == target
            if matches != (condition == "same"):
                continue
            if mediator == "AP":
                labels = candidate["choices"]["label"]
                if not (0 <= target < len(labels)):
                    continue
            eligible.append(candidate)
        if not eligible:
            raise ValueError(f"No recipient for {mediator}={target} ({condition})")
        schedule.append({"recipient": rng.choice(eligible), "mediator_target": target})
    return schedule


def prepare_recipient_schedules(banks, seed):
    pool = build_recipient_pool(banks)
    specs = (
        ("AP_only", "AT", banks["AP"]["pairs"], "AT_after_AP_intervene"),
        ("C_only", "AP", banks["C"]["pairs"], "AP_after_C_intervene"),
        ("C_only", "AT", banks["C"]["pairs"], "AT_after_C_intervene"),
    )
    schedules = {}
    for family, mediator, donor_pairs, target_name in specs:
        targets = [pair[target_name] for pair in donor_pairs]
        for condition in ("same", "different"):
            key = (family, mediator, condition)
            schedules[key] = make_recipient_schedule(
                donor_pairs, pool, mediator, targets, condition,
                seed + RECIPIENT_OFFSETS[key],
            )
    print(f"[recipient] {len(pool)} unique model-correct TEST prompts", flush=True)
    return schedules


def recipient_bank(tokenizer, schedule):
    """Build MCQA bank for recipients, tracking row order after create_bank shuffle."""
    pairs = []
    for donor_index, entry in enumerate(schedule):
        recipient = entry["recipient"]
        ap_target = entry["mediator_target"] if isinstance(entry["mediator_target"], int) else None
        if ap_target is not None:
            target_at = str(recipient["choices"]["label"][ap_target])
        else:
            target_at = str(entry["mediator_target"])

        # A factual synthetic pair: only base inputs and positions are replayed.
        # Source fields are made factual so create_bank can build its metadata.
        pair = {
            "pair_type": "recipient", "question": recipient["question"],
            "base_prompt": recipient["prompt"],
            "source_prompt": recipient["prompt"],
            "base_choices": recipient["choices"],
            "source_choices": recipient["choices"],
            "base_color": recipient["color"],
            "source_color": recipient["color"],
            "base_AP": recipient["AP"],
            "source_AP": recipient["AP"],
            "base_AT": recipient["AT"],
            "source_AT": recipient["AT"],
            "base_Y": recipient["AT"],
            "source_Y": recipient["AT"],
            "_donor_index": donor_index,
            "_target_ap": ap_target,
            "_target_at": target_at,
        }
        pair.update(intervention_targets(pair))
        pairs.append(pair)

    # create_bank shuffles unconditionally; recover the exact permutation.
    bank = create_bank(pairs, tokenizer, size=None, seed=0)
    order = torch.tensor([pair["_donor_index"] for pair in bank["pairs"]], dtype=torch.long)
    target_at = torch.tensor([LETTER_TO_CLASS[pair["_target_at"]] for pair in bank["pairs"]])
    target_ap = None
    if bank["pairs"][0]["_target_ap"] is not None:
        target_ap = torch.tensor([pair["_target_ap"] for pair in bank["pairs"]])
    return bank, order, target_ap, target_at


def prepare_recipient_banks(tokenizer, schedules):
    """Prepare all recipient token positions just once (independent of layers)."""
    return {key: recipient_bank(tokenizer, schedule) for key, schedule in schedules.items()}


@torch.no_grad()
def eval_at_recipient(model, tokenizer, recipient, induced_at,
                      Q_at, at_layer, at_pos, W_at, b_at, batch_size, output_space):
    """Replay induced AT; measure AT readout, Y, and their joint correctness."""
    bank, order, _, target_at = recipient
    donor_at = induced_at[order]
    base_at = collect_states(
        model, bank["base_input_ids"], bank["base_attention_mask"],
        bank["base_position_by_id"], at_layer, at_pos, batch_size,
    )
    post_at = projected_swap_state(base_at, donor_at, Q_at)
    at_ok = readout_prediction(post_at, Q_at, W_at, b_at) == target_at

    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    logits = run_frozen_at_mediator(
        model, bank, donor_at, Q_at, at_layer, output_ids, at_pos, batch_size, False,
    )
    y_ok = output_correct(logits, target_at, tokenizer, output_space)
    return {
        "AT_after_AT_replay": accuracy(at_ok),
        "Y_after_AT_replay": accuracy(y_ok),
        "Joint_AT_Y_after_AT_replay": accuracy(at_ok & y_ok),
    }


@torch.no_grad()
def eval_ap_recipient(model, tokenizer, recipient, induced_ap,
                      Q_ap, ap_layer, ap_pos, W_ap, b_ap,
                      Q_at, at_layer, at_pos, W_at, b_at,
                      batch_size, output_space):
    """Replay induced AP; measure AP, downstream AT, Y, and joint."""
    bank, order, target_ap, target_at = recipient
    if target_ap is None:
        raise ValueError("AP transfer requires AP targets")
    donor_ap = induced_ap[order]
    base_ap = collect_states(
        model, bank["base_input_ids"], bank["base_attention_mask"],
        bank["base_position_by_id"], ap_layer, ap_pos, batch_size,
    )
    post_ap = projected_swap_state(base_ap, donor_ap, Q_ap)
    ap_ok = readout_prediction(post_ap, Q_ap, W_ap, b_ap) == target_ap

    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)
    logits, generated_at = run_ap_intervention_and_capture_at(
        model, bank, donor_ap, Q_ap, ap_layer, at_layer, output_ids,
        ap_pos, at_pos, batch_size, False,
    )
    at_ok = readout_prediction(generated_at, Q_at, W_at, b_at) == target_at
    y_ok = output_correct(logits, target_at, tokenizer, output_space)
    return {
        "AP_after_AP_replay": accuracy(ap_ok),
        "AT_after_AP_replay": accuracy(at_ok),
        "Y_after_AP_replay": accuracy(y_ok),
        "Joint_AP_AT_Y_after_AP_replay": accuracy(ap_ok & at_ok & y_ok),
    }


def eval_induced_recipients(model, tokenizer, banks, recipient_banks, handles,
                            layers, batch_size, output_space, ap_induced_at=None):
    """Three induced mediator transfers, each evaluated in same/different groups."""
    at_layer, ap_layer, c_layer = layers
    Q_at, at_pos, W_at, b_at, _ = handles["AT"]
    Q_ap, ap_pos, W_ap, b_ap, _ = handles["AP"]
    Q_c, c_pos, _, _, _ = handles["C"]
    output_ids = output_label_ids(answer_label_ids(tokenizer), output_space)

    if ap_induced_at is None:
        source_ap = collect_states(
            model, banks["AP"]["source_input_ids"],
            banks["AP"]["source_attention_mask"],
            banks["AP"]["source_position_by_id"], ap_layer, ap_pos, batch_size,
        )
        _, ap_induced_at = run_ap_intervention_and_capture_at(
            model, banks["AP"], source_ap, Q_ap, ap_layer, at_layer,
            output_ids, ap_pos, at_pos, batch_size, False,
        )

    source_c = collect_states(
        model, banks["C"]["source_input_ids"],
        banks["C"]["source_attention_mask"],
        banks["C"]["source_position_by_id"], c_layer, c_pos, batch_size,
    )
    _, c_induced_ap, c_induced_at = run_c_intervention_and_capture(
        model, banks["C"], source_c, Q_c, c_layer, ap_layer, at_layer,
        output_ids, c_pos, ap_pos, at_pos, batch_size, False,
    )

    metrics = {"AP_only": {}, "C_only": {}}
    for condition in ("same", "different"):
        ap_at = eval_at_recipient(
            model, tokenizer, recipient_banks[("AP_only", "AT", condition)],
            ap_induced_at, Q_at, at_layer, at_pos, W_at, b_at,
            batch_size, output_space,
        )
        for key, value in ap_at.items():
            metrics["AP_only"][f"{key}_recipient_{condition}"] = value

        c_ap = eval_ap_recipient(
            model, tokenizer, recipient_banks[("C_only", "AP", condition)],
            c_induced_ap, Q_ap, ap_layer, ap_pos, W_ap, b_ap,
            Q_at, at_layer, at_pos, W_at, b_at, batch_size, output_space,
        )
        for key, value in c_ap.items():
            metrics["C_only"][f"{key}_recipient_{condition}"] = value

        c_at = eval_at_recipient(
            model, tokenizer, recipient_banks[("C_only", "AT", condition)],
            c_induced_at, Q_at, at_layer, at_pos, W_at, b_at,
            batch_size, output_space,
        )
        for key, value in c_at.items():
            metrics["C_only"][f"{key}_recipient_{condition}"] = value
    return metrics, ap_induced_at

def artifact_paths(root, variable, layer):
    folder = root / "trained_handles" / f"{variable}_L{layer}"
    return (
        folder / f"das_{variable.lower()}.pt",
        folder / f"{variable.lower()}_readout_LogisticRegression.pt",
    )


def config_signature(root, args, at_layer, ap_layer, c_layer):
    files = []
    for variable, layer in (("AT", at_layer), ("AP", ap_layer), ("C", c_layer)):
        for path in artifact_paths(root, variable, layer):
            if not path.is_file():
                raise FileNotFoundError(f"Missing handle/readout: {path}")
            stat = path.stat()
            files.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    return {
        "seed": args.seed,
        "data_path": args.data_path,
        "eval_size": args.eval_size,
        "output_space": args.output_space,
        "eval_pair_types": EVAL_PAIR_TYPES,
        "eval_changing_only": EVAL_CHANGING_ONLY,
        "eval_code_version": EVAL_CODE_VERSION,
        "recipient_version": RECIPIENT_VERSION,
        "artifacts": files,
    }


def read_cached(path, signature):
    if not path.is_file():
        return None
    try:
        with path.open() as file:
            result = json.load(file)
        if result.get("signature") == signature:
            return result
    except (OSError, ValueError):
        pass
    return None


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as file:
        json.dump(payload, file, indent=2)
    temp.replace(path)


def write_eval_report(output_dir, at_layer, ap_layer, c_layer, clean, variable_metrics, recipient_metrics):
    lines = [f"CONFIG: AT=L{at_layer} AP=L{ap_layer} C=L{c_layer}", ""]
    lines.append(f"EVAL_CHANGING_ONLY = {EVAL_CHANGING_ONLY}")
    for variable in ("AT", "AP", "C"):
        lines.append(f"{variable} evaluation pairs = {', '.join(EVAL_PAIR_TYPES[variable])}")
    lines.append("")
    for variable in ("AT", "AP", "C"):
        lines.append(f"=== {variable} INTERVENTION ===")
        lines.append(
            f"Clean base/source: {clean[f'{variable}_base']:.4f} / "
            f"{clean[f'{variable}_source']:.4f}"
        )
        for metric, score in variable_metrics[variable].items():
            lines.append(f"{metric} = {score:.4f}")
        lines.append("")
    for family in ("AP_only", "C_only"):
        lines.append(f"=== RECIPIENT TRANSFER — {family} ===")
        for metric, score in recipient_metrics[family].items():
            lines.append(f"{metric} = {score:.4f}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_faithfulness.txt").write_text("\n".join(lines).rstrip() + "\n")


def rebuild_summary(root, eval_dir, args):
    """Maintain plot-compatible summaries from completed, valid config JSON files."""
    summary_rows = []
    detailed_rows = []
    for path in sorted(eval_dir.glob("AT_L*__AP_L*__C_L*/metrics.json")):
        with path.open() as file:
            result = json.load(file)
        at_layer = result["at_layer"]
        ap_layer = result["ap_layer"]
        c_layer = result["c_layer"]
        signature = config_signature(root, args, at_layer, ap_layer, c_layer)
        if result.get("signature") != signature:
            continue

        row = {"at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer}
        for variable in ("AT", "AP", "C"):
            for metric, value in result["metrics"][variable].items():
                row[f"{variable}_{metric}"] = value
                category = "correctness"
                if metric.startswith("Y_via_"):
                    category = "mediator"
                if metric.startswith("Y_restored_"):
                    category = "restoration"
                add_metric(
                    detailed_rows, at_layer, ap_layer, c_layer, category,
                    variable, ",".join(EVAL_PAIR_TYPES[variable]), metric, value,
                )
        for family in ("AP_only", "C_only"):
            for metric, value in result.get("recipient", {}).get(family, {}).items():
                row[f"{family}_{metric}"] = value
                intervention = "AP" if family == "AP_only" else "C"
                add_metric(
                    detailed_rows, at_layer, ap_layer, c_layer, "recipient",
                    intervention, family, metric, value,
                )
        for key, value in result["clean"].items():
            row[f"clean_{key}"] = value
        summary_rows.append(row)

    summary_path = eval_dir / "config_summary.csv"
    metrics_path = eval_dir / "all_config_metrics.csv"
    if summary_rows:
        summary_rows.sort(key=lambda row: (row["at_layer"], row["ap_layer"], row["c_layer"]))
        columns = ["at_layer", "ap_layer", "c_layer"]
        columns.extend(key for key in summary_rows[0] if key not in columns)
        temp = summary_path.with_suffix(".csv.tmp")
        with temp.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(summary_rows)
        temp.replace(summary_path)
        save_csv(detailed_rows, metrics_path)
    else:
        # Prevent plots from using stale summaries from the old seed protocol.
        for stale in (summary_path, metrics_path):
            if stale.exists():
                stale.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results_standard_all_layers_controlled_seed0_no_distractor")
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--at-layers", nargs="+", required=True,
                        help="Explicit AT layers, e.g. 23 24 25 or 23,24,25")
    parser.add_argument("--ap-layers", nargs="+", required=True,
                        help="Explicit AP layers, e.g. 5 6 7 or 5,6,7")
    parser.add_argument("--c-layers", nargs="+", required=True,
                        help="Explicit C layers, e.g. 0 1 or 0,1")
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if valid cached config exists")
    args = parser.parse_args()

    validate_eval_settings()
    print(f"Evaluation pairs: {EVAL_PAIR_TYPES}; changing_only={EVAL_CHANGING_ONLY}")

    try:
        selected = {
            "AT": parse_layers(args.at_layers, "AT"),
            "AP": parse_layers(args.ap_layers, "AP"),
            "C": parse_layers(args.c_layers, "C"),
        }
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.result_dir)
    eval_dir = root / eval_profile_name(selected)
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected layers: {selected}", flush=True)
    configs = []
    for at_layer in selected["AT"]:
        for ap_layer in selected["AP"]:
            for c_layer in selected["C"]:
                if c_layer < ap_layer < at_layer:
                    configs.append((at_layer, ap_layer, c_layer))

    if not configs:
        parser.error(f"No valid C < AP < AT configurations for {selected}")

    pending = []
    for at_layer, ap_layer, c_layer in configs:
        name = f"AT_L{at_layer}__AP_L{ap_layer}__C_L{c_layer}"
        out_dir = eval_dir / name
        signature = config_signature(root, args, at_layer, ap_layer, c_layer)
        if not args.force and read_cached(out_dir / "metrics.json", signature):
            print(f"[skip cached] {name}")
            continue
        pending.append((at_layer, ap_layer, c_layer, out_dir, signature))

    print(f"Selected {len(configs)} valid configs; {len(pending)} require evaluation")
    rebuild_summary(root, eval_dir, args)
    if not pending:
        return

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    dataset = load_generated_dataset(args.data_path)
    test_split = dataset["test"]
    banks = {}
    for variable in ("AT", "AP", "C"):
        banks[variable] = build_eval_bank(
            model, tokenizer, test_split, variable, EVAL_PAIR_TYPES[variable],
            args.eval_size, args.batch_size, args.seed,
        )

    # Schedule and tokenize once, so every layer configuration receives the
    # exact same donor→recipient assignments (for a given seed).
    schedules = prepare_recipient_schedules(banks, args.seed)
    recipient_banks = prepare_recipient_banks(tokenizer, schedules)

    handles = {"AT": {}, "AP": {}, "C": {}}
    clean_cache = {"AT": {}, "AP": {}, "C": {}}
    at_eval_cache = {}
    ap_eval_cache = {}
    ap_induced_at_cache = {}

    def get_handle(variable, layer):
        if layer not in handles[variable]:
            checkpoint_path, readout_path = artifact_paths(root, variable, layer)
            Q, saved_layer, token_pos = load_ckpt(checkpoint_path, device)
            W, b, info = load_readout(readout_path, variable, device)
            if saved_layer != layer:
                raise ValueError(f"Checkpoint layer mismatch: {checkpoint_path}")
            handles[variable][layer] = (Q, token_pos, W, b, info)
        return handles[variable][layer]

    def get_clean(variable, layer, bank):
        if layer not in clean_cache[variable]:
            Q, position, W, b, info = get_handle(variable, layer)
            if variable == "AT":
                targets = (bank["base_answer_label_ids"], bank["source_answer_label_ids"])
            elif variable == "AP":
                targets = (bank["base_answer_pointer_ids"], bank["source_answer_pointer_ids"])
            else:
                targets = (
                    color_targets(bank, info["color_to_class"], "base"),
                    color_targets(bank, info["color_to_class"], "source"),
                )
            base = clean_readout_accuracy(
                model, bank, Q, layer, position, W, b, targets[0], args.batch_size, "base"
            )
            source = clean_readout_accuracy(
                model, bank, Q, layer, position, W, b, targets[1], args.batch_size, "source"
            )
            clean_cache[variable][layer] = (base, source)
        return clean_cache[variable][layer]

    for i, (at_layer, ap_layer, c_layer, out_dir, signature) in enumerate(pending, 1):
        print(f"\n=== EVAL {i}/{len(pending)}: {out_dir.name} ===")
        Q_at, pos_at, W_at, b_at, _ = get_handle("AT", at_layer)
        Q_ap, pos_ap, W_ap, b_ap, _ = get_handle("AP", ap_layer)
        Q_c, pos_c, W_c, b_c, info_c = get_handle("C", c_layer)
        if not isinstance(pos_ap, list):
            raise ValueError(f"AP L{ap_layer} checkpoint must use four color tokens")

        at_clean = get_clean("AT", at_layer, banks["AT"])
        ap_clean = get_clean("AP", ap_layer, banks["AP"])
        c_clean = get_clean("C", c_layer, banks["C"])
        clean = {
            "AT_base": at_clean[0], "AT_source": at_clean[1],
            "AP_base": ap_clean[0], "AP_source": ap_clean[1],
            "C_base": c_clean[0], "C_source": c_clean[1],
        }

        if at_layer not in at_eval_cache:
            at_eval_cache[at_layer] = eval_at(
                model, tokenizer, banks["AT"], Q_at, at_layer, pos_at,
                W_at, b_at, args.batch_size, args.output_space,
            )
        at_metrics = at_eval_cache[at_layer]

        ap_key = (at_layer, ap_layer)
        if ap_key not in ap_eval_cache:
            ap_eval_cache[ap_key] = eval_ap(
                model, tokenizer, banks["AP"], Q_ap, ap_layer, pos_ap,
                W_ap, b_ap, Q_at, at_layer, pos_at, W_at, b_at,
                args.batch_size, args.output_space,
            )
        ap_metrics = ap_eval_cache[ap_key]
        c_metrics = eval_c(
            model, tokenizer, banks["C"], Q_c, c_layer, pos_c, W_c, b_c,
            info_c["color_to_class"], Q_ap, ap_layer, pos_ap, W_ap, b_ap,
            Q_at, at_layer, pos_at, W_at, b_at, args.batch_size, args.output_space,
        )

        variable_metrics = {"AT": at_metrics, "AP": ap_metrics, "C": c_metrics}
        ap_key = (at_layer, ap_layer)
        recipient_metrics, ap_induced_at = eval_induced_recipients(
            model, tokenizer, banks, recipient_banks,
            {"AT": get_handle("AT", at_layer),
             "AP": get_handle("AP", ap_layer),
             "C": get_handle("C", c_layer)},
            (at_layer, ap_layer, c_layer), args.batch_size, args.output_space,
            ap_induced_at_cache.get(ap_key),
        )
        ap_induced_at_cache[ap_key] = ap_induced_at
        write_eval_report(
            out_dir, at_layer, ap_layer, c_layer,
            clean, variable_metrics, recipient_metrics,
        )
        rows = []
        for variable, metrics in variable_metrics.items():
            for metric, value in metrics.items():
                family = "correctness"
                if metric.startswith("Y_via_"):
                    family = "mediator"
                if metric.startswith("Y_restored_"):
                    family = "restoration"
                add_metric(
                    rows, at_layer, ap_layer, c_layer, family, variable,
                    ",".join(EVAL_PAIR_TYPES[variable]), metric, value,
                )
        for family in ("AP_only", "C_only"):
            for metric, value in recipient_metrics[family].items():
                intervention = "AP" if family == "AP_only" else "C"
                add_metric(
                    rows, at_layer, ap_layer, c_layer, "recipient",
                    intervention, family, metric, value,
                )
        save_csv(rows, out_dir / "eval_faithfulness.csv")
        payload = {
            "at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer,
            "clean": clean, "metrics": variable_metrics,
            "recipient": recipient_metrics,
            "eval_pair_types": EVAL_PAIR_TYPES,
            "eval_changing_only": EVAL_CHANGING_ONLY,
            "signature": signature,
        }
        save_json(out_dir / "metrics.json", payload)
        rebuild_summary(root, eval_dir, args)
        print(f"[saved] {out_dir / 'metrics.json'}")

    print(f"\nDone. Cumulative results: {eval_dir / 'config_summary.csv'}")


if __name__ == "__main__":
    main()
