"""Evaluate saved Standard DAS handles AFTER selecting layers by calibration IIA.

Pipeline:
  1. mcqa_train_all_layers_no_eval.py -> summary.csv + trained_handles/
  2. Recompute CAL IIA on validation using saved handles, or specify layers manually.
  3. Select layers within threshold percentage points of each fresh CAL IIA maximum.
  4. Evaluate the valid C < AP < AT configurations; save each result immediately.

Evaluation pair selection and shuffling match mcqa_eval_completed_gradual.py.
No training or optimization happens in this script.
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from mcqa_load_randomized_ap_at_c import (
    PAIR_FIELDS,
    create_bank,
    load_generated_dataset,
    predict_correct,
    row_to_pairs,
)
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_standard_das import (
    add_metric,
    balanced_counts,
    clean_readout_accuracy,
    collect_states,
    evaluate_standard_direct,
    color_targets,
    eval_ap,
    eval_at,
    eval_c,
    load_ckpt,
    load_readout,
    save_csv,
)

# Use the same settings as mcqa_eval_completed_gradual.py.
# To evaluate mixed pairs later, edit BOTH scripts to the same settings.

ALL_PAIRS = ["AT_only", "AP_only", "C_only", "AP_AT", "C_AP_AT"]
EVAL_PAIR_TYPES = {
    "AT": ALL_PAIRS,
    "AP": ALL_PAIRS,
    "C": ALL_PAIRS,
}
EVAL_CHANGING_ONLY = False

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
EVAL_CODE_VERSION = 2


def validate_eval_settings():
    for variable in ("AT", "AP", "C"):
        names = EVAL_PAIR_TYPES[variable]
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{variable}: pair types must be nonempty and unique")
        for name in names:
            if name not in PAIR_FIELDS:
                raise ValueError(f"Unknown pair type: {name}")


def eval_profile_name():
    settings = {"pairs": EVAL_PAIR_TYPES, "changing_only": EVAL_CHANGING_ONLY}
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return f"eval_configs_{digest}"


def build_eval_bank(model, tokenizer, dataset_split, variable,
                    pair_types, total_size, batch_size, seed, split_name="test"):
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
                for pair in row_to_pairs(row, split_name):
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
        print(f"[{split_name} bank] {variable}/{pair_type}: {needed} examples", flush=True)

    # This matches GDAS evaluation exactly, including AT seed+100.
    shuffle_seed = seed + BANK_SHUFFLE_OFFSETS[variable]
    return create_bank(mixed, tokenizer, size=None, seed=shuffle_seed)


def available_layers(root, variable):
    layers = []
    for folder in (root / "trained_handles").glob(f"{variable}_L*"):
        suffix = folder.name.split("_L", 1)[1]
        if not folder.is_dir() or not suffix.isdigit():
            continue
        layer = int(suffix)
        checkpoint, _ = artifact_paths(root, variable, layer)
        if checkpoint.is_file():
            layers.append(layer)
    if not layers:
        raise ValueError(f"No saved {variable} handles in {root}")
    return sorted(set(layers))


def recalibrate_variable(model, tokenizer, dataset, root, variable, args):
    bank = build_eval_bank(
        model, tokenizer, dataset["validation"], variable,
        EVAL_PAIR_TYPES[variable], args.cal_size, args.batch_size,
        args.seed, split_name="validation",
    )
    device = next(model.parameters()).device
    scored = []
    for layer in available_layers(root, variable):
        checkpoint, _ = artifact_paths(root, variable, layer)
        Q, saved_layer, position = load_ckpt(checkpoint, device)
        if saved_layer != layer:
            raise ValueError(f"Checkpoint layer mismatch: {checkpoint}")
        source_states = collect_states(
            model, bank["source_input_ids"], bank["source_attention_mask"],
            bank["source_position_by_id"], layer, position, args.batch_size,
        )
        score = float(evaluate_standard_direct(
            model, tokenizer, bank, source_states, Q, layer, position,
            TARGET_BY_VARIABLE[variable], args.batch_size, args.output_space,
        ))
        scored.append((layer, score))
        print(f"[CAL] {variable} L{layer}: IIA={score:.4f}", flush=True)
        del Q, source_states
    return scored


def select_from_scores(variable, scored, threshold_pct):
    highest = max(score for _, score in scored)
    cutoff = highest - threshold_pct / 100.0
    selected = []
    for layer, score in scored:
        if score >= cutoff - 1e-12:
            selected.append(layer)
    print(f"{variable}: max cal={highest:.4f}, cutoff={cutoff:.4f}, layers={selected}")
    return selected


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


def write_eval_report(output_dir, at_layer, ap_layer, c_layer, clean, variable_metrics):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_faithfulness.txt").write_text("\n".join(lines).rstrip() + "\n")


def rebuild_summary(root, eval_dir, args, configs):
    """Maintain plot-compatible summaries from completed, valid config JSON files."""
    summary_rows = []
    detailed_rows = []
    selected_configs = set(configs)
    for path in sorted(eval_dir.glob("AT_L*__AP_L*__C_L*/metrics.json")):
        with path.open() as file:
            result = json.load(file)
        at_layer = result["at_layer"]
        ap_layer = result["ap_layer"]
        c_layer = result["c_layer"]
        if (at_layer, ap_layer, c_layer) not in selected_configs:
            continue
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
    parser.add_argument("--threshold-pct", type=float, default=5.0,
                        help="Within this many percentage points of best CAL IIA")
    parser.add_argument("--at-layers", type=int, nargs="+", default=None)
    parser.add_argument("--ap-layers", type=int, nargs="+", default=None)
    parser.add_argument("--c-layers", type=int, nargs="+", default=None)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--cal-only", action="store_true",
                        help="Recompute calibration IIA and exit before test evaluation")
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if valid cached config exists")
    args = parser.parse_args()

    if not 0 <= args.threshold_pct <= 100:
        raise ValueError("--threshold-pct must be in [0, 100]")
    if args.cal_size <= 0 or args.eval_size <= 0 or args.batch_size <= 0:
        parser.error("--cal-size, --eval-size and --batch-size must be positive")
    validate_eval_settings()
    print(f"Evaluation pairs: {EVAL_PAIR_TYPES}; changing_only={EVAL_CHANGING_ONLY}")

    root = Path(args.result_dir)
    eval_dir = root / eval_profile_name()
    eval_dir.mkdir(parents=True, exist_ok=True)
    manual = {"AT": args.at_layers, "AP": args.ap_layers, "C": args.c_layers}
    if args.cal_only and any(layers is not None for layers in manual.values()):
        parser.error("--cal-only recalibrates all saved layers; do not pass manual layer options")
    selected = {}
    cal_rows = []
    model, tokenizer, dataset = None, None, None
    for variable in ("AT", "AP", "C"):
        if manual[variable] is not None:
            selected[variable] = sorted(set(manual[variable]))
            for layer in selected[variable]:
                for path in artifact_paths(root, variable, layer):
                    if layer < 0 or not path.is_file():
                        raise ValueError(f"Invalid manual layer or missing artifact: {path}")
            print(f"[manual] {variable}: {selected[variable]}")
            continue
        if model is None:
            model, tokenizer = load_gemma_model()
            model.eval()
            dataset = load_generated_dataset(args.data_path)
        scored = recalibrate_variable(model, tokenizer, dataset, root, variable, args)
        selected[variable] = select_from_scores(variable, scored, args.threshold_pct)
        for layer, score in scored:
            cal_rows.append({"variable": variable, "layer": layer, "cal_iia": score})

    # Never overwrite the training summary. Keep the latest selection separately.
    cal_path = eval_dir / "recalibration.csv"
    if cal_rows:
        with cal_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["variable", "layer", "cal_iia"])
            writer.writeheader()
            writer.writerows(cal_rows)
    elif cal_path.exists():
        cal_path.unlink()
    save_json(eval_dir / "selected_layers.json", {
        "layers": selected, "manual": manual, "threshold_pct": args.threshold_pct,
        "data_path": args.data_path, "split": "validation", "cal_size": args.cal_size,
        "seed": args.seed, "output_space": args.output_space,
        "pair_types": EVAL_PAIR_TYPES, "changing_only": EVAL_CHANGING_ONLY,
    })
    if args.cal_only:
        print(f"[cal-only] Calibration results: {cal_path}")
        return

    configs = []
    for at_layer in selected["AT"]:
        for ap_layer in selected["AP"]:
            for c_layer in selected["C"]:
                if c_layer < ap_layer < at_layer:
                    configs.append((at_layer, ap_layer, c_layer))

    if not configs:
        raise ValueError("No valid configurations satisfy C < AP < AT")

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
    rebuild_summary(root, eval_dir, args, configs)
    if not pending:
        return

    if model is None:
        model, tokenizer = load_gemma_model()
        model.eval()
        dataset = load_generated_dataset(args.data_path)
    device = next(model.parameters()).device
    test_split = dataset["test"]
    banks = {}
    for variable in ("AT", "AP", "C"):
        banks[variable] = build_eval_bank(
            model, tokenizer, test_split, variable, EVAL_PAIR_TYPES[variable],
            args.eval_size, args.batch_size, args.seed,
        )

    handles = {"AT": {}, "AP": {}, "C": {}}
    clean_cache = {"AT": {}, "AP": {}, "C": {}}
    at_eval_cache = {}
    ap_eval_cache = {}

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
        write_eval_report(out_dir, at_layer, ap_layer, c_layer, clean, variable_metrics)
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
        save_csv(rows, out_dir / "eval_faithfulness.csv")
        payload = {
            "at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer,
            "clean": clean, "metrics": variable_metrics,
            "eval_pair_types": EVAL_PAIR_TYPES,
            "eval_changing_only": EVAL_CHANGING_ONLY,
            "signature": signature,
        }
        save_json(out_dir / "metrics.json", payload)
        rebuild_summary(root, eval_dir, args, configs)
        print(f"[saved] {out_dir / 'metrics.json'}")

    print(f"\nDone. Cumulative results: {eval_dir / 'config_summary.csv'}")


if __name__ == "__main__":
    main()
