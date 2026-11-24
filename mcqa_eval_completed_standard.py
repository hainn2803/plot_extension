"""Evaluate saved Standard DAS handles AFTER selecting layers by calibration IIA.

Pipeline:
  1. mcqa_train_all_layers_no_eval.py -> summary.csv + trained_handles/
  2. Select AT/AP/C layers within 5 percentage points of each CAL IIA maximum.
  3. Evaluate the valid C < AP < AT configurations; save each result immediately.

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


def select_layers(summary_path, threshold_pct):
    """Choose each variable's layers using CAL IIA >= max(CAL IIA) - threshold_pct/100."""
    with summary_path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    selected = {}
    for variable in ("AT", "AP", "C"):
        subset = [row for row in rows if row.get("variable") == variable]
        if not subset:
            raise ValueError(f"No {variable} rows in {summary_path}")
        scored = [(int(row["layer"]), float(row["cal_iia"])) for row in subset]
        highest = max(score for _, score in scored)
        cutoff = highest - threshold_pct / 100.0
        selected[variable] = sorted({layer for layer, score in scored if score >= cutoff - 1e-12})
        print(f"{variable}: max cal={highest:.4f}, cutoff={cutoff:.4f}, layers={selected[variable]}")
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
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if valid cached config exists")
    args = parser.parse_args()

    if not 0 <= args.threshold_pct <= 100:
        raise ValueError("--threshold-pct must be in [0, 100]")
    validate_eval_settings()
    print(f"Evaluation pairs: {EVAL_PAIR_TYPES}; changing_only={EVAL_CHANGING_ONLY}")

    root = Path(args.result_dir)
    eval_dir = root / eval_profile_name()
    eval_dir.mkdir(parents=True, exist_ok=True)
    selected = select_layers(root / "summary.csv", args.threshold_pct)
    configs = []
    for at_layer in selected["AT"]:
        for ap_layer in selected["AP"]:
            for c_layer in selected["C"]:
                if c_layer < ap_layer < at_layer:
                    configs.append((at_layer, ap_layer, c_layer))

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
        rebuild_summary(root, eval_dir, args)
        print(f"[saved] {out_dir / 'metrics.json'}")

    print(f"\nDone. Cumulative results: {eval_dir / 'config_summary.csv'}")


if __name__ == "__main__":
    main()
