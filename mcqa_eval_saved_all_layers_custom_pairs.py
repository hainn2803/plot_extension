"""Evaluate saved Standard DAS handles, selecting layers by CAL IIA only.

Layer rule: cal_iia >= max_cal_iia(variable) - threshold_pct / 100.
Every completed configuration is saved immediately and reused on later runs,
including runs with a larger threshold (e.g. 5 -> 10).
"""

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

from mcqa_load_randomized_ap_at_c import PAIR_FIELDS, create_bank, load_generated_dataset, predict_correct, row_to_pairs
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_standard_das import (
    add_metric,
    clean_readout_accuracy,
    balanced_counts,
    color_targets,
    eval_ap,
    eval_at,
    eval_c,
    load_ckpt,
    load_readout,
    save_csv,
)


# ============================================================================
# EDIT THESE SETTINGS — no command-line flags needed for dataset selection.
# Each variable is evaluated on a BALANCED MIX of the selected pair types.
# EVAL_SIZE is the total number of examples PER VARIABLE (default: 400).
# ============================================================================
EVAL_PAIR_TYPES = {
    "AT": ["AT_only", "C_AP_AT"],                  # e.g. ["AT_only", "AP_AT", "C_AP_AT"]
    "AP": ["AP_only", "C_AP_AT"],                  # e.g. ["AP_only", "AP_AT", "C_AP_AT"]
    "C": ["C_only", "C_AP_AT"],                    # e.g. ["C_only", "C_AP_AT"]
}

# True: test examples must change the expected answer under THIS intervention.
# False: also include preservation examples.
EVAL_CHANGING_ONLY = True

TARGET_BY_VARIABLE = {"AT": "Y_after_AT_intervene", "AP": "Y_after_AP_intervene", "C": "Y_after_C_intervene"}
PAIR_SEEDS = {"AT_only": 3001, "AP_only": 3002, "C_only": 3004, "AP_AT": 3005, "C_AP_AT": 3006}


def validate_eval_settings():
    for variable in ("AT", "AP", "C"):
        names = EVAL_PAIR_TYPES.get(variable, [])
        if not names or len(names) != len(set(names)):
            raise ValueError(f"{variable}: pair types must be nonempty and unique")
        for name in names:
            if name not in PAIR_FIELDS:
                raise ValueError(f"Unknown pair type {name!r}; valid: {list(PAIR_FIELDS)}")


def eval_profile_name():
    settings = {"pairs": EVAL_PAIR_TYPES, "changing_only": EVAL_CHANGING_ONLY}
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return f"eval_configs_{digest}"


def build_eval_bank(model, tokenizer, dataset_split, variable, pair_types, total_size, batch_size, seed):
    """Balanced mix, optionally filter to changing examples BEFORE sampling quota."""
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

            correct = predict_correct(model, tokenizer, candidates, batch_size, prediction_cache) if candidates else []
            accepted.extend(correct)
            if len(accepted) >= needed:
                break

        if len(accepted) < needed:
            raise ValueError(
                f"{variable}/{pair_type}: found {len(accepted)} qualifying examples, need {needed}. "
                "This pair type may contain no changing examples for this intervention; "
                "remove it from EVAL_PAIR_TYPES or set EVAL_CHANGING_ONLY=False."
            )
        mixed.extend(accepted[:needed])
        print(f"[eval bank] {variable} {pair_type}: {needed} examples")

    return create_bank(mixed, tokenizer, size=None, seed=seed + 5000)


def write_eval_report(output_dir, at_layer, ap_layer, c_layer, clean, variable_metrics):
    lines = [f"CONFIG: AT=L{at_layer} AP=L{ap_layer} C=L{c_layer}", ""]
    lines.append(f"EVAL_CHANGING_ONLY = {EVAL_CHANGING_ONLY}")
    for variable in ("AT", "AP", "C"):
        lines.append(f"{variable} evaluation pairs = {', '.join(EVAL_PAIR_TYPES[variable])}")
    lines.append("")
    for variable in ("AT", "AP", "C"):
        lines.append(f"=== {variable} INTERVENTION ===")
        lines.append(f"Clean base/source: {clean[f'{variable}_base']:.4f} / {clean[f'{variable}_source']:.4f}")
        for metric, value in variable_metrics[variable].items():
            lines.append(f"{metric} = {value:.4f}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_faithfulness.txt").write_text("\n".join(lines).rstrip() + "\n")


def select_layers(summary_path, threshold_pct):
    with open(summary_path, newline="") as file:
        rows = list(csv.DictReader(file))

    selected = {}
    for variable in ("AT", "AP", "C"):
        subset = [row for row in rows if row["variable"] == variable]
        if not subset:
            raise ValueError(f"No {variable} rows in {summary_path}")

        best = max(float(row["cal_iia"]) for row in subset)
        cutoff = best - threshold_pct / 100.0
        layers = []
        for row in subset:
            if float(row["cal_iia"]) >= cutoff - 1e-12:
                layers.append(int(row["layer"]))
        selected[variable] = sorted(set(layers))
        print(f"{variable}: max CAL IIA={best:.4f}, cutoff={cutoff:.4f}, layers={selected[variable]}")
    return selected


def artifact_paths(root, variable, layer):
    folder = root / "trained_handles" / f"{variable}_L{layer}"
    return [folder / f"das_{variable.lower()}.pt", folder / f"{variable.lower()}_readout_LogisticRegression.pt"]


def config_signature(root, args, at_layer, ap_layer, c_layer):
    files = []
    for variable, layer in (("AT", at_layer), ("AP", ap_layer), ("C", c_layer)):
        for path in artifact_paths(root, variable, layer):
            if not path.is_file():
                raise FileNotFoundError(f"Missing saved handle/readout: {path}")
            info = path.stat()
            files.append([str(path.relative_to(root)), info.st_size, info.st_mtime_ns])

    return {
        "seed": args.seed,
        "data_path": args.data_path,
        "eval_size": args.eval_size,
        "output_space": args.output_space,
        "eval_pair_types": EVAL_PAIR_TYPES,
        "eval_changing_only": EVAL_CHANGING_ONLY,
        "eval_code_version": 1,
        "artifacts": files,
    }


def read_cached(path, signature):
    if not path.is_file():
        return None
    try:
        with open(path) as file:
            result = json.load(file)
        if result.get("signature") == signature:
            return result
    except (OSError, ValueError):
        pass
    return None


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with open(temp, "w") as file:
        json.dump(payload, file, indent=2)
    temp.replace(path)  # Acts as the completed-config marker.


def rebuild_summary(root, eval_dir, args):
    summary_rows = []
    detailed_rows = []
    for path in sorted(eval_dir.glob("AT_L*__AP_L*__C_L*/metrics.json")):
        with open(path) as file:
            result = json.load(file)

        at_layer, ap_layer, c_layer = result["at_layer"], result["ap_layer"], result["c_layer"]
        signature = config_signature(root, args, at_layer, ap_layer, c_layer)
        if result["signature"] != signature:
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
                dataset_label = ",".join(EVAL_PAIR_TYPES[variable])
                add_metric(detailed_rows, at_layer, ap_layer, c_layer, category, variable, dataset_label, metric, value)

        for key, value in result["clean"].items():
            row[f"clean_{key}"] = value
        summary_rows.append(row)

    if summary_rows:
        summary_rows.sort(key=lambda row: (row["at_layer"], row["ap_layer"], row["c_layer"]))
        fieldnames = ["at_layer", "ap_layer", "c_layer"]
        fieldnames.extend(key for key in summary_rows[0] if key not in fieldnames)
        with open(eval_dir / "config_summary.csv", "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
        save_csv(detailed_rows, eval_dir / "all_config_metrics.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results_standard_all_layers_mixedall_seed0")
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at")
    parser.add_argument("--threshold-pct", type=float, default=5.0, help="CAL IIA within x percentage points of variable max")
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--force", action="store_true", help="Recompute even if a config has a valid saved result")
    args = parser.parse_args()

    if not 0 <= args.threshold_pct <= 100:
        raise ValueError("--threshold-pct must be between 0 and 100")

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
        output_dir = eval_dir / name
        signature = config_signature(root, args, at_layer, ap_layer, c_layer)
        if not args.force and read_cached(output_dir / "metrics.json", signature):
            print(f"[skip cached] {name}")
            continue
        pending.append((at_layer, ap_layer, c_layer, output_dir, signature))

    print(f"Selected {len(configs)} valid configurations; {len(configs) - len(pending)} cached, {len(pending)} to evaluate")
    rebuild_summary(root, eval_dir, args)
    if not pending:
        return

    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    dataset = load_generated_dataset(args.data_path)
    test_split = dataset["test"]
    banks = {}
    for variable in ("AT", "AP", "C"):
        bank_seed = args.seed + {"AT": 0, "AP": 100, "C": 200}[variable]
        banks[variable] = build_eval_bank(model, tokenizer, test_split, variable, EVAL_PAIR_TYPES[variable], args.eval_size, args.batch_size, bank_seed)

    handles = {"AT": {}, "AP": {}, "C": {}}
    clean_cache = {"AT": {}, "AP": {}, "C": {}}
    at_eval_cache = {}
    ap_eval_cache = {}

    def get_handle(variable, layer):
        if layer not in handles[variable]:
            checkpoint_path, readout_path = artifact_paths(root, variable, layer)
            Q, saved_layer, token_position = load_ckpt(checkpoint_path, device)
            W, b, info = load_readout(readout_path, variable, device)
            if saved_layer != layer:
                raise ValueError(f"Checkpoint layer mismatch: {checkpoint_path}")
            handles[variable][layer] = (Q, token_position, W, b, info)
        return handles[variable][layer]

    def get_clean(variable, layer, bank):
        if layer not in clean_cache[variable]:
            Q, token_position, W, b, info = get_handle(variable, layer)
            if variable == "AT":
                targets = (bank["base_answer_label_ids"], bank["source_answer_label_ids"])
            elif variable == "AP":
                targets = (bank["base_answer_pointer_ids"], bank["source_answer_pointer_ids"])
            else:
                targets = (color_targets(bank, info["color_to_class"], "base"), color_targets(bank, info["color_to_class"], "source"))

            base_acc = clean_readout_accuracy(model, bank, Q, layer, token_position, W, b, targets[0], args.batch_size, "base")
            source_acc = clean_readout_accuracy(model, bank, Q, layer, token_position, W, b, targets[1], args.batch_size, "source")
            clean_cache[variable][layer] = (base_acc, source_acc)
        return clean_cache[variable][layer]

    for i, (at_layer, ap_layer, c_layer, output_dir, signature) in enumerate(pending, 1):
        name = output_dir.name
        print(f"\n=== EVAL {i}/{len(pending)}: {name} ===")
        Q_at, pos_at, W_at, b_at, _ = get_handle("AT", at_layer)
        Q_ap, pos_ap, W_ap, b_ap, _ = get_handle("AP", ap_layer)
        Q_c, pos_c, W_c, b_c, info_c = get_handle("C", c_layer)
        if not isinstance(pos_ap, list):
            raise ValueError(f"AP checkpoint must use 4 color tokens: AP L{ap_layer}")

        at_clean = get_clean("AT", at_layer, banks["AT"])
        ap_clean = get_clean("AP", ap_layer, banks["AP"])
        c_clean = get_clean("C", c_layer, banks["C"])
        clean = {
            "AT_base": at_clean[0], "AT_source": at_clean[1],
            "AP_base": ap_clean[0], "AP_source": ap_clean[1],
            "C_base": c_clean[0], "C_source": c_clean[1],
        }

        if at_layer not in at_eval_cache:
            at_eval_cache[at_layer] = eval_at(model, tokenizer, banks["AT"], Q_at, at_layer, pos_at, W_at, b_at, args.batch_size, args.output_space)
        at_metrics = at_eval_cache[at_layer]

        ap_key = (at_layer, ap_layer)
        if ap_key not in ap_eval_cache:
            ap_eval_cache[ap_key] = eval_ap(model, tokenizer, banks["AP"], Q_ap, ap_layer, pos_ap, W_ap, b_ap, Q_at, at_layer, pos_at, W_at, b_at, args.batch_size, args.output_space)
        ap_metrics = ap_eval_cache[ap_key]
        c_metrics = eval_c(model, tokenizer, banks["C"], Q_c, c_layer, pos_c, W_c, b_c, info_c["color_to_class"], Q_ap, ap_layer, pos_ap, W_ap, b_ap, Q_at, at_layer, pos_at, W_at, b_at, args.batch_size, args.output_space)

        variable_metrics = {"AT": at_metrics, "AP": ap_metrics, "C": c_metrics}
        write_eval_report(output_dir, at_layer, ap_layer, c_layer, clean, variable_metrics)
        rows = []
        for variable, metrics in variable_metrics.items():
            for metric, value in metrics.items():
                category = "correctness"
                if metric.startswith("Y_via_"):
                    category = "mediator"
                if metric.startswith("Y_restored_"):
                    category = "restoration"
                dataset_label = ",".join(EVAL_PAIR_TYPES[variable])
                add_metric(rows, at_layer, ap_layer, c_layer, category, variable, dataset_label, metric, value)
        save_csv(rows, output_dir / "eval_faithfulness.csv")

        payload = {
            "at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer,
            "clean": clean, "metrics": variable_metrics,
            "eval_pair_types": EVAL_PAIR_TYPES, "eval_changing_only": EVAL_CHANGING_ONLY,
            "signature": signature,
        }
        save_json(output_dir / "metrics.json", payload)
        rebuild_summary(root, eval_dir, args)
        print(f"[saved] {output_dir / 'metrics.json'}")

    print(f"\nDone. Cumulative results: {eval_dir / 'config_summary.csv'}")


if __name__ == "__main__":
    main()
