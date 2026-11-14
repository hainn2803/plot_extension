"""Evaluate ONLY complete saved Gradual DAS AT -> AP -> C configurations.

Works with checkpoints produced by mcqa_train_gradual_efficient_v2.py:
  trained_handles/AT_L23/das_at.pt
  trained_handles/AT_L23/at_readout_LogisticRegression.pt
  trained_handles/AT_L23__AP_L12/das_ap.pt
  trained_handles/AT_L23__AP_L12/ap_readout_LogisticRegression.pt
  trained_handles/AT_L23__AP_L12__C_L1/das_c.pt
  trained_handles/AT_L23__AP_L12__C_L1/c_readout_LogisticRegression.pt

No training, no need for all 165 combinations to exist. Incomplete or
incompatible configurations are skipped. Save after EACH evaluation, resume
based on the eval settings and exact saved artifact identities.

Run:
    python3 mcqa_eval_gradual_completed.py
    python3 mcqa_eval_gradual_completed.py --list-only
    python3 mcqa_eval_gradual_completed.py --force
"""

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import torch

from mcqa_load_randomized_ap_at_c import (
    PAIR_FIELDS, create_bank, load_generated_dataset, predict_correct, row_to_pairs,
)
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_gradual_das import (
    add_metric, clean_readout_accuracy, color_targets, eval_ap, eval_at, eval_c,
    load_ckpt, load_readout,
)


# ============================================================================
# EDIT SETTINGS HERE
# ============================================================================
RESULT_DIR = "results_gradual_2.02.04.0_selected_layers_mixedall_seed0_no_distractor"
DATA_PATH = "randomized_copycolors_include_ap_at_no_distractor"
EVAL_SIZE = 400  # total test examples PER variable, NOT per pair type
BATCH_SIZE = 32
SEED = 0
OUTPUT_SPACE = "full"  # "full" or "az"

# Each variable can use its own (possibly mixed) test pair types.
EVAL_PAIR_TYPES = {
    "AT": ["AT_only", "C_AP_AT"],
    "AP": ["AP_only", "C_AP_AT"],
    "C": ["C_only", "C_AP_AT"],
}
EVAL_CHANGING_ONLY = True  # require target Y_after_variable != base_Y

# Optional: skip candidates whose saved v2 metadata does not match upstream
# checkpoints/readouts. Legacy checkpoints without metadata remain evaluable.
VERIFY_SAVED_DEPENDENCIES = True

TARGETS = {"AT": "Y_after_AT_intervene", "AP": "Y_after_AP_intervene", "C": "Y_after_C_intervene"}
PAIR_SEEDS = {"AT_only": 3001, "AP_only": 3002, "C_only": 3004, "AP_AT": 3005, "C_AP_AT": 3006}
VARS = ("AT", "AP", "C")
PATTERN = re.compile(r"^AT_L(\d+)__AP_L(\d+)__C_L(\d+)$")
METRICS_VERSION = 1


def validate_settings():
    if EVAL_SIZE <= 0 or BATCH_SIZE <= 0:
        raise ValueError("EVAL_SIZE and BATCH_SIZE must be positive")
    if OUTPUT_SPACE not in ("full", "az"):
        raise ValueError("OUTPUT_SPACE must be 'full' or 'az'")
    for variable in VARS:
        choices = EVAL_PAIR_TYPES[variable]
        if not choices or len(choices) != len(set(choices)):
            raise ValueError(f"{variable}: pair types must be nonempty and unique")
        for name in choices:
            if name not in PAIR_FIELDS:
                raise ValueError(f"Unknown pair type: {name}")


def paths_for(root, at_layer, ap_layer, c_layer):
    parent = root / "trained_handles"
    names = {
        "AT": parent / f"AT_L{at_layer}",
        "AP": parent / f"AT_L{at_layer}__AP_L{ap_layer}",
        "C": parent / f"AT_L{at_layer}__AP_L{ap_layer}__C_L{c_layer}",
    }
    result = {}
    for variable in VARS:
        folder = names[variable]
        result[variable] = {
            "checkpoint": folder / f"das_{variable.lower()}.pt",
            "readout": folder / f"{variable.lower()}_readout_LogisticRegression.pt",
            "metadata": folder / "readout_meta.json",
        }
    return result


def discover_configs(root):
    """Discover configurations from C directories; no dependence on summary.csv."""
    trained_dir = root / "trained_handles"
    if not trained_dir.is_dir():
        raise FileNotFoundError(f"No trained_handles directory: {trained_dir}")
    completed, incomplete = [], []
    for folder in sorted(trained_dir.iterdir()):
        if not folder.is_dir():
            continue
        match = PATTERN.fullmatch(folder.name)
        if match is None:
            continue
        at_layer, ap_layer, c_layer = [int(x) for x in match.groups()]
        missing = []
        if not c_layer < ap_layer < at_layer:
            missing.append("invalid layer order (need C < AP < AT)")
        paths = paths_for(root, at_layer, ap_layer, c_layer)
        for variable in VARS:
            for key in ("checkpoint", "readout"):
                file = paths[variable][key]
                if not file.is_file() or file.stat().st_size == 0:
                    missing.append(str(file.relative_to(root)))
        item = {"name": folder.name, "layers": (at_layer, ap_layer, c_layer), "paths": paths}
        if missing:
            incomplete.append({"name": folder.name, "reason": "; ".join(missing)})
        else:
            completed.append(item)
    completed.sort(key=lambda x: x["layers"])
    return completed, incomplete


def stable_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path):
    hash_value = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def stage_identity(stage_paths, expected_variable, expected_layer, cache):
    """Load/save a small, validated identity for an existing handle and readout."""
    checkpoint = stage_paths["checkpoint"]
    readout = stage_paths["readout"]
    key = (str(checkpoint), str(readout))
    if key in cache:
        return cache[key]

    info = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(info, dict) or "basis" not in info:
        raise ValueError(f"invalid checkpoint: {checkpoint}")
    if info.get("variable") != expected_variable or int(info.get("layer", -1)) != expected_layer:
        raise ValueError(f"checkpoint variable/layer mismatch: {checkpoint}")

    basis = info["basis"].detach().cpu().contiguous()
    basis_hash = hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    if info.get("basis_sha256") and info["basis_sha256"] != basis_hash:
        raise ValueError(f"basis checksum mismatch: {checkpoint}")
    if info.get("training_signature") and info.get("hyperparameters"):
        if stable_digest(info["hyperparameters"]) != info["training_signature"]:
            raise ValueError(f"checkpoint hyperparameter signature mismatch: {checkpoint}")

    readout_info = torch.load(readout, map_location="cpu", weights_only=False)
    if not isinstance(readout_info, dict) or f"W_{expected_variable}" not in readout_info:
        raise ValueError(f"invalid readout: {readout}")

    meta_path = stage_paths["metadata"]
    readout_identity = None
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        expected_sha = meta.get("file_sha256")
        actual_sha = file_sha256(readout)
        if expected_sha != actual_sha:
            raise ValueError(f"readout checksum mismatch: {readout}")
        if meta.get("hyperparameters"):
            if stable_digest(meta["hyperparameters"]) != meta.get("readout_signature"):
                raise ValueError(f"readout signature mismatch: {meta_path}")
            correct_handle = meta["hyperparameters"].get("handle")
            if correct_handle is not None:
                actual_handle = {"training_signature": info.get("training_signature"), "basis_sha256": basis_hash}
                if correct_handle != actual_handle:
                    raise ValueError(f"readout fitted to different handle: {readout}")
        readout_identity = {"readout_signature": meta.get("readout_signature"), "file_sha256": actual_sha}

    handle_identity = {"training_signature": info.get("training_signature"), "basis_sha256": basis_hash}
    # Keep only metadata in the scan cache, not hundreds of MB of basis tensors.
    info.pop("basis", None)
    stage = {"info": info, "handle": handle_identity, "readout": readout_identity}
    cache[key] = stage
    return stage


def check_dependencies(item, cache):
    """Check that AP/C really reference the saved upstream v2 handles."""
    layers = item["layers"]
    verified = {}
    for i, variable in enumerate(VARS):
        layer = layers[i]
        verified[variable] = stage_identity(item["paths"][variable], variable, layer, cache)
        cond = verified[variable]["info"].get("conditioning")
        if isinstance(cond, dict):
            for j, field in enumerate(("AT", "AP", "C")):
                if field not in cond or cond[field] is None:
                    continue
                if int(cond[field]) != layers[j]:
                    raise ValueError(f"{variable} conditioning {field} mismatch for {item['name']}")

    if VERIFY_SAVED_DEPENDENCIES:
        for variable in ("AP", "C"):
            info = verified[variable]["info"]
            hparams = info.get("hyperparameters")
            if not isinstance(hparams, dict):
                continue  # legacy checkpoint: no upstream fingerprint was stored
            deps = hparams.get("dependencies")
            if not isinstance(deps, dict):
                continue
            expected_variables = ["AT"] if variable == "AP" else ["AT", "AP"]
            for upstream in expected_variables:
                if upstream not in deps:
                    raise ValueError(f"missing dependency {upstream} in {item['name']}")
                identity = {"handle": verified[upstream]["handle"], "readout": verified[upstream]["readout"]}
                if identity != deps[upstream]:
                    raise ValueError(f"{variable} depends on a DIFFERENT {upstream} checkpoint/readout: {item['name']}")
    return verified


def build_eval_bank(model, tokenizer, test_split, variable):
    """Balanced per-type quota; apply changing filter BEFORE taking N examples."""
    choices = EVAL_PAIR_TYPES[variable]
    base_count = EVAL_SIZE // len(choices)
    quotas = {name: base_count + int(i < EVAL_SIZE % len(choices)) for i, name in enumerate(choices)}
    prediction_cache = {}
    all_pairs = []
    for pair_type in choices:
        indices = list(range(len(test_split)))
        random.Random(SEED + PAIR_SEEDS[pair_type]).shuffle(indices)
        accepted = []
        for start in range(0, len(indices), 64):
            candidates = []
            for index in indices[start:start + 64]:
                row = test_split[int(index)]
                for pair in row_to_pairs(row, "test"):
                    if pair["pair_type"] != pair_type:
                        continue
                    if EVAL_CHANGING_ONLY and pair[TARGETS[variable]] == pair["base_Y"]:
                        continue
                    candidates.append(pair)
            if candidates:
                accepted.extend(predict_correct(model, tokenizer, candidates, BATCH_SIZE, prediction_cache))
            if len(accepted) >= quotas[pair_type]:
                break
        if len(accepted) < quotas[pair_type]:
            raise ValueError(f"Not enough test pairs: {variable}/{pair_type}: need {quotas[pair_type]}, found {len(accepted)}")
        all_pairs.extend(accepted[:quotas[pair_type]])
        print(f"[test bank] {variable}/{pair_type} = {quotas[pair_type]}", flush=True)
    return create_bank(all_pairs, tokenizer, size=None, seed=SEED + {"AT": 100, "AP": 200, "C": 300}[variable])


def data_stamp(data_path):
    path = Path(data_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset: {path}")
    result = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            stat = file.stat()
            result.append([str(file.relative_to(path)), stat.st_size, stat.st_mtime_ns])
    return stable_digest(result)


def eval_settings(data_path):
    return {
        "version": METRICS_VERSION,
        "data_path": str(Path(data_path).resolve()),
        "data_stamp": data_stamp(data_path),
        "pair_types": EVAL_PAIR_TYPES,
        "changing_only": EVAL_CHANGING_ONLY,
        "eval_size": EVAL_SIZE,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "output_space": OUTPUT_SPACE,
    }


def artifact_signature(item, settings):
    artifacts = []
    for variable in VARS:
        for key in ("checkpoint", "readout", "metadata"):
            path = item["paths"][variable][key]
            if not path.is_file():
                continue
            info = path.stat()
            artifacts.append([str(path), info.st_size, info.st_mtime_ns])
    return stable_digest({"settings": settings, "artifacts": artifacts})


def read_cache(path, signature):
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        if data.get("signature") == signature:
            return data
    except (OSError, ValueError):
        pass
    return None


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def write_csv_atomic(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_summaries(output_root, records, status_rows):
    summaries = []
    detailed = []
    for name in sorted(records):
        record = records[name]
        at_layer, ap_layer, c_layer = record["at_layer"], record["ap_layer"], record["c_layer"]
        row = {"at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer}
        for variable in VARS:
            for metric, value in record["metrics"][variable].items():
                row[f"{variable}_{metric}"] = float(value)
                category = "correctness"
                if metric.startswith("Y_via_"):
                    category = "mediator"
                if metric.startswith("Y_restored_"):
                    category = "restoration"
                dataset_label = ",".join(EVAL_PAIR_TYPES[variable])
                add_metric(detailed, at_layer, ap_layer, c_layer, category, variable, dataset_label, metric, value)
        for name_clean, value in record["clean"].items():
            row[f"clean_{name_clean}"] = float(value)
        summaries.append(row)
    if summaries:
        summaries.sort(key=lambda x: (x["at_layer"], x["ap_layer"], x["c_layer"]))
        cols = list(summaries[0])
        write_csv_atomic(output_root / "config_summary.csv", summaries, cols)
    elif (output_root / "config_summary.csv").exists():
        (output_root / "config_summary.csv").unlink()
    write_csv_atomic(output_root / "all_config_metrics.csv", detailed, list(detailed[0]) if detailed else ["at_layer", "ap_layer", "c_layer", "family", "intervention", "dataset", "metric", "value"])
    write_csv_atomic(output_root / "config_status.csv", status_rows, ["config", "status", "reason"])


def write_report(output_dir, record):
    lines = [f"CONFIG: AT=L{record['at_layer']} AP=L{record['ap_layer']} C=L{record['c_layer']}", ""]
    lines.append(f"Evaluation: changing_only={EVAL_CHANGING_ONLY}, size={EVAL_SIZE} per variable")
    for var in VARS:
        lines.append(f"Test pairs {var}: {', '.join(EVAL_PAIR_TYPES[var])}")
    lines.append("")
    for var in VARS:
        lines.append(f"=== {var} INTERVENTION ===")
        lines.append(f"clean_base={record['clean'][f'{var}_base']:.4f} clean_source={record['clean'][f'{var}_source']:.4f}")
        for metric, value in record["metrics"][var].items():
            lines.append(f"{metric}={value:.4f}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_faithfulness.txt").write_text("\n".join(lines).rstrip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--list-only", action="store_true", help="List completed/incomplete configs, no GPU needed")
    parser.add_argument("--force", action="store_true", help="Reevaluate all complete configurations")
    args = parser.parse_args()

    validate_settings()
    root = Path(args.result_dir)
    completed, incomplete = discover_configs(root)
    print(f"[scan] {len(completed)} file-complete, {len(incomplete)} incomplete C configurations", flush=True)
    for item in incomplete:
        print(f"[incomplete] {item['name']}: {item['reason']}", flush=True)

    verified = []
    invalid = []
    identity_cache = {}
    for item in completed:
        try:
            check_dependencies(item, identity_cache)
            verified.append(item)
        except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
            invalid.append({"name": item["name"], "reason": str(error)})
            print(f"[invalid] {item['name']}: {error}", flush=True)

    print(f"[verified] {len(verified)} valid complete configurations", flush=True)
    status = []
    for item in incomplete:
        status.append({"config": item["name"], "status": "incomplete", "reason": item["reason"]})
    for item in invalid:
        status.append({"config": item["name"], "status": "invalid", "reason": item["reason"]})
    if args.list_only:
        for item in verified:
            print(f"[complete] {item['name']}")
        return

    settings = eval_settings(DATA_PATH)
    profile = stable_digest(settings)[:10]
    out_root = root / f"eval_configs_gradual_{profile}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "eval_settings.json").write_text(json.dumps(settings, indent=2) + "\n")

    results = {}
    pending = []
    for item in verified:
        layers = item["layers"]
        signature = artifact_signature(item, settings)
        out_dir = out_root / item["name"]
        cached = None if args.force else read_cache(out_dir / "metrics.json", signature)
        if cached is not None:
            results[item["name"]] = cached
            status.append({"config": item["name"], "status": "cached", "reason": "signature matched"})
        else:
            pending.append((item, out_dir, signature))
    print(f"[evaluation] {len(results)} cached, {len(pending)} pending", flush=True)
    save_summaries(out_root, results, status)
    if not pending:
        print(f"Done. Results: {out_root}")
        return

    model, tokenizer = load_gemma_model()
    model.eval()
    device = next(model.parameters()).device
    dataset = load_generated_dataset(DATA_PATH)
    banks = {}
    for variable in VARS:
        banks[variable] = build_eval_bank(model, tokenizer, dataset["test"], variable)

    handle_cache = {"AT": {}, "AP": {}}
    clean_cache = {"AT": {}, "AP": {}, "C": {}}
    at_eval_cache = {}
    ap_eval_cache = {}

    def get_handle(item, variable):
        key = (variable, item["layers"][:VARS.index(variable) + 1])
        if variable != "C" and key in handle_cache[variable]:
            return handle_cache[variable][key]
        ckpt = item["paths"][variable]["checkpoint"]
        readout = item["paths"][variable]["readout"]
        Q, layer, pos = load_ckpt(ckpt, device)
        W, b, info = load_readout(readout, variable, device)
        result = (Q, layer, pos, W, b, info)
        if variable != "C":
            handle_cache[variable][key] = result
        return result

    def get_clean(item, variable, handle):
        key = item["layers"][:VARS.index(variable) + 1]
        if key in clean_cache[variable]:
            return clean_cache[variable][key]
        Q, layer, pos, W, b, info = handle
        bank = banks[variable]
        if variable == "AT":
            base, source = bank["base_answer_label_ids"], bank["source_answer_label_ids"]
        elif variable == "AP":
            base, source = bank["base_answer_pointer_ids"], bank["source_answer_pointer_ids"]
        else:
            base = color_targets(bank, info["color_to_class"], "base")
            source = color_targets(bank, info["color_to_class"], "source")
        base_acc = clean_readout_accuracy(model, bank, Q, layer, pos, W, b, base, BATCH_SIZE, "base")
        source_acc = clean_readout_accuracy(model, bank, Q, layer, pos, W, b, source, BATCH_SIZE, "source")
        clean_cache[variable][key] = (base_acc, source_acc)
        return base_acc, source_acc

    for i, (item, out_dir, signature) in enumerate(pending, 1):
        at_layer, ap_layer, c_layer = item["layers"]
        print(f"\n[EVAL {i}/{len(pending)}] {item['name']}", flush=True)
        at_handle = get_handle(item, "AT")
        ap_handle = get_handle(item, "AP")
        c_handle = get_handle(item, "C")
        Q_at, saved_at, at_pos, W_at, b_at, _ = at_handle
        Q_ap, saved_ap, ap_pos, W_ap, b_ap, _ = ap_handle
        Q_c, saved_c, c_pos, W_c, b_c, color_info = c_handle
        if (saved_at, saved_ap, saved_c) != (at_layer, ap_layer, c_layer):
            raise ValueError(f"Unexpected checkpoint layers: {item['name']}")
        if not isinstance(ap_pos, list):
            raise ValueError(f"AP token positions should be a list: {item['name']}")

        clean = {}
        for variable, handle in (("AT", at_handle), ("AP", ap_handle), ("C", c_handle)):
            base_acc, source_acc = get_clean(item, variable, handle)
            clean[f"{variable}_base"] = float(base_acc)
            clean[f"{variable}_source"] = float(source_acc)

        at_key = at_layer
        if at_key not in at_eval_cache:
            at_eval_cache[at_key] = eval_at(model, tokenizer, banks["AT"], Q_at, at_layer, at_pos, W_at, b_at, BATCH_SIZE, OUTPUT_SPACE)
        ap_key = (at_layer, ap_layer)
        if ap_key not in ap_eval_cache:
            ap_eval_cache[ap_key] = eval_ap(model, tokenizer, banks["AP"], Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, BATCH_SIZE, OUTPUT_SPACE)
        c_metrics = eval_c(model, tokenizer, banks["C"], Q_c, c_layer, c_pos, W_c, b_c, color_info["color_to_class"], Q_ap, ap_layer, ap_pos, W_ap, b_ap, Q_at, at_layer, at_pos, W_at, b_at, BATCH_SIZE, OUTPUT_SPACE)

        record = {
            "at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer,
            "metrics": {"AT": at_eval_cache[at_key], "AP": ap_eval_cache[ap_key], "C": c_metrics},
            "clean": clean,
            "signature": signature,
        }
        save_json(out_dir / "metrics.json", record)
        write_report(out_dir, record)
        results[item["name"]] = record
        status.append({"config": item["name"], "status": "evaluated", "reason": "saved"})
        save_summaries(out_root, results, status)
        print(f"[saved] {out_dir / 'metrics.json'}", flush=True)
        del c_handle, Q_c, W_c, b_c

    print(f"\nDone. Evaluated {len(results)} complete configs. Output: {out_root}")


if __name__ == "__main__":
    main()
