"""Evaluate saved Gradual DAS configurations, including induced-mediator recipient transfer.

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

Recipient experiment (controlled sensitivity donors; DIFFERENT-only recipients):
  - AP intervention -> induced AT -> recipient: AT/Y/joint
  - C intervention  -> induced AP -> recipient: AP/AT/Y/joint
  - C intervention  -> induced AT -> recipient: AT/Y/joint
  - Only DIFFERENT recipient mediator values (no same-recipient evaluation).

Original evaluation uses controlled sensitivity pairs (AT_only/AP_only/C_only),
matching mcqa_eval_selected_layers.py. Recipient donors use the same selection
from the TEST split for fair comparison against Standard DAS.

Run:
    python3 mcqa_eval_gradual_completed_recipient.py
    python3 mcqa_eval_gradual_completed_recipient.py --list-only
    python3 mcqa_eval_gradual_completed_recipient.py --force
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
    PAIR_FIELDS, LETTER_TO_CLASS, intervention_targets,
    create_bank, load_generated_dataset, predict_correct, row_to_pairs,
)
from mcqa_neural_net import load_gemma_model
from mcqa_gradual_das import answer_label_ids, output_label_ids
from mcqa_gradual_das_ap_cross_tokens import (
    run_ap_intervention_and_capture_at, run_frozen_at_mediator,
)
from mcqa_gradual_das_c import run_c_intervention_and_capture
from mcqa_train_grid_gradual_das import (
    add_metric, clean_readout_accuracy, color_targets, eval_ap, eval_at, eval_c,
    load_ckpt, load_readout, collect_states, accuracy, projected_swap_state,
    readout_prediction, output_correct,
)


# ============================================================================
# EDIT SETTINGS HERE
# ============================================================================
RESULT_DIR = "results_gradual_222222_selected_layers_controlled_seed0_no_distractor"
DATA_PATH = "randomized_copycolors_include_ap_at_no_distractor"
EVAL_SIZE = 400  # total test examples PER variable, NOT per pair type
BATCH_SIZE = 32
SEED = 0
OUTPUT_SPACE = "full"  # "full" or "az"
# Controlled sensitivity evaluation: same pair types as Standard DAS.
EVAL_PAIR_TYPES = {
    "AT": ["AT_only"],
    "AP": ["AP_only"],
    "C": ["C_only"],
}
EVAL_CHANGING_ONLY = True  # require counterfactual Y != base Y

# Optional: skip candidates whose saved v2 metadata does not match upstream
# checkpoints/readouts. Legacy checkpoints without metadata remain evaluable.
VERIFY_SAVED_DEPENDENCIES = False

TARGETS = {"AT": "Y_after_AT_intervene", "AP": "Y_after_AP_intervene", "C": "Y_after_C_intervene"}
PAIR_SEEDS = {"AT_only": 3001, "AP_only": 3002, "C_only": 3004, "AP_AT": 3005, "C_AP_AT": 3006}
VARS = ("AT", "AP", "C")
PATTERN = re.compile(r"^AT_L(\d+)__AP_L(\d+)__C_L(\d+)$")
METRICS_VERSION = 3  # v3 retains only recipient_different metrics


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



def build_recipient_donor_bank(model, tokenizer, test_split, variable):
    """Match Standard controlled sensitivity donors, independent of original eval settings.

    This matches mcqa_eval_selected_layers.py when its EVAL_PAIR_TYPES are
    AT_only/AP_only/C_only, EVAL_CHANGING_ONLY=True, size=EVAL_SIZE and seed=SEED.
    """
    pair_type = {"AT": "AT_only", "AP": "AP_only", "C": "C_only"}[variable]
    indices = list(range(len(test_split)))
    random.Random(SEED + PAIR_SEEDS[pair_type]).shuffle(indices)
    accepted = []
    correct_cache = {}
    for start in range(0, len(indices), 64):
        candidates = []
        for index in indices[start:start + 64]:
            row = test_split[int(index)]
            for pair in row_to_pairs(row, "test"):
                if pair["pair_type"] != pair_type:
                    continue
                if pair[TARGETS[variable]] == pair["base_Y"]:
                    continue
                candidates.append(pair)
        if candidates:
            correct = predict_correct(model, tokenizer, candidates, BATCH_SIZE, correct_cache)
            accepted.extend(correct)
        if len(accepted) >= EVAL_SIZE:
            break
    if len(accepted) < EVAL_SIZE:
        raise ValueError(
            f"Not enough sensitivity recipient donors for {variable}/{pair_type}: "
            f"need {EVAL_SIZE}, found {len(accepted)}"
        )
    shuffle_seed = SEED + {"AT": 100, "AP": 200, "C": 300}[variable]
    bank = create_bank(accepted[:EVAL_SIZE], tokenizer, size=None, seed=shuffle_seed)
    print(f"[recipient donors] {variable}/{pair_type} = {EVAL_SIZE}", flush=True)
    return bank


RECIPIENT_VERSION = 2
RECIPIENT_OFFSETS = {
    ("AP_only", "AT", "different"): 4102,
    ("C_only", "AP", "different"): 4202,
    ("C_only", "AT", "different"): 4302,
}


def build_recipient_pool(banks):
    """Only model-correct TEST questions, since donor_bank filters both sides.

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
    """One independent question per donor with a different AP/AT value."""
    if mediator not in ("AP", "AT") or condition != "different":
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
            if candidate[mediator] == target:
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
        for condition in ("different",):
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


@torch.no_grad()
def eval_induced_recipients(model, tokenizer, banks, recipient_banks, handles,
                            layers, batch_size, output_space, ap_induced_at=None):
    """Three induced mediator transfers; only different-recipient groups."""
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
    for condition in ("different",):
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
        "recipient_version": RECIPIENT_VERSION,
        "recipient_conditions": ["different"],
        "recipient_donors": {"AP": "AP_only", "C": "C_only", "AT": "AT_only"},
        "recipient_donors_changing_only": True,
        "recipient_size": EVAL_SIZE,
        "recipient_seed": SEED,
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
        for family in ("AP_only", "C_only"):
            for metric, value in record.get("recipient", {}).get(family, {}).items():
                row[f"{family}_{metric}"] = float(value)
                intervention = "AP" if family == "AP_only" else "C"
                add_metric(
                    detailed, at_layer, ap_layer, c_layer, "recipient",
                    intervention, family, metric, value,
                )
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
    lines.append("Recipient donors: AP_only and C_only, sensitivity-only")
    lines.append("Recipient different: natural recipient mediator != induced mediator target")
    lines.append("")
    for family in ("AP_only", "C_only"):
        lines.append(f"=== RECIPIENT TRANSFER — {family} ===")
        for metric, value in record.get("recipient", {}).get(family, {}).items():
            lines.append(f"{metric}={value:.4f}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_faithfulness.txt").write_text("\n".join(lines).rstrip() + "\n")


def write_config_csv(output_dir, record):
    """Detailed per-config CSV covering both original and recipient metrics."""
    at_layer = record["at_layer"]
    ap_layer = record["ap_layer"]
    c_layer = record["c_layer"]
    rows = []
    for variable in VARS:
        for metric, value in record["metrics"][variable].items():
            category = "correctness"
            if metric.startswith("Y_via_"):
                category = "mediator"
            elif metric.startswith("Y_restored_"):
                category = "restoration"
            add_metric(
                rows, at_layer, ap_layer, c_layer, category, variable,
                ",".join(EVAL_PAIR_TYPES[variable]), metric, value,
            )
    for family in ("AP_only", "C_only"):
        for metric, value in record.get("recipient", {}).get(family, {}).items():
            variable = "AP" if family == "AP_only" else "C"
            add_metric(
                rows, at_layer, ap_layer, c_layer, "recipient", variable,
                family, metric, value,
            )
    if rows:
        write_csv_atomic(
            output_dir / "eval_faithfulness.csv", rows, list(rows[0]),
        )


def main():
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=RESULT_DIR)
    parser.add_argument("--list-only", action="store_true", help="List completed/incomplete configs, no GPU needed")
    parser.add_argument("--force", action="store_true", help="Reevaluate all complete configurations")
    parser.add_argument("--seed", type=int, default=SEED, help="Dataset sampling/recipient seed")
    args = parser.parse_args()

    SEED = args.seed
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

    # Controlled donors and DIFFERENT-only recipient schedules.
    # The original evaluation may still use mixed/all pairs as configured above.
    recipient_donors = {}
    for variable in VARS:
        recipient_donors[variable] = build_recipient_donor_bank(
            model, tokenizer, dataset["test"], variable,
        )
    schedules = prepare_recipient_schedules(recipient_donors, SEED)
    recipient_banks = prepare_recipient_banks(tokenizer, schedules)

    handle_cache = {"AT": {}, "AP": {}}
    clean_cache = {"AT": {}, "AP": {}, "C": {}}
    at_eval_cache = {}
    ap_eval_cache = {}
    ap_induced_at_cache = {}

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

        # Induce mediator on the original controlled donor, then replay that
        # ACTUAL hidden representation into a different TEST question.
        ap_key = (at_layer, ap_layer)
        recipient_metrics, ap_induced_at = eval_induced_recipients(
            model, tokenizer, recipient_donors, recipient_banks,
            {
                "AT": (Q_at, at_pos, W_at, b_at, at_handle[-1]),
                "AP": (Q_ap, ap_pos, W_ap, b_ap, ap_handle[-1]),
                "C": (Q_c, c_pos, W_c, b_c, c_handle[-1]),
            },
            (at_layer, ap_layer, c_layer), BATCH_SIZE, OUTPUT_SPACE,
            ap_induced_at_cache.get(ap_key),
        )
        ap_induced_at_cache[ap_key] = ap_induced_at

        record = {
            "at_layer": at_layer, "ap_layer": ap_layer, "c_layer": c_layer,
            "metrics": {"AT": at_eval_cache[at_key], "AP": ap_eval_cache[ap_key], "C": c_metrics},
            "clean": clean,
            "recipient": recipient_metrics,
            "recipient_version": RECIPIENT_VERSION,
            "signature": signature,
        }
        save_json(out_dir / "metrics.json", record)
        write_report(out_dir, record)
        write_config_csv(out_dir, record)
        results[item["name"]] = record
        status.append({"config": item["name"], "status": "evaluated", "reason": "saved"})
        save_summaries(out_root, results, status)
        print(f"[saved] {out_dir / 'metrics.json'}", flush=True)
        del c_handle, Q_c, W_c, b_c

    print(f"\nDone. Evaluated {len(results)} complete configs. Output: {out_root}")


if __name__ == "__main__":
    main()
