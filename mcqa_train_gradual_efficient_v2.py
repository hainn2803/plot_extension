"""Efficient, resumable, train-only Gradual DAS layer grid.

Runs AT -> AP | AT -> C | AP, AT.  Uses the exact Gradual DAS losses
from mcqa_train_grid_gradual_das.py, but does NOT repeatedly search for
full-vector layers and caches donor activations for reused layers.

The defaults train on mixed examples and calibrate on all five pair types.
The six loss coefficients can be overridden using command-line arguments.

Run on Delta:
    python3 mcqa_train_gradual_efficient.py
    python3 mcqa_train_gradual_efficient.py --result-dir results_gradual_mixed_seed0
    python3 mcqa_train_gradual_efficient.py --force   # retrain all checkpoints
    python3 mcqa_train_gradual_efficient.py --ap-lambda-med 3 --c-lambda-ap-med 2

Resume checks stage-local hyperparameters and upstream handle/readout identities.
AT is reused by layer, AP is reused by (AT, AP), C by (AT, AP, C).
Changing AP hyperparameters retrains affected AP and dependent C, but not AT.
Older checkpoints without training_signature are retrained once (fail closed).
Checkpoints/readouts are saved immediately per handle.
No 165-configuration causal-faithfulness evaluation is performed in this file.
"""

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from mcqa_gradual_das import (
    LearnedSubspace, answer_label_ids, output_iia, output_label_ids,
    output_targets, run_das_intervention, set_seed,
)
from mcqa_gradual_das_ap_cross_tokens import (
    evaluate_ap, run_ap_intervention_and_capture_at, run_frozen_at_mediator,
)
from mcqa_gradual_das_at import evaluate_at
from mcqa_gradual_das_c import (
    evaluate_c, run_c_intervention_and_capture, run_frozen_mediator,
)
from mcqa_load_randomized_ap_at_c import (
    PAIR_FIELDS, build_shared_readout_banks, create_bank,
    load_generated_dataset, predict_correct, row_to_pairs,
)
from mcqa_neural_net import load_gemma_model
from mcqa_train_grid_gradual_das import (
    balanced_counts, collect_states, fit_one_readout, load_ckpt,
    load_readout, subset_base_bank,
)

# ===========================================================================
# DATA/LAYER SETTINGS (loss coefficients also have CLI overrides).
# ===========================================================================
AT_LAYERS = [23, 24, 25]
AP_LAYERS = list(range(4, 16))
C_LAYERS = list(range(3))

# Training: any mix of pair types, including preservation examples.
ALL_PAIRS = ["AT_only", "AP_only", "C_only", "AP_AT", "C_AP_AT"]

TRAIN_PAIR_TYPES = {"AT": ALL_PAIRS, "AP": ALL_PAIRS, "C": ALL_PAIRS}

# Validation/test: all five pair types by default.
# Set CHANGING_ONLY_CAL_TEST=True to keep only output-changing pairs.
CAL_TEST_PAIR_TYPES = {
    "AT": ["AT_only"],
    "AP": ["AP_only"],
    "C": ["C_only"],
}
CHANGING_ONLY_CAL_TEST = True

# Default Gradual DAS objective coefficients (all six support CLI overrides).
AP_LAMBDA_MED = 2.0
AP_LAMBDA_READOUT = 2.0
C_LAMBDA_AP_MED = 4.0
C_LAMBDA_AP_READOUT = 4.0
C_LAMBDA_AT_MED = 0.0
C_LAMBDA_AT_READOUT = 0.0

AT_TOKEN = "last_token"
AP_TOKENS = ["color_0", "color_1", "color_2", "color_3"]
C_TOKEN = "color"
TARGETS = {"AT": "Y_after_AT_intervene", "AP": "Y_after_AP_intervene", "C": "Y_after_C_intervene"}
VARIABLES = ("AT", "AP", "C")


def validate_settings():
    for var in VARIABLES:
        for names in (TRAIN_PAIR_TYPES[var], CAL_TEST_PAIR_TYPES[var]):
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{var}: pair types must be nonempty and unique")
            for name in names:
                if name not in PAIR_FIELDS:
                    raise ValueError(f"Unknown pair type: {name}")
    for a in AT_LAYERS:
        if a < 0:
            raise ValueError("Layer must be nonnegative")
    for p in AP_LAYERS:
        if p < 0:
            raise ValueError("Layer must be nonnegative")
    for c in C_LAYERS:
        if c < 0:
            raise ValueError("Layer must be nonnegative")
    if not AT_LAYERS or not AP_LAYERS or not C_LAYERS:
        raise ValueError("Layer lists must be nonempty")


def all_configs():
    configs = []
    for at_layer in AT_LAYERS:
        for ap_layer in AP_LAYERS:
            for c_layer in C_LAYERS:
                if c_layer < ap_layer < at_layer:
                    configs.append((at_layer, ap_layer, c_layer))
    return configs


def make_banks(model, tokenizer, data_path, train_size, cal_size, test_size, batch_size, seed, with_test):
    """One pass per split; share model correctness cache across all pair types."""
    dataset = load_generated_dataset(data_path)
    sizes = {"train": train_size, "validation": cal_size}
    if with_test:
        sizes["test"] = test_size

    by_split = {}
    for split, size in sizes.items():
        allowed = TRAIN_PAIR_TYPES if split == "train" else CAL_TEST_PAIR_TYPES
        quotas = {}
        for var in VARIABLES:
            quotas[var] = balanced_counts(size, allowed[var])

        selected = {}
        for var in VARIABLES:
            selected[var] = {name: [] for name in quotas[var]}

        indices = list(range(len(dataset[split])))
        offset = {"train": 0, "validation": 1000, "test": 2000}[split]
        random.Random(seed + offset).shuffle(indices)
        prediction_cache = {}

        for start in range(0, len(indices), 64):
            candidates = []
            for index in indices[start:start + 64]:
                for pair in row_to_pairs(dataset[split][index], split):
                    name = pair["pair_type"]
                    needed = False
                    for var in VARIABLES:
                        if name not in quotas[var] or len(selected[var][name]) >= quotas[var][name]:
                            continue
                        if split != "train" and CHANGING_ONLY_CAL_TEST:
                            if pair[TARGETS[var]] == pair["base_Y"]:
                                continue
                        needed = True
                        break
                    if needed:
                        candidates.append(pair)

            good = predict_correct(model, tokenizer, candidates, batch_size, prediction_cache) if candidates else []
            for pair in good:
                name = pair["pair_type"]
                for var in VARIABLES:
                    if name not in quotas[var] or len(selected[var][name]) >= quotas[var][name]:
                        continue
                    if split != "train" and CHANGING_ONLY_CAL_TEST:
                        if pair[TARGETS[var]] == pair["base_Y"]:
                            continue
                    selected[var][name].append(pair)

            complete = True
            for var in VARIABLES:
                for name, quota in quotas[var].items():
                    if len(selected[var][name]) < quota:
                        complete = False
            if complete:
                break

        by_split[split] = {}
        for var in VARIABLES:
            pairs = []
            for name, quota in quotas[var].items():
                available = len(selected[var][name])
                if available < quota:
                    raise ValueError(
                        f"Not enough {split} changing examples for {var}/{name}: "
                        f"need {quota}, found {available}. Edit CAL_TEST_PAIR_TYPES."
                    )
                pairs.extend(selected[var][name])
                print(f"[bank] {split} {var} {name}: {quota}", flush=True)
            by_split[split][var] = create_bank(pairs, tokenizer, size=None, seed=seed + offset + {"AT": 1, "AP": 2, "C": 3}[var])

    banks = {}
    for var in VARIABLES:
        test = by_split["test"][var] if with_test else None
        banks[var] = (by_split["train"][var], by_split["validation"][var], test)
    return banks


class DonorCache:
    """Lazily cache donor features on CPU; reuse across downstream configurations."""

    def __init__(self, model, banks, batch_size, half=False):
        self.model = model
        self.banks = banks
        self.batch_size = batch_size
        self.half = half
        self.states = {}

    def get(self, var, layer, split):
        key = (var, layer, split)
        if key not in self.states:
            bank_id = {"train": 0, "validation": 1, "test": 2}[split]
            bank = self.banks[var][bank_id]
            if bank is None:
                return None
            pos = {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[var]
            data = collect_states(self.model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], layer, pos, self.batch_size)
            if self.half:
                data = data.half()
            self.states[key] = data
            print(f"[donor cached] {var} L{layer} {split}: {tuple(data.shape)}", flush=True)
        return self.states[key]

    @torch.no_grad()
    def prewarm(self, var, layers):
        """Collect MANY layers from one forward pass per input batch."""
        layers = sorted(set(layers))
        pos = {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[var]
        for split, bank_id in (("train", 0), ("validation", 1), ("test", 2)):
            bank = self.banks[var][bank_id]
            if bank is None:
                continue
            pending = [layer for layer in layers if (var, layer, split) not in self.states]
            if not pending:
                continue
            stored = {layer: [] for layer in pending}
            device = next(self.model.parameters()).device
            ids_all = bank["source_input_ids"]
            for start in range(0, len(ids_all), self.batch_size):
                end = min(start + self.batch_size, len(ids_all))
                ids = ids_all[start:end].to(device)
                mask = bank["source_attention_mask"][start:end].to(device)
                row_ids = torch.arange(len(ids), device=device)
                pad = (mask == 0).sum(dim=1)
                if isinstance(pos, list):
                    positions = []
                    for name in pos:
                        positions.append(pad + bank["source_position_by_id"][name][start:end].to(device))
                    positions = torch.stack(positions, dim=1)
                else:
                    positions = pad + bank["source_position_by_id"][pos][start:end].to(device)

                handles = []
                for layer in pending:
                    def hook(_module, _inputs, output, layer=layer):
                        hidden = output[0] if isinstance(output, tuple) else output
                        if positions.ndim == 1:
                            state = hidden[row_ids, positions, :]
                        else:
                            state = hidden[row_ids[:, None], positions, :]
                        state = state.detach().float().cpu()
                        if self.half:
                            state = state.half()
                        stored[layer].append(state)
                    handles.append(self.model.model.layers[layer].register_forward_hook(hook))
                try:
                    position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
                    self.model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
                finally:
                    for handle in handles:
                        handle.remove()

            for layer in pending:
                self.states[(var, layer, split)] = torch.cat(stored[layer], dim=0)
                print(f"[donor cached] {var} L{layer} {split}: {tuple(self.states[(var, layer, split)].shape)}", flush=True)

    def clear(self):
        self.states.clear()


def load_dependency(path, variable, device):
    Q, layer, position = load_ckpt(path, device)
    return Q, layer, position


def eval_metrics_at(model, tokenizer, bank, donors, Q, layer, batch_size, output_space):
    return (evaluate_at(model, tokenizer, bank, donors, Q, layer, AT_TOKEN, batch_size, output_space),)


def eval_metrics_ap(model, tokenizer, bank, donors, Q, ap_layer, at_layer, Q_at, W_at, b_at, batch_size, output_space):
    return evaluate_ap(model, tokenizer, bank, donors, Q, ap_layer, Q_at, at_layer, W_at, b_at, AP_TOKENS, AT_TOKEN, batch_size, output_space)


def eval_metrics_c(model, tokenizer, bank, donors, Q, c_layer, ap_layer, at_layer, Q_ap, W_ap, b_ap, Q_at, W_at, b_at, batch_size, output_space, active):
    return evaluate_c(
        model, tokenizer, bank, donors, Q, c_layer, Q_ap, ap_layer,
        AP_TOKENS, W_ap, b_ap, Q_at, at_layer, AT_TOKEN, W_at, b_at,
        C_TOKEN, batch_size, output_space, *active,
    )


def train_fixed(model, tokenizer, variable, layers, banks, donor_cache, paths, args, training_signature, hyperparameters):
    """One stage-conditional handle, same training loss and epoch score as original."""
    at_layer, ap_layer, c_layer = layers
    device = next(model.parameters()).device
    ckpt_path, readout_path = paths
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_bank, cal_bank, test_bank = banks[variable]
    layer = {"AT": at_layer, "AP": ap_layer, "C": c_layer}[variable]
    train_src = donor_cache.get(variable, layer, "train")
    cal_src = donor_cache.get(variable, layer, "validation")
    test_src = donor_cache.get(variable, layer, "test") if args.eval_test else None

    seed = args.seed
    set_seed(seed)
    model.eval()
    labels = answer_label_ids(tokenizer)
    output_ids = output_label_ids(labels, args.output_space)
    dim = {"AT": args.at_dim, "AP": args.ap_dim, "C": args.c_dim}[variable]
    epochs = {"AT": args.at_epochs, "AP": args.ap_epochs, "C": args.c_epochs}[variable]
    width = train_src.reshape(len(train_src), -1).shape[1]
    if dim > width:
        raise ValueError(f"{variable}: subspace_dim={dim} > width={width}")
    alignment = LearnedSubspace(width, dim).to(device)
    optimizer = torch.optim.Adam(alignment.parameters(), lr=args.lr)

    Q_at = W_at = b_at = Q_ap = W_ap = b_ap = None
    if variable in ("AP", "C"):
        at_ckpt = handle_path(args.root, "AT", at_layer)
        at_readout = readout_file(args.root, "AT", at_layer)
        Q_at, _, _ = load_dependency(at_ckpt, "AT", device)
        W_at, b_at, _ = load_readout(at_readout, "AT", device)
    if variable == "C":
        ap_ckpt = handle_path(args.root, "AP", at_layer, ap_layer)
        ap_readout = readout_file(args.root, "AP", at_layer, ap_layer)
        Q_ap, _, saved_pos = load_dependency(ap_ckpt, "AP", device)
        if saved_pos != AP_TOKENS:
            raise ValueError(f"AP position mismatch: {saved_pos} vs {AP_TOKENS}")
        W_ap, b_ap, _ = load_readout(ap_readout, "AP", device)

    active = (args.c_lambda_ap_med > 0, args.c_lambda_ap_readout > 0, args.c_lambda_at_med > 0, args.c_lambda_at_readout > 0)

    def evaluate(bank, donors, basis):
        if variable == "AT":
            return eval_metrics_at(model, tokenizer, bank, donors, basis, at_layer, args.batch_size, args.output_space)
        if variable == "AP":
            return eval_metrics_ap(model, tokenizer, bank, donors, basis, ap_layer, at_layer, Q_at, W_at, b_at, args.batch_size, args.output_space)
        return eval_metrics_c(model, tokenizer, bank, donors, basis, c_layer, ap_layer, at_layer, Q_ap, W_ap, b_ap, Q_at, W_at, b_at, args.batch_size, args.output_space, active)

    best_score = -1.0
    best_basis = None
    best_epoch = 0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(train_bank["base_input_ids"]), generator=torch.Generator().manual_seed(seed + epoch))
        epoch_loss = 0.0
        for start in range(0, len(perm), args.train_batch_size):
            idx = perm[start:start + args.train_batch_size]
            mini = subset_base_bank(train_bank, idx)
            basis = alignment.basis()
            optimizer.zero_grad(set_to_none=True)
            size = len(idx)
            if variable == "AT":
                logits = run_das_intervention(model, mini, train_src[idx], basis, layer, output_ids, AT_TOKEN, size, True)
                y = train_bank["intervention_targets"][TARGETS["AT"]][idx]
                loss = F.cross_entropy(logits, output_targets(y, labels, args.output_space, device))
            elif variable == "AP":
                logits, generated_at = run_ap_intervention_and_capture_at(model, mini, train_src[idx], basis, ap_layer, at_layer, output_ids, AP_TOKENS, AT_TOKEN, size, True)
                y = train_bank["intervention_targets"][TARGETS["AP"]][idx]
                target = output_targets(y, labels, args.output_space, device)
                loss = F.cross_entropy(logits, target)
                if args.ap_lambda_med > 0:
                    mediated = run_frozen_at_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, AT_TOKEN, size, True)
                    loss = loss + args.ap_lambda_med * F.cross_entropy(mediated, target)
                if args.ap_lambda_readout > 0:
                    at_target = train_bank["intervention_targets"]["AT_after_AP_intervene"][idx].to(device)
                    readout_logits = (generated_at @ Q_at) @ W_at.T + b_at
                    loss = loss + args.ap_lambda_readout * F.cross_entropy(readout_logits, at_target)
            else:
                logits, generated_ap, generated_at = run_c_intervention_and_capture(model, mini, train_src[idx], basis, c_layer, ap_layer, at_layer, output_ids, C_TOKEN, AP_TOKENS, AT_TOKEN, size, True)
                y = train_bank["intervention_targets"][TARGETS["C"]][idx]
                target = output_targets(y, labels, args.output_space, device)
                loss = F.cross_entropy(logits, target)
                if active[0]:
                    mediated_ap = run_frozen_mediator(model, mini, generated_ap, Q_ap, ap_layer, output_ids, AP_TOKENS, size, True)
                    loss = loss + args.c_lambda_ap_med * F.cross_entropy(mediated_ap, target)
                if active[1]:
                    ap_target = train_bank["intervention_targets"]["AP_after_C_intervene"][idx].to(device)
                    ap_flat = generated_ap.reshape(len(idx), -1)
                    ap_logits = (ap_flat @ Q_ap) @ W_ap.T + b_ap
                    loss = loss + args.c_lambda_ap_readout * F.cross_entropy(ap_logits, ap_target)
                if active[2]:
                    mediated_at = run_frozen_mediator(model, mini, generated_at, Q_at, at_layer, output_ids, [AT_TOKEN], size, True)
                    loss = loss + args.c_lambda_at_med * F.cross_entropy(mediated_at, target)
                if active[3]:
                    at_target = train_bank["intervention_targets"]["AT_after_C_intervene"][idx].to(device)
                    at_logits = (generated_at @ Q_at) @ W_at.T + b_at
                    loss = loss + args.c_lambda_at_readout * F.cross_entropy(at_logits, at_target)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * size

        with torch.no_grad():
            metrics = evaluate(cal_bank, cal_src, alignment.basis())
        enabled = [metrics[0]]
        if variable == "AP":
            if args.ap_lambda_med > 0:
                enabled.append(metrics[1])
            if args.ap_lambda_readout > 0:
                enabled.append(metrics[2])
        if variable == "C":
            for use, val in zip(active, metrics[1:]):
                if use:
                    enabled.append(val)
        score = min(enabled)
        print(f"[{variable} AT={at_layer} AP={ap_layer} C={c_layer} epoch={epoch:02d}] loss={epoch_loss / len(perm):.4f} cal={tuple(round(float(x), 4) for x in metrics)} score={score:.4f}", flush=True)

        if score > best_score or (variable == "C" and score == best_score):
            best_score = score
            best_basis = alignment.basis().detach().cpu().clone()
            best_epoch = epoch
            best_metrics = metrics

    test_metrics = None
    if args.eval_test:
        with torch.no_grad():
            test_metrics = evaluate(test_bank, test_src, best_basis.to(device))

    result = {
        "variable": variable, "layer": int(layer), "basis": best_basis,
        "token_position": {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[variable],
        "subspace_dim": int(dim), "best_epoch": best_epoch,
        "cal": tuple(float(x) for x in best_metrics),
        "test": tuple(float(x) for x in test_metrics) if test_metrics is not None else None,
        "cal_score": float(best_score),
        "cal_iia": float(best_metrics[0]),
        "test_iia": float(test_metrics[0]) if test_metrics is not None else None,
        "conditioning": {"AT": at_layer, "AP": ap_layer if variable != "AT" else None, "C": c_layer if variable == "C" else None},
        "training_signature": training_signature,
        "hyperparameters": hyperparameters,
        "basis_sha256": hashlib.sha256(best_basis.contiguous().numpy().tobytes()).hexdigest(),
    }
    temp = ckpt_path.with_suffix(".tmp")
    torch.save(result, temp)
    temp.replace(ckpt_path)
    print(f"[saved] {ckpt_path}", flush=True)


def handle_dir(root, variable, at_layer, ap_layer=None, c_layer=None):
    base = root / "trained_handles"
    if variable == "AT":
        return base / f"AT_L{at_layer}"
    if variable == "AP":
        return base / f"AT_L{at_layer}__AP_L{ap_layer}"
    return base / f"AT_L{at_layer}__AP_L{ap_layer}__C_L{c_layer}"


def handle_path(root, variable, at_layer, ap_layer=None, c_layer=None):
    return handle_dir(root, variable, at_layer, ap_layer, c_layer) / f"das_{variable.lower()}.pt"


def readout_file(root, variable, at_layer, ap_layer=None, c_layer=None):
    return handle_dir(root, variable, at_layer, ap_layer, c_layer) / f"{variable.lower()}_readout_LogisticRegression.pt"


SUMMARY_FIELDS = [
    "variable", "layer", "at_layer", "ap_layer", "c_layer", "cal_iia",
    "cal_score", "test_iia", "best_epoch", "cal_metrics", "test_metrics", "training_signature",
]


def load_summary(root):
    """Read the small summary once, rather than reloading every .pt after each job."""
    path = root / "training_summary.csv"
    summary = {}
    if path.is_file():
        with open(path, newline="") as file:
            for row in csv.DictReader(file):
                key = (row["variable"], row["at_layer"], row["ap_layer"], row["c_layer"])
                row.setdefault("training_signature", "legacy")
                summary[key] = row
    return summary


def update_summary(root, checkpoint, summary):
    info = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cond = info["conditioning"]
    row = {
        "variable": info["variable"], "layer": info["layer"],
        "at_layer": cond["AT"], "ap_layer": cond["AP"], "c_layer": cond["C"],
        "cal_iia": info["cal_iia"], "cal_score": info["cal_score"],
        "test_iia": info["test_iia"], "best_epoch": info["best_epoch"],
        "cal_metrics": json.dumps(info["cal"]),
        "test_metrics": json.dumps(info["test"]),
        "training_signature": info.get("training_signature", "legacy"),
    }
    key = (str(row["variable"]), str(row["at_layer"]), str(row["ap_layer"]), str(row["c_layer"]))
    summary[key] = row
    path = root / "training_summary.csv"
    temp = path.with_suffix(".tmp")
    with open(temp, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(summary.values(), key=lambda r: (str(r["variable"]), int(r["at_layer"]), -1 if r["ap_layer"] in (None, "", "None") else int(r["ap_layer"]), -1 if r["c_layer"] in (None, "", "None") else int(r["c_layer"]))))
    temp.replace(path)


# ---------------------------------------------------------------------------
# Cache validity: stage-local training hyperparameters + upstream dependencies.
# Change the per-variable version if you modify that stage's optimization logic.
# ---------------------------------------------------------------------------
TRAINER_VERSIONS = {"AT": 2, "AP": 2, "C": 2}
READOUT_VERSION = 2


def stable_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            data = file.read(1024 * 1024)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def dataset_stamp(data_path):
    """Detect dataset changes without hashing all Arrow data in full."""
    root = Path(data_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset not found: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append([str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns])
    return stable_digest({"path": str(root), "files": files})


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def checkpoint_matches(path, desired_signature, desired_hparams=None):
    if not path.is_file():
        return False, "missing checkpoint"
    try:
        info = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(info, dict) or "basis" not in info:
            return False, "invalid checkpoint"
        if "training_signature" not in info:
            return False, "legacy checkpoint has no hyperparameter metadata"
        if stable_digest(info.get("hyperparameters")) != info["training_signature"]:
            return False, "checkpoint hyperparameter metadata is inconsistent"
        basis = info["basis"].detach().cpu().contiguous()
        basis_hash = hashlib.sha256(basis.numpy().tobytes()).hexdigest()
        if info.get("basis_sha256") != basis_hash:
            return False, "checkpoint basis checksum missing or incorrect"
        if info["training_signature"] != desired_signature:
            changed = []
            if desired_hparams is not None:
                for key, value in desired_hparams.items():
                    if info["hyperparameters"].get(key) != value:
                        changed.append(key)
            return False, "changed: " + ", ".join(changed) if changed else "hyperparameters changed"
        return True, "signature matched"
    except Exception as exc:
        return False, f"cannot read checkpoint ({type(exc).__name__})"


def checkpoint_identity(path):
    """Hash only the causal handle: metadata/test metrics do not affect dependents."""
    info = torch.load(path, map_location="cpu", weights_only=False)
    basis = info["basis"].detach().cpu().contiguous()
    return {
        "training_signature": info["training_signature"],
        "basis_sha256": hashlib.sha256(basis.numpy().tobytes()).hexdigest(),
    }


def readout_meta_file(readout_path):
    return readout_path.parent / "readout_meta.json"


def readout_matches(readout_path, expected_signature):
    meta_file = readout_meta_file(readout_path)
    if not readout_path.is_file() or not meta_file.is_file():
        return False, "readout or metadata missing"
    try:
        meta = json.loads(meta_file.read_text())
        if meta.get("readout_signature") != expected_signature:
            return False, "readout hyperparameters or handle changed"
        if meta.get("file_sha256") != file_sha256(readout_path):
            return False, "readout file content changed"
        return True, "readout signature matched"
    except (OSError, ValueError) as exc:
        return False, f"unreadable readout metadata ({exc})"


def readout_identity(readout_path):
    meta = json.loads(readout_meta_file(readout_path).read_text())
    return {"readout_signature": meta["readout_signature"], "file_sha256": meta["file_sha256"]}


def stage_hyperparameters(variable, layers, args, dependencies):
    at_layer, ap_layer, c_layer = layers
    dim = {"AT": args.at_dim, "AP": args.ap_dim, "C": args.c_dim}[variable]
    epochs = {"AT": args.at_epochs, "AP": args.ap_epochs, "C": args.c_epochs}[variable]
    position = {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[variable]
    value = {
        "training_version": TRAINER_VERSIONS[variable],
        "variable": variable,
        "layer": {"AT": at_layer, "AP": ap_layer, "C": c_layer}[variable],
        "model_id": args.model_id,
        "dataset": args.dataset_identity,
        "seed": args.seed,
        "train_pair_types": list(TRAIN_PAIR_TYPES[variable]),
        "cal_pair_types": list(CAL_TEST_PAIR_TYPES[variable]),
        "changing_only_cal": CHANGING_ONLY_CAL_TEST,
        "train_size": args.ft_size,
        "cal_size": args.cal_size,
        "test_enabled": args.eval_test,
        "test_size": args.te_size if args.eval_test else None,
        "dim": dim,
        "epochs": epochs,
        "lr": args.lr,
        "train_batch_size": args.train_batch_size,
        "batch_size": args.batch_size,
        "output_space": args.output_space,
        "donor_half": args.donor_half,
        "token_position": position,
        "dependencies": dependencies,
    }
    if variable == "AP":
        value["at_layer"] = at_layer
        value["lambda_med"] = args.ap_lambda_med
        value["lambda_readout"] = args.ap_lambda_readout
    if variable == "C":
        value["at_layer"] = at_layer
        value["ap_layer"] = ap_layer
        value["lambda_ap_med"] = args.c_lambda_ap_med
        value["lambda_ap_readout"] = args.c_lambda_ap_readout
        value["lambda_at_med"] = args.c_lambda_at_med
        value["lambda_at_readout"] = args.c_lambda_at_readout
    return value


def readout_hyperparameters(variable, args, handle_id):
    return {
        "readout_version": READOUT_VERSION,
        "variable": variable,
        "model_id": args.model_id,
        "dataset": args.dataset_identity,
        "seed": args.seed,
        "readout_train_size": args.readout_train_size,
        "readout_val_size": args.readout_val_size,
        "batch_size": args.batch_size,
        "token_position": {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[variable],
        "handle": handle_id,
    }


def save_run_settings(root, args):
    """Informational only: each handle has its OWN authoritative signature."""
    settings = {
        "data_path": args.data_path, "dataset": args.dataset_identity,
        "model_id": args.model_id, "seed": args.seed,
        "sizes": [args.ft_size, args.cal_size, args.te_size],
        "train_pair_types": TRAIN_PAIR_TYPES, "cal_test_pair_types": CAL_TEST_PAIR_TYPES,
        "changing_only_cal_test": CHANGING_ONLY_CAL_TEST,
        "positions": [AT_TOKEN, AP_TOKENS, C_TOKEN],
        "lambdas": [args.ap_lambda_med, args.ap_lambda_readout, args.c_lambda_ap_med, args.c_lambda_ap_readout, args.c_lambda_at_med, args.c_lambda_at_readout],
        "dimensions": [args.at_dim, args.ap_dim, args.c_dim],
        "epochs": [args.at_epochs, args.ap_epochs, args.c_epochs],
        "lr": args.lr, "output_space": args.output_space,
        "eval_test": args.eval_test, "donor_half": args.donor_half,
        "readout_sizes": [args.readout_train_size, args.readout_val_size],
    }
    write_json_atomic(root / "training_settings.json", settings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="randomized_copycolors_include_ap_at_no_distractor")
    parser.add_argument("--result-dir", default="results_gradual_2.04.00.0_selected_layers_mixedall_seed0_no_distractor")
    parser.add_argument("--ft-size", type=int, default=800)
    parser.add_argument("--cal-size", type=int, default=400)
    parser.add_argument("--te-size", type=int, default=400)
    parser.add_argument("--at-epochs", type=int, default=10)
    parser.add_argument("--ap-epochs", type=int, default=10)
    parser.add_argument("--c-epochs", type=int, default=20)
    parser.add_argument("--at-dim", type=int, default=128)
    parser.add_argument("--ap-dim", type=int, default=128)
    parser.add_argument("--c-dim", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32, help="Donor/readout/eval batch size")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--ap-lambda-med", type=float, default=AP_LAMBDA_MED)
    parser.add_argument("--ap-lambda-readout", type=float, default=AP_LAMBDA_READOUT)
    parser.add_argument("--c-lambda-ap-med", type=float, default=C_LAMBDA_AP_MED)
    parser.add_argument("--c-lambda-ap-readout", type=float, default=C_LAMBDA_AP_READOUT)
    parser.add_argument("--c-lambda-at-med", type=float, default=C_LAMBDA_AT_MED)
    parser.add_argument("--c-lambda-at-readout", type=float, default=C_LAMBDA_AT_READOUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--readout-train-size", type=int, default=1300)
    parser.add_argument("--readout-val-size", type=int, default=520)
    parser.add_argument("--output-space", choices=("full", "az"), default="full")
    parser.add_argument("--eval-test", action="store_true", help="Compute per-stage test scores; slower")
    parser.add_argument("--donor-half", action="store_true", help="Keep donor cache in CPU float16 instead of float32")
    parser.add_argument("--force", action="store_true", help="Retrain and replace all checkpoints")
    args = parser.parse_args()
    for name in (
        "ap_lambda_med", "ap_lambda_readout",
        "c_lambda_ap_med", "c_lambda_ap_readout",
        "c_lambda_at_med", "c_lambda_at_readout",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    args.root = Path(args.result_dir)
    args.root.mkdir(parents=True, exist_ok=True)
    validate_settings()
    configs = all_configs()
    if not configs:
        raise ValueError("No valid C < AP < AT combinations")
    print(f"[grid] {len(configs)} valid configurations; AT={AT_LAYERS}, AP={AP_LAYERS}, C={C_LAYERS}", flush=True)
    set_seed(args.seed)
    model, tokenizer = load_gemma_model()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    n_layers = len(model.model.layers)
    for l in set(AT_LAYERS + AP_LAYERS + C_LAYERS):
        if not 0 <= l < n_layers:
            raise ValueError(f"Invalid layer L{l}; model has {n_layers} layers")

    args.model_id = str(getattr(model.config, "_name_or_path", getattr(model, "name_or_path", "unknown")))
    args.dataset_identity = dataset_stamp(args.data_path)
    save_run_settings(args.root, args)

    # Build tokenized data and donor activations ONLY if at least one handle
    # actually requires training. Fully cached reruns skip bank construction.
    runtime = {"banks": None, "donor": None, "shared_readout": None}
    identities = {}
    summary = load_summary(args.root)

    def ensure_donor():
        if runtime["donor"] is None:
            print("[setup] building shared training/validation banks", flush=True)
            banks = make_banks(model, tokenizer, args.data_path, args.ft_size, args.cal_size, args.te_size, args.batch_size, args.seed, args.eval_test)
            runtime["banks"] = banks
            runtime["donor"] = DonorCache(model, banks, args.batch_size, args.donor_half)
        return runtime["donor"]

    def ensure_readout_data():
        if runtime["shared_readout"] is None:
            print("[setup] building shared readout data", flush=True)
            runtime["shared_readout"] = build_shared_readout_banks(model, tokenizer, args.data_path, args.readout_train_size, args.readout_val_size, args.batch_size, args.seed)
        return runtime["shared_readout"]

    def do_stage(variable, stage_keys):
        plans = []
        pending_layers = set()
        for at_layer, ap_layer, c_layer in stage_keys:
            layers = (at_layer, ap_layer, c_layer)
            dependencies = {}
            if variable in ("AP", "C"):
                dependencies["AT"] = identities[("AT", at_layer)]
            if variable == "C":
                dependencies["AP"] = identities[("AP", at_layer, ap_layer)]

            ckpt = handle_path(args.root, variable, at_layer, ap_layer, c_layer)
            readout = readout_file(args.root, variable, at_layer, ap_layer, c_layer)
            hparams = stage_hyperparameters(variable, layers, args, dependencies)
            signature = stable_digest(hparams)
            reuse, reason = checkpoint_matches(ckpt, signature, hparams)
            if args.force:
                reuse, reason = False, "--force"
            if not reuse:
                layer = {"AT": at_layer, "AP": ap_layer, "C": c_layer}[variable]
                pending_layers.add(layer)
            plans.append((layers, ckpt, readout, hparams, signature, reuse, reason))

        if pending_layers:
            ensure_donor().prewarm(variable, sorted(pending_layers))

        for layers, ckpt, readout, hparams, signature, reuse, reason in plans:
            at_layer, ap_layer, c_layer = layers
            name = f"{variable} AT={at_layer} AP={ap_layer} C={c_layer}"
            if reuse:
                print(f"[reuse handle] {name} ({reason})", flush=True)
            else:
                print(f"[train handle] {name} ({reason})", flush=True)
                train_fixed(model, tokenizer, variable, layers, runtime["banks"], runtime["donor"], (ckpt, readout), args, signature, hparams)

            handle_id = checkpoint_identity(ckpt)
            rparams = readout_hyperparameters(variable, args, handle_id)
            rsig = stable_digest(rparams)
            reuse_readout, readout_reason = readout_matches(readout, rsig)
            if args.force or not reuse:
                reuse_readout, readout_reason = False, "handle retrained"

            if reuse_readout:
                print(f"[reuse readout] {name}", flush=True)
            else:
                print(f"[fit readout] {name} ({readout_reason})", flush=True)
                if readout.is_file():
                    readout.unlink()
                meta_file = readout_meta_file(readout)
                if meta_file.is_file():
                    meta_file.unlink()
                shared = ensure_readout_data()
                token_pos = {"AT": AT_TOKEN, "AP": AP_TOKENS, "C": C_TOKEN}[variable]
                fit_one_readout(model, ckpt, readout, variable, shared, token_pos, args.batch_size, args.seed)
                write_json_atomic(meta_file, {
                    "readout_signature": rsig,
                    "hyperparameters": rparams,
                    "file_sha256": file_sha256(readout),
                })

            own_identity = {"handle": handle_id, "readout": readout_identity(readout)}
            if variable == "AT":
                identities[("AT", at_layer)] = own_identity
            elif variable == "AP":
                identities[("AP", at_layer, ap_layer)] = own_identity
            update_summary(args.root, ckpt, summary)

        if runtime["donor"] is not None:
            runtime["donor"].clear()

    at_needed = sorted({at for at, ap, c in configs})
    ap_needed = sorted({(at, ap) for at, ap, c in configs})
    print("\n=== STAGE 1: AT ===", flush=True)
    do_stage("AT", [(at, None, None) for at in at_needed])
    print("\n=== STAGE 2: AP | AT ===", flush=True)
    do_stage("AP", [(at, ap, None) for at, ap in ap_needed])
    print("\n=== STAGE 3: C | AP, AT ===", flush=True)
    do_stage("C", configs)

    print(f"Done. Checkpoints: {args.root / 'trained_handles'}; summary: {args.root / 'training_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
