"""Guarded final-block tail, cumulative execution ceiling and chunk receipts."""
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import descriptive as desc, run as old
from experiments.hai_c_at_evaluation import core, schema
from experiments.mcqa_c_at_localization import runtime


class Budget:
    def __init__(self, root, seconds=5400):
        self.root, self.limit = Path(root), seconds
        p = self.root / "engineering/gpu_budget.json"
        previous = old.read(p) if p.exists() else {}
        if previous.get("active"):
            raise RuntimeError("Unclosed GPU session requires accounting verification before resume")
        self.previous = previous.get("consumed_seconds", 0.)
        self.started = time.perf_counter()
        self.active = True
        self.tick()

    def elapsed(self):
        return self.previous + time.perf_counter() - self.started

    def tick(self):
        elapsed = self.elapsed()
        old.write(self.root / "engineering/gpu_budget.json", dict(limit_seconds=self.limit,
            consumed_seconds=elapsed, active=self.active, pid=os.getpid(), time=time.time(),
            accounting="Elapsed GPU-session wall time including loading, checks, I/O and cooling"), immutable=False)
        runtime.guard()
        if elapsed >= self.limit - 10:
            raise runtime.Paused("90 GPU-minute ceiling reached; progress committed")

    def close(self):
        self.active = False
        old.write(self.root / "engineering/gpu_budget.json", dict(limit_seconds=self.limit,
            consumed_seconds=self.elapsed(), active=False, pid=os.getpid(), time=time.time()), immutable=False)


def cooling(root, budget):
    """Keep both the original safety timer and the budget active while idling."""
    start = None
    while True:
        budget.tick()
        path = Path(root) / "engineering/thermal_state.json"
        if not path.exists() or not old.read(path)["active"]:
            break
        if start is None:
            start = time.time()
        time.sleep(1)
    if start is not None:
        old.write(Path(root) / "engineering/cooling" / f"{time.time_ns()}.json", dict(start=start,
            end=time.time(), guard_reset=False, guard_changed=False, pause_request_cleared=False,
            reason="Physically idled while unchanged monitor reported thermal warning"))


class Tail:
    def __init__(self, backend, root, budget):
        self.backend, self.root, self.budget = backend, Path(root), budget
        self.model, self.device = backend.model, backend.device
        self.q = backend.at_basis
        self.letters, self.label_ids = backend.letters, backend.label_ids
        self.tie_tolerance = backend.tie_tolerance
        if backend.at_layer != len(self.model.model.layers)-1:
            raise ValueError("Final-layer shortcut is invalid for an earlier AT site")

    def decode(self, token):
        return self.backend.decode(token)

    def guard(self):
        self.budget.tick()
        cooling(self.root, self.budget)
        self.budget.tick()

    @torch.no_grad()
    def state(self, base, donor=None, restore=None):
        self.guard()
        h = base.to(self.device).float()
        if donor is not None:
            h = core.patch(h, donor.to(self.device).float(), self.q).to(torch.bfloat16).float()
        if restore is not None:
            h = core.patch(h, restore.to(self.device).float(), self.q).to(torch.bfloat16).float()
        return h

    @torch.no_grad()
    def logits(self, h, head_size=None):
        h = h.to(self.device).to(torch.bfloat16)
        size = len(h) if head_size is None else head_size
        if not 0 < len(h) <= size <= 32:
            raise ValueError("Unregistered physical output-head shape")
        if len(h) < size:
            h = torch.cat((h, h[:1].expand(size-len(h), -1)), dim=0)
        hidden = self.model.model.norm(h[:, None, :])[:, 0]
        y = self.model.lm_head(hidden).float()
        cap = self.model.config.final_logit_softcapping
        if cap is not None:
            y = torch.tanh(y / cap) * cap
        return y[:len(h) if head_size is None else size]

    @torch.no_grad()
    def predict(self, base, targets, donor=None, restore=None, head_sizes=None, baseline=None):
        h = self.state(base, donor, restore)
        head_sizes = np.full(len(base), 32) if head_sizes is None else np.asarray(head_sizes)
        result = None
        for size in sorted(set(head_sizes.tolist())):
            ids = np.flatnonzero(head_sizes == size)
            if len(ids) > size:
                raise ValueError("Tail microbatch exceeds recorded physical shape")
            logits = self.logits(h[torch.as_tensor(ids, device=self.device)], int(size))[:len(ids)]
            pred = core.score(logits, [targets[i] for i in ids], self)
            if result is None:
                result = {k: np.empty((len(base), *v.shape[1:]), dtype=v.dtype) for k, v in pred.items()}
            for key, value in pred.items():
                result[key][ids] = value
        if baseline is not None:
            result["label_probability_change"] = result["label_probabilities"] - baseline
        if restore is not None:
            reference = restore.to(self.device).float()
            error = (h-reference) @ self.q
            result["coordinate_max_error"] = error.abs().amax(-1).cpu().numpy()
            result["coordinate_l2_error"] = error.norm(dim=-1).cpu().numpy()
        return result


def head_sizes(indices, total):
    indices = np.asarray(indices)
    return np.minimum(32, total - (indices // 32)*32)


def save_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def read_operation(root, name):
    folder = Path(root) / "operations" / name
    completion = old.read(folder / "completion.json")
    blocks = []
    for entry in completion["chunks"]:
        p = folder / entry["file"]
        if schema.file_hash(p) != entry["sha256"]:
            raise ValueError("Tail prediction chunk changed")
        with np.load(p, allow_pickle=False) as arrays:
            blocks.append({k: arrays[k] for k in arrays.files})
    if sum(len(p["correct"]) for p in blocks) != completion["count"]:
        raise ValueError("Missing tail prediction rows")
    return {k: np.concatenate([p[k] for p in blocks]) for k in blocks[0]}


def operation(root, name, tail, base_bank, base_idx, targets, *, donor_bank=None, donor_idx=None,
              restore_bank=None, restore_idx=None, sizes=None, baseline_probs=None, chunk=512):
    root = Path(root)
    folder = root / "operations" / name
    ids, targets = np.asarray(base_idx), np.asarray(targets)
    sizes = np.full(len(ids), 32) if sizes is None else np.asarray(sizes)
    identity = dict(name=name, plan=schema.file_hash(root / "plan.json"), count=len(ids), chunk=chunk,
        base_indices=ids.tolist(), targets=targets.tolist(), head_sizes=sizes.tolist(),
        donor_indices=None if donor_idx is None else np.asarray(donor_idx).tolist(),
        restore_indices=None if restore_idx is None else np.asarray(restore_idx).tolist(),
        base_bank=old.tensor_digest(base_bank), donor_bank=None if donor_bank is None else old.tensor_digest(donor_bank),
        restore_bank=None if restore_bank is None else old.tensor_digest(restore_bank))
    old.write(folder / "identity.json", identity)
    entries = []
    for start in range(0, len(ids), chunk):
        stop = min(start+chunk, len(ids))
        path = folder / f"chunk_{start:07d}.npz"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            record = old.read(receipt)
            if record["sha256"] != schema.file_hash(path) or record["start"] != start or record["stop"] != stop:
                raise ValueError("Committed tail chunk changed")
        else:
            blocks = []
            for a in range(start, stop, 32):
                b = min(a+32, stop)
                part = slice(a, b)
                blocks.append(tail.predict(base_bank[torch.as_tensor(ids[part])], targets[part].tolist(),
                    None if donor_bank is None else donor_bank[torch.as_tensor(np.asarray(donor_idx)[part])],
                    None if restore_bank is None else restore_bank[torch.as_tensor(np.asarray(restore_idx)[part])],
                    sizes[part], None if baseline_probs is None else baseline_probs[part]))
            save_npz(path, {k: np.concatenate([p[k] for p in blocks]) for k in blocks[0]})
            record = dict(file=path.name, sha256=schema.file_hash(path), start=start, stop=stop)
            old.write(receipt, record)
        entries.append(record)
        old.write(root / "status.json", dict(stage=name, committed=stop, total=len(ids),
            gpu_seconds=tail.budget.elapsed(), pid=os.getpid(), time=time.time()), immutable=False)
        tail.budget.tick()
    old.write(folder / "completion.json", dict(count=len(ids), identity=schema.digest(identity), chunks=entries))
    print(json.dumps(dict(operation=name, rows=len(ids), gpu_minutes=tail.budget.elapsed()/60)), flush=True)
    return read_operation(root, name)
