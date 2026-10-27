"""Immutable proposal registration, sealed final bank, and quota-verified filtering."""
from collections import Counter, defaultdict
from pathlib import Path
import time

from . import banks
from .protocol import PROTOCOL, BANKS, MODEL_PATH, allocate_groups, order_key
from .storage import read, write, digest, file_hash, source_hashes


def prepare(root, query_mode):
    from transformers import AutoTokenizer
    from .verification import alignment
    root = Path(root)
    if query_mode != "preserve":
        raise ValueError("The user selected color preservation for the one-color requested-object family")
    registration_path = root / "registration.json"
    identity = {"protocol": PROTOCOL, "one_color_query_mode": query_mode,
                "source": source_hashes(), "groups": allocate_groups()}
    if registration_path.exists():
        return verify_registration(root, query_mode)
    token_map = alignment(AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True))
    from .verification import asset_hashes
    from .lens import LENS_PATH
    identity["model_files"] = asset_hashes()
    identity["lens_sha256"] = file_hash(LENS_PATH)
    write(root / "token_map.json", token_map, immutable=True)
    hashes = {}
    prompt_ids = set()
    for bank in BANKS:
        print(f"Freezing proposals: {bank}", flush=True)
        rows = banks.readout_proposals(bank) if bank.startswith("readout") else banks.generate_bank(bank, query_mode)
        path = root / "proposals" / f"{bank}.json"
        write(path, rows, immutable=True)
        hashes[bank] = file_hash(path)
        prompt_ids.update(p["id"] for p in flatten_prompts(rows, bank.startswith("readout")))
    identity["proposals"] = hashes
    identity["token_map"] = file_hash(root / "token_map.json")
    from .overlap import check
    overlap = check(prompt_ids, root)
    write(root / "historical_overlap.json", overlap, immutable=True)
    identity["historical_overlap"] = file_hash(root / "historical_overlap.json")
    write(registration_path, identity, immutable=True)
    return identity


def verify_registration(root, query_mode=None):
    root = Path(root)
    registration = read(root / "registration.json")
    if digest(registration["protocol"]) != digest(PROTOCOL) or registration["source"] != source_hashes():
        raise ValueError("Source/protocol differs from frozen registration")
    if query_mode is not None and query_mode != registration["one_color_query_mode"]:
        raise ValueError("Requested-object semantics differ from registration")
    for bank, expected in registration["proposals"].items():
        if file_hash(root / "proposals" / f"{bank}.json") != expected:
            raise ValueError("Frozen proposals changed: " + bank)
    if file_hash(root / "token_map.json") != registration["token_map"]:
        raise ValueError("Token slots changed")
    if file_hash(root / "historical_overlap.json") != registration["historical_overlap"]:
        raise ValueError("Historical overlap receipt changed")
    from .lens import LENS_PATH
    if file_hash(LENS_PATH) != registration["lens_sha256"]:
        raise ValueError("Pinned J-lens artifact changed")
    return registration


def require_final_lock(root):
    root = Path(root)
    lock = read(root / "selections" / "all_locked.json")
    if lock["registration"] != file_hash(root / "registration.json"):
        raise ValueError("Final-data lock belongs to another registration")
    for filename, expected in lock["files"].items():
        if file_hash(root / filename) != expected:
            raise ValueError("Selected checkpoint or selection file changed")
    return lock


def flatten_prompts(proposals, readout=False):
    result = {}
    for item in proposals:
        for candidate in item["alternatives"]:
            rows = [candidate] if readout else [candidate["base"], candidate["source"]]
            for row in rows:
                if row["id"] in result and result[row["id"]] != row:
                    raise ValueError("Prompt ID collision")
                result[row["id"]] = row
    return [result[k] for k in sorted(result, key=order_key)]


def select_readout(proposals, clean):
    assignment, choices, rows = {}, {}, {}
    for item in proposals:
        slot = item["slot"]
        choices[slot] = sorted([p["id"] for p in item["alternatives"] if clean.get(p["id"]) is True], key=order_key)
        rows.update({p["id"]: p for p in item["alternatives"]})
    def augment(slot, visited):
        for prompt_id in choices[slot]:
            if prompt_id in visited:
                continue
            visited.add(prompt_id)
            if prompt_id not in assignment or augment(assignment[prompt_id], visited):
                assignment[prompt_id] = slot
                return True
        return False
    for slot in sorted(choices):
        if not augment(slot, set()):
            raise ValueError("Finite readout pool cannot meet clean quota: " + slot)
    selected = [rows[k] for k in sorted(assignment, key=order_key)]
    counts = Counter((r["condition"], r["C"], r["AT"]) for r in selected)
    if len(counts) != 32 or len(set(counts.values())) != 1:
        raise ValueError("Readout condition/C/AT balance failed")
    return selected


def select_localization(rows):
    """Pick twelve reciprocal pairs per cell by hashes, retaining transition coverage."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    by_cell = defaultdict(dict)
    for row in rows:
        key = tuple(sorted((row["base"]["id"], row["source"]["id"])))
        by_cell[row["family"], row["condition"]].setdefault(key, []).append(row)
    result = []
    for cell, pairs in sorted(by_cell.items()):
        groups = [pairs[k] for k in sorted(pairs, key=lambda k: order_key(str(k))) ]
        if any(len(g) != 2 for g in groups):
            raise ValueError("Localization input lacks reciprocal pairs")
        transition_keys = sorted({(v, r["base"][v], r["source"][v]) for g in groups for r in g for v in ("C", "AT")})
        matrix = np.array([[1.] * len(groups)] + [
            [float(any((v, r["base"][v], r["source"][v]) == key for r in g for v in ("C", "AT"))) for g in groups]
            for key in transition_keys])
        lower = np.array([12] + [1] * len(transition_keys))
        upper = np.array([12] + [np.inf] * len(transition_keys))
        cost = np.array([int(order_key(g[0]["slot"])[:12], 16) / 16**12 for g in groups])
        solution = milp(cost, integrality=np.ones(len(groups)), bounds=Bounds(0, 1),
                        constraints=LinearConstraint(matrix, lower, upper), options={"time_limit": 60, "mip_rel_gap": 0})
        if solution.status != 0:
            raise ValueError(f"Cannot certify fixed localization subset for {cell}: {solution.message}")
        for group, value in zip(groups, solution.x):
            if value > .5:
                result.extend(group)
    if len(result) != 384:
        raise ValueError("Localization count differs")
    return result


def filter_bank(model, tokenizer, root, bank, batch):
    import torch
    from .backend import tokenize, forward, summarize_logits
    root = Path(root)
    registration = verify_registration(root)
    if bank == "test":
        require_final_lock(root)
    proposals = read(root / "proposals" / f"{bank}.json")
    readout = bank.startswith("readout")
    prompts = flatten_prompts(proposals, readout)
    label_ids = read(root / "token_map.json")["label_ids"]
    id_for_label = dict(zip(PROTOCOL["labels"], label_ids))
    cache = root / "clean" / bank
    cache.mkdir(parents=True, exist_ok=True)
    from .expansion import clean_chunks, imported_measurement
    clean, processed, reused = {}, 0, 0
    for chunk, inherited in clean_chunks(root, bank, prompts, batch):
        start = processed
        path = cache / f"batch_{start:07d}.json"
        identity = {"registration": file_hash(root / "registration.json"),
                    "ids": [p["id"] for p in chunk], "batch": batch}
        if path.exists():
            receipt = read(path)
            if receipt["identity"] != identity:
                raise ValueError("Clean-forward cache identity differs")
        else:
            imported_from = None
            if inherited:
                measured, imported_from = imported_measurement(inherited, identity["ids"])
            else:
                with torch.no_grad():
                    ids = tokenize(tokenizer, [p["text"] for p in chunk]).to("cuda")
                    logits, _ = forward(model, ids)
                    measured = summarize_logits(logits, label_ids)
            accepted = [prediction == id_for_label[p["AT"]] for prediction, p in zip(measured["prediction"], chunk)]
            receipt = {"identity": identity, **measured, "accepted": accepted}
            if imported_from:
                receipt["imported_from"] = imported_from
            write(path, receipt, immutable=True)
        clean.update(zip(receipt["identity"]["ids"], receipt["accepted"]))
        processed += len(chunk)
        reused += len(chunk) if "imported_from" in receipt else 0
        write(cache / "progress.json", {"bank": bank, "processed_prompts": processed,
              "reused_prompts": reused, "newly_evaluated_prompts": processed-reused,
              "total_prompts": len(prompts), "clean_correct": sum(clean.values()), "time": time.time()})
        if (root / "pause.request").exists():
            from .runtime import guard
            guard()
            raise RuntimeError("Paused at clean-forward batch boundary; all completed predictions saved")
    selected = select_readout(proposals, clean) if readout else banks.select_pairs(proposals, clean)
    coverage = {"prompts": len(selected)} if readout else banks.audit_pairs(selected, bank, registration["one_color_query_mode"])
    write(root / "banks" / f"{bank}.json", selected, immutable=True)
    write(root / "coverage" / f"{bank}.json", {"coverage": coverage, "evaluated_prompts": len(clean),
          "clean_correct": sum(clean.values()), "proposal_sha256": registration["proposals"][bank],
          "bank_sha256": file_hash(root / "banks" / f"{bank}.json")}, immutable=True)
    if bank == "train":
        write(root / "banks" / "localization.json", select_localization(selected), immutable=True)
    return selected
