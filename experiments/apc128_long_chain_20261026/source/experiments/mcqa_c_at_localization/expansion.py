"""Explicit 32-to-128 proposal amendment and exact-batch clean-cache provenance."""
from pathlib import Path
import copy

from .protocol import PROTOCOL, BANKS
from .storage import read, write, digest, file_hash, source_hashes

APPROVAL = "Expand to 128 alternatives per slot"


def same_family_roots(parent, root, original):
    exclusions = []
    for candidate in sorted(root.parent.glob("mcqa_c_at_localization*")):
        registration = candidate / "registration.json"
        if registration.exists() and read(registration).get("proposals") == original["proposals"]:
            exclusions.append(candidate)
            continue
        request = candidate / "provenance/expansion_request.json"
        if not request.exists():
            continue
        value = read(request)
        if value.get("approval") != APPROVAL or value.get("parent_registration") != file_hash(parent / "registration.json") or digest(value.get("protocol")) != digest(PROTOCOL):
            continue
        files = {"parent_registration.json": "registration.json", "token_map.json": "token_map.json",
                 "approved_examples.json": "approved_examples.json", "concrete_bank_examples.json": "concrete_bank_examples.json"}
        if all((candidate / destination).exists() and file_hash(candidate / destination) == file_hash(parent / source) for destination, source in files.items()):
            exclusions.append(candidate)
    return exclusions


def assert_extension(old, new, readout=False):
    if len(old) != len(new):
        raise ValueError("Expansion changed the number of quota slots")
    for previous, expanded in zip(old, new):
        if digest({k: v for k, v in previous.items() if k != "alternatives"}) != digest({k: v for k, v in expanded.items() if k != "alternatives"}):
            raise ValueError("Expansion changed a quota slot")
        old_rows = {r["id"]: r for r in previous["alternatives"]}
        new_rows = {r["id"]: r for r in expanded["alternatives"]}
        if len(old_rows) != 32 or len(new_rows) != 128 or len(expanded["alternatives"]) != 128:
            raise ValueError("Expansion must contain exactly 32 old and 128 distinct new alternatives")
        if any(digest(new_rows.get(key)) != digest(row) for key, row in old_rows.items()):
            raise ValueError("Expansion removed or changed an original proposal")


def cache_manifest(parent, original):
    from . import backend
    from .preparation import flatten_prompts
    profile = read(parent / "engineering/profile.json")
    if original["source"]["backend.py"] != file_hash(Path(backend.__file__)):
        raise ValueError("Cannot import clean predictions from a different forward implementation")
    result = {"parent": str(parent), "parent_registration": file_hash(parent / "registration.json"),
              "environment": profile["environment"], "batch": profile["selected_batch"],
              "backend_sha256": original["source"]["backend.py"], "model_files": original["model_files"],
              "token_map_sha256": original["token_map"], "banks": {}}
    labels = dict(zip(PROTOCOL["labels"], read(parent / "token_map.json")["label_ids"]))
    # Never inspect final-test predictions before the new selections are locked.
    for bank in ("train", "calibration", "readout_fit", "readout_validation"):
        files = sorted((parent / "clean" / bank).glob("batch_*.json"))
        if not files:
            continue
        prompts = {p["id"]: p for p in flatten_prompts(read(parent / "proposals" / f"{bank}.json"), bank.startswith("readout"))}
        batches, seen = [], set()
        for path in files:
            receipt = read(path)
            identity = receipt["identity"]
            ids = identity["ids"]
            if identity["registration"] != result["parent_registration"] or identity["batch"] != result["batch"]:
                raise ValueError("Parent clean-cache identity differs")
            if not ids or len(ids) > result["batch"] or any(i in seen or i not in prompts for i in ids):
                raise ValueError("Parent clean-cache prompt coverage differs")
            if len(receipt["prediction"]) != len(ids) or receipt["accepted"] != [y == labels[prompts[i]["AT"]] for i, y in zip(ids, receipt["prediction"])]:
                raise ValueError("Parent clean-cache scoring differs")
            seen.update(ids)
            batches.append({"path": str(path.resolve()), "sha256": file_hash(path), "identity": identity})
        result["banks"][bank] = batches
    return result


def register(parent, root, approval):
    from . import banks
    from .preparation import verify_registration, flatten_prompts
    from .overlap import check
    if approval != APPROVAL:
        raise ValueError("Missing explicit approval for the 128-alternative amendment")
    original = read(parent / "registration.json")
    if digest(original["protocol"]) != digest(dict(PROTOCOL, candidate_alternatives_per_slot=32)):
        raise ValueError("Expansion parent differs beyond the approved alternative count")
    identity = {"approval": approval, "parent": str(parent), "parent_registration": file_hash(parent / "registration.json"),
                "source": source_hashes(), "protocol": PROTOCOL}
    write(root / "provenance/expansion_request.json", identity, immutable=True)
    if (root / "registration.json").exists():
        return verify_registration(root)
    for name, expected in original["proposals"].items():
        if file_hash(parent / "proposals" / f"{name}.json") != expected:
            raise ValueError("Parent proposal hash changed")
    if file_hash(parent / "token_map.json") != original["token_map"]:
        raise ValueError("Parent token map changed")
    for filename in ("registration.json", "token_map.json", "approved_examples.json", "concrete_bank_examples.json"):
        destination = root / ("parent_registration.json" if filename == "registration.json" else filename)
        write(destination, read(parent / filename), immutable=True)
    value = copy.deepcopy(original)
    value.update(protocol=copy.deepcopy(PROTOCOL), source=source_hashes(), proposals={})
    prompt_ids, proofs = set(), {}
    for bank in BANKS:
        print(f"Freezing approved 128-alternative pool: {bank}", flush=True)
        old = read(parent / "proposals" / f"{bank}.json")
        path = root / "proposals" / f"{bank}.json"
        if path.exists():
            expanded = read(path)
        elif bank.startswith("readout"):
            expanded = banks.readout_proposals(bank, 128)
        else:
            expanded = [{"slot": item["slot"], "alternatives": banks.proposal_pairs(item["slot"], original["one_color_query_mode"], 128)} for item in old]
        assert_extension(old, expanded, bank.startswith("readout"))
        write(path, expanded, immutable=True)
        value["proposals"][bank] = file_hash(path)
        proofs[bank] = {"parent_sha256": original["proposals"][bank], "new_sha256": value["proposals"][bank],
                        "slots": len(expanded), "old_alternatives": 32, "new_alternatives": 128, "all_original_records_preserved": True}
        prompt_ids.update(p["id"] for p in flatten_prompts(expanded, bank.startswith("readout")))
        del old, expanded
    exclusions = same_family_roots(parent, root, original)
    overlap = check(prompt_ids, root, exclude_roots=exclusions)
    write(root / "historical_overlap.json", overlap, immutable=True)
    value["historical_overlap"] = file_hash(root / "historical_overlap.json")
    imports = cache_manifest(parent, original)
    write(root / "provenance/clean_import.json", imports, immutable=True)
    value["proposal_expansion"] = {**identity, "banks": proofs,
                                  "clean_import_sha256": file_hash(root / "provenance/clean_import.json"),
                                  "only_protocol_change": "candidate_alternatives_per_slot: 32 -> 128",
                                  "same_experiment_overlap_exclusions": list(map(str, exclusions))}
    value["execution_amendment"] = {"parent": str(parent), "local_approval": approval,
                                     "scientific_protocol_unchanged_except_approved_pool_size": True,
                                     "original_proposals_preserved": True}
    write(root / "registration.json", value, immutable=True)
    write(root / "engineering/local_approval.json", {"approval": approval, "date": "2026-10-17",
          "long_local_run_approved": True, "prior_local_approval": "Execute our plan locally and be very thorough.",
          "parent": str(parent), "scientific_protocol_sha256": digest(PROTOCOL),
          "power_acknowledgment": "User is mindful of battery and will connect the full-power Lenovo charger",
          "no_automatic_delta_or_gradual_experiment": True}, immutable=True)
    verify_registration(root)
    return value


def clean_chunks(root, bank, prompts, batch):
    """Retain every inherited physical batch, including the partial final batch."""
    registration = read(root / "registration.json")
    amendment = registration.get("proposal_expansion")
    inherited = []
    if amendment:
        from .verification import environment
        from . import backend
        path = root / "provenance/clean_import.json"
        if file_hash(path) != amendment["clean_import_sha256"]:
            raise ValueError("Clean-import provenance changed")
        proof = read(path)
        if proof["backend_sha256"] != file_hash(Path(backend.__file__)) or proof["model_files"] != registration["model_files"] or proof["token_map_sha256"] != registration["token_map"]:
            raise ValueError("Imported clean-forward implementation/assets/targets differ")
        if digest(proof["environment"]) != digest(environment()):
            raise ValueError("Imported clean-forward environment differs")
        if bank == "test" and bank in proof["banks"]:
            raise ValueError("Final clean-cache import is forbidden")
        if proof["batch"] == batch:
            inherited = proof["banks"].get(bank, [])
    by_id, seen = {p["id"]: p for p in prompts}, set()
    for spec in inherited:
        ids = spec["identity"]["ids"]
        if any(i not in by_id or i in seen for i in ids):
            raise ValueError("Imported batch is not a unique subset of the expanded bank")
        seen.update(ids)
        yield [by_id[i] for i in ids], spec
    remaining = [p for p in prompts if p["id"] not in seen]
    for start in range(0, len(remaining), batch):
        yield remaining[start:start+batch], None


def imported_measurement(spec, ids):
    path = Path(spec["path"])
    if file_hash(path) != spec["sha256"]:
        raise ValueError("Parent clean receipt changed")
    receipt = read(path)
    if receipt["identity"] != spec["identity"] or receipt["identity"]["ids"] != ids:
        raise ValueError("Imported physical batch differs")
    measured = {k: receipt[k] for k in ("prediction", "label_logits", "label_probabilities")}
    return measured, {"path": str(path), "sha256": spec["sha256"], "parent_registration": spec["identity"]["registration"]}
