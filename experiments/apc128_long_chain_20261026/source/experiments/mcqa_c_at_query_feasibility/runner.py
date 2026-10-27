"""Exact finite-domain pair classification with budgeted, conditional inference."""
import argparse
from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import tarfile
import threading
import time
import traceback

from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write
from experiments.mcqa_c_at_localization.protocol import LABELS
from experiments.mcqa_c_at_localization.runtime import Paused
from experiments.mcqa_c_at_prompt_diagnostic import runner as diagnostic

ROOT = Path("eval/mcqa_c_at_query_feasibility_20261017_v1")
PARENT = diagnostic.PARENT
BATCH = 16
PROTOCOL = {
    "scope": "All original-template, training-group query-only pairs for BDCA/CDBA/DCBA; no bank adoption, DAS, or final data",
    "physical_batch": BATCH, "tokens_including_BOS": 38,
    "budget": "Original shared cumulative 900-second ledger, including previous diagnostics",
    "atol": 2e-4, "rtol": 2e-4, "cleanup_reserve_seconds": 35,
    "cache": "Preserve complete original cohort identity; import only unpadded 16-row measurements; fresh exact-cohort parity gate",
    "schedule": "One hash-first unknown prompt per unresolved pair, then the other only if needed; freeze each phase before inference",
    "pruning": "One known wrong natural answer disproves pair eligibility; never infer correctness of an unmeasured prompt",
    "filler": "Repeat fixed hash-first panel prompt to physical batch 16; filler predictions never scored or cached as measurements",
    "completion": "Every candidate pair classified; individual unmeasured partners of rejected pairs remain explicitly unmeasured",
}


def pair_ids(pair):
    return [pair[side]["id"] for side in ("base", "source")]


def status(ids, known):
    values = [known[key]["accepted"] for key in ids if key in known]
    if False in values:
        return "rejected"
    return "valid" if len(values) == 2 else "unresolved"


def required(pairs, known):
    result = []
    for pair in pairs:
        ids = pair_ids(pair)
        if status(ids, known) == "unresolved":
            result.append(min((key for key in ids if key not in known), key=digest))
    if len(result) != len(set(result)):
        raise ValueError("Expected disjoint two-prompt contexts")
    return sorted(result, key=digest)


def physical_ids(real, filler):
    if not 0 < len(real) <= BATCH or len(real) != len(set(real)):
        raise ValueError("Invalid scored batch")
    return list(real) + [filler] * (BATCH-len(real))


def source_hashes():
    return {p.as_posix(): file_hash(p) for p in sorted(Path(__file__).parent.rglob("*.py"))}


def load_inputs(root):
    reg = read(root / "candidate_registration.json")
    if reg["parent_registration"] != file_hash(PARENT / "registration.json"):
        raise ValueError("Parent changed")
    for name, field in (("pairs.json", "pairs_sha256"), ("prompts.json", "prompts_sha256")):
        if file_hash(root / name) != reg[field]:
            raise ValueError("Frozen candidate domain changed")
    if file_hash(root / "prepare_inventory.py") != reg["source_sha256"]:
        raise ValueError("Candidate generation source changed")
    inventory = read(root / "inventory.json")
    if file_hash(root / "cache_import.json") != inventory["cache_import_sha256"]:
        raise ValueError("Imported cache index changed")
    pairs = read(root / "pairs.json")
    prompts = {p["id"]: p for p in read(root / "prompts.json")}
    if len(pairs) != 27648 or len(prompts) != 55296 or len({p["id"] for p in pairs}) != 27648:
        raise ValueError("Finite domain count differs")
    flattened = [key for pair in pairs for key in pair_ids(pair)]
    if len(set(flattened)) != len(flattened) or set(flattened) != set(prompts):
        raise ValueError("Pair/domain identity differs")
    if {len(p["input_ids"]) for p in prompts.values()} != {38}:
        raise ValueError("Unexpected token shape")
    labels = read(diagnostic.ROOT / "registration.json")["label_ids"]
    imported = read(root / "cache_import.json")
    cache_files, known, excluded = {}, {}, []
    for key, record in imported.items():
        path = record["cache"]
        cache_files.setdefault(path, record["cache_sha256"])
        if cache_files[path] != record["cache_sha256"]:
            raise ValueError("Conflicting cache hashes")
        if record["registered_batch"] != BATCH or len(record["full_measured_batch_ids"]) != BATCH:
            excluded.append(key)
            continue
        if record["full_measured_batch_ids"][record["row"]] != key:
            raise ValueError("Cache row provenance differs")
        accepted = record["prediction"] == labels[prompts[key]["AT"]]
        if accepted != record["accepted"]:
            raise ValueError("Cached natural target is inconsistent")
        known[key] = record
    for path, expected in cache_files.items():
        if file_hash(path) != expected:
            raise ValueError("Cached cohort changed")
    return reg, pairs, prompts, known, excluded


def prepare(root):
    from transformers import AutoTokenizer
    from experiments.mcqa_c_at_localization.protocol import MODEL_PATH
    from experiments.mcqa_c_at_localization.preparation import flatten_prompts
    from experiments.mcqa_c_at_localization.verification import asset_hashes
    diagnostic.verify(diagnostic.ROOT)
    candidates, pairs, prompts, known, excluded = load_inputs(root)
    old = read(PARENT / "registration.json")
    if asset_hashes() != old["model_files"]:
        raise ValueError("Pinned assets differ")
    # Keep the complete old reference cohort, including rows outside this diagnostic.
    sample = known[min(known, key=digest)]
    saved = read(sample["cache"])
    old_prompts = {p["id"]: p for p in flatten_prompts(read(PARENT / "proposals/train.json"))}
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    reference = {"cache": sample["cache"], "cache_sha256": sample["cache_sha256"],
                 "ids": saved["identity"]["ids"], "predictions": saved["prediction"],
                 "label_logits": saved["label_logits"],
                 "input_ids": [tok.encode(old_prompts[key]["text"], add_special_tokens=True) for key in saved["identity"]["ids"]]}
    if len(reference["input_ids"]) != BATCH or {len(v) for v in reference["input_ids"]} != {38}:
        raise ValueError("Reference cohort physical shape mismatch")
    write(root / "reference_cohort.json", reference, immutable=True)
    first = required(pairs, known)
    write(root / "phase_1.json", first, immutable=True)
    reg = {"protocol": PROTOCOL, "candidate_registration": digest(candidates),
           "cache_import": file_hash(root / "cache_import.json"), "source": source_hashes(),
           "reference_cohort": file_hash(root / "reference_cohort.json"),
           "phase_1": file_hash(root / "phase_1.json"), "imported_count": len(known),
           "excluded_nonmatching_cohorts": excluded,
           "initial_budget": read(diagnostic.ROOT / "gpu_budget.json")}
    write(root / "execution_registration.json", reg, immutable=True)
    if not (root / "source.tar.gz").exists():
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for path in reg["source"]:
                archive.add(path, arcname=path)
            archive.add(root / "prepare_inventory.py", arcname="prepare_inventory.py")
    print({"status": "CPU_ready", "phase_1_prompts": len(first), "imported_prompts": len(known),
           "excluded_nonmatching_cohorts": len(excluded), "GPU_started": False}, flush=True)


class Safety:
    def __init__(self, root, battery_approval):
        self.root, self.battery_approval = root, battery_approval
        self.stop = threading.Event()
        self.failure, self.thread = None, None

    def start(self):
        from experiments.mcqa_c_at_localization.runtime import power_status
        power = power_status()
        if power["ACLineStatus"] != 1 and not self.battery_approval:
            raise Paused("AC power or explicit diagnostic battery approval required")
        write(self.root / "engineering/power_authorization.json", {
            "initial_power": power, "battery_approval": self.battery_approval,
            "scope": "This bounded query-feasibility diagnostic only; not the full scientific run"})
        self.thread = threading.Thread(target=self.watch, daemon=True, name="query-feasibility-safety")
        self.thread.start()
        self.guard()

    def watch(self):
        from experiments.mcqa_c_at_localization.runtime import ThermalPolicy, power_status
        from experiments.mcqa_c_at_localization.profiling import thermals
        policy = ThermalPolicy()
        try:
            while not self.stop.is_set():
                sample = thermals()
                sample["power"] = power_status()
                write(self.root / "engineering/thermal_state.json", sample)
                with (self.root / "engineering/thermal_samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, sort_keys=True)+"\n")
                if policy.observe(sample):
                    self.failure = "Thermal throttling persisted for at least 60 seconds"
                if sample["power"]["ACLineStatus"] != 1 and not self.battery_approval:
                    self.failure = "AC disconnected without diagnostic battery approval"
                self.stop.wait(15)
        except BaseException as error:
            self.failure = "Safety monitoring failed: " + str(error)

    def guard(self):
        if self.failure:
            raise Paused(self.failure)
        if (self.root / "pause.request").exists():
            raise Paused("Pause requested")

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=20)


def verify(root):
    reg = read(root / "execution_registration.json")
    if reg["source"] != source_hashes() or reg["protocol"] != PROTOCOL:
        raise ValueError("Execution source/protocol changed")
    for name, field in (("phase_1.json", "phase_1"), ("reference_cohort.json", "reference_cohort"), ("cache_import.json", "cache_import")):
        if file_hash(root / name) != reg[field]:
            raise ValueError("Execution inputs changed")
    candidates, pairs, prompts, known, excluded = load_inputs(root)
    if digest(candidates) != reg["candidate_registration"] or excluded != reg["excluded_nonmatching_cohorts"]:
        raise ValueError("Candidate/cache eligibility changed")
    if read(root / "phase_1.json") != required(pairs, known):
        raise ValueError("First phase schedule changed")
    budget, initial = read(diagnostic.ROOT / "gpu_budget.json"), reg["initial_budget"]
    if budget["spent_seconds"] < initial["spent_seconds"] or budget["sessions"][:len(initial["sessions"])] != initial["sessions"]:
        raise ValueError("Shared GPU budget was reset or changed")
    return reg, pairs, prompts, known


def validate_chunk(saved, reg, expected, filler, phase, start):
    identity = {"registration": digest(reg), "phase": phase, "start": start,
                "scored_ids": expected, "physical_ids": physical_ids(expected, filler)}
    if saved["identity"] != identity or [r["id"] for r in saved["rows"]] != expected:
        raise ValueError("Checkpoint or filler/scored-row identity changed")
    return {r["id"]: r for r in saved["rows"]}


def run(root, approval, battery_approval=None):
    if not approval:
        raise ValueError("Explicit approval is required before GPU inference")
    from experiments.mcqa_c_at_localization.local_run import single_worker
    with single_worker(root):
        return execute(root, approval, battery_approval)


def execute(root, approval, battery_approval):
    reg, pairs, prompts, known = verify(root)
    diagnostic.verify(diagnostic.ROOT)
    from experiments.mcqa_c_at_localization.verification import asset_hashes, environment
    if asset_hashes() != read(PARENT / "registration.json")["model_files"]:
        raise ValueError("Pinned model changed")
    import torch
    from experiments.mcqa_c_at_localization import backend
    torch.set_num_threads(8)
    budget = diagnostic.Budget(diagnostic.ROOT / "gpu_budget.json")
    guard = Safety(root, battery_approval)
    model = None
    outcome = "failed"
    try:
        guard.start()
        write(root / "engineering/inference_authorization.json", {"approval": approval, "battery_approval": battery_approval})
        budget.start()
        budget.check(60)
        model, tokenizer = backend.load_model()
        versions = {n: p._version for n, p in model.named_parameters()}
        write(root / "environment.json", environment(), immutable=True)
        reference = read(root / "reference_cohort.json")
        ref_ids = torch.tensor(reference["input_ids"], device="cuda", dtype=torch.long)
        labels = read(diagnostic.ROOT / "registration.json")["label_ids"]
        label_ids = [labels[letter] for letter in LABELS]
        with torch.no_grad():
            guard.guard()
            checks = diagnostic.numeric_check(model, ref_ids, "query_original_template_batch16", budget, torch, backend)
            logits, _ = backend.forward(model, ref_ids)
            delta = float((logits[:, label_ids]-torch.tensor(reference["label_logits"], device="cuda")).abs().max())
            if not torch.allclose(logits[:, label_ids], torch.tensor(reference["label_logits"], device="cuda"), atol=2e-4, rtol=2e-4):
                raise RuntimeError("Cached exact-cohort logit parity failed; tolerances unchanged")
            if logits.argmax(-1).cpu().tolist() != reference["predictions"]:
                raise RuntimeError("Cached exact-cohort predictions differ")
            write(root / "numerical.json", {"status": "passed", "native_reference": checks, "cached_logit_max_abs_difference": delta}, immutable=True)
            del logits
            torch.cuda.synchronize()
            start = time.monotonic()
            for _ in range(6):
                guard.guard()
                logits, _ = backend.forward(model, ref_ids)
                del logits
            torch.cuda.synchronize()
            seconds = (time.monotonic()-start)/6
            write(root / "profile.json", {"batch_seconds": seconds, "GPU_budget_remaining": budget.remaining(),
                                          "phase_1_upper_seconds_with_allowance": len(required(pairs, known))/BATCH*seconds*1.25})
            filler = min(prompts, key=digest)
            for phase in (1, 2):
                schedule = read(root / "phase_1.json") if phase == 1 else required(pairs, known)
                write(root / f"phase_{phase}.json", schedule, immutable=True)
                remaining_batches = sum(not (root / f"phase_{phase}" / f"{i:06d}.json").exists() for i in range(0, len(schedule), BATCH))
                budget.check(20 + remaining_batches*seconds*1.25)
                for index in range(0, len(schedule), BATCH):
                    expected = schedule[index:index+BATCH]
                    path = root / f"phase_{phase}" / f"{index:06d}.json"
                    if path.exists():
                        saved = read(path)
                    else:
                        guard.guard()
                        budget.check(20 + 4*seconds)
                        physical = physical_ids(expected, filler)
                        inputs = torch.tensor([prompts[key]["input_ids"] for key in physical], device="cuda", dtype=torch.long)
                        logits, _ = backend.forward(model, inputs)
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("Nonfinite natural-answer logits")
                        measured = backend.summarize_logits(logits, label_ids)
                        del logits
                        saved = {"identity": {"registration": digest(reg), "phase": phase, "start": index,
                                               "scored_ids": expected, "physical_ids": physical}, "rows": []}
                        for offset, key in enumerate(expected):
                            prediction = measured["prediction"][offset]
                            saved["rows"].append({"id": key, "prediction": prediction, "accepted": prediction == labels[prompts[key]["AT"]],
                                                  "label_logits": measured["label_logits"][offset],
                                                  "label_probabilities": measured["label_probabilities"][offset]})
                        write(path, saved, immutable=True)
                    known.update(validate_chunk(saved, reg, expected, filler, phase, index))
                    if index % (BATCH*64) == 0:
                        progress = {"status": "running", "phase": phase, "phase_prompts_committed": min(index+BATCH, len(schedule)),
                                    "phase_prompts_total": len(schedule), "cumulative_GPU_seconds": budget.elapsed()}
                        write(root / "status.json", progress)
                        print(progress, flush=True)
        if required(pairs, known):
            raise RuntimeError("Two-phase classification left unresolved pairs")
        backend.assert_frozen(model, versions)
        if model.lm_head._forward_hooks or any(layer._forward_hooks for layer in model.model.layers):
            raise RuntimeError("Hook cleanup failed")
        write(root / "completion.json", {"registration": digest(reg), "all_pairs_classified": True,
                                        "frozen_backbone_verified": True, "hooks_clean": True,
                                        "result_hashes": {p.relative_to(root).as_posix(): file_hash(p)
                                                          for phase in (1, 2) for p in sorted((root / f"phase_{phase}").glob("*.json"))}}, immutable=True)
        outcome = "completed"
    except BaseException as error:
        outcome = "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt, Paused)) else "failed"
        write(root / "error.json", {"status": outcome, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        guard.close()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        budget.close()
        write(root / "status.json", {"status": outcome, "cumulative_GPU_seconds": budget.state["spent_seconds"]})


def report(root):
    reg, pairs, prompts, known = verify(root)
    filler = min(prompts, key=digest)
    for phase in (1, 2):
        schedule_path = root / f"phase_{phase}.json"
        if not schedule_path.exists():
            continue
        schedule = read(schedule_path)
        for index in range(0, len(schedule), BATCH):
            path = root / f"phase_{phase}" / f"{index:06d}.json"
            if path.exists():
                known.update(validate_chunk(read(path), reg, schedule[index:index+BATCH], filler, phase, index))
    counts = defaultdict(Counter)
    examples = {}
    labels = read(diagnostic.ROOT / "registration.json")["label_ids"]
    for key, record in known.items():
        if record["accepted"] != (record["prediction"] == labels[prompts[key]["AT"]]):
            raise ValueError("Saved acceptance differs from full-vocabulary prediction")
    for pair in pairs:
        order = "".join(pair["base"]["labels"])
        verdict = status(pair_ids(pair), known)
        counts[order][verdict] += 1
        if verdict == "valid":
            examples.setdefault(order, pair)
    coverage_complete = all(v["unresolved"] == 0 for v in counts.values())
    complete = False
    if (root / "completion.json").exists():
        receipt = read(root / "completion.json")
        if not coverage_complete or receipt["registration"] != digest(reg) or read(root / "status.json")["status"] != "completed":
            raise ValueError("Completion receipt contradicts verified coverage/status")
        for name, sha in receipt["result_hashes"].items():
            if file_hash(root / name) != sha:
                raise ValueError("Completed predictions changed")
        complete = True
    result = {"complete_domain_classification": complete, "raw_pair_coverage_complete": coverage_complete,
              "label_orders": dict(counts), "valid_examples": examples,
              "measured_prompts": len(known), "unevaluated_prompts": len(prompts)-len(known),
              "meaning": "Zero valid pairs in a completed label-order domain rules out the required query-only bank for that order under this finite template/model/scoring domain, not the existence of a color representation.",
              "no_bank_adopted": True, "new_model_or_template": False}
    write(root / ("results.json" if complete else "partial_results.json"), result, immutable=complete)
    print(json.dumps({k: v for k, v in result.items() if k != "valid_examples"}, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approval-text")
    parser.add_argument("--battery-approval-text")
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.approval_text, args.battery_approval_text)
    elif args.command == "verify":
        reg, pairs, prompts, known = verify(args.root)
        print({"status": "verified", "pairs": len(pairs), "prompts": len(prompts), "cached": len(known)})
    else:
        globals()[args.command](args.root)
