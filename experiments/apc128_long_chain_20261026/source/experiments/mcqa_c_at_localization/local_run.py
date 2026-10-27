"""Checkpointed full local pipeline using the original frozen proposal banks."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback

from .protocol import PROTOCOL
from .storage import read, write, file_hash, digest, source_hashes


def import_parent(parent, root, approval):
    parent, root = Path(parent).resolve(), Path(root).resolve()
    if parent == root:
        raise ValueError("Local execution must not overwrite its preserved parent")
    original = read(parent / "registration.json")
    if digest(original["protocol"]) != digest(PROTOCOL):
        expected = dict(PROTOCOL, candidate_alternatives_per_slot=32)
        if digest(original["protocol"]) == digest(expected) and PROTOCOL["candidate_alternatives_per_slot"] == 128:
            from .expansion import register
            return register(parent, root, approval)
        raise ValueError("Scientific protocol differs from the frozen parent")
    files = {f"proposals/{name}.json": sha for name, sha in original["proposals"].items()}
    files.update({"token_map.json": original["token_map"], "historical_overlap.json": original["historical_overlap"]})
    for name in ("approved_examples.json", "concrete_bank_examples.json", "registration.json"):
        files[name] = file_hash(parent / name)
    if "proposal_expansion" in original:
        files["provenance/clean_import.json"] = original["proposal_expansion"]["clean_import_sha256"]
    for filename, expected in files.items():
        destination_name = {"registration.json": "parent_registration.json", "token_map.json": "provenance/parent_token_map.json"}.get(filename, filename)
        source, destination = parent / filename, root / destination_name
        if file_hash(source) != expected:
            raise ValueError("Frozen parent artifact changed: " + filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if file_hash(destination) != expected:
                raise ValueError("Imported artifact changed: " + str(destination))
        else:
            shutil.copy2(source, destination)
    from transformers import AutoTokenizer
    from .protocol import MODEL_PATH
    from .answers import corrected_token_map
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    token_map = corrected_token_map(read(parent / "token_map.json"), tokenizer)
    write(root / "token_map.json", token_map, immutable=True)
    value = copy.deepcopy(original)
    value["token_map"] = file_hash(root / "token_map.json")
    value["source"] = source_hashes()
    value["execution_amendment"] = {
        "parent": str(parent), "parent_registration_sha256": files["registration.json"],
        "imported_artifacts": files, "scientific_protocol_unchanged": True,
        "proposals_regenerated": False, "local_approval": approval,
        "answer_token_correction": {"old_ids": read(parent / "token_map.json")["label_ids"],
                                    "new_ids": token_map["label_ids"], "continuation": "space followed by answer letter",
                                    "evidence": "v3 clean predictions decoded as space-prefixed correct letters; bare-token scoring falsely rejected them",
                                    "scope": "One canonical full-vocabulary answer token; no stripping, alternate-token acceptance, or restricted-label scoring"},
        "scope": "Local runtime-cap approval, continuous safeguards, crash recovery, distinct-question replay recipients, matched-batch structural baselines, and corrected answer continuation tokenization; no prompt, intervention equation, loss objective, selection, or quota change"}
    write(root / "registration.json", value, immutable=True)
    write(root / "engineering/local_approval.json", {
        "approval": approval, "date": "2026-10-17", "long_local_run_approved": True,
        "original_cap_hours": 12, "original_conservative_estimate_hours": 16.67,
        "scientific_protocol_sha256": digest(PROTOCOL), "parent": str(parent),
        "no_automatic_delta_or_gradual_experiment": True}, immutable=True)
    from .preparation import verify_registration
    verify_registration(root)
    return value


@contextmanager
def single_worker(root):
    path = Path(root) / "worker.lock"
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another local worker owns this run") from error
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def sustained_profile(model, tokenizer, root):
    import torch
    from .profiling import profile, profile_batch
    path = root / "engineering/profile.json"
    if path.exists():
        result = read(path)
        from .verification import environment
        from . import backend
        if result["environment"] != environment() or result["backend_sha256"] != file_hash(Path(backend.__file__)):
            raise ValueError("Saved local profile belongs to a different environment/backend")
        if result["sustained_thermal_gate"] == "passed":
            return result
        from .verification import numerical
        numerical(model, tokenizer, result["selected_batch"], root)
    else:
        progress = root / "engineering/profile_progress.json"
        if progress.exists():
            shutil.copy2(progress, root / "engineering" / f"profile_progress_interrupted_{time.time_ns()}.json")
        result = profile(root, model, tokenizer)
    before = root / "engineering/profile_before_sustained_check.json"
    if not before.exists():
        write(before, result, immutable=True)
    begun, wall_start, updates = time.monotonic(), time.time(), 0
    print("Beginning five-minute sustained early-layer rank-512 safety check", flush=True)
    while time.monotonic() - begun < 300:
        from .runtime import guard
        guard()
        profile_batch(model, tokenizer, result["selected_batch"])
        updates += 1
        torch.cuda.empty_cache()
    samples = [json.loads(line) for line in (root / "engineering/thermal_samples.jsonl").read_text().splitlines()]
    samples = [s for s in samples if wall_start <= s["time"] <= time.time()]
    if len(samples) < 3 or samples[-1]["time"] - samples[0]["time"] < 60:
        raise RuntimeError("Insufficient sustained thermal observations")
    if any(s["active"] or s["power"]["ACLineStatus"] == 0 for s in samples):
        raise RuntimeError("Sustained thermal/AC gate failed; scientific execution not started")
    result.update(sustained_thermal_gate="passed", launch_allowed=True,
                  sustained_check={"seconds": time.monotonic()-begun, "iterations": updates,
                                   "observations": len(samples), "max_temperature_C": max(s["temperature_C"] for s in samples),
                                   "continuous_runtime_monitor": "15-second samples; pause on AC loss or persistent thermal throttling"})
    write(root / "engineering/profile.json", result)
    return result


def run(root, parent, approval):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with single_worker(root):
        import_parent(parent, root, approval)
        if (root / "completed.json").exists():
            completed = read(root / "completed.json")
            if completed["registration"] != file_hash(root / "registration.json"):
                raise ValueError("Completed run registration changed")
            for filename, sha in completed["result_hashes"].items():
                if file_hash(root / filename) != sha:
                    raise ValueError("Completed result changed: " + filename)
            from .preparation import require_final_lock
            require_final_lock(root)
            print("Run already completed; all saved result hashes verified", flush=True)
            return completed
        return pipeline(root, approval)


def pipeline(root, approval):
    import torch
    from . import backend, runtime
    from .verification import asset_hashes, alignment, environment
    from .preparation import verify_registration, filter_bank, require_final_lock
    from .execution import lock, check
    from .localization import run as localize
    from .training import run_search
    from .evaluation import run as evaluate
    from .reporting import report
    torch.set_num_threads(8)
    backend.settings()
    registration = file_hash(root / "registration.json")
    worker_start = time.time()
    status_path = root / "worker_status.json"
    state = {"pid": os.getpid(), "host": socket.gethostname(), "registration": registration,
             "started_at": worker_start, "status": "starting", "final_bank_opened": False}
    model = tokenizer = None
    versions = None

    def stage(name, function):
        runtime.guard()
        verify_registration(root)
        path = root / "stages" / (name + ".json")
        if path.exists():
            saved = read(path)
            if saved["registration"] != registration or saved["status"] != "completed":
                raise ValueError("Stage identity differs: " + name)
            return
        begun = time.monotonic()
        state.update(status="running", stage=name, updated_at=time.time())
        write(status_path, state)
        print(f"Starting stage {name}", flush=True)
        function()
        if model is not None:
            backend.assert_frozen(model, versions)
        write(path, {"status": "completed", "registration": registration,
                     "seconds": time.monotonic()-begun, "completed_at": time.time()}, immutable=True)
        print(f"Completed stage {name} in {time.monotonic()-begun:.1f}s", flush=True)

    try:
        runtime.configure(root)
        def cpu_tests():
            result = subprocess.run([sys.executable, "-B", "-m", "pytest", "experiments/mcqa_c_at_localization/tests", "-q",
                                     "-p", "no:cacheprovider", "--basetemp", str(root / "engineering/pytest_worker")],
                                    capture_output=True, text=True, check=False)
            write(root / "engineering/cpu_tests.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError("Local CPU tests failed; inspect engineering/cpu_tests.json")
        stage("01_cpu_tests", cpu_tests)
        model, tokenizer = backend.load_model()
        versions = {name: parameter._version for name, parameter in model.named_parameters()}
        write(root / "engineering/current_environment.json", environment())
        def assets():
            write(root / "engineering/assets.json", asset_hashes(), immutable=True)
            write(root / "engineering/token_map.json", alignment(tokenizer), immutable=True)
            if digest(read(root / "engineering/token_map.json")) != digest(read(root / "token_map.json")):
                raise ValueError("Fresh token alignment differs from frozen parent")
        stage("02_assets_alignment", assets)
        stage("03_profile_and_numerics", lambda: sustained_profile(model, tokenizer, root))
        stage("04_execution_lock", lambda: lock(root, "local", approval, allow_long_local=True))
        batch = check(root)
        for i, bank in enumerate(("train", "calibration", "readout_fit", "readout_validation"), 5):
            stage(f"{i:02d}_filter_{bank}", lambda bank=bank: filter_bank(model, tokenizer, root, bank, batch))
        stage("09_localization", lambda: localize(model, tokenizer, root, batch))
        stage("10_training_and_readouts", lambda: run_search(model, tokenizer, root, batch))
        require_final_lock(root)
        state["final_bank_opened"] = True
        stage("11_filter_test", lambda: filter_bank(model, tokenizer, root, "test", batch))
        stage("12_evaluation", lambda: evaluate(model, tokenizer, root, batch))
        stage("13_report", lambda: report(root))
        require_final_lock(root)
        selections = read(root / "selections/seed1.json")
        chains = read(root / "selections/chain_seeds.json")
        chosen = {c["checkpoint"] for methods in selections["method_winners"].values() for c in methods.values()}
        chosen.update(c["checkpoint"] for pair in chains.values() if pair is not None for c in pair.values())
        direct = list((root / "evaluation/direct").glob("*.json"))
        if len(direct) != len(chosen) or not (root / "evaluation/wording_filter.json").exists():
            raise ValueError("Final evaluation deliverables incomplete")
        if any(pair is not None and not (root / "evaluation" / f"chain_seed_{seed}.json").exists() for seed, pair in chains.items()):
            raise ValueError("Chain evaluation is incomplete")
        timings = {p.stem: read(p) for p in (root / "stages").glob("*.json")}
        write(root / "runtime_accounting.json", {"stages": timings, "total_stage_seconds": sum(v["seconds"] for v in timings.values()),
              "scope": "Includes engineering, filtering, localization, training/readouts, heldout evaluation, and reporting; interrupted attempts remain separate"})
        report(root)
        state.update(status="completed", stage="completed", completed_at=time.time(), updated_at=time.time(),
                     report_sha256=file_hash(root / "STATUS_REPORT.md"),
                     runtime_sha256=file_hash(root / "runtime_accounting.json"),
                     direct_handles=len(direct), no_chain_compatible_pair=read(root / "selections/all_locked.json")["no_chain_compatible_pair"])
        results = list((root / "evaluation").rglob("*.json")) + [root / "STATUS_REPORT.md", root / "runtime_accounting.json"]
        state["result_hashes"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(results)}
        write(root / "completed.json", state, immutable=True)
    except BaseException as error:
        state.update(status="paused" if isinstance(error, runtime.Paused) else "failed", updated_at=time.time(), error=traceback.format_exc())
        write(root / "attempts" / f"worker_{time.time_ns()}.json", state, immutable=True)
        print(state["error"], flush=True)
        raise
    finally:
        write(status_path, state)
        runtime.close()
    return state
