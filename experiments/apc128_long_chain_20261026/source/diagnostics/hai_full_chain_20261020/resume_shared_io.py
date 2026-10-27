"""Windows shared-delete metadata reads; no change to scientific or safety rules."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import shutil
import threading
import time

from diagnostics.hai_frozen_eval_20261020 import run as old
from experiments.hai_c_at_full_chain import data, worker, measurements, report
from experiments.hai_c_at_evaluation import schema
from experiments.mcqa_c_at_localization import runtime

PARENT = Path("eval/hai_c_at_full_chain_20261020_v2")
ROOT = Path("eval/hai_c_at_full_chain_20261020_v3")
original_write = old.write
original_replace = os.replace
_locks = {}


def metadata_lock(path):
    key = os.path.abspath(os.fspath(path)).casefold()
    return _locks.setdefault(key, threading.RLock())


def _read_shared(path):
    path = Path(path)
    if os.name != "nt":
        return json.loads(path.read_text(encoding="utf-8"))
    import msvcrt
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    for attempt in range(7):
        # READ | WRITE | DELETE sharing permits atomic publication by the watchdog.
        handle = create(str(path.resolve()), 0x80000000, 7, None, 3, 0x80, None)
        if handle != ctypes.c_void_p(-1).value:
            try:
                fd = msvcrt.open_osfhandle(handle, os.O_RDONLY)
            except BaseException:
                kernel.CloseHandle(wintypes.HANDLE(handle))
                raise
            with os.fdopen(fd, "r", encoding="utf-8") as stream:
                return json.load(stream)
        error = ctypes.get_last_error()
        if error not in (5, 32, 33, 303) or attempt == 6:
            raise ctypes.WinError(error)
        runtime.guard()
        time.sleep(min(.02*2**attempt, .4))


def read_shared(path):
    # This filesystem rejects replacement of an open file even with delete sharing.
    with metadata_lock(path):
        return _read_shared(path)


def retry_write(path, value, immutable=True):
    for attempt in range(7):
        try:
            return original_write(path, value, immutable=immutable)
        except PermissionError:
            if attempt == 6:
                raise
            runtime.guard()
            time.sleep(min(.02*2**attempt, .4))


def shared_replace(source, destination, *args, **kwargs):
    target = Path(destination).resolve()
    local = target.is_relative_to(ROOT.resolve()) or target.is_relative_to(Path(".tmp/hai_full_chain_20261020_metadata_io").resolve())
    if not local:
        return original_replace(source, destination, *args, **kwargs)
    for attempt in range(7):
        try:
            with metadata_lock(destination):
                return original_replace(source, destination, *args, **kwargs)
        except PermissionError:
            if attempt == 6:
                raise
            runtime.guard()
            time.sleep(min(.02*2**attempt, .4))


def install():
    old.read, old.write = read_shared, original_write
    os.replace = shared_replace


def prepare():
    install()
    data.load(PARENT)
    budget = old.read(PARENT / "engineering/gpu_budget.json")
    status = old.read(PARENT / "status.json")
    if budget["active"] or (PARENT / "pause.request").exists():
        raise RuntimeError("No reset or resume of an active/unclosed or safety-paused worker")
    if status["stage"] != "failed" or "Permission denied" not in status["reason"]:
        raise RuntimeError("Engineering child requires the recorded metadata-read failure")
    data.prepare(ROOT)
    imported = {}
    for branch in ("operations", "states"):
        parent = PARENT / branch
        if not parent.exists():
            continue
        for receipt in sorted(parent.rglob("*.json")):
            record = old.read(receipt)
            if "sha256" not in record:
                continue
            source = receipt.with_suffix(".pt")
            if "file" in record:
                source = receipt.parent / record["file"]
            if not source.exists() or schema.file_hash(source) != record["sha256"]:
                raise ValueError("Parent committed artifact hash failed: " + str(source))
            for p in (source, receipt):
                dest = ROOT / p.relative_to(PARENT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
                imported[str(p)] = schema.file_hash(p)
        for p in sorted(parent.rglob("*.json")):
            if p.name in ("identity.json", "completion.json"):
                dest = ROOT / p.relative_to(PARENT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, dest)
                imported[str(p)] = schema.file_hash(p)
    shutil.copyfile(PARENT / "profile.json", ROOT / "profile.json")
    old.write(ROOT / "engineering/gpu_budget.json", dict(active=False, limit_seconds=5400,
        consumed_seconds=budget["consumed_seconds"], imported_parent_accounting=str(PARENT / "engineering/gpu_budget.json")))
    old.write(ROOT / "engineering/child_registration.json", dict(parent=str(PARENT), driver_sha256=schema.file_hash(__file__),
        parent_status_sha256=schema.file_hash(PARENT / "status.json"), parent_budget_sha256=schema.file_hash(PARENT / "engineering/gpu_budget.json"),
        imported=imported, cumulative_90_minute_budget=True,
        correction="Coordinate metadata reads/atomic replacements with per-file process locks; shared-delete readers and root-scoped replacement retries bounded to 1.42 seconds",
        unchanged_scientific_source=True, unchanged_safety_policy=True, no_pause_cleared=True,
        no_cached_or_stale_thermal_fallback=True, numerical_validation_repeated=True))
    from experiments.hai_c_at_full_chain.verification import verify
    verify(ROOT)


def main():
    if not (ROOT / "engineering/child_registration.json").exists():
        prepare()
    else:
        if old.read(ROOT / "engineering/child_registration.json")["driver_sha256"] != schema.file_hash(__file__):
            raise ValueError("Engineering driver changed")
        install()
    worker.run(ROOT)
    measurements.run(ROOT)
    report.run(ROOT)


if __name__ == "__main__":
    main()
