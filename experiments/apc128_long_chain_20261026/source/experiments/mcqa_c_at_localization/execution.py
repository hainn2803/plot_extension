"""Bind scientific execution to the approved environment and passed numerical profile."""
from pathlib import Path
import socket

from .protocol import PROTOCOL
from .storage import read, write, file_hash, digest


def local_gates(profile, override=False):
    if profile["hours_with_25_percent_allowance"] > 12 and not override:
        raise ValueError("Local runtime exceeds the original cap without explicit renewed approval")
    if profile["sustained_thermal_gate"] != "passed":
        raise ValueError("Sustained-thermal gate does not pass")


def check(root):
    from .backend import settings
    from .verification import environment, asset_hashes
    from .preparation import verify_registration
    settings()
    root = Path(root)
    verify_registration(root)
    gate = read(root / "engineering/execution_lock.json")
    if not gate["approved"] or gate["registration_sha256"] != file_hash(root / "registration.json"):
        raise ValueError("Execution approval differs from this registered experiment")
    if gate["host"] != socket.gethostname() or digest(gate["environment"]) != digest(environment()):
        raise ValueError("Execution host/environment differs; fresh migration validation is required")
    if gate["profile_sha256"] != file_hash(root / "engineering/profile.json"):
        raise ValueError("Execution profile changed")
    profile = read(root / "engineering/profile.json")
    if gate["kind"] == "local":
        local_gates(profile, gate.get("long_local_run_approved", False))
        if gate.get("long_local_run_approved"):
            approval = read(root / "engineering/local_approval.json")
            if not approval.get("long_local_run_approved") or file_hash(root / "engineering/local_approval.json") != gate.get("approval_sha256"):
                raise ValueError("Renewed local approval changed")
    receipt = root / gate["numerical_receipt"]
    if file_hash(receipt) != gate["numerical_sha256"]:
        raise ValueError("Numerical receipt changed")
    passed = read(receipt)
    batch = gate["physical_batch"]
    if passed["status"] != "passed" or passed["batch"] != batch or not all(c["passed"] for c in passed["checks"]):
        raise ValueError("No passed numerical gate for this physical batch")
    if passed["atol"] != 2e-4 or passed["rtol"] != 2e-4:
        raise ValueError("Numerical tolerances differ")
    from . import backend
    if passed["backend_sha256"] != file_hash(Path(backend.__file__)):
        raise ValueError("Intervention implementation changed after numerical validation")
    if not any(s["batch"] == batch and s["status"] == "passed" for s in profile["batches"]):
        raise ValueError("Physical memory/batch profile does not pass")
    if asset_hashes() != read(root / "engineering/assets.json"):
        raise ValueError("Pinned model assets changed")
    return batch


def lock(root, kind, approval, allow_long_local=False):
    """Called only after the human approves the concrete full scientific schedule."""
    from .backend import settings
    from .verification import environment
    from .preparation import verify_registration
    settings()
    root = Path(root)
    verify_registration(root)
    profile = read(root / "engineering/profile.json")
    batch = profile["selected_batch"]
    receipts = sorted((root / "engineering").glob(f"numerical_batch_{batch}_attempt_*.json"))
    matching = [p for p in receipts if read(p).get("status") == "passed"]
    if not matching:
        raise ValueError("Missing passed numerical receipt")
    receipt = matching[-1]
    value = {"approved": True, "approval": approval, "kind": kind, "host": socket.gethostname(),
             "environment": environment(), "physical_batch": batch,
             "registration_sha256": file_hash(root / "registration.json"),
             "profile_sha256": file_hash(root / "engineering/profile.json"),
             "numerical_receipt": receipt.relative_to(root).as_posix(), "numerical_sha256": file_hash(receipt)}
    if kind == "local":
        local_gates(profile, allow_long_local)
        if allow_long_local:
            renewed = read(root / "engineering/local_approval.json")
            if not renewed.get("long_local_run_approved") or renewed["approval"] != approval:
                raise ValueError("Missing explicit renewed local approval")
            value.update(long_local_run_approved=True, approval_sha256=file_hash(root / "engineering/local_approval.json"))
    write(root / "engineering/execution_lock.json", value, immutable=True)
    check(root)
