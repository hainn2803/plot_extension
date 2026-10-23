"""Atomic artifacts, content identity, and fail-closed resume."""
import hashlib
import json
import os
from pathlib import Path


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if digest(read(path)) != digest(value):
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def source_hashes():
    here = Path(__file__).parent
    return {p.relative_to(here).as_posix(): file_hash(p) for p in sorted(here.rglob("*.py"))}


def require_identity(path, identity):
    row = read(path)
    if row["identity"] != identity:
        raise ValueError(f"Resume identity differs: {path}")
    return row


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def checked_torch(path, expected_hash=None):
    import torch
    if expected_hash is not None and file_hash(path) != expected_hash:
        raise ValueError("Checkpoint hash mismatch: " + str(path))
    return torch.load(path, map_location="cpu", weights_only=True)
