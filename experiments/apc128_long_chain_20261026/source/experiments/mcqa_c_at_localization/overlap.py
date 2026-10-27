"""Exact text checks against available historical MCQA JSON banks."""
import json
from pathlib import Path

from .storage import digest, file_hash


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings(child)


def check(new_ids, new_root, history_root=Path("eval"), exclude_roots=()):
    new_root = Path(new_root).resolve()
    new_ids = set(new_ids)
    checked, matches = {}, []
    excluded = {Path(p).resolve() for p in exclude_roots}
    for run in sorted(Path(history_root).glob("mcqa*")):
        if run.resolve() == new_root or run.resolve() in excluded or not run.is_dir():
            continue
        paths = sorted({p for p in run.rglob("*.json") if "banks" in p.parts or "bank" in p.name or "proposals" in p.name})
        for path in paths:
            data = path.read_bytes()
            checked[path.as_posix()] = {"sha256": file_hash(path), "bytes": len(data)}
            if b". The " not in data:
                continue
            value = json.loads(data)
            for text in strings(value):
                if digest(text) in new_ids:
                    matches.append({"file": path.as_posix(), "prompt_id": digest(text)})
    if matches:
        raise ValueError("Exact historical prompt overlap: " + repr(matches[:10]))
    return {"matching_prompts": 0, "files": checked, "excluded_same_experiment_roots": sorted(map(str, excluded)), "scope": "available historical MCQA JSON banks and proposal files",
            "limitation": "Remote-only or removed historical banks require their own check before scientific filtering"}
