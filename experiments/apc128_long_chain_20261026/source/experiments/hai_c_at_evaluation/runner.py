"""CPU-only preparation commands. Real-model loading waits for verified artifacts."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from . import audit, backends, core, schema

ROOT = Path("eval/hai_c_at_evaluation_prep_20261019_v2")
SNAPSHOT = Path(".external/hai_c_at_audit_latest_20261019")
TOY = Path("eval/toy_c_at_composition_20261019_v1")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Preserve existing artifact rather than overwrite: {path}")
        return
    with path.open("xb") as stream:
        stream.write(data)


def source_hashes():
    directory = Path(__file__).parent
    return {p.relative_to(directory).as_posix(): schema.file_hash(p)
            for p in sorted(directory.rglob("*")) if p.suffix in (".py", ".md")}


def identity(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["source"] != source_hashes():
        raise ValueError("Preparation source changed; preserve this version")
    for name, sha in manifest["audit_files"].items():
        if schema.file_hash(Path(manifest["snapshot"]) / name) != sha:
            raise ValueError("Pinned audit snapshot changed")
    return manifest


def prepare(root, snapshot):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=snapshot, text=True).strip()
    if commit != audit.COMMIT:
        raise ValueError("Audit snapshot is not the registered commit")
    result = audit.inspect(snapshot)
    write(root / "audit_checks.json", result)
    write(root / "BUNDLE_TEMPLATE.json", dict(
        description="Fill with Hai's actual artifacts; do not regenerate or refilter his pairs",
        code_commit=None, commands=None, model_id=None, model_revision=None,
        tokenizer_id=None, tokenizer_revision=None, label_token_ids=None,
        pairs=[dict(path=None, sha256=None)],
        methods={method: dict(c=dict(path=None, sha256=None), at=dict(path=None, sha256=None))
                 for method in ("standard", "gradual")},
        readouts=[], readout_note="Record whether each readout entered training/selection"))
    write(root / "manifest.json", dict(source=source_hashes(), snapshot=str(snapshot.resolve()),
        commit=commit, audit_files=result["files"],
        engineering_parent=dict(path="eval/hai_c_at_evaluation_prep_20261019_v1",
            reason="Match Hai's first-index argmax on ties for the secondary reproduction score. Primary ties still fail; original sources/results preserved."),
        environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__),
        state="Prepared; external checkpoints/pairs and real-model numerical checks pending",
        rules=dict(ce_selection="none; evaluate supplied frozen checkpoints",
            targets="C uses source color in base options; AT uses donor natural letter",
            natural_filter="stop on mismatch; never refilter imported pairs",
            scoring="exact stripped letter; ties fail; Hai NFKC-uppercase score reported separately",
            controls="all natural donor/recipient combinations; matched control states cached exactly",
            recipients="all imported contexts with exact color-masked first-line template distinct from both original templates",
            assignments="each imported C pair crossed with every natural AT donor; conflicts reported separately",
            gaps="absent preservation/letters/recipient controls reported, no generated replacements",
            bootstrap="not performed before exact pairs and independent grouping are known",
            model_loading="no pretrained model or GPU is loaded by these preparation commands")))
    return result


def validate_bundle(path, output):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    import re
    for name in ("code_commit", "model_revision", "tokenizer_revision"):
        if not isinstance(value.get(name), str) or not re.fullmatch(r"[0-9a-f]{40}", value[name]):
            raise ValueError(f"Explicit immutable {name} is required")
    for name in ("model_id", "tokenizer_id", "commands"):
        if not value.get(name):
            raise ValueError(f"Missing {name}")
    ids = value.get("label_token_ids")
    if not isinstance(ids, list) or len(ids) != 26 or len(set(ids)) != 26 or any(
            not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("26 explicit, distinct nonnegative label token IDs required")
    def imported(entry):
        location = (path.parent / entry["path"]).resolve()
        if schema.file_hash(location) != entry["sha256"]:
            raise ValueError(f"Imported file hash changed: {location}")
        return location
    rows = []
    for entry in value["pairs"]:
        location = imported(entry)
        if location.suffix == ".jsonl":
            rows.extend(json.loads(line) for line in location.read_text().splitlines() if line.strip())
        else:
            content = json.loads(location.read_text())
            rows.extend(content["pairs"] if isinstance(content, dict) else content)
    contexts, pairs = schema.import_pairs(rows)
    methods = {}
    at_hashes = []
    for method in ("standard", "gradual"):
        entries = value["methods"][method]
        info = {}
        width = None
        for variable in ("c", "at"):
            location = imported(entries[variable])
            checkpoint = backends.load_checkpoint(location, entries[variable]["sha256"])
            q = checkpoint["basis"]
            width = q.shape[0] if width is None else width
            backends.checked_basis(checkpoint, width, variable.upper())
            info[variable] = dict(layer=int(checkpoint["layer"]), rank=q.shape[1],
                token_position=checkpoint.get("token_position"), sha256=entries[variable]["sha256"])
        if info["c"]["layer"] >= info["at"]["layer"]:
            raise ValueError("C layer must precede AT layer")
        at_hashes.append(info["at"]["sha256"])
        methods[method] = info
    readouts = []
    for entry in value.get("readouts", []):
        imported(entry)
        readouts.append({key: entry.get(key) for key in ("sha256", "role", "used_in_training", "used_in_selection")})
    result = dict(bundle_sha256=schema.file_hash(path), coverage=schema.coverage(contexts, pairs),
                  methods=methods, common_at_handle=len(set(at_hashes)) == 1, readouts=readouts,
                  real_model_validation="pending; matching arithmetic, repeated forwards and thermal preflight still required")
    n, p = len(contexts), len(pairs)
    result["workload_per_method"] = dict(clean=n, primary=4*p, natural_donor_controls=n*n,
        independent_assignments=p*n, cross_recipient_before_question_exclusion=p*n,
        compact_extended_prediction_upper_bytes=(n*n + 2*p*n) * (2*26*8 + 64),
        note="Profile time/memory after import before running the exhaustive recipient/control schedule")
    write(output, result)
    return result


def verify(root):
    identity(root)
    command = [sys.executable, "-B", "-m", "pytest", str(Path(__file__).parent / "tests"),
               "-q", "-p", "no:cacheprovider", "--basetemp", str(root / "pytest_temporary")]
    result = subprocess.run(command, capture_output=True, text=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    write(root / "verification.json", dict(exit_code=result.returncode, stdout=result.stdout,
        stderr=result.stderr, command=command, source=source_hashes(), device="cpu", threads=2))
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return dict(passed=True, output=result.stdout)


def toy(root, toy_root):
    identity(root)
    start = time.monotonic()
    results = {}
    for name in ("incompatible", "compatible", "standard", "replay", "combined"):
        backend = backends.toy_saved_backend(toy_root, name)
        scores, arrays = core.evaluate(backend, backend.pairs(), extended=True, batch=2048)
        # Compare the new evaluator's primary predictions against independently
        # audited saved predictions, rather than deriving expected values from it.
        if name in ("incompatible", "compatible"):
            path = toy_root / "references" / f"{name}.npz"
        else:
            path = toy_root / "fits" / f"discovery_S01_{name}" / "predictions.npz"
        with np.load(path) as expected:
            maximum = 0.
            for operation in ("direct", "replay", "restoration", "natural_at"):
                # Primary compact records retain full-vocabulary decisions and label logits.
                error = float(np.max(np.abs(arrays[f"{operation}_label_logits"] - expected[f"{operation}_logits"])))
                maximum = max(maximum, error)
                if error > 1e-10:
                    raise AssertionError(f"New evaluator differs from saved {name}/{operation}")
        scores["saved_prediction_max_logit_error"] = maximum
        scores["comparison_artifact_sha256"] = schema.file_hash(path)
        write(root / "toy" / name / "scores.json", scores)
        destination = root / "toy" / name / "predictions.npz"
        with destination.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        results[name] = scores
        print(f"{name}: direct={scores['operations']['direct']['changing']['accuracy']:.4f}, "
              f"replay={scores['operations']['replay']['changing']['accuracy']:.4f}, "
              f"restoration={scores['operations']['restoration']['changing']['accuracy']:.4f}", flush=True)
    write(root / "toy_verification.json", dict(results=results, seconds=time.monotonic() - start,
        note="Primary bank: all 9216 pairs. Extended checks use every actual natural donor, including equal-valued duplicates; toy context identities exclude the original base/source. Language import uses exact color-masked templates with original question keys retained. No neural weights were trained."))
    return dict(passed=True, seconds=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "toy", "validate-bundle"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--toy-root", type=Path, default=TOY)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        result = prepare(args.root, args.snapshot)
    elif args.command == "verify":
        result = verify(args.root)
    elif args.command == "toy":
        result = toy(args.root, args.toy_root)
    else:
        if args.bundle is None:
            parser.error("--bundle is required")
        identity(args.root)
        result = validate_bundle(args.bundle, args.root / "bundle_validation.json")
    print(json.dumps(result, indent=2))
