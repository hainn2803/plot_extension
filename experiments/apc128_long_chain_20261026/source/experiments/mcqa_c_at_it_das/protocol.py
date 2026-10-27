"""Preserve the completed localization; import only its four unsealed banks."""
from pathlib import Path
import tarfile

from experiments.mcqa_c_at_it_localization import runner as parent
from experiments.mcqa_c_at_localization.storage import read, write, file_hash, digest
from experiments.mcqa_c_at_localization.data import from_record, targets

ROOT = Path("eval/mcqa_c_at_it_das_20261018_v1")
RANKS = (8, 32, 128, 512)
PROTOCOL = {
    "model": parent.it.MODEL, "revision": parent.it.REVISION, "format": "chat_explicit",
    "scope": "All deduplicated locked C/AT sites, including J-lens; standard output-only DAS",
    "ranks": RANKS, "search_seed": 1, "repeat_seeds": (0, 2), "epochs": 20,
    "optimizer": "Adam", "learning_rate": .01, "effective_batch": 32,
    "backbone": "Frozen BF16 Gemma-2-2B-it, SDPA, disabled TF32, full-sequence head and FP32 softcap",
    "intervention": "h_base + ((h_source-h_base) @ Q) @ Q.T; FP32 QR basis",
    "loss": "Full-vocabulary counterfactual output CE; C uses source color and base options/labels; AT uses source letter",
    "selection": "Equal-family/equal-condition calibration output accuracy; smaller rank, latest epoch, earlier layer/token ties",
    "chain": "C layer < AT layer and C position <= AT position; max minimum calibration, mean, smaller total rank, deterministic order",
    "training_pairing": "Hai seed initialization and separate seeded epoch permutation, paired across comparable configurations",
    "readouts": "Independent measurement classifiers only after selection, never used to select handles",
    "final_data": "No final-test access by prepare/profile/train commands; future final filtering requires all selection/readout locks",
    "checkpoints": "Every epoch, every 25 complete updates, and at a safe pause; exact optimizer/RNG/order/progress state",
    "verification": "atol=rtol=2e-4; source and configuration hashes; matched-arithmetic real-model gradients; effective-batch accumulation",
    "execution": "Local preflight only initially; measured <=12h conservative full-sweep estimate required for automatic local launch; no Delta submission",
    "safety": "AC required; unchanged 60-second persistent thermal stop; 12h per-session safe-boundary cap; numerical failures stop",
}


def sources():
    values = parent.sources()
    values.update({p.as_posix(): file_hash(p) for p in Path(__file__).parent.rglob("*.py")})
    return values


def candidate_pool(shortlists):
    return [{"variable": variable, "layer": layer, "position": position, "rank": rank, "seed": 1}
            for variable in ("C", "AT")
            for layer, position in sorted({(s["layer"], s["position"]) for sites in shortlists[variable].values() for s in sites})
            for rank in RANKS]


def prepare(root):
    reg = parent.verify(parent.ROOT, assets=True)
    completed = read(parent.ROOT / "completion.json")
    assert completed["registration"] == digest(reg) and completed["all_2106_sites_complete"] and completed["PLOT_converged"]
    assert not completed["final_data_used"] and not completed["DAS_training_started"]
    for p, sha in completed["artifacts"].items():
        assert file_hash(parent.ROOT / p) == sha
    lock = read(parent.ROOT / "selections/shortlists.json")
    assert lock["registration"] == digest(reg)
    assert lock["rankings_sha256"] == file_hash(parent.ROOT / "localization/rankings.json")
    imported = {}
    for name in parent.banks.NAMES:
        logical = read(parent.ROOT / "banks" / (name + ".json"))["logical_rows"]
        actual = read(parent.ROOT / "banks" / (name + "_inputs.json"))
        for suffix, value in (("", logical), ("_inputs", actual)):
            path = root / "banks" / (name + suffix + ".json")
            write(path, value, immutable=True)
            imported[path.relative_to(root).as_posix()] = file_hash(path)
        if name in ("train", "calibration"):
            for row in logical:
                assert targets(from_record(row["base"]), from_record(row["source"])) == row["targets"]
    for filename in ("token_map.json", "selections/shortlists.json"):
        write(root / filename, read(parent.ROOT / filename), immutable=True)
        imported[filename] = file_hash(root / filename)
    pool = candidate_pool(lock["shortlists"])
    assert len(pool) == 28 and len({(c['variable'], c['layer'], c['position'], c['rank']) for c in pool}) == 28
    write(root / "candidate_pool.json", pool, immutable=True)
    train = read(root / "banks/train.json")
    index = {p["id"]: p for p in read(root / "banks/train_inputs.json")}
    earliest = min(c["layer"] for c in pool)
    pos = next(c["position"] for c in pool if c["layer"] == earliest)
    profile_rows = [r for r in train if index[r["base"]["id"]]["input_ids"][pos] != index[r["source"]["id"]]["input_ids"][pos]][:32]
    assert len(profile_rows) == 32
    write(root / "engineering/profile_rows.json", profile_rows, immutable=True)
    registration = {"protocol": PROTOCOL, "source": sources(), "parent": str(parent.ROOT),
                    "parent_registration": file_hash(parent.ROOT / "registration.json"),
                    "parent_completion": file_hash(parent.ROOT / "completion.json"), "imported": imported,
                    "candidate_pool": file_hash(root / "candidate_pool.json"), "profile_rows": file_hash(root / "engineering/profile_rows.json"),
                    "model_files": reg["model_files"], "lens_files": reg["lens_files"],
                    "counts": {"train": 1536, "calibration": 768, "seed1_trainings": 28, "maximum_repeats": 4},
                    "no_final_data_used": True}
    write(root / "registration.json", registration, immutable=True)
    if not (root / "source.tar.gz").exists():
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for p in registration["source"]:
                archive.add(p, arcname=p)
    return registration


def verify(root, assets=False):
    reg = read(root / "registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == sources(), "Source/config changed; resume refused"
    assert reg["parent_registration"] == file_hash(parent.ROOT / "registration.json")
    assert reg["parent_completion"] == file_hash(parent.ROOT / "completion.json")
    for p, sha in reg["imported"].items():
        assert file_hash(root / p) == sha, "Imported artifact changed: " + p
    assert reg["candidate_pool"] == file_hash(root / "candidate_pool.json")
    assert reg["profile_rows"] == file_hash(root / "engineering/profile_rows.json")
    if assets:
        assert parent.it.assets() == reg["model_files"]
        for p, sha in reg["lens_files"].items():
            assert file_hash(p) == sha
    return reg


def ids_for(root, name, rows=None, side="base"):
    import torch
    actual = read(root / "banks" / (name + "_inputs.json"))
    lookup = {p["id"]: p["input_ids"] for p in actual}
    if rows is None:
        rows = read(root / "banks" / (name + ".json"))
    prompts = [r[side] for r in rows] if name in ("train", "calibration") else rows
    values = [lookup[p["id"]] for p in prompts]
    assert len({len(v) for v in values}) == 1 and len(values[0]) == 82
    assert all(v[0] == 2 and v.count(2) == 1 for v in values)
    return torch.tensor(values, dtype=torch.long)
