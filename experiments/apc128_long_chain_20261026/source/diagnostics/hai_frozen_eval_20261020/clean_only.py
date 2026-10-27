"""CPU-only, baseline-selected paired analysis; original bank/results remain immutable."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import descriptive as desc, run
from experiments.hai_c_at_evaluation import core, schema

ROOT = Path("eval/hai_frozen_clean_only_20261020_v1")
EXPECTED_COUNTS = dict(AP_only=368, AT_only=371, C_only=379, C_AP_AT=358)


def baseline_mask(natural_correct, base, source, matched_base_correct):
    natural_correct = np.asarray(natural_correct, dtype=bool)
    base, source = np.asarray(base), np.asarray(source)
    matched_base_correct = np.asarray(matched_base_correct, dtype=bool)
    if base.ndim != 1 or source.shape != base.shape or matched_base_correct.shape != base.shape:
        raise ValueError("Misaligned baseline arrays")
    if len(base) == 0 or base.min() < 0 or source.min() < 0 or max(base.max(), source.max()) >= len(natural_correct):
        raise ValueError("Invalid context indices")
    return natural_correct[base] & natural_correct[source] & matched_base_correct


def paired_summary(standard, gradual):
    masks = {family: standard["family"] == family for family in run.FAMILIES}
    masks.update(C_changing=standard["changing"], C_preservation=~standard["changing"])
    return {name: core.paired_comparison(standard, gradual, mask=mask, resamples=2000, seed=20261019)
            for name, mask in masks.items()}


def independent_bootstrap_check(standard, gradual, comparisons):
    """Recompute intervals with explicit cluster multiplicities, not indexed sums."""
    checked = 0
    for name, result in comparisons.items():
        mask = standard["family"] == name if name in run.FAMILIES else (
            standard["changing"] if name == "C_changing" else ~standard["changing"])
        groups, membership = np.unique(standard["cluster"][mask], return_inverse=True)
        sizes = np.bincount(membership)
        rng = np.random.default_rng(20261019)
        draws = rng.integers(0, len(groups), size=(2000, len(groups)))
        multiplicities = np.array([np.bincount(draw, minlength=len(groups)) for draw in draws])
        for endpoint in ("direct", "replay", "joint"):
            s = standard["direct_correct"] & standard["replay_correct"] if endpoint == "joint" else standard[endpoint+"_correct"]
            g = gradual["direct_correct"] & gradual["replay_correct"] if endpoint == "joint" else gradual[endpoint+"_correct"]
            delta = g[mask].astype(int)-s[mask].astype(int)
            totals = np.bincount(membership, weights=delta, minlength=len(groups))
            distribution = (multiplicities @ totals)/(multiplicities @ sizes)
            actual = result["endpoints"][endpoint]
            if not np.allclose(np.quantile(distribution, [.025, .975]), actual["percentile_95"], atol=1e-12, rtol=0):
                raise ValueError("Independent cluster-bootstrap interval mismatch")
            if actual["standard"]["correct"] != int(s[mask].sum()) or actual["gradual"]["correct"] != int(g[mask].sum()):
                raise ValueError("Endpoint counts do not match saved predictions")
            checked += 1
    return checked


def save_npz(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[k], v) for k, v in arrays.items()):
                raise ValueError("Immutable clean-only predictions differ")
    else:
        np.savez_compressed(path, **arrays)


def report(root, results):
    lines = ["# Clean-Only Frozen-Handle Comparison", "",
        "The same 1,476 baseline-selected pairs are used for both frozen methods. No new inference, retraining, checkpoint selection, replacement cases or rebatching was performed. The original 1,600-pair results remain unchanged.", "",
        "## Main Result: C-Changing Pairs", "",
        "| Endpoint | Standard | Gradual | Gain (percentage points), paired 95% interval |",
        "|---|---:|---:|---:|"]
    pooled = results["clean_only"]["C_changing"]
    for endpoint in ("direct", "replay", "joint"):
        e = pooled["endpoints"][endpoint]
        lo, hi = e["percentile_95"]
        lines.append(f"| {endpoint} | {e['standard']['accuracy']:.2%} ({e['standard']['correct']}/{pooled['cases']}) | "
            f"{e['gradual']['accuracy']:.2%} ({e['gradual']['correct']}/{pooled['cases']}) | "
            f"{100*e['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "Joint correctness requires both direct output and same-base AT replay to be correct. Preservation cases do not enter the changing endpoint.", "",
        "## Family Breakdown", "", "| Family | Retained pairs | Standard replay | Gradual replay | Standard joint | Gradual joint |",
        "|---|---:|---:|---:|---:|---:|"]
    for family in run.FAMILIES:
        row = results["clean_only"][family]
        replay, joint = row["endpoints"]["replay"], row["endpoints"]["joint"]
        lines.append(f"| {family} | {row['cases']} | {replay['standard']['accuracy']:.2%} | {replay['gradual']['accuracy']:.2%} | "
            f"{joint['standard']['accuracy']:.2%} | {joint['gradual']['accuracy']:.2%} |")
    lines += ["", "## Original Results Preserved", "",
        "| Population | C-changing pairs | Standard replay | Gradual replay | Standard joint | Gradual joint |",
        "|---|---:|---:|---:|---:|---:|"]
    for name, comparison in (("Original all-case bank", results["original_all_case"]["C_changing"]), ("Shared clean-only subset", pooled)):
        r, j = comparison["endpoints"]["replay"], comparison["endpoints"]["joint"]
        lines.append(f"| {name} | {comparison['cases']} | {r['standard']['accuracy']:.2%} | {r['gradual']['accuracy']:.2%} | "
            f"{j['standard']['accuracy']:.2%} | {j['gradual']['accuracy']:.2%} |")
    lines += ["", "## Selection and Limits", "",
        "The mask requires correct, untied full-vocabulary predictions for both cached natural prompts AND the batch-matched clean base. It is computed from baseline outputs only and saved before intervention predictions are read for this analysis. The exact mask and excluded IDs are recorded in selection.json.",
        "There are 737 changing pairs and 739 preservation pairs. Family sizes are no longer equal, so per-family results accompany the pooled estimate. No balancing, replacement, or intervention-success filtering was introduced.",
        "Both methods have 100% retained-baseline correctness in the recorded layouts. Saved predictions are sliced, never rerun in a new batch layout. This does not prove numerical stability under every other layout.",
        f"Intervals use 2,000 paired color-masked-question-template cluster resamples; the changing subset has {pooled['clusters']} clusters. These are not independent trained-model replications.",
        "This is an explicitly approved post hoc clean-only analysis of a reconstructed bank, not a fresh heldout certification. There is one uploaded checkpoint per method, checkpoint-selection rules differ, and the uploaded teaching readout is not independent.",
        "The parent clean-bank gate remains failed for the original bank. The retained-baseline check passes for this subset; full-chain certification remains unestablished. The full-chain test has NOT been started.", "",
        "All original artifacts are hash-checked before and after this analysis. Verification recomputes endpoint counts and all 18 bootstrap intervals independently."]
    content = "\n".join(lines)+"\n"
    path = root / "RESULTS.md"
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise ValueError("Immutable clean-only report differs")
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def main():
    if torch.cuda.is_initialized():
        raise RuntimeError("This analysis must not initialize CUDA")
    torch.set_num_threads(2)
    root, start = ROOT, time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    parent = desc.ROOT
    completion = run.read(parent / "completion.json")
    original_hashes = {name: schema.file_hash(parent / name) for name in completion["artifacts"]}
    if original_hashes != completion["artifacts"] or schema.file_hash(desc.__file__) != run.read(parent / "identity.json")["source"]:
        raise ValueError("Original descriptive artifacts/source changed")
    contexts, pairs = schema.import_pairs(run.read(parent / "selected_pairs.json"))
    natural = desc.load_observed(run.ROOT / "operations/natural")
    matched = desc.load_observed(parent / "operations/matched_base")
    base, source = np.array([p.base for p in pairs]), np.array([p.source for p in pairs])
    for path, ids, targets in ((run.ROOT / "operations/natural", list(range(len(contexts))), [c.answer for c in contexts]),
                              (parent / "operations/matched_base", base.tolist(), [contexts[i].answer for i in base])):
        saved = run.read(path / "identity.json")
        if saved["indices"] != ids or saved["targets"] != targets:
            raise ValueError("Baseline context alignment changed")
    mask = baseline_mask(natural.predictions["correct"], base, source, matched.predictions["correct"])
    counts = {f: sum(bool(keep) and p.family == f for keep, p in zip(mask, pairs)) for f in run.FAMILIES}
    if int(mask.sum()) != 1476 or counts != EXPECTED_COUNTS:
        raise ValueError(f"The previously specified shared subset changed: {counts}")
    selection = dict(rule="natural_correct[base] AND natural_correct[source] AND matched_clean_base_correct; full vocabulary, ties fail",
        original_pairs=len(pairs), retained=int(mask.sum()), excluded=int((~mask).sum()), families=counts,
        selected_indices=np.flatnonzero(mask).tolist(), selected_pair_ids=[p.id for p, keep in zip(pairs, mask) if keep],
        excluded_pair_ids=[p.id for p, keep in zip(pairs, mask) if not keep],
        intervention_outputs_used_for_selection=False, same_mask_for_both_methods=True)
    run.write(root / "selection.json", selection)
    run.write(root / "selected_pairs.json", [row for row, keep in zip(run.read(parent / "selected_pairs.json"), mask) if keep])
    imports = [parent / "completion.json", parent / "selected_pairs.json", run.ROOT / "operations/natural/identity.json",
        parent / "operations/matched_base/identity.json"]
    imports += sorted((run.ROOT / "operations/natural").glob("chunk_*.json"))
    imports += sorted((parent / "operations/matched_base").glob("chunk_*.json"))
    identity = dict(source_sha256=schema.file_hash(__file__), core_sha256=schema.file_hash(core.__file__),
        original_artifacts=original_hashes, baseline_receipts={str(p): schema.file_hash(p) for p in imports},
        selection_sha256=schema.file_hash(root / "selection.json"), cpu_only=True,
        bootstrap=dict(resamples=2000, seed=20261019, unit="exact color-masked first-line template"),
        authorization="User requested the identical 1476-pair clean-only comparison and preservation of original results",
        no_new_inference=True, no_rebatching=True, full_chain_not_started=True)
    run.write(root / "identity.json", identity)
    original, selected = {}, {}
    for method in ("standard", "gradual"):
        with np.load(parent / method / "predictions.npz", allow_pickle=False) as data:
            original[method] = {key: data[key] for key in data.files}
        a = original[method]
        if not np.array_equal(a["pair_id"], [p.id for p in pairs]) or not np.array_equal(a["matched_base_correct"], matched.predictions["correct"]):
            raise ValueError("Saved intervention/baseline pair alignment changed")
        if not np.array_equal(a["base"], base) or not np.array_equal(a["source"], source):
            raise ValueError("Intervention context indices changed")
        expected_target = [contexts[p.base].label_for(contexts[p.source].color) for p in pairs]
        if not np.array_equal(a["target"], expected_target):
            raise ValueError("C targets disagree with the base options")
        selected[method] = {key: value[mask] for key, value in a.items()}
        if not selected[method]["matched_base_correct"].all():
            raise ValueError("Retained baseline is not clean")
        save_npz(root / f"{method}_predictions.npz", selected[method])
    original_summary = paired_summary(original["standard"], original["gradual"])
    if original_summary != run.read(parent / "paired_comparison.json"):
        raise ValueError("Original all-case results no longer reproduce")
    clean_summary = paired_summary(selected["standard"], selected["gradual"])
    intervals = independent_bootstrap_check(selected["standard"], selected["gradual"], clean_summary)
    results = dict(clean_only=clean_summary, original_all_case=original_summary,
        retained_pairs=1476, excluded_pairs=124, conditional_retained_baseline_correct=True,
        parent_clean_gate_passed=False, full_chain_certified=False, full_chain_test_started=False)
    run.write(root / "results.json", results)
    report(root, results)
    unchanged = all(schema.file_hash(parent / name) == value for name, value in original_hashes.items())
    if not unchanged or torch.cuda.is_initialized():
        raise RuntimeError("Original artifacts changed or CUDA was initialized")
    artifacts = {p.name: schema.file_hash(p) for p in root.iterdir() if p.is_file() and p.name != "completion.json"}
    run.write(root / "completion.json", dict(complete=True, identity=schema.digest(identity), artifacts=artifacts,
        originals_unchanged=True, independent_intervals_verified=intervals, cuda_initialized=False,
        retained_pairs=1476, full_chain_test_started=False))
    print(json.dumps(dict(completed=True, seconds=time.perf_counter()-start,
        C_changing=clean_summary["C_changing"], families=counts), indent=2))


if __name__ == "__main__":
    main()
