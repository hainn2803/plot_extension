"""Paired template-cluster analysis of all locked diagnostic rows."""
from pathlib import Path

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import clean_only, run as old
from experiments.hai_c_at_evaluation import core, schema
from . import data, gpu
from .worker import cache_checked


def summary(pred, masks):
    return {name: core.measure(pred, mask) for name, mask in masks.items()}


def paired(a, b, groups, mask=None):
    mask = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    a, b, groups = np.asarray(a)[mask], np.asarray(b)[mask], np.asarray(groups)[mask]
    labels, membership = np.unique(groups, return_inverse=True)
    count = np.bincount(membership)
    rng = np.random.default_rng(20261020)
    draws = rng.integers(0, len(labels), size=(2000, len(labels)))
    change = b.astype(int)-a.astype(int)
    summed = np.bincount(membership, weights=change)
    distribution = summed[draws].sum(1)/count[draws].sum(1)
    return dict(standard=core.rate(a), gradual=core.rate(b), difference=float(change.mean()),
        percentile_95=np.quantile(distribution, [.025, .975]).tolist(), clusters=len(labels),
        resamples=2000, seed=20261020, grouping="Base color-masked question template; common donor/recipient schedule held fixed")


def run(root=data.ROOT):
    root = Path(root)
    contexts, pairs, plan, ro = data.load(root)
    if (root / "completion.json").exists():
        for name, expected in old.read(root / "completion.json")["artifacts"].items():
            if schema.file_hash(root / name) != expected:
                raise ValueError("Completed diagnostic artifact changed")
        return old.read(root / "results.json")
    if not old.read(root / "gpu_completion.json")["complete"]:
        raise ValueError("GPU diagnostic has not completed")
    readouts = old.read(root / "readouts/results.json")
    n, p = len(plan["bases"]), len(pairs)
    bases = np.asarray(plan["bases"])
    groups = np.asarray([contexts[i].question for i in bases])
    pair_groups = np.asarray([p.cluster for p in pairs])
    changing = np.asarray([contexts[p.base].color != contexts[p.source].color for p in pairs])
    letters = np.asarray(data.LETTERS)
    natural_letters = np.asarray([contexts[i].answer for i in bases])
    target_matrix = np.asarray([contexts[i].labels for i in bases])
    contents = np.asarray([contexts[i].contents for i in bases])
    color_change = contents != np.asarray([contexts[i].color for i in bases])[:, None]
    in_options = (target_matrix[:, :, None] == letters[None, None, :]).any(axis=1)
    control_masks = dict(all=np.ones(n*52, bool), changing=np.repeat(natural_letters, 52) != np.tile(np.repeat(letters, 2), n),
        preservation=np.repeat(natural_letters, 52) == np.tile(np.repeat(letters, 2), n),
        in_options=np.repeat(in_options, 2, axis=1).reshape(-1), outside_options=~np.repeat(in_options, 2, axis=1).reshape(-1))
    clean = gpu.read_operation(root, "clean_AT")
    roundtrip = gpu.read_operation(root, "clean_roundtrip")
    cross_control = gpu.read_operation(root, "cross_controls")
    pair_control = gpu.read_operation(root, "pair_controls")
    all_results = dict(scope="Frozen one-fact reconstructed-bank diagnostic; one checkpoint per method",
        coverage=plan["coverage"], controls=summary(clean, control_masks),
        clean_roundtrip=core.measure(roundtrip),
        clean_roundtrip_coordinate_max_error=float(roundtrip["coordinate_max_error"].max()),
        original_pair_donor_controls=summary(pair_control, dict(changing=np.repeat(changing, 2), preservation=np.repeat(~changing, 2))),
        cross_clean_controls=summary(cross_control, dict(all=np.ones(p*8, bool),
            recipient_preservation=np.tile(np.array([True, True, False, False]), p).repeat(2),
            recipient_changing=np.tile(np.array([False, False, True, True]), p).repeat(2))),
        methods={}, paired={}, readouts=readouts,
        original_clean_comparison=old.read(clean_only.ROOT / "results.json")["clean_only"])
    saved = {}
    allrows = n*4*52
    conflict = np.tile(np.repeat(letters, 2), n*4) != np.repeat(target_matrix.reshape(-1), 52)
    masks = dict(all=np.ones(allrows, bool), conflict=conflict, consistent=~conflict,
        C_changing=np.repeat(color_change.reshape(-1), 52), C_preservation=np.repeat(~color_change.reshape(-1), 52),
        AT_changing=np.tile(np.repeat(letters, 2), n*4) != np.repeat(natural_letters, 4*52),
        AT_preservation=np.tile(np.repeat(letters, 2), n*4) == np.repeat(natural_letters, 4*52),
        in_options=np.repeat(np.repeat(in_options[:, None, :], 4, axis=1), 2, axis=2).reshape(-1),
        outside_options=~np.repeat(np.repeat(in_options[:, None, :], 4, axis=1), 2, axis=2).reshape(-1))
    for method in data.METHODS:
        cross = gpu.read_operation(root, method+"_cross")
        fact = cache_checked(root / "states" / (method+"_factorial.pt"))
        fact_pred = {k: v.numpy() for k, v in fact["predictions"].items()}
        replay = gpu.read_operation(root, method+"_factorial_replay")
        restore = gpu.read_operation(root, method+"_factorial_restore")
        assignments = gpu.read_operation(root, method+"_assignments")
        cross_masks = dict(all=np.ones(p*4, bool), C_changing=np.repeat(changing, 4),
            C_preservation=np.repeat(~changing, 4), recipient_changing=np.tile([False, False, True, True], p),
            recipient_preservation=np.tile([True, True, False, False], p))
        color_masks = dict(all=np.ones(n*4, bool), changing=color_change.reshape(-1), preservation=~color_change.reshape(-1))
        control_success = cross_control["correct"].reshape(p*4, 2).all(1)
        pair_controls_ok = pair_control["correct"].reshape(p, 2).all(1)
        cross_all = cross["correct"].reshape(p, 4).all(1)
        errors = cache_checked(root / "states" / (method+"_restoration_errors.pt"))["coordinate_max_error"].numpy()
        result = dict(cross=summary(cross, cross_masks), direct=summary(fact_pred, color_masks),
            factorial_replay=summary(replay, color_masks), factorial_restoration=summary(restore, color_masks),
            factorial_joint={k: core.rate((fact_pred["correct"] & replay["correct"])[v]) for k, v in color_masks.items()},
            assignments=summary(assignments, masks), restoration_coordinate_max_error=float(errors.max()),
            cross_failures_despite_two_successful_clean_donors=int((~cross["correct"] & control_success).sum()),
            cross_control_failure_rows=int((~control_success).sum()),
            all_four_cross_recipients_correct=core.rate(cross_all[changing]),
            pair_clean_donor_success=core.rate(pair_controls_ok[changing]), families={}, directions={})
        for family in old.FAMILIES:
            fm = np.asarray([p.family == family for p in pairs])
            result["families"][family] = dict(cross=core.measure(cross, np.repeat(fm, 4)),
                pair_controls=core.measure(pair_control, np.repeat(fm, 2)))
        for direction, count in plan["coverage"]["factorial_direction_template_counts"].items():
            c0, c1 = direction.split("->")
            m = np.array([contexts[b].color == c0 and x == c1 for b in bases for x in contexts[b].contents])
            result["directions"][direction] = dict(distinct_templates=count, direct=core.measure(fact_pred, m),
                replay=core.measure(replay, m), restoration=core.measure(restore, m))
        all_results["methods"][method] = result
        saved[method] = dict(cross=cross["correct"], assignments=assignments["correct"],
            direct=fact_pred["correct"], replay=replay["correct"], restore=restore["correct"],
            joint=fact_pred["correct"] & replay["correct"])
    for endpoint, grouping, selected in (
        ("cross", pair_groups.repeat(4), np.repeat(changing, 4)),
        ("assignments", groups.repeat(4*52), np.ones(allrows, bool)),
        ("direct", groups.repeat(4), color_change.reshape(-1)),
        ("replay", groups.repeat(4), color_change.reshape(-1)),
        ("joint", groups.repeat(4), color_change.reshape(-1)),
        ("restore", groups.repeat(4), color_change.reshape(-1))):
        all_results["paired"][endpoint] = paired(saved["standard"][endpoint], saved["gradual"][endpoint], grouping, selected)
    failed = []
    for name, values in all_results["controls"].items():
        if values["accuracy"] < .9:
            failed.append("clean_AT/"+name)
    for method, result in all_results["methods"].items():
        for metric in ("direct", "factorial_replay", "factorial_restoration", "assignments", "cross"):
            for cell, value in result[metric].items():
                if value["accuracy"] is not None and value["accuracy"] < .9:
                    failed.append(method+"/"+metric+"/"+cell)
        for family, result0 in all_results["original_clean_comparison"].items():
            if family in ("C_only", "C_AP_AT") and result0["endpoints"]["joint"][method]["accuracy"] < .9:
                failed.append(method+"/original_joint/"+family)
    coverage_ok = len(plan["coverage"]["factorial_direction_template_counts"]) == 23*22 and min(plan["coverage"]["factorial_direction_template_counts"].values()) >= 25
    for name, receipt in readouts.items():
        if not receipt["available"]:
            failed.append("readout/"+name+"/unavailable")
        else:
            for population in ("validation", "natural_evaluation"):
                if receipt[population]["accuracy"] < .9:
                    failed.append("readout/"+name+"/"+population)
    all_results["decision"] = dict(discrete_90_percent_failures=failed, directional_coverage_passed=coverage_ok,
        full_chain_certified=False, reason="Finite reconstructed bank, insufficient per-direction coverage; discrete failures listed separately",
        continuous_exactness="Reported coordinate errors and matched natural roundtrips; no claim of exact BF16 restoration",
        no_two_fact_claim=True, independently_retrained_replication=False)
    old.write(root / "results.json", all_results)
    lines = ["# Frozen One-Color Full-Chain Diagnostic", "",
        "Same frozen standard/gradual C handles and shared AT anchor. Original 1600-pair and 1476-pair reports preserved. No handle training or selection.", "",
        f"Coverage: {p} original pairs, {n} distinct bases, {len(plan['valid_contexts'])} natural contexts; 23 colors, A-Z labels.", "",
        "| Test | Standard | Gradual | Gradual - standard, paired 95% interval |", "|---|---:|---:|---:|"]
    for endpoint, result in all_results["paired"].items():
        lo, hi = result["percentile_95"]
        lines.append(f"| {endpoint} | {result['standard']['accuracy']:.2%} | {result['gradual']['accuracy']:.2%} | {100*result['difference']:+.2f} pp [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "Factorial direct/replay/joint/restoration rows above are C-changing only. Cross replay uses original C-changing pairs. Assignments include all 4 colors x 26 independent AT letters x 2 donors per base.", "",
        "## Shared AT Controls", "", "| Cell | Correct | Count |", "|---|---:|---:|"]
    for name, result in all_results["controls"].items():
        lines.append(f"| {name} | {result['accuracy']:.2%} | {result['count']} |")
    lines += ["", f"Natural AT swap-then-restore: {all_results['clean_roundtrip']['accuracy']:.2%}; maximum AT-coordinate residual {all_results['clean_roundtrip_coordinate_max_error']:.6g}.", "",
        "## Family-Specific Cross-Question Replay", "", "| Family | Standard | Gradual |", "|---|---:|---:|"]
    for family in old.FAMILIES:
        lines.append(f"| {family} | {all_results['methods']['standard']['families'][family]['cross']['accuracy']:.2%} | {all_results['methods']['gradual']['families'][family]['cross']['accuracy']:.2%} |")
    lines += ["", "## Independent Measurements", "", "| Readout | Validation | Natural evaluation |", "|---|---:|---:|"]
    for name, receipt in readouts.items():
        lines.append(f"| {name} | " + (f"{receipt['validation']['accuracy']:.2%} | {receipt['natural_evaluation']['accuracy']:.2%} |" if receipt['available'] else "Unavailable: frozen class coverage shortfall | n/a |"))
    lines += ["", "## Interpretation and Limits", "",
        "Full-chain certification is not established. Inspect results.json for every failed 90% cell and color-direction coverage. Even a passing aggregate cannot replace missing directional evidence.",
        "Shared clean donors test the AT anchor independently of C. Cross replay failures with successful clean controls are counted explicitly; controls never filter the primary intervention bank.",
        "AT letters outside the base's options are independent interventions; their results are separated from in-option assignments. Ties and nonletter outputs fail strict scoring.",
        "Continuous restoration errors are compared with natural swap-then-restore controls. An imperfect BF16 restoration does not by itself demonstrate an AT bypass.",
        "Intervals resample base question templates with a shared fixed donor/recipient schedule. There is one uploaded checkpoint per method and different checkpoint-selection rules; this is evaluation of those checkpoints, not an independently retrained method replication.",
        "The C handle is at the literal color token in a one-fact prompt. This does not establish requested-object resolution in two-fact prompts.", "",
        f"GPU session accounting: {old.read(root / 'engineering/gpu_budget.json')['consumed_seconds']/60:.2f} minutes, ceiling 90 minutes. See profile.json and preflight.json."]
    content = "\n".join(lines)+"\n"
    path = root / "RESULTS.md"
    if path.exists() and path.read_text() != content:
        raise ValueError("Immutable diagnostic report changed")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    artifacts = {str(p.relative_to(root)): schema.file_hash(p) for p in root.rglob("*") if p.is_file()
        and (p.suffix in (".pt", ".npz") or p.name in ("identity.json", "completion.json", "results.json", "RESULTS.md", "preflight.json", "profile.json"))}
    old.write(root / "completion.json", dict(complete=True, artifacts=artifacts, originals_unchanged=data.verify_originals(),
        diagnostic_complete=True, full_chain_certified=False, no_new_training=True))
    old.write(root / "status.json", dict(stage="complete", diagnostic_complete=True, full_chain_certified=False), immutable=False)
    return all_results
