"""Preservation-separated metrics, distinct-case gates, and paired group uncertainty."""
from collections import defaultdict
import numpy as np


def rate(correct, mask=None):
    correct = np.asarray(correct, dtype=float)
    if mask is not None:
        correct = correct[np.asarray(mask, dtype=bool)]
    return float(correct.mean()) if len(correct) else None


def breakdown(rows, correct, variable):
    changed = [r[variable + "_changed"] for r in rows]
    result = {"changing": rate(correct, changed), "preservation": rate(correct, np.logical_not(changed)),
              "overall_supplement": rate(correct), "families_conditions": {}, "directions": {}, "label_orders": {}}
    for field, key in (("families_conditions", lambda r: r["family"] + "/" + r["condition"]),
                       ("directions", lambda r: r["base"][variable] + "->" + r["source"][variable]),
                       ("label_orders", lambda r: "".join(r["base"]["labels"]))):
        indices = defaultdict(list)
        for i, row in enumerate(rows):
            indices[key(row)].append(i)
        for name, selected in sorted(indices.items()):
            result[field][name] = {"accuracy": rate(np.asarray(correct)[selected]), "n": len(selected),
                                   "distinct_pairs": len({rows[i]["id"] for i in selected}),
                                   "distinct_groups": len({tuple(rows[i]["group"]) for i in selected})}
    return result


def paired_group_bootstrap(groups, left, right=None, repetitions=2000, seed=1):
    left = np.asarray(left, dtype=float)
    delta = left if right is None else left - np.asarray(right, dtype=float)
    if len(groups) != len(delta):
        raise ValueError("Bootstrap data length mismatch")
    grouped = defaultdict(list)
    for group, value in zip(groups, delta):
        grouped[tuple(group)].append(value)
    if not grouped:
        raise ValueError("Empty bootstrap")
    values = list(grouped.values())
    sums = np.array([sum(v) for v in values])
    sizes = np.array([len(v) for v in values])
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(values), (repetitions, len(values)))
    estimates = sums[selected].sum(1) / sizes[selected].sum(1)
    return {"estimate": float(delta.mean()), "interval": np.quantile(estimates, [.025, .975]).tolist(),
            "clusters": len(values), "resamples": repetitions, "unit": "fixed two-object group",
            "donor_recipient_library": "held fixed across bootstrap draws"}


def discrete_gate(summary, threshold=.9, minimum=25):
    failures = []
    if len(summary["families_conditions"]) != 16:
        failures.append("missing_family_condition_coverage")
    if len(summary["directions"]) != 16:
        failures.append("missing_state_transition_coverage")
    for key in ("changing", "preservation"):
        if summary[key] is None or summary[key] < threshold:
            failures.append(key)
    for key, cell in summary["families_conditions"].items():
        if cell["accuracy"] < threshold:
            failures.append("family/condition:" + key)
    for key, cell in summary["directions"].items():
        if cell["accuracy"] < threshold or cell["distinct_pairs"] < minimum:
            failures.append("direction:" + key)
    return {"passed": not failures, "failed_checks": failures, "threshold": threshold, "minimum_distinct_cases": minimum}


def structural_gates(rows, threshold=.9, minimum=25):
    fields = {"joint_assignments": lambda r: r["joint_correct"],
              "AT_directed_control": lambda r: r["AT_only_correct"],
              "reverse_invariance": lambda r: r["upstream_decoded_C"] == r["base_C"] and r["upstream_max_abs_change"] <= 2e-4,
              "restoration_output": lambda r: r["restoration_correct"],
              "restoration_upstream_C": lambda r: r["restored_decoded_C"] == r["color"],
              "restoration_downstream_AT": lambda r: r["restored_decoded_AT"] == r["base_AT"]}
    results = {}
    for name, value in fields.items():
        cells = defaultdict(list)
        for row in rows:
            cells[row["color"], row["letter"]].append(row)
        by_assignment = {}
        for (color, letter), selected in sorted(cells.items()):
            cases = {(r["base"], r["color"], r["letter"]) for r in selected}
            by_assignment[color + "/" + letter] = {"rate": rate([value(r) for r in selected]), "distinct_cases": len(cases)}
        results[name] = {"rate": rate([value(r) for r in rows]), "assignments": by_assignment,
                         "passed": len(cells) == 16 and all(c["rate"] >= threshold and c["distinct_cases"] >= minimum for c in by_assignment.values())}
    return results
