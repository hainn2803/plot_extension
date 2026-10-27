"""Finite, fixed-slot proposals; correctness-only selection and explicit coverage audits."""
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations, permutations, product
import random

from .data import Prompt, family_check, from_record, pair_record
from .protocol import BANKS, COLORS, FAMILIES, LABELS, PROTOCOL, allocate_groups, order_key
from .storage import digest

COLOR_PAIRS = tuple(combinations(COLORS, 2))
LETTER_PAIRS = tuple(combinations(LABELS, 2))
COLOR_ORDERS = tuple(permutations(COLORS))
LABEL_ORDERS = tuple(permutations(LABELS))
CONDITIONS = ("one_color", "two_colors")


def changes(family, condition, query_mode):
    c = family in ("C_change_AT_fixed", "both_options_fixed", "four_way", "requested_object")
    at = family in ("labels_only", "contents_only", "both_options_fixed", "four_way", "requested_object")
    if family == "requested_object" and condition == "one_color":
        c = at = query_mode == "change"
    return c, at


def slots(bank, family, condition, query_mode):
    groups = allocate_groups()[bank]
    c_change, at_change = changes(family, condition, query_mode)
    count = BANKS[bank]["pairs_per_cell"] // 2
    # Hash the group assignment independently of the cyclic transition quotas.
    # Every group still receives the same number of slots in this cell.
    assignments = [groups[k % len(groups)] for k in range(count)]
    assignment_order = sorted(range(count), key=lambda k: order_key(f"groups:{bank}:{family}:{condition}:{k}"))
    group_for_slot = {slot: group for slot, group in zip(assignment_order, assignments)}
    for k in range(count):
        cs = COLOR_PAIRS[k % 6] if c_change else (COLORS[k % 4],) * 2
        if at_change:
            index = (k % 6 + k // 6) % 6 if c_change else k % 6
            ats = LETTER_PAIRS[index]
        else:
            index = k % 4 if c_change else (k % 4 + k // 4) % 4
            ats = (LABELS[index],) * 2
        yield {"id": f"{bank}:{family}:{condition}:{k:03d}", "bank": bank,
               "family": family, "condition": condition, "group": group_for_slot[k],
               "C": cs, "AT": ats}


def option_pairs(slot):
    result = []
    for labels in LABEL_ORDERS:
        result.extend(_option_pairs_for_labels(slot, labels))
    return result


def _option_pairs_for_labels(slot, lb):
    cb, cs = slot["C"]
    ab, ass = slot["AT"]
    family = slot["family"]
    out = []
    for ob in COLOR_ORDERS:
        if lb[ob.index(cb)] != ab:
            continue
        if family in ("both_options_fixed", "irrelevant_fact", "requested_object"):
            candidates = [(ob, lb)]
        elif family == "labels_only":
            candidates = [(ob, ls) for ls in LABEL_ORDERS if ls != lb]
        elif family == "contents_only":
            candidates = [(os, lb) for os in COLOR_ORDERS if os != ob]
        elif family == "row_order_only":
            mapping = dict(zip(ob, lb))
            candidates = [(os, tuple(mapping[c] for c in os)) for os in COLOR_ORDERS if os.index(cb) != ob.index(cb)]
        else:
            candidates = product(COLOR_ORDERS, LABEL_ORDERS)
        for os, ls in candidates:
            if ls[os.index(cs)] != ass:
                continue
            if family == "four_way":
                forward = (lb[ob.index(cs)], ass, lb[os.index(cs)], ab)
                backward = (ls[os.index(cb)], ab, ls[ob.index(cb)], ass)
                if len(set(forward)) != 4 or len(set(backward)) != 4:
                    continue
            out.append((ob, lb, os, ls))
    return out


def proposal_pairs(slot, query_mode, alternatives=None):
    """Extend the same frozen random stream without looking at model outputs."""
    alternatives = alternatives or PROTOCOL["candidate_alternatives_per_slot"]
    seed = int(order_key(slot["id"]), 16)
    rng = random.Random(seed)
    options = option_pairs(slot)
    if not options:
        raise ValueError("Abstract quota slot is infeasible: " + slot["id"])
    cb, cs = slot["C"]
    nuisance = ["here"] if slot["condition"] == "one_color" else [c for c in COLORS if c not in (cb, cs)]
    variants = list(product(options, (0, 1), ((0, 1), (1, 0)), nuisance))
    rng.shuffle(variants)
    seen, result = set(), []
    for (ob, lb, os, ls), query, fact_order, other in variants:
        facts = [other, other]
        facts[query] = cb
        base = Prompt(tuple(slot["group"]), tuple(facts), query, fact_order, ob, lb)
        sfacts = list(facts)
        sfacts[query] = cs
        source = replace(base, facts=tuple(sfacts), contents=os, labels=ls)
        family = slot["family"]
        if family == "irrelevant_fact":
            if base.condition == "one_color":
                source = replace(source, fact_order=tuple(reversed(fact_order)))
            else:
                choices = [c for c in COLORS if c not in (cb, other)]
                sfacts[1-query] = choices[seed % len(choices)]
                source = replace(source, facts=tuple(sfacts))
        elif family == "requested_object":
            if base.condition == "two_colors":
                facts[1-query] = cs
                base = replace(base, facts=tuple(facts))
                source = replace(base, query=1-query)
            else:
                sfacts[query], sfacts[1-query] = "here", cs
                source = replace(source, facts=tuple(sfacts), query=1-query)
        if not family_check(base, source, family, query_mode) or not family_check(source, base, family, query_mode):
            raise AssertionError("Proposal violates its family: " + slot["id"])
        key = tuple(sorted((base.id, source.id)))
        if key in seen:
            continue
        seen.add(key)
        result.append(pair_record(base, source, family, slot["id"]))
        if len(result) == alternatives:
            return sorted(result, key=lambda p: order_key(p["id"]))
    raise ValueError(f"Only {len(result)} distinct alternatives for {slot['id']}; no expansion permitted")


def generate_bank(bank, query_mode):
    if query_mode not in ("preserve", "change"):
        raise ValueError("One-color requested-object semantics must be explicitly resolved")
    return [{"slot": slot, "alternatives": proposal_pairs(slot, query_mode)}
            for family in FAMILIES for condition in CONDITIONS
            for slot in slots(bank, family, condition, query_mode)]


def select_pairs(proposals, clean):
    """Small per-cell binary programs enforce only declared quotas and uniqueness."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    cells = defaultdict(list)
    for item in proposals:
        cells[item["slot"]["family"], item["slot"]["condition"]].append(item)
    selected = []
    for cell, items in sorted(cells.items()):
        candidates = [r for item in items for r in item["alternatives"]
                      if clean.get(r["base"]["id"]) is True and clean.get(r["source"]["id"]) is True]
        candidates.sort(key=lambda r: (r["slot"], order_key(r["id"])))
        slot_keys = [item["slot"]["id"] for item in items]
        pair_keys = sorted({tuple(sorted((r["base"]["id"], r["source"]["id"]))) for r in candidates})
        if not candidates or set(slot_keys) != {r["slot"] for r in candidates}:
            raise ValueError(f"Finite clean pool has empty slots in {cell}")
        keys = [("slot", s) for s in slot_keys] + [("labels", p) for p in LABEL_ORDERS] + [("pair", p) for p in pair_keys]
        indices = {key: i for i, key in enumerate(keys)}
        matrix = lil_matrix((len(keys), len(candidates)), dtype=float)
        for j, row in enumerate(candidates):
            matrix[indices["slot", row["slot"]], j] = 1
            # Both orientations are retained, so coverage applies equally to both sides.
            for labels in {tuple(row[side]["labels"]) for side in ("base", "source")}:
                matrix[indices["labels", labels], j] = 1
            key = tuple(sorted((row["base"]["id"], row["source"]["id"])))
            matrix[indices["pair", key], j] = 1
        lower = np.array([1 if k[0] in ("slot", "labels") else 0 for k in keys], float)
        upper = np.array([np.inf if k[0] == "labels" else 1 for k in keys], float)
        cost = np.array([int(order_key(r["slot"] + r["id"])[:12], 16) / 16**12 for r in candidates])
        result = milp(cost, integrality=np.ones(len(candidates)), bounds=Bounds(0, 1),
                      constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                      options={"time_limit": 120, "mip_rel_gap": 0})
        if result.status == 2:
            raise ValueError(f"Finite clean pool cannot satisfy declared coverage in {cell}")
        if result.status != 0:
            raise RuntimeError(f"Quota solver did not finish in {cell}; feasibility is unknown: {result.message}")
        for row, x in zip(candidates, result.x):
            if x > 0.5:
                base, source = from_record(row["base"]), from_record(row["source"])
                selected.extend([row, pair_record(source, base, row["family"], row["slot"])])
    return selected


def audit_pairs(rows, bank, query_mode):
    expected = BANKS[bank]["pairs_per_cell"]
    if len(rows) != expected * 16 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Wrong size or duplicate directed pairs")
    ids = {r["id"] for r in rows}
    cells = defaultdict(list)
    for row in rows:
        b, s = from_record(row["base"]), from_record(row["source"])
        if pair_record(s, b, row["family"])["id"] not in ids:
            raise ValueError("Missing exact reciprocal pair")
        if not family_check(b, s, row["family"], query_mode):
            raise ValueError("Family mismatch")
        if row["targets"] != pair_record(b, s, row["family"])["targets"]:
            raise ValueError("Stale counterfactual target")
        cells[row["family"], row["condition"]].append(row)
    coverage = {}
    allocated = {tuple(g) for g in allocate_groups()[bank]}
    for family in FAMILIES:
        for condition in CONDITIONS:
            cell = cells[family, condition]
            if len(cell) != expected or {tuple(r["group"]) for r in cell} != allocated:
                raise ValueError("Missing cell or group coverage")
            rates = {}
            for variable, states, changing in zip(("C", "AT"), (COLORS, LABELS), changes(family, condition, query_mode)):
                counts = Counter((r["base"][variable], r["source"][variable]) for r in cell)
                required = {(a, b) for a in states for b in states if (a != b) == changing}
                if set(counts) != required or len(set(counts.values())) != 1:
                    raise ValueError(f"Unbalanced {variable}: {family}/{condition}: {counts}")
                rates[variable] = {f"{a}->{b}": n for (a, b), n in sorted(counts.items())}
            for side in ("base", "source"):
                if {tuple(r[side]["labels"]) for r in cell} != set(LABEL_ORDERS):
                    raise ValueError(f"Missing label order: {family}/{condition}/{side}")
            joint = Counter((r["base"]["C"], r["source"]["C"], r["base"]["AT"], r["source"]["AT"]) for r in cell)
            coverage[f"{family}/{condition}"] = {"pairs": len(cell), "groups": len(allocated),
                "transitions": rates, "joint_counts": {"/".join(k): n for k, n in sorted(joint.items())}}
    return coverage


def readout_proposals(bank, alternatives=None):
    alternatives = alternatives or PROTOCOL["candidate_alternatives_per_slot"]
    groups = allocate_groups()[bank]
    total = BANKS[bank]["prompts"]
    result = []
    for condition, color, label in product(CONDITIONS, COLORS, LABELS):
        for index in range(total // 32):
            key = f"{bank}:{condition}:{color}:{label}:{index}"
            group = tuple(groups[index % len(groups)])
            variants = []
            for labels, contents, query, fact_order in product(LABEL_ORDERS, COLOR_ORDERS, (0, 1), ((0, 1), (1, 0))):
                if labels[contents.index(color)] != label:
                    continue
                other = "here" if condition == "one_color" else COLORS[(COLORS.index(color) + 1 + index % 3) % 4]
                facts = [other, other]
                facts[query] = color
                variants.append(Prompt(group, tuple(facts), query, fact_order, contents, labels))
            chosen = sorted(variants, key=lambda p: order_key(key + p.id))[:alternatives]
            result.append({"slot": key, "C": color, "AT": label, "condition": condition,
                           "alternatives": [p.record() for p in chosen]})
    return result
