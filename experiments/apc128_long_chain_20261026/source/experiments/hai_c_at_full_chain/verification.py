"""Independent metadata and exact toy checks before any GPU work."""
from pathlib import Path

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import run as old
from experiments.hai_c_at_evaluation import schema, backends
from experiments.toy_c_at_composition import core as toy
from . import data


def toy_check():
    model = toy.FrozenNetwork()
    states = toy.prepared(model, toy.bank())
    results = {}
    for reference in ("incompatible", "compatible"):
        result, _ = toy.evaluate(model, states, model.reference_basis(reference), structural=True)
        for field in ("direct_changing", "direct_preservation", "at_changing", "at_preservation"):
            if result[field]["accuracy"] != 1:
                raise ValueError("Toy individual-handle oracle failed")
        expected = 0. if reference == "incompatible" else 1.
        if result["replay_changing"]["accuracy"] != expected or result["restoration_changing"]["accuracy"] != expected:
            raise ValueError("Toy replay/restoration counterexample not detected")
        if result["joint_assignments"]["accuracy"] != (0.4375 if reference == "incompatible" else 1.):
            raise ValueError("Toy conflict oracle failed")
        if result["upstream_invariance_max_error"] > 1e-10:
            raise ValueError("Toy downstream intervention affected upstream state")
        results[reference] = {k: result[k] for k in ("direct_changing", "replay_changing", "restoration_changing", "joint_assignments")}
    return results


def validate_schedule(contexts, pairs, plan, ro):
    if len(pairs) != 1476 or len(plan["bases"]) != 1065 or len(plan["valid_contexts"]) != 2541:
        raise ValueError("Registered population sizes changed")
    for bi, b in enumerate(plan["bases"]):
        base = contexts[b]
        for color, donor in zip(base.contents, plan["c_donors"][bi]):
            source = contexts[donor]
            if source.color != color or source.question == base.question or source.answer == base.label_for(color):
                raise ValueError("C donor does not isolate source color from source natural letter")
        excluded = {base.question, *(contexts[i].question for i in plan["c_donors"][bi])}
        for letter, donors in zip(data.LETTERS, plan["at_donors"][bi]):
            a, z = [contexts[i] for i in donors]
            if a.answer != letter or z.answer != letter or a.color == z.color or a.question == z.question or {a.question, z.question} & excluded:
                raise ValueError("Invalid independent AT donor pair")
    for j, p in enumerate(pairs):
        b, s = contexts[p.base], contexts[p.source]
        t = b.label_for(s.color)
        rr = plan["recipients"][j]
        if len(set(contexts[i].question for i in rr)) != 4:
            raise ValueError("Cross recipients repeat a question template")
        for k, r in enumerate(rr):
            c = contexts[r]
            if c.question in (b.question, s.question) or t not in c.labels or (c.answer == t) != (k < 2):
                raise ValueError("Cross recipient changing/preservation assignment incorrect")
            for donor in plan["cross_donors"][j][k]:
                d = contexts[donor]
                if d.answer != t or d.question in (b.question, s.question, c.question):
                    raise ValueError("Cross clean donor target/independence incorrect")
    fitting, validation = set(ro["fit_templates"]), set(ro["validation_templates"])
    evaluation = {contexts[i].question for i in plan["valid_contexts"]}
    if fitting & validation or (fitting | validation) & evaluation or len(fitting) != 16 or len(validation) != 7:
        raise ValueError("Independent readout split leakage")
    return dict(passed=True, pairs=len(pairs), factorial_assignments_per_method=len(plan["bases"])*4*26*2,
        same_schedule_for_both_methods=True, all_donors_clean_by_parent_mask=True,
        C_source_letter_excluded=True, independent_AT_donor_colors=True,
        cross_recipients_per_pair=4, readout_templates=[16, 7])


def verify(root=data.ROOT):
    root = Path(root)
    contexts, pairs, plan, ro = data.load(root)
    torch.set_num_threads(2)
    result = validate_schedule(contexts, pairs, plan, ro)
    natural, base, _ = data.caches(plan)
    if data.schedule(contexts, pairs, plan["original_pair_indices"]) != plan:
        raise ValueError("Hash schedule is not reproducible")
    for variable in ("C", "AT"):
        for method in data.METHODS:
            backends.checked_basis(old.checkpoint(method, variable), 2304, variable)
    if not torch.equal(old.checkpoint("standard", "AT")["basis"], old.checkpoint("gradual", "AT")["basis"]):
        raise ValueError("Frozen AT handles differ across methods")
    result["toy"] = toy_check()
    result["cuda_initialized"] = torch.cuda.is_initialized()
    result["source_identity"] = schema.file_hash(root / "identity.json")
    if result["cuda_initialized"]:
        raise ValueError("CPU verification unexpectedly initialized CUDA")
    old.write(root / "cpu_verification.json", result)
    print({"cpu_verification": "passed", "toy": "both exact reference outcomes recovered", "schedule": result["factorial_assignments_per_method"]}, flush=True)
    return result
