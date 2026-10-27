"""Matched-layout references, workload profile and bounded scientific execution."""
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import descriptive as desc, run as old
from experiments.hai_c_at_evaluation import core, schema, backends
from experiments.mcqa_c_at_localization import runtime
from . import data, gpu


class Full(old.GuardedBackend):
    def __init__(self, *args, budget, root, **kwargs):
        self.budget = budget
        self.root = Path(root)
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.budget.tick()
        gpu.cooling(self.root, self.budget)
        result = super().run(*args, **kwargs)
        self.budget.tick()
        return result


def compare_predictions(name, actual, expected, checks):
    for field in ("correct", "token_id", "tied", "other"):
        if not np.array_equal(actual[field], expected[field]):
            raise ValueError(f"{name}: full-vocabulary {field} differs")
    for field in ("label_logits", "label_probabilities", "margin"):
        old.compare(name + "/" + field, torch.as_tensor(actual[field]), torch.as_tensor(expected[field]), checks)


@torch.no_grad()
def preflight(root, fulls, natural, base, original_pairs, contexts, plan, budget):
    checks = []
    selected = plan["valid_contexts"]
    tail = gpu.Tail(fulls["standard"], root, budget)
    for a in range(0, len(selected), 32):
        ii = selected[a:a+32]
        pred = tail.predict(natural.at[ii], [contexts[i].answer for i in ii],
            head_sizes=gpu.head_sizes(ii, len(contexts)))
        expected = {k: v[ii] for k, v in natural.predictions.items()}
        compare_predictions(f"natural_tail/{a}", pred, expected, checks)
    all_b = np.array([p.base for p in original_pairs])
    all_s = np.array([p.source for p in original_pairs])
    for method, full in fulls.items():
        tail = gpu.Tail(full, root, budget)
        for a in (800, 1216):
            ii, ss = all_b[a:a+32], all_s[a:a+32]
            targets = [contexts[b].label_for(contexts[s].color) for b, s in zip(ii, ss)]
            clean = full.run(ii)
            direct = full.run(ii, c_source=natural.c[ss])
            donor = natural.at[np.roll(ss, 1)]
            for name, h, d, ref, expected_targets in (
                ("clean", clean.at, None, clean, [contexts[i].answer for i in ii]),
                ("direct", direct.at, None, direct, targets),
                ("replay", clean.at, direct.at, full.run(ii, at_source=direct.at), targets),
                ("conflict", direct.at, donor, full.run(ii, c_source=natural.c[ss], at_source=donor),
                    [contexts[i].answer for i in np.roll(ss, 1)]),
                ("restore", direct.at, clean.at, full.run(ii, c_source=natural.c[ss], at_source=clean.at),
                    [contexts[i].answer for i in ii])):
                state = tail.state(h, d)
                y = tail.logits(state, 32)[:len(ii)]
                old.compare(f"{method}/{a}/{name}/logits", y, ref.logits, checks)
                if d is not None:
                    old.compare(f"{method}/{a}/{name}/AT", state, ref.at, checks)
                compare_predictions(f"{method}/{a}/{name}", core.score(y, expected_targets, tail),
                    core.score(ref.logits, expected_targets, full), checks)
            restored = full.run(ii, c_source=natural.c[ss], at_source=clean.at)
            old.compare(f"{method}/{a}/upstream_invariance", restored.c, direct.c, checks)
        if any(layer._forward_hooks for layer in full.model.model.layers) or any(p.requires_grad for p in full.model.parameters()):
            raise ValueError("Backbone mutation or hook leak")
    receipt = dict(passed=True, checks=checks, baseline_contexts=len(selected),
        full_vocabulary_winners_and_ties_identical=True, atol=2e-4, rtol=2e-4,
        frozen_backbone=True, hook_cleanup=True,
        note="Final block output -> unchanged normalization/head/FP32 softcap; original physical head sizes retained")
    old.write(root / "preflight.json", receipt)
    print(json.dumps(dict(preflight_passed=True, checks=len(checks), clean_contexts=len(selected))), flush=True)


def workload(plan, readouts):
    b, p = len(plan["bases"]), len(plan["pairs"])
    counts = dict(clean_AT=b*26*2, clean_roundtrip=b*26*2, pair_controls=p*2,
        cross_controls=p*4*2, cross_both_methods=2*p*4,
        factorial_assignments_both_methods=2*b*4*26*2,
        factorial_replay_restore_both_methods=2*2*b*4)
    return dict(tail_rows=sum(counts.values()), components=counts,
        full_backbone_rows=2*4*1600+len(readouts["contexts"]))


@torch.no_grad()
def profile(root, full, natural, base, plan, ro, budget):
    tail = gpu.Tail(full, root, budget)
    b0 = np.asarray(plan["original_pair_indices"])[np.asarray(plan["base_pair_occurrences"])][:32]
    h, d = base.at[b0], natural.at[np.asarray(plan["at_donors"])[:32, 0, 0]]
    labels = ["A"]*len(h)
    full_ids = old.read(desc.ROOT / "operations/matched_base/identity.json")["indices"][:32]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(3):
        full.run(full_ids)
    full_seconds = (time.perf_counter()-start)/(3*32)
    started, rows, repeats = time.perf_counter(), 0, 0
    while time.perf_counter()-started < 60:
        pieces = [tail.predict(h, labels, d) for _ in range(16)]
        gpu.save_npz(root / "engineering/profile_tail_sample.npz", {k: np.concatenate([v[k] for v in pieces]) for k in pieces[0]})
        rows += 16*len(h)
        repeats += 1
    seconds = time.perf_counter()-started
    work = workload(plan, ro)
    remaining = (work["tail_rows"]*seconds/rows + work["full_backbone_rows"]*full_seconds)*1.25
    receipt = dict(passed=True, physical_batch=32, profile_seconds=seconds, profile_tail_rows=rows,
        seconds_per_tail_row_including_guard_scoring_disk=seconds/rows, seconds_per_full_row=full_seconds,
        workload=work, allowance=1.25, estimated_remaining_seconds=remaining,
        consumed_gpu_session_seconds=budget.elapsed(), estimated_total_gpu_minutes=(budget.elapsed()+remaining)/60,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        dedicated_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        launch_allowed=budget.elapsed()+remaining <= budget.limit)
    if receipt["peak_reserved_bytes"] > receipt["dedicated_memory_bytes"]:
        raise ValueError("Profile exceeds dedicated GPU memory")
    old.write(root / "profile.json", receipt)
    print(json.dumps(dict(profile=receipt)), flush=True)
    if not receipt["launch_allowed"]:
        raise runtime.Paused("Measured full workload plus 25% allowance exceeds approved 90 GPU-minute budget")
    return receipt


def cache_checked(path):
    receipt = old.read(path.with_suffix(".json"))
    if schema.file_hash(path) != receipt["sha256"]:
        raise ValueError("Cached diagnostic states changed")
    return torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def factorial_cache(root, method, full, natural, base, plan, contexts):
    path = root / "states" / (method + "_factorial.pt")
    if path.with_suffix(".json").exists():
        return cache_checked(path)
    all_indices = old.read(desc.ROOT / "operations/matched_base/identity.json")["indices"]
    first = np.asarray(plan["original_pair_indices"])[np.asarray(plan["base_pair_occurrences"])]
    mapping = {int(row): i for i, row in enumerate(first)}
    n = len(first)
    cs, ats, predictions = [None]*(n*4), [None]*(n*4), [None]*(n*4)
    donors = np.asarray(plan["c_donors"])
    for variant in range(4):
        for start in range(0, len(all_indices), 32):
            stop = min(start+32, len(all_indices))
            chunk = root / "states" / method / f"v{variant}_batch{start:04d}.pt"
            if chunk.with_suffix(".json").exists():
                payload = cache_checked(chunk)
            else:
                ii = all_indices[start:stop]
                source = natural.c[ii].clone()
                for row in range(start, stop):
                    if row in mapping:
                        source[row-start] = natural.c[donors[mapping[row], variant]]
                trace = full.run(ii, c_source=source)
                target = [contexts[i].labels[variant] for i in ii]
                pred = core.score(trace.logits, target, full)
                payload = dict(c=trace.c, at=trace.at,
                    predictions={k: torch.from_numpy(v) for k, v in pred.items()})
                old.save_tensor(chunk, payload)
                old.write(chunk.with_suffix(".json"), dict(sha256=schema.file_hash(chunk)))
            for row in range(start, stop):
                if row in mapping:
                    out = mapping[row]*4+variant
                    cs[out], ats[out] = payload["c"][row-start], payload["at"][row-start]
                    predictions[out] = {k: v[row-start].numpy() for k, v in payload["predictions"].items()}
            old.write(root / "status.json", dict(stage=method+"_factorial_capture", variant=variant,
                committed=stop, total=len(all_indices), time=time.time(), pid=os.getpid()), immutable=False)
    if any(h is None for h in ats):
        raise ValueError("Factorial capture missing a base/color assignment")
    saved = dict(c=torch.stack(cs), at=torch.stack(ats), predictions={k: torch.from_numpy(np.stack([p[k] for p in predictions])) for k in predictions[0]})
    old.save_tensor(path, saved)
    old.write(path.with_suffix(".json"), dict(sha256=schema.file_hash(path), count=n*4,
        layout="Full original 32-row base batches including unscored filler rows; first retained occurrence per distinct base"))
    return saved


@torch.no_grad()
def readout_capture(root, full, ro):
    path = root / "states/readout_natural.pt"
    if path.with_suffix(".json").exists():
        return cache_checked(path)
    contexts = [schema.Context(**r) for r in ro["contexts"]]
    backend = Full(full.model, full.tokenizer, contexts, old.checkpoint("standard", "C"), old.checkpoint("standard", "AT"),
        full.label_ids, root=root, budget=full.budget)
    out = old.operation(root, "readout_natural", backend, np.arange(len(contexts)), [c.answer for c in contexts], 32)
    saved = dict(c=out.c, at=out.at, correct=torch.from_numpy(out.predictions["correct"]))
    old.save_tensor(path, saved)
    old.write(path.with_suffix(".json"), dict(sha256=schema.file_hash(path)))
    return saved


def evaluate(root, fulls, contexts, pairs, plan, ro, natural, base, direct, budget):
    originals = np.asarray(plan["original_pair_indices"])
    first = originals[np.asarray(plan["base_pair_occurrences"])]
    b = np.asarray(plan["bases"])
    n, p = len(b), len(pairs)
    adb = np.asarray(plan["at_donors"])
    letters = np.asarray(data.LETTERS)
    tail = gpu.Tail(fulls["standard"], root, budget)
    control_base = np.repeat(first, 52)
    control_donor = adb.reshape(-1)
    control_targets = np.tile(np.repeat(letters, 2), n)
    control_baseline = np.repeat(base.predictions["label_probabilities"][first], 52, axis=0)
    gpu.operation(root, "clean_AT", tail, base.at, control_base, control_targets,
        donor_bank=natural.at, donor_idx=control_donor, baseline_probs=control_baseline)
    gpu.operation(root, "clean_roundtrip", tail, base.at, control_base,
        np.repeat([contexts[i].answer for i in b], 52), donor_bank=natural.at, donor_idx=control_donor,
        restore_bank=base.at, restore_idx=control_base, baseline_probs=control_baseline)
    target = np.array([contexts[x.base].label_for(contexts[x.source].color) for x in pairs])
    gpu.operation(root, "pair_controls", tail, base.at, np.repeat(originals, 2), np.repeat(target, 2),
        donor_bank=natural.at, donor_idx=np.asarray(plan["pair_donors"]).reshape(-1),
        baseline_probs=np.repeat(base.predictions["label_probabilities"][originals], 2, axis=0))
    recipients = np.asarray(plan["recipients"]).reshape(-1)
    rc_targets = np.repeat(target, 4)
    gpu.operation(root, "cross_controls", tail, natural.at, np.repeat(recipients, 2), np.repeat(rc_targets, 2),
        donor_bank=natural.at, donor_idx=np.asarray(plan["cross_donors"]).reshape(-1),
        sizes=gpu.head_sizes(np.repeat(recipients, 2), len(contexts)),
        baseline_probs=np.repeat(natural.predictions["label_probabilities"][recipients], 2, axis=0))
    readout_capture(root, fulls["standard"], ro)
    for method in data.METHODS:
        tail = gpu.Tail(fulls[method], root, budget)
        gpu.operation(root, method+"_cross", tail, natural.at, recipients, rc_targets,
            donor_bank=direct[method].at, donor_idx=np.repeat(originals, 4),
            sizes=gpu.head_sizes(recipients, len(contexts)),
            baseline_probs=natural.predictions["label_probabilities"][recipients])
        factor = factorial_cache(root, method, fulls[method], natural, base, plan, contexts)
        color_first = np.repeat(first, 4)
        color_idx = np.arange(n*4)
        color_targets = np.asarray([x for i in b for x in contexts[i].labels])
        gpu.operation(root, method+"_factorial_replay", tail, base.at, color_first, color_targets,
            donor_bank=factor["at"], donor_idx=color_idx,
            baseline_probs=base.predictions["label_probabilities"][color_first])
        gpu.operation(root, method+"_factorial_restore", tail, factor["at"], color_idx,
            [contexts[i].answer for i in np.repeat(b, 4)], donor_bank=base.at, donor_idx=color_first,
            baseline_probs=base.predictions["label_probabilities"][color_first],
            restore_bank=None)
        # Restoration is a single AT injection from the clean base into C-induced states.
        restore_file = root / "states" / (method+"_restoration_errors.pt")
        if not restore_file.with_suffix(".json").exists():
            errors = []
            for start in range(0, n*4, 32):
                stop = min(start+32, n*4)
                hh = tail.state(factor["at"][start:stop], base.at[color_first[start:stop]])
                e = (hh-base.at[color_first[start:stop]].to(tail.device)) @ tail.q
                errors.append(e.abs().amax(-1).cpu())
            old.save_tensor(restore_file, dict(coordinate_max_error=torch.cat(errors)))
            old.write(restore_file.with_suffix(".json"), dict(sha256=schema.file_hash(restore_file)))
        factor_donor = np.repeat(adb[:, None, :, :], 4, axis=1).reshape(-1)
        gpu.operation(root, method+"_assignments", tail, factor["at"], np.repeat(color_idx, 52),
            np.tile(np.repeat(letters, 2), n*4), donor_bank=natural.at, donor_idx=factor_donor,
            baseline_probs=np.repeat(base.predictions["label_probabilities"][color_first], 52, axis=0))
    old.write(root / "gpu_completion.json", dict(complete=True, gpu_seconds=budget.elapsed(),
        methods=list(data.METHODS), no_handle_training=True, full_vocabulary_scoring=True))


def run(root=data.ROOT, command="run"):
    root = Path(root)
    contexts, pairs, plan, ro = data.load(root)
    if not old.read(root / "cpu_verification.json")["passed"]:
        raise ValueError("CPU protocol verification has not passed")
    natural, base, direct = data.caches(plan)
    _, original_pairs = schema.import_pairs(old.read(desc.ROOT / "selected_pairs.json"))
    budget = None
    try:
        runtime.configure(root)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        budget = gpu.Budget(root)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import transformers
        tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(old.MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        fulls = {m: Full(model, tokenizer, contexts, old.checkpoint(m, "C"), old.checkpoint(m, "AT"),
            old.read(old.DATA / "protocol.json")["label_token_ids"], root=root, budget=budget) for m in data.METHODS}
        old.write(root / "environment.json", dict(torch=torch.__version__, transformers=transformers.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), model_revision=old.MODEL.name,
            python=__import__("platform").python_version()))
        if not (root / "preflight.json").exists():
            preflight(root, fulls, natural, base, original_pairs, contexts, plan, budget)
        if not (root / "profile.json").exists():
            profile(root, fulls["standard"], natural, base, plan, ro, budget)
        if not old.read(root / "profile.json")["launch_allowed"]:
            raise runtime.Paused("Saved measured profile did not permit full execution")
        if command != "profile":
            evaluate(root, fulls, contexts, pairs, plan, ro, natural, base, direct, budget)
        fulls.clear()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException as error:
        import traceback
        old.write(root / f"stop_{time.time_ns()}.json", dict(type=type(error).__name__, error=str(error),
            traceback=traceback.format_exc(), time=time.time()))
        old.write(root / "status.json", dict(stage="paused" if isinstance(error, runtime.Paused) else "failed",
            reason=str(error), pid=os.getpid(), time=time.time()), immutable=False)
        raise
    finally:
        if budget is not None:
            budget.close()
        runtime.close()
