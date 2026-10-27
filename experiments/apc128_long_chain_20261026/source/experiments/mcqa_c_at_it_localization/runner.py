"""Immutable IT inputs, complete-site localization, and unattended safe resume."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path
import tarfile
import time
import traceback

import numpy as np

from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write, save_torch, checked_torch
from experiments.mcqa_c_at_localization.protocol import COLORS, LABELS
from experiments.mcqa_c_at_localization.data import from_record, targets, effect
from experiments.mcqa_c_at_localization.localization import equal_cell_score, balanced_decoding
from experiments.mcqa_c_at_localization.matching import signature_cost, one_sided_uot, rank_sites
from experiments.mcqa_c_at_it_feasibility import runner as it
from experiments.mcqa_c_at_bank_feasibility import runner as banks

ROOT = Path("eval/mcqa_c_at_it_localization_20261017_v2")
ENGINEERING_PARENT = Path("eval/mcqa_c_at_it_localization_20261017_v1")
CHECK_POSITION = 45  # Fixed first displayed fact value, independent of the query.
LENS_REVISION = "a4114d7752d11eb546e6cf372213d7e75526d3a1"
LENS_DIR = Path(".assets_c_at/gemma-2-2b-it/jlens/Salesforce-wikitext")
LENS_PATH = LENS_DIR / "gemma-2-2b-it_jacobian_lens.pt"
PROTOCOL = {
    "model": it.MODEL, "revision": it.REVISION, "format": "chat_explicit",
    "amendment": "User adopted verified IT checkpoint and official chat template; bare next-letter targets; unchanged semantic banks",
    "scope": "All-site full-vector, canonical PLOT and matching IT J-lens localization only; no DAS training or final-test access",
    "pairs": 384, "layers": 26, "positions": "Every non-BOS actual chat-input token, including instructions and special tokens",
    "settings": "Frozen BF16, SDPA, FP32 full-sequence head softcap, TF32 disabled, no padding",
    "atol": 2e-4, "rtol": 2e-4,
    "engineering_correction": "Fixed preflight token45 (first fact value), not token47 (article); nontrivial single-token embedding control; release reference output heads promptly",
    "full_vector": "Strength-one direct replacement; equal family/condition counterfactual accuracy",
    "plot": {"signature": "four-label softmax(swap) minus four-label softmax(matched base)",
             "abstract": "onehot(counterfactual label) minus onehot(base label)",
             "distance": "squared concatenated signature distance divided by number of pairs",
             "epsilon": .05, "beta": .5, "tolerance": 1e-10, "max_iter": 10000,
             "marginals": "fixed equal C/AT mass; relaxed uniform neural; no signature normalization"},
    "lens": {"repo": "neuronpedia/jacobian-lens", "revision": LENS_REVISION,
             "file": "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt",
             "application": "block output @ J.T FP32, cast BF16, final RMSNorm, full-sequence head, FP32 softcap; layer25 identity",
             "ranking": "balanced four-class accuracy; full-vocabulary reciprocal-rank tie-break",
             "color_tokens": "single space-prefixed colors", "letter_tokens": "bare official assistant continuation"},
    "selection": "Top two per method per variable; layer/token ties; J-lens reciprocal-rank tie-break",
    "future_DAS": {"ranks": [8, 32, 128, 512], "seeds": [1, 0, 2], "not_launched_by_this_worker": True},
    "execution": {"session_hours": 12, "allowance": .25, "checkpoint": "Every activation chunk, lens chunk and complete swap site",
                  "safety": "AC required; pause on 60 seconds thermal throttling, battery <=20%, monitor or numerical failure",
                  "batches": [1, 2, 4, 8, 16, 32], "no_shared_memory_spill": True},
}


def sources():
    result = banks.sources()
    result.update({p.as_posix(): file_hash(p) for p in Path(__file__).parent.rglob("*.py")})
    return result


def audit_rows(rows):
    assert len(rows) == 384 and len({r["id"] for r in rows}) == 384
    cells = {}
    for row in rows:
        b, s = from_record(row["base"]), from_record(row["source"])
        assert digest(b.record()) == digest(row["base"]) and digest(s.record()) == digest(row["source"])
        assert targets(b, s) == row["targets"]
        assert row["id"] == digest([b.id, s.id])
        cell = row["family"] + "/" + row["condition"]
        cells[cell] = cells.get(cell, 0) + 1
    assert len(cells) == 16 and set(cells.values()) == {24}
    by_id = {(r["base"]["id"], r["source"]["id"]) for r in rows}
    assert all((s, b) in by_id for b, s in by_id)
    return {"pairs": len(rows), "cells": cells, "groups": sorted({tuple(r["group"]) for r in rows})}


def check_input(tokenizer, logical, actual):
    assert actual["id"] == logical["id"]
    text = it.render(tokenizer, logical["text"], "chat_explicit")
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert actual["text"] == text and actual["input_ids"] == ids
    assert ids[0] == tokenizer.bos_token_id and ids.count(tokenizer.bos_token_id) == 1
    assert tokenizer.encode(text + logical["AT"], add_special_tokens=False) == ids + [actual["expected_token"]]
    assert actual["expected_label"] == logical["AT"]
    return ids


def prepare(root):
    from transformers import AutoTokenizer
    banks.verify(banks.ROOT)
    proof = read(banks.ROOT / "completion.json")
    assert proof["all_four_unsealed_banks_passed"] and proof["no_final_data_used"]
    for path, sha in proof["selected_hashes"].items():
        assert file_hash(banks.ROOT / path) == sha
    tokenizer = AutoTokenizer.from_pretrained(it.MODEL_PATH, local_files_only=True)
    rows = read(banks.ROOT / "selected/localization.json")
    coverage = audit_rows(rows)
    imported, all_lengths, logical_inputs = {}, set(), {}
    for name in banks.NAMES:
        selected = read(banks.ROOT / f"selected/{name}.json")
        logical = selected["logical_rows"]
        prompts = logical if name.startswith("readout") else [r[side] for r in logical for side in ("base", "source")]
        inputs = read(banks.ROOT / f"selected/{name}_inputs.json")
        index = {p["id"]: p for p in inputs}
        assert len(index) == len(inputs) and set(index) == {p["id"] for p in prompts}
        for p in prompts:
            all_lengths.add(len(check_input(tokenizer, p, index[p["id"]])))
        for suffix, value in (("", selected), ("_inputs", inputs)):
            dest = root / "banks" / f"{name}{suffix}.json"
            write(dest, value, immutable=True)
            imported[dest.relative_to(root).as_posix()] = file_hash(dest)
        if name == "train":
            logical_inputs = index
    assert all_lengths == {82}, "No padding, truncation, or template replacement is allowed"
    write(root / "banks/localization.json", rows, immutable=True)
    imported["banks/localization.json"] = file_hash(root / "banks/localization.json")
    ref = logical_inputs[rows[0]["base"]["id"]]
    encoded = tokenizer(ref["text"], add_special_tokens=False, return_offsets_mapping=True)
    original = rows[0]["base"]["text"]
    task_start = ref["text"].index(original)
    slots = []
    for p, (token, offset) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
        if not p:
            continue
        a, b = offset
        region = "task" if task_start <= a < task_start + len(original) else "chat_or_instruction"
        slots.append({"position": p, "reference_token_id": token, "piece": tokenizer.convert_ids_to_tokens(token),
                      "offset": [a, b], "region": region})
    labels = read(banks.ROOT / "registration.json")["token_map"]
    color_tokens = [tokenizer.encode(" " + c, add_special_tokens=False) for c in COLORS]
    assert all(len(v) == 1 for v in color_tokens)
    token_map = {"tokens_including_bos": 82, "slots": slots,
                 "sites": [{"layer": l, "position": p} for l in range(26) for p in range(1, 82)],
                 "label_ids": [labels[l] for l in LABELS], "color_ids": [v[0] for v in color_tokens],
                 "reference": ref, "candidate_count": 26 * 81,
                 "alignment": "Independent source/base ID lookup to verified actual chat input; fixed ordinal slots, no queried-color-aware selection"}
    write(root / "token_map.json", token_map, immutable=True)
    import yaml
    config = yaml.safe_load((LENS_DIR / "config.yaml").read_text())
    assert config["hf_model_name"] == it.MODEL
    from experiments.mcqa_c_at_localization.lens import load_lens
    matrices, lens_meta = load_lens(LENS_PATH)
    del matrices
    lens_meta.update(model=config["hf_model_name"], config=config,
                     caveat="Publisher does not specify the underlying model commit; exact checkpoint-revision provenance is unverified")
    write(root / "coverage.json", coverage, immutable=True)
    reg = {"protocol": PROTOCOL, "source": sources(), "parent": str(banks.ROOT),
           "engineering_parent": {"path": str(ENGINEERING_PARENT),
                                  "registration": file_hash(ENGINEERING_PARENT / "registration.json"),
                                  "pause": file_hash(ENGINEERING_PARENT / "pause.request"),
                                  "source_archive": file_hash(ENGINEERING_PARENT / "source.tar.gz"),
                                  "science_unchanged": True},
           "parent_registration": file_hash(banks.ROOT / "registration.json"),
           "parent_completion": file_hash(banks.ROOT / "completion.json"), "imported": imported,
           "model_files": it.assets(), "token_map": file_hash(root / "token_map.json"),
           "lens_files": {p.as_posix(): file_hash(p) for p in sorted(LENS_DIR.glob("*")) if p.is_file()},
           "lens_metadata": lens_meta, "coverage": coverage, "no_final_data_used": True}
    write(root / "registration.json", reg, immutable=True)
    if not (root / "source.tar.gz").exists():
        with tarfile.open(root / "source.tar.gz", "w:gz") as archive:
            for p in reg["source"]:
                archive.add(p, arcname=p)
    print(json.dumps({"prepared": True, "pairs": 384, "sites": 2106, "model": it.MODEL}), flush=True)


def verify(root, assets=False):
    reg = read(root / "registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == sources(), "Source/protocol identity changed"
    assert reg["parent_registration"] == file_hash(banks.ROOT / "registration.json")
    assert reg["parent_completion"] == file_hash(banks.ROOT / "completion.json")
    for name, filename in (("registration", "registration.json"), ("pause", "pause.request"), ("source_archive", "source.tar.gz")):
        assert reg["engineering_parent"][name] == file_hash(ENGINEERING_PARENT / filename)
    for name, sha in reg["imported"].items():
        assert file_hash(root / name) == sha, "Imported bank changed: " + name
    assert reg["token_map"] == file_hash(root / "token_map.json")
    for name, sha in reg["lens_files"].items():
        assert file_hash(name) == sha, "Lens changed"
    if assets:
        assert it.assets() == reg["model_files"]
    audit_rows(read(root / "banks/localization.json"))
    return reg


class Session:
    def __init__(self, root, hours=12):
        self.root, self.started = root, time.monotonic()
        self.deadline = self.started + hours * 3600

    def check(self, required=0):
        from experiments.mcqa_c_at_localization import runtime
        runtime.guard()
        if time.monotonic() + required + 30 >= self.deadline:
            raise TimeoutError("Overnight session limit reached; committed progress is resumable")
        if runtime.power_status().get("battery_percent", 100) <= 20:
            raise runtime.Paused("Battery at or below 20%; stopping at saved boundary")


def checked_json(path, identity):
    saved = read(path)
    assert saved["identity"] == identity, "Resume identity mismatch: " + str(path)
    return saved


def progress(root, stage, **extra):
    row = {"stage": stage, "time": time.time(), "pid": os.getpid(), **extra}
    write(root / "status.json", row)
    print(json.dumps(row), flush=True)


def actual_ids(root, rows, side):
    import torch
    index = {r["id"]: r for r in read(root / "banks/train_inputs.json")}
    return torch.tensor([index[r[side]["id"]]["input_ids"] for r in rows], dtype=torch.long)


def metric_arrays(logits, chunk, token_map):
    import torch
    from experiments.mcqa_c_at_localization.lens import decoding_metrics
    result = {"correct": {}, "reciprocal_rank": {}}
    for var, names, candidates in (("C", COLORS, token_map["color_ids"]), ("AT", LABELS, token_map["label_ids"])):
        target = torch.tensor([candidates[names.index(r["base"][var])] for r in chunk], device=logits.device)[:, None].expand(logits.shape[:2])
        acc, rr = decoding_metrics(logits, target, candidates)
        result["correct"][var] = acc.cpu().tolist()
        result["reciprocal_rank"][var] = rr.cpu().tolist()
    guesses = torch.tensor(token_map["color_ids"], device=logits.device)[logits[..., token_map["color_ids"]].argmax(-1)].cpu().tolist()
    irrelevant = []
    for row, prediction in zip(chunk, guesses):
        p = from_record(row["base"])
        other = p.facts[1 - p.query]
        other_id = token_map["color_ids"][COLORS.index(other)] if other in COLORS else None
        irrelevant.append([int(g == other_id) if other_id is not None else None for g in prediction])
    result["irrelevant_color_correct"] = irrelevant
    return result


def profile_batch(model, root, base, source, batch, token_map, matrices, session):
    import torch
    from experiments.mcqa_c_at_localization import backend
    from experiments.mcqa_c_at_localization.lens import decode
    from experiments.mcqa_c_at_prompt_diagnostic.runner import numeric_check
    ids, src = base[:batch].to("cuda"), source[:batch].to("cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        checks = numeric_check(model, ids, f"IT_profile_batch{batch}", session, torch, backend)
        _, hidden = backend.forward(model, src, capture=[0])
        patch = backend.Patch(0, CHECK_POSITION, hidden[0][:, CHECK_POSITION])
        timings = []
        for _ in range(4):
            torch.cuda.synchronize()
            t = time.monotonic()
            out, _ = backend.forward(model, ids, [patch])
            assert torch.isfinite(out).all()
            del out
            torch.cuda.synchronize()
            timings.append(time.monotonic() - t)
        t = time.monotonic()
        decoded = decode(model, hidden[0], 0, matrices)
        assert torch.isfinite(decoded).all()
        metric_arrays(decoded, read(root / "banks/localization.json")[:batch], token_map)
        del decoded
        torch.cuda.synchronize()
        lens_seconds = time.monotonic() - t
    peak, total_memory = torch.cuda.max_memory_reserved(), torch.cuda.get_device_properties(0).total_memory
    return {"batch": batch, "status": "passed" if peak <= total_memory else "exceeds_dedicated_memory",
            "peak_reserved_bytes": peak, "device_bytes": total_memory, "checks": checks,
            "swap_seconds": float(np.median(timings[1:])), "lens_seconds": lens_seconds}


def profile(model, root, base, source, token_map, matrices, session):
    import torch
    from experiments.mcqa_c_at_localization import backend
    samples = []
    total_memory = torch.cuda.get_device_properties(0).total_memory
    parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    for batch in PROTOCOL["execution"]["batches"]:
        session.check()
        # Three simultaneous FP32 full-sequence softcap tensors are an unavoidable peak.
        lower_bound = parameter_bytes + batch * base.shape[1] * model.config.vocab_size * 12
        if lower_bound > total_memory:
            samples.append({"batch": batch, "status": "not_attempted_dedicated_memory_lower_bound_exceeds_device",
                            "lower_bound_bytes": lower_bound, "device_bytes": total_memory})
            continue
        try:
            gc.collect()
            torch.cuda.empty_cache()
            sample = profile_batch(model, root, base, source, batch, token_map, matrices, session)
        except torch.cuda.OutOfMemoryError:
            sample = {"batch": batch, "status": "CUDA_OOM_not_validated"}
        samples.append(sample)
        gc.collect()
        torch.cuda.empty_cache()
        write(root / "engineering/profile_progress.json", {"batches": samples})
    passing = [s for s in samples if s["status"] == "passed"]
    assert passing, "No validated dedicated-memory batch"
    chosen = max(passing, key=lambda s: s["batch"])
    batch = chosen["batch"]
    ids, src = base[:batch].to("cuda"), source[:batch].to("cuda")
    with torch.no_grad():
        _, hidden = backend.forward(model, src, capture=[0])
        patch = backend.Patch(0, CHECK_POSITION, hidden[0][:, CHECK_POSITION])
        t, count = time.monotonic(), 0
        while time.monotonic() - t < 180:
            session.check()
            out, _ = backend.forward(model, ids, [patch])
            assert torch.isfinite(out).all()
            del out
            count += 1
        torch.cuda.synchronize()
    duration = time.monotonic() - t
    sustained = duration / count
    swap_seconds = max(chosen["swap_seconds"], sustained)
    components = {"full_vector": len(token_map["sites"]) * (384 // batch) * swap_seconds,
                  "J_lens": 26 * (384 // batch) * chosen["lens_seconds"],
                  "activation_cache_and_baselines": 3 * (384 // batch) * swap_seconds}
    result = {"identity": digest(read(root / "registration.json")), "batch": batch, "batches": samples,
              "sustained_seconds": duration, "sustained_forwards": count,
              "sustained_swap_seconds": sustained, "components_seconds": components,
              "remaining_hours_with_allowance": sum(components.values()) * 1.25 / 3600,
              "scope": "Localization only; early-layer DAS/backward profile required before any future DAS training",
              "safety_passed": True}
    write(root / "engineering/profile.json", result, immutable=True)
    return result


def numerical(model, root, base, source, batch, token_map, matrices, session):
    import torch
    from experiments.mcqa_c_at_localization import backend
    from experiments.mcqa_c_at_localization.lens import decode
    from experiments.mcqa_c_at_localization.reference import jonathan_patch
    checks = []
    def check(name, left, right):
        ok = torch.allclose(left, right, atol=2e-4, rtol=2e-4)
        checks.append({"name": name, "passed": ok, "max_abs_difference": float((left.float() - right.float()).abs().max())})
        write(root / "engineering/numerical_progress.json", {"checks": checks})
        if not ok:
            raise RuntimeError("Numerical failure at " + name + "; tolerances unchanged")
    ids, sid = base[:batch].to("cuda"), source[:batch].to("cuda")
    with torch.no_grad():
        clean, states = backend.forward(model, ids, capture=range(26))
        src, sstates = backend.forward(model, sid, capture=range(26))
        check("IT_J_lens_last_block_identity_full_sequence_head", decode(model, states[25], 25, matrices)[:, -1], clean)
        for layer in range(26):
            for pos in (CHECK_POSITION, ids.shape[1] - 1):
                session.check()
                out, _ = backend.forward(model, ids, [backend.Patch(layer, pos, states[layer][:, pos])])
                check(f"identity_L{layer}_P{pos}", out, clean)
        out, _ = backend.forward(model, ids, [backend.Patch(25, CHECK_POSITION, sstates[25][:, CHECK_POSITION])])
        check("final_layer_earlier_token_no_path", out, clean)
        out, _ = backend.forward(model, ids, [backend.Patch(25, ids.shape[1] - 1, sstates[25][:, -1])])
        check("final_layer_last_token_equals_source", out, src)
        controlled_ids = ids.clone()
        controlled_ids[:, CHECK_POSITION] = torch.where(ids[:, CHECK_POSITION] == token_map["color_ids"][0],
                    token_map["color_ids"][1], token_map["color_ids"][0])
        assert torch.all(controlled_ids[:, CHECK_POSITION] != ids[:, CHECK_POSITION])
        controlled, _ = backend.forward(model, controlled_ids)
        embeddings = model.model.embed_tokens(controlled_ids).detach()
        def embedding_swap(_module, _inputs, output):
            changed = output.clone()
            changed[:, CHECK_POSITION] = embeddings[:, CHECK_POSITION]
            return changed
        hook = model.model.embed_tokens.register_forward_hook(embedding_swap)
        try:
            embedded, _ = backend.forward(model, ids)
        finally:
            hook.remove()
        check("single_token_embedding_swap_equals_controlled_clean_source", embedded, controlled)
        del out, src, clean, controlled_ids, controlled, embeddings, embedded
    torch.manual_seed(1)
    for rank in (8, 32, 128, 512):
        subspace = backend.LearnedSubspace(2304, rank).to("cuda")
        q = subspace.basis()
        b, s = states[0][:, CHECK_POSITION].float(), sstates[0][:, CHECK_POSITION].float()
        direct, reference = backend.project(b, s, q), jonathan_patch(b, s, q)
        check(f"Jonathan_patch_rank{rank}", direct, reference)
        probe = torch.randn_like(b)
        gd = torch.autograd.grad((direct * probe).sum(), subspace.raw, retain_graph=True)[0]
        gr = torch.autograd.grad((reference * probe).sum(), subspace.raw)[0]
        check(f"Jonathan_basis_gradient_rank{rank}", gd, gr)
        check(f"orthogonality_rank{rank}", q.T @ q, torch.eye(rank, device=q.device))
        with torch.no_grad():
            q = subspace.basis()
            out, _ = backend.forward(model, ids, [backend.Patch(0, CHECK_POSITION, sstates[0][:, CHECK_POSITION], q)])
            fixed = backend.same_shape_reference(states[0][:, CHECK_POSITION], sstates[0][:, CHECK_POSITION], q, jonathan_patch)
            ref, _ = backend.forward(model, ids, [backend.Patch(0, CHECK_POSITION, fixed)])
            check(f"matched_order_logits_rank{rank}", out, ref)
            del out, ref
    assert not any(l._forward_hooks for l in model.model.layers) and not model.lm_head._forward_hooks
    result = {"identity": digest(read(root / "registration.json")), "batch": batch, "checks": checks,
              "atol": 2e-4, "rtol": 2e-4, "passed": True, "full_sequence_head": True}
    write(root / "engineering/numerical.json", result, immutable=True)
    return result


def cache_states(model, root, base, source, batch, identity, session):
    import torch
    from experiments.mcqa_c_at_localization import backend
    all_states, baselines = {}, []
    for side, ids in (("base", base), ("source", source)):
        result = {l: torch.empty((384, 82, 2304), dtype=torch.bfloat16) for l in range(26)}
        for start in range(0, 384, batch):
            session.check()
            path = root / "cache" / side / f"{start:03d}.pt"
            receipt = path.with_suffix(".json")
            ident = {**identity, "side": side, "start": start}
            if receipt.exists():
                saved = checked_json(receipt, ident)
                chunk = checked_torch(path, saved["sha256"])
                measurements = saved["logits"]
            else:
                with torch.no_grad():
                    logits, states = backend.forward(model, ids[start:start + batch].to("cuda"), capture=range(26))
                    assert torch.isfinite(logits).all()
                    measurements = backend.summarize_logits(logits, identity["label_ids"])
                    rows = read(root / "banks/localization.json")[start:start + batch]
                    expected = [identity["label_ids"][LABELS.index(r[side]["AT"])] for r in rows]
                    assert measurements["prediction"] == expected, "Selected clean-correct bank changed under execution batch; do not refilter"
                    chunk = {l: states[l].detach().cpu() for l in range(26)}
                    assert all(torch.isfinite(t).all() for t in chunk.values())
                    del states, logits
                save_torch(path, chunk)
                write(receipt, {"identity": ident, "sha256": file_hash(path), "logits": measurements}, immutable=True)
            for l in range(26):
                result[l][start:start + batch] = chunk[l]
            if side == "base":
                baselines.append(measurements)
            del chunk
            progress(root, "activation_cache", side=side, done=start + batch, total=384)
        all_states[side] = result
    baseline = {key: sum((v[key] for v in baselines), []) for key in baselines[0]}
    write(root / "localization/clean_baseline.json", {"identity": identity, **baseline}, immutable=True)
    return all_states, baseline


def lens_map(model, root, rows, states, batch, token_map, matrices, identity, session):
    import torch
    from experiments.mcqa_c_at_localization.lens import decode
    scores, mrr = {v: [] for v in ("C", "AT")}, {v: [] for v in ("C", "AT")}
    for layer in range(26):
        chunks = []
        for start in range(0, 384, batch):
            session.check()
            path = root / "localization/lens_chunks" / f"L{layer:02d}_{start:03d}.json"
            ident = {**identity, "layer": layer, "start": start}
            if path.exists():
                chunk = checked_json(path, ident)
            else:
                with torch.no_grad():
                    logits = decode(model, states[layer][start:start + batch].to("cuda"), layer, matrices)
                    assert torch.isfinite(logits).all()
                    chunk = {"identity": ident, **metric_arrays(logits, rows[start:start + batch], token_map)}
                    del logits
                write(path, chunk, immutable=True)
            chunks.append(chunk)
        combined = {"correct": {}, "reciprocal_rank": {}, "identity": {**identity, "layer": layer}}
        for var in ("C", "AT"):
            for key in ("correct", "reciprocal_rank"):
                combined[key][var] = sum((c[key][var] for c in chunks), [])
            for pos in range(1, 82):
                scores[var].append(balanced_decoding([r[pos] for r in combined["correct"][var]],
                    [r["base"][var] for r in rows], [r["condition"] for r in rows]))
                mrr[var].append(float(np.mean([r[pos] for r in combined["reciprocal_rank"][var]])))
        combined["irrelevant_color_correct"] = sum((c["irrelevant_color_correct"] for c in chunks), [])
        write(root / "localization/lens" / f"L{layer:02d}.json", combined, immutable=True)
        progress(root, "J_lens", completed_layers=layer + 1, total_layers=26)
    write(root / "localization/lens_scores.json", {"identity": identity, "scores": scores, "mrr": mrr}, immutable=True)
    return scores, mrr


def swap_map(model, root, rows, base, sources, baseline, batch, token_map, identity, session):
    import torch
    from experiments.mcqa_c_at_localization import backend
    started, newly_done = time.monotonic(), 0
    for i, site in enumerate(token_map["sites"]):
        path = root / "localization/swaps" / f"L{site['layer']:02d}_P{site['position']:02d}.json"
        ident = {**identity, "site": site}
        if path.exists():
            checked_json(path, ident)
            continue
        session.check()
        chunks = []
        with torch.no_grad():
            for start in range(0, 384, batch):
                session.check()
                patch = backend.Patch(site["layer"], site["position"], sources[site["layer"]][start:start + batch, site["position"]])
                logits, _ = backend.forward(model, base[start:start + batch].to("cuda"), [patch])
                assert torch.isfinite(logits).all()
                chunks.append(backend.summarize_logits(logits, token_map["label_ids"]))
                del logits
        measured = {key: sum((c[key] for c in chunks), []) for key in chunks[0]}
        measured["delta_probabilities"] = (np.array(measured["label_probabilities"]) - baseline["label_probabilities"]).tolist()
        measured.update(identity=ident, site=site)
        write(path, measured, immutable=True)
        newly_done += 1
        elapsed = time.monotonic() - started
        progress(root, "full_vector", completed_sites=i + 1, total_sites=len(token_map["sites"]), site=site,
                 measured_seconds_per_site=elapsed / newly_done,
                 remaining_hours=(len(token_map["sites"]) - i - 1) * elapsed / newly_done / 3600,
                 J_lens_complete=True, final_data_sealed=True)


def finish(root):
    from scipy.stats import spearmanr
    reg = verify(root)
    rows, tm = read(root / "banks/localization.json"), read(root / "token_map.json")
    lens = read(root / "localization/lens_scores.json")
    identity = lens["identity"]
    scores, neural = {"C": [], "AT": []}, []
    for site in tm["sites"]:
        measured = checked_json(root / "localization/swaps" / f"L{site['layer']:02d}_P{site['position']:02d}.json", {**identity, "site": site})
        neural.append(measured["delta_probabilities"])
        for var in ("C", "AT"):
            target = np.array([tm["label_ids"][LABELS.index(r["targets"][var])] for r in rows])
            scores[var].append(equal_cell_score(np.array(measured["prediction"]) == target, rows))
    abstract = [[effect(from_record(r["base"]), from_record(r["source"]), var) for r in rows] for var in ("C", "AT")]
    cost = signature_cost(abstract, neural)
    coupling, convergence = one_sided_uot(cost)
    write(root / "localization/plot.json", {"identity": identity, "cost": cost.tolist(), "coupling": coupling.tolist(), "convergence": convergence}, immutable=True)
    maps, rankings, shortlists, comparisons = {}, {}, {}, {}
    for vi, var in enumerate(("C", "AT")):
        maps[var] = {"full_vector": scores[var], "plot": coupling[vi].tolist(), "j_lens": lens["scores"][var]}
        rankings[var], shortlists[var], comparisons[var] = {}, {}, {}
        for method, values in maps[var].items():
            order = rank_sites(tm["sites"], values, lens["mrr"][var] if method == "j_lens" else None)
            rankings[var][method] = [{**tm["sites"][j], "score": float(values[j]), "rank": rank + 1,
                                       **({"mrr": lens["mrr"][var][j]} if method == "j_lens" else {})} for rank, j in enumerate(order)]
            shortlists[var][method] = [tm["sites"][j] for j in order[:2]]
        methods = list(maps[var])
        for a in range(3):
            for b in range(a + 1, 3):
                first, second = methods[a], methods[b]
                top = lambda method: {(s["layer"], s["position"]) for s in rankings[var][method][:10]}
                rho = float(spearmanr(maps[var][first], maps[var][second]).statistic)
                comparisons[var][first + "__" + second] = {"spearman": rho if np.isfinite(rho) else None,
                                                          "top10_overlap": len(top(first) & top(second))}
    write(root / "localization/rankings.json", rankings, immutable=True)
    write(root / "localization/maps.json", {"maps": maps, "comparisons": comparisons}, immutable=True)
    write(root / "selections/shortlists.json", {"shortlists": shortlists, "registration": digest(reg),
          "rankings_sha256": file_hash(root / "localization/rankings.json"), "not_a_heldout_result": True}, immutable=True)
    with (root / "localization/site_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variable", "method", "layer", "position", "score", "rank"])
        for var, methods in rankings.items():
            for method, records in methods.items():
                for record in records:
                    writer.writerow([var, method, record["layer"], record["position"], record["score"], record["rank"]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(20, 9), constrained_layout=True)
    for vi, var in enumerate(("C", "AT")):
        for mi, (method, values) in enumerate(maps[var].items()):
            ax = axes[vi, mi]
            image = ax.imshow(np.asarray(values).reshape(26, 81), aspect="auto", origin="lower", cmap="cividis",
                              vmin=0, vmax=None if method == "plot" else 1, extent=(.5, 81.5, -.5, 25.5))
            ax.set_title(f"{var}: {method} (384 development pairs)")
            ax.set_xlabel("Actual chat token position (BOS excluded)")
            ax.set_ylabel("Decoder block output (0-based)")
            fig.colorbar(image, ax=ax, label="UOT coupling mass" if method == "plot" else "Balanced accuracy")
    fig.savefig(root / "localization/heatmaps.png", dpi=150)
    plt.close(fig)
    artifacts = ["localization/plot.json", "localization/rankings.json", "localization/maps.json",
                 "localization/site_scores.csv", "localization/heatmaps.png", "selections/shortlists.json"]
    write(root / "completion.json", {"registration": digest(reg), "all_2106_sites_complete": True, "all_26_lens_layers_complete": True,
          "PLOT_converged": True, "final_data_used": False, "DAS_training_started": False,
          "lens_provenance_caveat": reg["lens_metadata"]["caveat"],
          "artifacts": {p: file_hash(root / p) for p in artifacts},
          "raw_swap_hashes": {p.name: file_hash(p) for p in sorted((root / "localization/swaps").glob("*.json"))},
          "raw_lens_hashes": {p.name: file_hash(p) for p in sorted((root / "localization/lens").glob("*.json"))}}, immutable=True)
    progress(root, "completed", completed_sites=2106, shortlists=shortlists, final_data_sealed=True, DAS_training_started=False)


def run(root):
    import torch
    from transformers import AutoModelForCausalLM
    from experiments.mcqa_c_at_localization import backend, runtime
    from experiments.mcqa_c_at_localization.local_run import single_worker
    from experiments.mcqa_c_at_localization.lens import load_lens
    from experiments.mcqa_c_at_localization.verification import environment
    reg = verify(root, assets=True)
    session, model = Session(root), None
    torch.set_num_threads(8)
    with single_worker(root):
        try:
            runtime.configure(root)
            progress(root, "loading_IT_model", final_data_sealed=True)
            backend.settings()
            model = AutoModelForCausalLM.from_pretrained(it.MODEL_PATH, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
            for p in model.parameters():
                p.requires_grad_(False)
            assert model.config.hidden_size == 2304 and model.config.num_hidden_layers == 26
            assert model.config._attn_implementation == "sdpa"
            versions = {n: p._version for n, p in model.named_parameters()}
            write(root / "environment.json", environment(), immutable=True)
            rows, tm = read(root / "banks/localization.json"), read(root / "token_map.json")
            base, source = actual_ids(root, rows, "base"), actual_ids(root, rows, "source")
            matrices, _ = load_lens(LENS_PATH)
            path = root / "engineering/profile.json"
            progress(root, "profiling_actual_IT_inputs")
            prof = read(path) if path.exists() else profile(model, root, base, source, tm, matrices, session)
            assert prof["identity"] == digest(reg) and prof["safety_passed"]
            batch = prof["batch"]
            numerical(model, root, base, source, batch, tm, matrices, session)
            progress(root, "numerical_gates_passed", batch=batch, estimated_localization_hours=prof["remaining_hours_with_allowance"])
            identity = {"registration": digest(reg), "batch": batch, "label_ids": tm["label_ids"],
                        "numerical": file_hash(root / "engineering/numerical.json"), "environment": file_hash(root / "environment.json")}
            states, baseline = cache_states(model, root, base, source, batch, identity, session)
            lens_map(model, root, rows, states["base"], batch, tm, matrices, identity, session)
            del matrices, states["base"]
            gc.collect()
            backend.assert_frozen(model, versions)
            swap_map(model, root, rows, base, states["source"], baseline, batch, tm, identity, session)
            backend.assert_frozen(model, versions)
            assert not any(l._forward_hooks for l in model.model.layers)
            del states, model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            finish(root)
        except BaseException as error:
            state = "paused" if isinstance(error, (runtime.Paused, TimeoutError, KeyboardInterrupt)) else "failed"
            path = root / "attempts" / f"{time.time_ns()}.json"
            write(path, {"state": state, "error": str(error), "traceback": traceback.format_exc(), "time": time.time()}, immutable=True)
            progress(root, state, error=str(error), evidence=str(path), final_data_sealed=True)
            raise
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root)
    elif args.command == "verify":
        reg = verify(args.root, assets=True)
        print(json.dumps({"verified": True, "registration": digest(reg)}), flush=True)
    elif args.command == "run":
        run(args.root)
    else:
        finish(args.root)
