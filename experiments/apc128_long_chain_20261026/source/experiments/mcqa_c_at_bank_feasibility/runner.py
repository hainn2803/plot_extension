"""Prove unchanged bank quotas using only pinned-pool natural correctness."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import tarfile
import time
import traceback

from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write
from experiments.mcqa_c_at_localization.protocol import BANKS, LABELS, allocate_groups, order_key
from experiments.mcqa_c_at_localization.preparation import flatten_prompts, select_readout, select_localization
from experiments.mcqa_c_at_localization.banks import audit_pairs, select_pairs
from experiments.mcqa_c_at_prompt_diagnostic import runner as diagnostic
from experiments.mcqa_c_at_query_feasibility.runner import Safety
from experiments.mcqa_c_at_task_search.runner import Budget, tasks
from experiments.mcqa_c_at_it_feasibility import runner as it

ROOT = Path("eval/mcqa_c_at_bank_feasibility_20261017_it_v1")
NAMES = ("train", "calibration", "readout_fit", "readout_validation")
FRONTIERS = (1, 2, 4, 8, 16, 32, 64, 128)
PROTOCOL = {"purpose": "Exact feasibility proof, not a replacement of the original scientific run",
            "banks": NAMES, "frontiers": FRONTIERS,
            "pool": "Unchanged parent 128-alternative pools, groups, states, quotas, and both orientations",
            "selection": "Unchanged select_pairs/audit_pairs/select_readout; no quota relaxation or intervention-based selection",
            "inference": "Pinned instruction-tuned model and frozen winning chat format; strict full-vocabulary single-letter prediction",
            "cache": "No base-model predictions imported; exact source/config/phase identity, scored rows only",
            "final_data": "No test proposals, filtering, inference, or selection",
            "settings": "Frozen BF16, SDPA, TF32 disabled, FP32 softcap, full sequence head, atol=rtol=2e-4",
            "authorization": "works as long as needed to resolve this. you don't need to keep asking me"}


def prefix_pool(pool, n):
    if n not in FRONTIERS:
        raise ValueError("Unregistered proposal frontier")
    return [{"slot": item["slot"], "alternatives": item["alternatives"][:n]} for item in pool]


def audit_readout(rows, name):
    expected = BANKS[name]["prompts"]
    assert len(rows) == expected and len({r["id"] for r in rows}) == expected
    counts = Counter((r["condition"], r["C"], r["AT"]) for r in rows)
    assert len(counts) == 32 and set(counts.values()) == {expected//32}
    assert {tuple(r["objects"]) for r in rows} == {tuple(g) for g in allocate_groups()[name]}
    return {"prompts": len(rows), "groups": len(allocate_groups()[name]),
            "condition_C_AT_counts": {"/".join(k): v for k, v in sorted(counts.items())}}


def sources():
    result = it.sources()
    result.update({p.as_posix(): file_hash(p) for p in Path(__file__).parent.rglob("*.py")})
    return result


def prepare(root):
    it.report(it.ROOT)
    selected = read(it.ROOT/"selection_lock.json")
    parent = read(diagnostic.PARENT/"registration.json")
    hashes = {}
    for name in NAMES:
        path = diagnostic.PARENT/"proposals"/f"{name}.json"
        assert file_hash(path) == parent["proposals"][name]
        pool = read(path)
        assert all(len(item["alternatives"]) == 128 for item in pool)
        hashes[name] = file_hash(path)
    reg = {"protocol": PROTOCOL, "source": sources(), "parent_registration": file_hash(diagnostic.PARENT/"registration.json"),
           "it_registration": file_hash(it.ROOT/"registration.json"), "selection_lock": file_hash(it.ROOT/"selection_lock.json"),
           "it_results": file_hash(it.ROOT/"results.json"), "format": selected["winner"],
           "model_files": read(it.ROOT/"registration.json")["model_files"], "pools": hashes,
           "token_map": read(it.ROOT/"registration.json")["token_maps"][selected["winner"]]}
    write(root/"registration.json", reg, immutable=True)
    if not (root/"source.tar.gz").exists():
        with tarfile.open(root/"source.tar.gz", "w:gz") as tar:
            for p in reg["source"]:
                tar.add(p, arcname=p)
    print(json.dumps({"prepared": True, "model": it.MODEL, "format": reg["format"], "banks": NAMES, "final_data_used": False}), flush=True)


def verify(root):
    reg = read(root/"registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == sources()
    assert reg["parent_registration"] == file_hash(diagnostic.PARENT/"registration.json")
    for filename, field in (("registration.json", "it_registration"), ("selection_lock.json", "selection_lock"), ("results.json", "it_results")):
        assert file_hash(it.ROOT/filename) == reg[field]
    for name, sha in reg["pools"].items():
        assert file_hash(diagnostic.PARENT/"proposals"/f"{name}.json") == sha
    return reg


def run(root):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.mcqa_c_at_localization import backend
    from experiments.mcqa_c_at_localization.local_run import single_worker
    from experiments.mcqa_c_at_localization.runtime import Paused
    from experiments.mcqa_c_at_localization.verification import environment
    import torch
    reg = verify(root)
    assert it.assets() == reg["model_files"]
    model, outcome = None, "failed"
    guard, budget = Safety(root, None), Budget(root/"gpu_budget.json")
    torch.set_num_threads(8)
    with single_worker(root):
        try:
            guard.start()
            budget.start()
            budget.check(50)
            backend.settings()
            tokenizer = AutoTokenizer.from_pretrained(it.MODEL_PATH, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(it.MODEL_PATH, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
            for p in model.parameters():
                p.requires_grad_(False)
            versions = {n: p._version for n, p in model.named_parameters()}
            write(root/"environment.json", environment(), immutable=True)
            checked, timing, selected_ids = {}, {}, set()
            with torch.no_grad():
                for name in NAMES:
                    pool = read(diagnostic.PARENT/"proposals"/f"{name}.json")
                    readout = name.startswith("readout")
                    known = {}
                    solved = False
                    for frontier in FRONTIERS:
                        proposals = prefix_pool(pool, frontier)
                        prompts = flatten_prompts(proposals, readout)
                        rows = []
                        for p in prompts:
                            if p["id"] in known:
                                continue
                            text = it.render(tokenizer, p["text"], reg["format"])
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            assert ids[0] == tokenizer.bos_token_id and ids.count(tokenizer.bos_token_id) == 1
                            for letter in LABELS:
                                assert tokenizer.encode(text+it.continuation(reg["format"], letter), add_special_tokens=False) == ids+[reg["token_map"][letter]]
                            rows.append({"id": p["id"], "prompt": p, "text": text, "input_ids": {reg["format"]: ids}})
                        phase = {"registration": digest(reg), "bank": name, "frontier": frontier, "rows": rows}
                        write(root/"phases"/f"{name}_{frontier}.json", phase, immutable=True)
                        phase_hash = digest(phase)
                        index = {r["id"]: r for r in rows}
                        queue = tasks(rows, reg["format"])

                        def inputs(task):
                            return torch.tensor([index[i]["input_ids"][reg["format"]] for i in task["physical"]], device="cuda")

                        for task in queue:
                            shape = tuple(task["shape"])
                            if shape not in checked:
                                guard.guard()
                                ids = inputs(task)
                                checked[shape] = diagnostic.numeric_check(model, ids, f"{name}/{frontier}", budget, torch, backend)
                                write(root/"numerical.json", {"passed": True, "checks": list(checked.values())})
                                torch.cuda.synchronize()
                                start = time.monotonic()
                                for _ in range(3):
                                    logits, _ = backend.forward(model, ids)
                                    del logits
                                torch.cuda.synchronize()
                                timing[shape] = (time.monotonic()-start)/3
                        remaining = [(i, t) for i, t in enumerate(queue) if not (root/"predictions"/name/str(frontier)/f"{i:05d}.json").exists()]
                        estimate = sum(timing[tuple(t["shape"])] for _, t in remaining)*1.25
                        write(root/"profiles"/f"{name}_{frontier}.json", {"remaining_seconds_with_allowance": estimate,
                              "budget_remaining": budget.remaining(), "new_prompts": len(rows)})
                        budget.check(20+estimate)
                        for i, task in enumerate(queue):
                            path = root/"predictions"/name/str(frontier)/f"{i:05d}.json"
                            identity = {"registration": digest(reg), "phase": phase_hash, "task": task}
                            if path.exists():
                                saved = read(path)
                                assert saved["identity"] == identity
                            else:
                                guard.guard()
                                budget.check(20+4*timing[tuple(task["shape"])])
                                logits, _ = backend.forward(model, inputs(task))
                                assert torch.isfinite(logits).all()
                                measurement = backend.summarize_logits(logits, [reg["token_map"][l] for l in LABELS])
                                del logits
                                records = []
                                for offset, item in enumerate(task["ids"]):
                                    pred = measurement["prediction"][offset]
                                    target = index[item]["prompt"]["AT"]
                                    records.append({"id": item, "prediction": pred, "target": target,
                                                    "accepted": pred == reg["token_map"][target],
                                                    "label_logits": measurement["label_logits"][offset]})
                                saved = {"identity": identity, "rows": records}
                                write(path, saved, immutable=True)
                            assert [r["id"] for r in saved["rows"]] == task["ids"]
                            for record in saved["rows"]:
                                assert record["accepted"] == (record["prediction"] == reg["token_map"][index[record["id"]]["prompt"]["AT"]])
                                assert record["id"] not in known
                                known[record["id"]] = record["accepted"]
                            if (i+1) % 128 == 0:
                                progress = {"status": "filtering", "bank": name, "frontier": frontier,
                                            "new_done": min((i+1)*task["shape"][0], len(rows)), "new_total": len(rows), "GPU_seconds": budget.elapsed()}
                                write(root/"status.json", progress)
                                print(json.dumps(progress), flush=True)
                        try:
                            guard.guard()
                            budget.check(20)
                            selected = select_readout(proposals, known) if readout else select_pairs(proposals, known)
                            coverage = audit_readout(selected, name) if readout else audit_pairs(selected, name, "preserve")
                            guard.guard()
                            budget.check(20)
                        except ValueError as error:
                            write(root/"frontiers"/f"{name}_{frontier}.json", {"status": "insufficient_known_clean_candidates", "reason": str(error),
                                  "measured": len(known), "accepted": sum(known.values()), "remaining_frozen_alternatives": 128-frontier}, immutable=True)
                            print(json.dumps({"bank": name, "frontier": frontier, "reason": str(error), "measured": len(known)}), flush=True)
                            continue
                        selected_prompts = selected if readout else [p[side] for p in selected for side in ("base", "source")]
                        unique = {p["id"]: p for p in selected_prompts}
                        assert all(known.get(i) is True for i in unique)
                        assert not selected_ids & set(unique)
                        selected_ids.update(unique)
                        write(root/"selected"/f"{name}.json", {"registration": digest(reg), "model": it.MODEL,
                              "revision": it.REVISION, "format": reg["format"], "frontier": frontier,
                              "logical_rows": selected, "coverage": coverage, "not_adopted_into_original_run": True}, immutable=True)
                        rendered = [{"id": p["id"], "text": it.render(tokenizer, p["text"], reg["format"]),
                                     "input_ids": tokenizer.encode(it.render(tokenizer, p["text"], reg["format"]), add_special_tokens=False),
                                     "expected_label": p["AT"], "expected_token": reg["token_map"][p["AT"]]} for p in unique.values()]
                        write(root/"selected"/f"{name}_inputs.json", rendered, immutable=True)
                        if name == "train":
                            subset = select_localization(selected)
                            write(root/"selected/localization.json", subset, immutable=True)
                        write(root/"frontiers"/f"{name}_{frontier}.json", {"status": "exact_quotas_passed", "retained": len(selected),
                              "measured": len(known), "accepted": sum(known.values())}, immutable=True)
                        print(json.dumps({"bank": name, "frontier": frontier, "exact_quotas_passed": True, "retained": len(selected), "measured": len(known)}), flush=True)
                        solved = True
                        break
                    if not solved:
                        raise RuntimeError(f"Complete fixed pool cannot fill {name}; no requirements relaxed")
                backend.assert_frozen(model, versions)
                assert not model.lm_head._forward_hooks and not any(l._forward_hooks for l in model.model.layers)
                write(root/"completion.json", {"registration": digest(reg), "all_four_unsealed_banks_passed": True,
                      "no_final_data_used": True, "frozen_parameters": True, "hooks_clean": True,
                      "prediction_hashes": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root/"predictions").rglob("*.json"))},
                      "selected_hashes": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root/"selected").glob("*.json"))}}, immutable=True)
                outcome = "completed"
        except BaseException as error:
            outcome = "interrupted" if isinstance(error, (Paused, TimeoutError, KeyboardInterrupt)) else "failed"
            write(root/"error.json", {"status": outcome, "error": str(error), "traceback": traceback.format_exc()})
            raise
        finally:
            guard.close()
            if model is not None:
                del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            budget.close()
            write(root/"status.json", {"status": outcome, "GPU_seconds": budget.state["spent_seconds"]})


def report(root):
    reg = verify(root)
    receipt = read(root/"completion.json")
    assert receipt["registration"] == digest(reg) and read(root/"status.json")["status"] == "completed"
    for field in ("prediction_hashes", "selected_hashes"):
        for name, sha in receipt[field].items():
            assert file_hash(root/name) == sha
    summary, all_groups, all_ids = {}, set(), set()
    for name in NAMES:
        artifact = read(root/"selected"/f"{name}.json")
        selected = artifact["logical_rows"]
        readout = name.startswith("readout")
        coverage = audit_readout(selected, name) if readout else audit_pairs(selected, name, "preserve")
        assert digest(coverage) == digest(artifact["coverage"])
        groups = {tuple(g) for g in allocate_groups()[name]}
        assert not all_groups & groups
        all_groups.update(groups)
        rendered = read(root/"selected"/f"{name}_inputs.json")
        assert not all_ids & {p["id"] for p in rendered}
        all_ids.update(p["id"] for p in rendered)
        accepted = {}
        for phase_path in sorted((root/"phases").glob(f"{name}_*.json")):
            phase = read(phase_path)
            if phase["frontier"] > artifact["frontier"]:
                continue
            assert phase["registration"] == digest(reg)
            phase_hash = digest(phase)
            ids = {r["id"]: r for r in phase["rows"]}
            for i, task in enumerate(tasks(phase["rows"], reg["format"])):
                saved = read(root/"predictions"/name/str(phase["frontier"])/f"{i:05d}.json")
                assert saved["identity"] == {"registration": digest(reg), "phase": phase_hash, "task": task}
                assert [r["id"] for r in saved["rows"]] == task["ids"]
                for r in saved["rows"]:
                    correct = r["prediction"] == reg["token_map"][ids[r["id"]]["prompt"]["AT"]]
                    assert correct == r["accepted"] and r["id"] not in accepted
                    accepted[r["id"]] = correct
        assert all(accepted.get(p["id"]) is True for p in rendered)
        summary[name] = {"retained": len(selected), "groups": len(groups), "frontier": artifact["frontier"],
                         "prompts_measured": len(accepted), "clean_prompts": sum(accepted.values()), "quotas_verified": True}
    localization = read(root/"selected/localization.json")
    assert len(localization) == 384
    train_ids = {p["id"] for p in read(root/"selected/train.json")["logical_rows"]}
    assert all(p["id"] in train_ids for p in localization)
    output = {"status": "verified", "model": it.MODEL, "revision": it.REVISION, "format": reg["format"],
              "banks": summary, "localization_pairs": len(localization), "final_test_used": False,
              "historical_run_or_banks_modified": False, "DAS_started": False,
              "meaning": "Exact clean-bank feasibility under an alternate, identified checkpoint/chat format, not evidence of a causal handle or a completed scientific comparison"}
    write(root/"results.json", output, immutable=True)
    print(json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.root)
        print({"verified": True})
    else:
        globals()[args.command](args.root)
