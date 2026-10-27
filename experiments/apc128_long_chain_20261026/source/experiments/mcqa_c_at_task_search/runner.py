"""Strict next-token task search with immutable evidence and thermal protection."""
import argparse
from collections import defaultdict
import gc
from pathlib import Path
import json
import tarfile
import time
import traceback

from experiments.mcqa_c_at_localization.data import from_record
from experiments.mcqa_c_at_localization.protocol import LABELS, MODEL_PATH
from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write
from experiments.mcqa_c_at_prompt_diagnostic import runner as diagnostic
from experiments.mcqa_c_at_query_feasibility.runner import Safety
from experiments.mcqa_c_at_format_search.runner import metrics
from .formats import FORMS, continuation, render

ROOT = Path("eval/mcqa_c_at_task_search_20261017_v1")
FIRST = Path("eval/mcqa_c_at_format_search_20261017_v1")
CAP = 3600
PROTOCOL = {"scope": "Autonomous training-only prompt exploration; no DAS, final data, or scientific-bank replacement",
            "model": "Same pinned frozen Gemma-2-2B, BF16, SDPA, TF32 disabled, FP32 softcap, full sequence head",
            "formats": FORMS, "GPU_lease_seconds": CAP,
            "authorization": "works as long as needed to resolve this. you don't need to keep asking me",
            "previous_budget": "Preserved separately; this additional authorized allocation does not reset its ledger",
            "scoring": "Full-vocabulary next-token argmax equals registered, prefix-specific, single letter token; no constrained decoding or whitespace stripping",
            "panels": "Same 768 matched training prompts for search; 1536 on disjoint training objects for confirmation; frozen first alternatives for 768 original bank slots",
            "selection": "Fewest empty order/condition pair cells, then best minimum pair accuracy, mean pair accuracy, natural accuracy, listed order",
            "batch": "No padding; exact length buckets: 8 through 110 tokens, 4 through 260, 2 through 520, otherwise 1; filler never scored",
            "atol": 2e-4, "rtol": 2e-4, "cleanup_reserve_seconds": 35}


class Budget(diagnostic.Budget):
    def start(self):
        if self.state["spent_seconds"] >= CAP or self.state.get("active"):
            raise RuntimeError("Diagnostic GPU lease exhausted or active")
        self.started = self.clock()
        self.state.update(active=True, started_utc=time.time(), reserved_seconds=CAP-self.state["spent_seconds"], cap=CAP)
        write(self.path, self.state)

    def remaining(self):
        return CAP-self.elapsed()

    def check(self, required=0):
        if self.remaining() <= 15+required:
            raise TimeoutError("Stopping before the registered additional GPU lease cap")


def sources():
    paths = set(Path(__file__).parent.rglob("*.py"))
    for name in ("mcqa_c_at_localization", "mcqa_c_at_prompt_diagnostic", "mcqa_c_at_format_search"):
        paths.update(Path("experiments", name).rglob("*.py"))
    paths.add(Path("experiments/mcqa_c_at_query_feasibility/runner.py"))
    return {p.as_posix(): file_hash(p) for p in sorted(paths)}


def tasks(rows, form):
    buckets = defaultdict(list)
    for row in rows:
        buckets[len(row["input_ids"][form])].append(row["id"])
    result = []
    for length, ids in sorted(buckets.items()):
        ids.sort(key=digest)
        batch = 8 if length <= 110 else 4 if length <= 260 else 2 if length <= 520 else 1
        for start in range(0, len(ids), batch):
            scored = ids[start:start+batch]
            result.append({"ids": scored, "physical": scored+[ids[0]]*(batch-len(scored)), "shape": [batch, length]})
    return result


def prepare(root):
    from transformers import AutoTokenizer
    from experiments.mcqa_c_at_localization.verification import asset_hashes
    from experiments.mcqa_c_at_format_search.runner import verify as verify_first
    verify_first(FIRST)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    token_maps, hashes, lengths = {}, {}, {}
    for stage in ("screen", "confirm", "bank_probe"):
        rows = []
        for old in read(FIRST / f"{stage}.json"):
            row = {k: old[k] for k in ("id", "prompt", "pair") if k in old}
            p = from_record(row["prompt"])
            row.update(texts={}, input_ids={})
            for form in FORMS:
                text = render(p, form)
                ids = tokenizer.encode(text, add_special_tokens=True)
                token_map = {}
                for letter in LABELS:
                    continued = tokenizer.encode(text+continuation(form, letter), add_special_tokens=True)
                    if continued[:len(ids)] != ids or len(continued) != len(ids)+1:
                        raise ValueError(f"Non-single-token or retokenized continuation at {form}/{letter}")
                    token_map[letter] = continued[-1]
                if form in token_maps:
                    assert token_map == token_maps[form]
                token_maps[form] = token_map
                row["texts"][form], row["input_ids"][form] = text, ids
            rows.append(row)
        write(root/f"{stage}.json", rows, immutable=True)
        hashes[stage] = file_hash(root/f"{stage}.json")
        lengths[stage] = {f: sorted({len(r["input_ids"][f]) for r in rows}) for f in FORMS}
    write(root / "probe_pairs.json", read(FIRST / "probe_pairs.json"), immutable=True)
    assets = asset_hashes()
    assert assets == read(diagnostic.PARENT / "registration.json")["model_files"]
    reg = {"protocol": PROTOCOL, "source": sources(), "banks": hashes, "token_maps": token_maps,
           "model_files": assets, "lengths": lengths, "first_registration": file_hash(FIRST/"registration.json"),
           "probe_pairs": file_hash(root/"probe_pairs.json"),
           "previous_budget": read(diagnostic.ROOT / "gpu_budget.json")}
    write(root / "registration.json", reg, immutable=True)
    if not (root / "source.tar.gz").exists():
        with tarfile.open(root / "source.tar.gz", "w:gz") as tar:
            for p in reg["source"]:
                tar.add(p, arcname=p)
    print(json.dumps({"status": "prepared", "lengths": lengths, "token_maps": token_maps}), flush=True)


def verify(root):
    reg = read(root/"registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == sources()
    assert reg["first_registration"] == file_hash(FIRST/"registration.json")
    assert reg["probe_pairs"] == file_hash(root/"probe_pairs.json")
    banks = {}
    for stage, sha in reg["banks"].items():
        assert sha == file_hash(root/f"{stage}.json")
        banks[stage] = read(root/f"{stage}.json")
    return reg, banks


def ranking(m, form):
    return (m["empty_pair_cells"], -m["worst_pair_accuracy"], -m["mean_pair_accuracy"], -m["accuracy"], FORMS.index(form))


def run(root):
    from experiments.mcqa_c_at_localization import backend
    from experiments.mcqa_c_at_localization.verification import asset_hashes, environment
    from experiments.mcqa_c_at_localization.local_run import single_worker
    from experiments.mcqa_c_at_localization.runtime import Paused
    import torch
    reg, banks = verify(root)
    assert asset_hashes() == reg["model_files"]
    torch.set_num_threads(8)
    budget, guard = Budget(root/"gpu_budget.json"), Safety(root, None)
    model, outcome = None, "failed"
    with single_worker(root):
        try:
            guard.start()
            budget.start()
            budget.check(50)
            model, tokenizer = backend.load_model()
            versions = {n: p._version for n, p in model.named_parameters()}
            write(root/"environment.json", environment(), immutable=True)
            indexed = {name: {r["id"]: r for r in rows} for name, rows in banks.items()}
            checks, timings = {}, {}

            def forward_input(stage, form, task):
                return torch.tensor([indexed[stage][i]["input_ids"][form] for i in task["physical"]], device="cuda")

            def evaluate(stage, form):
                queue = tasks(banks[stage], form)
                for task in queue:
                    shape = tuple(task["shape"])
                    if shape not in checks:
                        guard.guard()
                        x = forward_input(stage, form, task)
                        checks[shape] = diagnostic.numeric_check(model, x, f"{stage}/{form}", budget, torch, backend)
                        write(root/"numerical.json", {"status": "passed", "checks": list(checks.values())})
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        for _ in range(3):
                            logits, _ = backend.forward(model, x)
                            del logits
                        torch.cuda.synchronize()
                        timings[shape] = (time.monotonic()-start)/3
                estimate = sum(timings[tuple(t["shape"])] for i, t in enumerate(queue)
                               if not (root/"predictions"/stage/form/f"{i:04d}.json").exists())*1.25
                write(root/"profiles"/f"{stage}_{form}.json", {"estimated_remaining_seconds_with_25pct": estimate,
                      "budget_remaining": budget.remaining(), "timings": {str(k): v for k, v in timings.items()}})
                budget.check(20+estimate)
                predictions = {}
                tokens = reg["token_maps"][form]
                for i, task in enumerate(queue):
                    path = root/"predictions"/stage/form/f"{i:04d}.json"
                    identity = {"registration": digest(reg), "stage": stage, "format": form, "task": task}
                    if path.exists():
                        saved = read(path)
                        assert saved["identity"] == identity
                    else:
                        guard.guard()
                        budget.check(20+4*timings[tuple(task["shape"])])
                        logits, _ = backend.forward(model, forward_input(stage, form, task))
                        assert torch.isfinite(logits).all()
                        measured = backend.summarize_logits(logits, [tokens[l] for l in LABELS])
                        del logits
                        records = []
                        for offset, item in enumerate(task["ids"]):
                            p = indexed[stage][item]["prompt"]
                            pred = measured["prediction"][offset]
                            records.append({"id": item, "prediction": pred, "correct": pred == tokens[p["AT"]],
                                            "target": p["AT"], "decoded": tokenizer.decode([pred]),
                                            "label_logits": measured["label_logits"][offset]})
                        saved = {"identity": identity, "rows": records}
                        write(path, saved, immutable=True)
                    assert [r["id"] for r in saved["rows"]] == task["ids"]
                    predictions.update({r["id"]: r for r in saved["rows"]})
                    if (i+1) % 64 == 0:
                        write(root/"status.json", {"status": "running", "stage": stage, "format": form,
                              "prompts_done": len(predictions), "prompts_total": len(banks[stage]), "GPU_seconds": budget.elapsed()})
                assert len(predictions) == len(banks[stage])
                m = metrics(banks[stage], predictions)
                write(root/"metrics"/f"{stage}_{form}.json", m, immutable=True)
                print(json.dumps({"stage": stage, "format": form, "accuracy": m["accuracy"],
                      "empty_pair_cells": m.get("empty_pair_cells"), "worst_pair_accuracy": m.get("worst_pair_accuracy"),
                      "GPU_seconds": budget.elapsed()}), flush=True)
                return m, predictions

            with torch.no_grad():
                screening = {}
                for form in FORMS:
                    screening[form] = evaluate("screen", form)[0]
                winner = min(FORMS, key=lambda f: ranking(screening[f], f))
                write(root/"selection_lock.json", {"registration": digest(reg), "winner": winner,
                      "criterion": PROTOCOL["selection"], "metrics": {f: file_hash(root/"metrics"/f"screen_{f}.json") for f in FORMS}}, immutable=True)
                evaluate("confirm", winner)
                _, predictions = evaluate("bank_probe", winner)
                pair_cells = defaultdict(lambda: [0, 0])
                for p in read(root/"probe_pairs.json"):
                    cell = p["family"]+"/"+p["condition"]
                    pair_cells[cell][0] += int(all(predictions[p[s]["id"]]["correct"] for s in ("base", "source")))
                    pair_cells[cell][1] += 1
                write(root/"probe_pair_metrics.json", dict(pair_cells), immutable=True)
                backend.assert_frozen(model, versions)
                assert not model.lm_head._forward_hooks and not any(l._forward_hooks for l in model.model.layers)
                write(root/"completion.json", {"registration": digest(reg), "winner": winner,
                      "frozen_parameters": True, "hooks_clean": True,
                      "prediction_hashes": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root/"predictions").rglob("*.json"))}}, immutable=True)
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
    reg, banks = verify(root)
    result = {"scope": PROTOCOL["scope"], "stages": {}, "complete": False}
    for stage, rows in banks.items():
        index = {r["id"]: r for r in rows}
        for form in FORMS:
            predictions, complete = {}, True
            for i, task in enumerate(tasks(rows, form)):
                path = root/"predictions"/stage/form/f"{i:04d}.json"
                if not path.exists():
                    complete = False
                    continue
                saved = read(path)
                assert saved["identity"] == {"registration": digest(reg), "stage": stage, "format": form, "task": task}
                assert [r["id"] for r in saved["rows"]] == task["ids"]
                for r in saved["rows"]:
                    assert r["id"] not in predictions
                    assert r["correct"] == (r["prediction"] == reg["token_maps"][form][index[r["id"]]["prompt"]["AT"]])
                    predictions[r["id"]] = r
            if complete:
                m = metrics(rows, predictions)
                assert m == read(root/"metrics"/f"{stage}_{form}.json")
                result["stages"][stage+"/"+form] = m
    if (root/"completion.json").exists():
        receipt = read(root/"completion.json")
        assert receipt["registration"] == digest(reg) and read(root/"status.json")["status"] == "completed"
        for name, sha in receipt["prediction_hashes"].items():
            assert file_hash(root/name) == sha
        result["complete"] = True
    write(root/("results.json" if result["complete"] else "partial_results.json"), result, immutable=result["complete"])
    print(json.dumps({"complete": result["complete"], "stages": {k: {f: v[f] for f in ("accuracy", "empty_pair_cells", "worst_pair_accuracy", "mean_pair_accuracy") if f in v} for k, v in result["stages"].items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "verify":
        _, banks = verify(args.root)
        print({"verified": True, "banks": {k: len(v) for k, v in banks.items()}})
    else:
        globals()[args.command](args.root)
