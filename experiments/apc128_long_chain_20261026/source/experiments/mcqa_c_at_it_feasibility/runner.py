"""Natural MCQA checks with the official pinned instruction-tuned checkpoint."""
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import tarfile
import time
import traceback

from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write
from experiments.mcqa_c_at_localization.protocol import LABELS
from experiments.mcqa_c_at_query_feasibility.runner import Safety
from experiments.mcqa_c_at_prompt_diagnostic.runner import numeric_check
from experiments.mcqa_c_at_task_search.runner import Budget, tasks
from experiments.mcqa_c_at_format_search.runner import metrics

MODEL = "google/gemma-2-2b-it"
REVISION = "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
MODEL_PATH = Path(".cache_mcqa_c_at/hub/models--google--gemma-2-2b-it/snapshots")/REVISION
ROOT = Path("eval/mcqa_c_at_it_feasibility_20261017_v1")
FIRST = Path("eval/mcqa_c_at_format_search_20261017_v1")
FORMS = ("chat_original", "chat_explicit", "chat_prefill")
INSTRUCTION = ("Answer with exactly one option label: A, B, C, or D. "
               "The option labels may appear in any order. Use the label printed next to the requested color.")
PROTOCOL = {"scope": "Alternate-checkpoint development diagnostic only; no scientific banks changed, no DAS or final data",
            "model": MODEL, "revision": REVISION, "forms": FORMS, "instruction": INSTRUCTION,
            "chat": "Official tokenizer.apply_chat_template, one user turn, add_generation_prompt=True, no duplicate BOS",
            "numerics": "Frozen BF16, SDPA, TF32 off, FP32 softcap, full-sequence head; atol=rtol=2e-4",
            "selection": "Fewest empty order/condition pair cells, best worst pair rate, mean pair rate, natural accuracy, listed order",
            "scoring": "Full-vocabulary next-token argmax equals declared prefix-specific token; no constrained decoding",
            "panels": "Same 768 screen, disjoint-object 1536 confirmation, and 768 frozen first-alternative training pairs",
            "authorization": "works as long as needed to resolve this. you don't need to keep asking me",
            "thermal_and_budget": "Same persistent-throttling and AC guards; separate recorded 3600-second diagnostic lease"}


def render(tokenizer, original, form):
    content = original if form == "chat_original" else INSTRUCTION+"\n\n"+original
    text = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    return text+("The answer is" if form == "chat_prefill" else "")


def continuation(form, letter):
    return (" " if form == "chat_prefill" else "")+letter


def assets():
    paths = sorted(p for p in MODEL_PATH.iterdir() if p.is_file())
    index = read(MODEL_PATH/"model.safetensors.index.json")
    assert set(index["weight_map"].values()) <= {p.name for p in paths}
    return {p.name: file_hash(p) for p in paths}


def sources():
    paths = set(Path(__file__).parent.rglob("*.py"))
    for name in ("mcqa_c_at_localization", "mcqa_c_at_prompt_diagnostic", "mcqa_c_at_task_search", "mcqa_c_at_format_search"):
        paths.update(Path("experiments", name).rglob("*.py"))
    paths.add(Path("experiments/mcqa_c_at_query_feasibility/runner.py"))
    return {p.as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(root):
    from transformers import AutoTokenizer
    from experiments.mcqa_c_at_format_search.runner import verify as verify_first
    verify_first(FIRST)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    hashes, token_maps, shapes = {}, {}, {}
    for stage in ("screen", "confirm", "bank_probe"):
        rows = []
        for old in read(FIRST/f"{stage}.json"):
            row = {k: old[k] for k in ("id", "prompt", "pair") if k in old}
            row.update(texts={}, input_ids={})
            for form in FORMS:
                text = render(tokenizer, row["prompt"]["text"], form)
                ids = tokenizer.encode(text, add_special_tokens=False)
                assert ids[0] == tokenizer.bos_token_id and ids.count(tokenizer.bos_token_id) == 1
                tokens = {}
                for letter in LABELS:
                    after = tokenizer.encode(text+continuation(form, letter), add_special_tokens=False)
                    assert after[:len(ids)] == ids and len(after) == len(ids)+1
                    tokens[letter] = after[-1]
                if form in token_maps:
                    assert tokens == token_maps[form]
                token_maps[form] = tokens
                row["texts"][form], row["input_ids"][form] = text, ids
            rows.append(row)
        write(root/f"{stage}.json", rows, immutable=True)
        hashes[stage] = file_hash(root/f"{stage}.json")
        shapes[stage] = {f: sorted({len(r["input_ids"][f]) for r in rows}) for f in FORMS}
    write(root/"probe_pairs.json", read(FIRST/"probe_pairs.json"), immutable=True)
    reg = {"protocol": PROTOCOL, "source": sources(), "model_files": assets(), "banks": hashes,
           "token_maps": token_maps, "shapes": shapes, "first_registration": file_hash(FIRST/"registration.json"),
           "probe_pairs": file_hash(root/"probe_pairs.json")}
    write(root/"registration.json", reg, immutable=True)
    if not (root/"source.tar.gz").exists():
        with tarfile.open(root/"source.tar.gz", "w:gz") as tar:
            for p in reg["source"]:
                tar.add(p, arcname=p)
    print(json.dumps({"prepared": True, "model": MODEL, "revision": REVISION, "shapes": shapes, "token_maps": token_maps}), flush=True)


def verify(root):
    reg = read(root/"registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == sources()
    assert reg["first_registration"] == file_hash(FIRST/"registration.json")
    assert reg["probe_pairs"] == file_hash(root/"probe_pairs.json")
    banks = {}
    for stage, sha in reg["banks"].items():
        assert file_hash(root/f"{stage}.json") == sha
        banks[stage] = read(root/f"{stage}.json")
    return reg, banks


def rank(m, form):
    return (m["empty_pair_cells"], -m["worst_pair_accuracy"], -m["mean_pair_accuracy"], -m["accuracy"], FORMS.index(form))


def run(root):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.mcqa_c_at_localization import backend
    from experiments.mcqa_c_at_localization.local_run import single_worker
    from experiments.mcqa_c_at_localization.verification import environment
    from experiments.mcqa_c_at_localization.runtime import Paused
    import torch
    reg, banks = verify(root)
    assert assets() == reg["model_files"]
    guard, budget = Safety(root, None), Budget(root/"gpu_budget.json")
    model, outcome = None, "failed"
    torch.set_num_threads(8)
    with single_worker(root):
        try:
            guard.start()
            budget.start()
            budget.check(50)
            backend.settings()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True,
                    torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
            for p in model.parameters():
                p.requires_grad_(False)
            assert model.config.hidden_size == 2304 and model.config.num_hidden_layers == 26
            assert model.config._attn_implementation == "sdpa"
            versions = {n: p._version for n, p in model.named_parameters()}
            write(root/"environment.json", environment(), immutable=True)
            checked, timing = {}, {}
            indexed = {s: {r["id"]: r for r in rows} for s, rows in banks.items()}

            def inputs(stage, form, task):
                return torch.tensor([indexed[stage][i]["input_ids"][form] for i in task["physical"]], device="cuda")

            def evaluate(stage, form):
                queue = tasks(banks[stage], form)
                for task in queue:
                    shape = tuple(task["shape"])
                    if shape not in checked:
                        guard.guard()
                        ids = inputs(stage, form, task)
                        checked[shape] = numeric_check(model, ids, f"{stage}/{form}", budget, torch, backend)
                        write(root/"numerical.json", {"passed": True, "checks": list(checked.values())})
                        torch.cuda.synchronize()
                        start = time.monotonic()
                        for _ in range(3):
                            logits, _ = backend.forward(model, ids)
                            del logits
                        torch.cuda.synchronize()
                        timing[shape] = (time.monotonic()-start)/3
                estimate = sum(timing[tuple(t["shape"])] for i, t in enumerate(queue)
                    if not (root/"predictions"/stage/form/f"{i:04d}.json").exists())*1.25
                budget.check(20+estimate)
                predictions, tokens = {}, reg["token_maps"][form]
                for i, task in enumerate(queue):
                    path = root/"predictions"/stage/form/f"{i:04d}.json"
                    identity = {"registration": digest(reg), "stage": stage, "format": form, "task": task}
                    if path.exists():
                        saved = read(path)
                        assert saved["identity"] == identity
                    else:
                        guard.guard()
                        budget.check(20+4*timing[tuple(task["shape"])])
                        logits, _ = backend.forward(model, inputs(stage, form, task))
                        assert torch.isfinite(logits).all()
                        summary = backend.summarize_logits(logits, [tokens[l] for l in LABELS])
                        del logits
                        records = []
                        for offset, item in enumerate(task["ids"]):
                            p = indexed[stage][item]["prompt"]
                            pred = summary["prediction"][offset]
                            records.append({"id": item, "prediction": pred, "correct": pred == tokens[p["AT"]],
                                            "target": p["AT"], "decoded": tokenizer.decode([pred]),
                                            "label_logits": summary["label_logits"][offset]})
                        saved = {"identity": identity, "rows": records}
                        write(path, saved, immutable=True)
                    assert [r["id"] for r in saved["rows"]] == task["ids"]
                    predictions.update({r["id"]: r for r in saved["rows"]})
                assert len(predictions) == len(banks[stage])
                m = metrics(banks[stage], predictions)
                write(root/"metrics"/f"{stage}_{form}.json", m, immutable=True)
                print(json.dumps({"stage": stage, "format": form, "accuracy": m["accuracy"],
                      "empty_pair_cells": m.get("empty_pair_cells"), "worst_pair_accuracy": m.get("worst_pair_accuracy")}), flush=True)
                return m, predictions

            with torch.no_grad():
                screened = {f: evaluate("screen", f)[0] for f in FORMS}
                winner = min(FORMS, key=lambda f: rank(screened[f], f))
                write(root/"selection_lock.json", {"registration": digest(reg), "winner": winner,
                      "metrics_hashes": {f: file_hash(root/"metrics"/f"screen_{f}.json") for f in FORMS}}, immutable=True)
                evaluate("confirm", winner)
                _, predictions = evaluate("bank_probe", winner)
                cells = defaultdict(lambda: [0, 0])
                for p in read(root/"probe_pairs.json"):
                    cell = p["family"]+"/"+p["condition"]
                    cells[cell][0] += int(all(predictions[p[s]["id"]]["correct"] for s in ("base", "source")))
                    cells[cell][1] += 1
                write(root/"probe_pair_metrics.json", dict(cells), immutable=True)
                backend.assert_frozen(model, versions)
                assert not model.lm_head._forward_hooks and not any(l._forward_hooks for l in model.model.layers)
                write(root/"completion.json", {"registration": digest(reg), "winner": winner, "frozen_parameters": True,
                      "hooks_clean": True, "prediction_hashes": {p.relative_to(root).as_posix(): file_hash(p)
                      for p in sorted((root/"predictions").rglob("*.json"))}}, immutable=True)
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
    receipt = read(root/"completion.json")
    assert receipt["registration"] == digest(reg) and read(root/"status.json")["status"] == "completed"
    for name, sha in receipt["prediction_hashes"].items():
        assert file_hash(root/name) == sha
    output = {"model": MODEL, "revision": REVISION, "winner": receipt["winner"], "stages": {}}
    for stage, rows in banks.items():
        indexed = {r["id"]: r for r in rows}
        forms = FORMS if stage == "screen" else (receipt["winner"],)
        for form in forms:
            predictions = {}
            for i, task in enumerate(tasks(rows, form)):
                saved = read(root/"predictions"/stage/form/f"{i:04d}.json")
                assert saved["identity"] == {"registration": digest(reg), "stage": stage, "format": form, "task": task}
                assert [r["id"] for r in saved["rows"]] == task["ids"]
                for r in saved["rows"]:
                    assert r["correct"] == (r["prediction"] == reg["token_maps"][form][indexed[r["id"]]["prompt"]["AT"]])
                    assert r["id"] not in predictions
                    predictions[r["id"]] = r
            m = metrics(rows, predictions)
            assert m == read(root/"metrics"/f"{stage}_{form}.json")
            output["stages"][stage+"/"+form] = m
    output["probe_pairs"] = read(root/"probe_pair_metrics.json")
    write(root/"results.json", output, immutable=True)
    print(json.dumps({"model": MODEL, "winner": output["winner"], "stages": {k: {f: v[f] for f in
          ("accuracy", "empty_pair_cells", "worst_pair_accuracy", "mean_pair_accuracy") if f in v} for k, v in output["stages"].items()},
          "probe_pairs": output["probe_pairs"]}), flush=True)


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
