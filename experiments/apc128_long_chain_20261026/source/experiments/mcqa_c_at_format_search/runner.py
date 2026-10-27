"""Frozen matched format search with independent development confirmation."""
import argparse
from collections import Counter, defaultdict
import gc
from functools import lru_cache
from itertools import permutations, product
import json
from pathlib import Path
import tarfile
import time
import traceback

from experiments.mcqa_c_at_localization.data import Prompt, from_record
from experiments.mcqa_c_at_localization.protocol import COLORS, LABELS, MODEL_PATH, allocate_groups
from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write
from experiments.mcqa_c_at_prompt_diagnostic import runner as diagnostic
from experiments.mcqa_c_at_query_feasibility.runner import Safety

NAME = "mcqa_c_at_format_search_20261017_v1"
ROOT = Path("eval") / NAME
FORMS = ("original", "query_last", "label_suffix", "arrow_key", "equals_key",
         "object_table", "json_key", "fewshot_original", "fewshot_arrow")
PROTOCOL = {
    "scope": "Training objects only; explore natural task competence, no DAS, bank adoption, or final data",
    "model": "Unchanged pinned Gemma-2-2B, frozen BF16, SDPA, TF32 off, FP32 softcap, full sequence head",
    "formats": FORMS, "screen_groups": 2, "confirm_groups": 4, "demo_groups": 2,
    "screen_prompts": 768, "confirmation_prompts": 1536,
    "label_orders": 24, "conditions": ["one_color", "two_colors"],
    "pair": "Both queries; one-color moves the colored fact and preserves C, two-color changes query only",
    "selection": "Fewest empty condition/order pair cells, then highest worst pair rate, mean pair rate, natural accuracy, listed format order",
    "bank_probe": "First frozen alternative in every parent training slot, no selection using predictions",
    "budget": "Remaining original cumulative 900-second shared diagnostic ledger; includes loading/checks/profile/cleanup",
    "execution": "Exact length buckets, no padding, physical batch 8 through 110 tokens, otherwise 4; fixed filler not scored",
    "atol": 2e-4, "rtol": 2e-4, "cleanup_reserve_seconds": 35,
    "success_scope": "Natural correctness and necessary coverage only; exact bank feasibility remains a separate requirement",
}


def key(value):
    return digest([NAME, value])


def groups():
    return sorted(allocate_groups()["train"], key=key)


@lru_cache(maxsize=1)
def demos():
    result = []
    ps = tuple(permutations(range(4)))
    ls = tuple(permutations(LABELS))
    for i in range(4):
        query = i % 2
        contents = min((tuple(COLORS[j] for j in p) for p in ps if p[i] == i), key=key)
        target = LABELS[(i+2) % 4]
        labels = min((p for p in ls if p[i] == target), key=key)
        other = "here" if i < 2 else COLORS[(i+1) % 4]
        facts = [other, other]
        facts[query] = COLORS[i]
        result.append(Prompt(tuple(groups()[-2 + i % 2]), tuple(facts), query,
                             (0, 1) if i % 2 == 0 else (1, 0), contents, labels))
    return result


def core(prompt, form):
    facts = "\n".join(f"The {prompt.objects[i]} is {prompt.facts[i]}." for i in prompt.fact_order)
    options = "\n".join(f"{letter}. {color}" for letter, color in zip(prompt.labels, prompt.contents))
    obj = prompt.objects[prompt.query]
    question = f"What color is the {obj}?"
    if form == "original":
        return prompt.text
    if form == "query_last":
        return f"Answer with the option label.\nFacts:\n{facts}\nOptions:\n{options}\nQuestion: {question}\nAnswer:"
    if form == "label_suffix":
        return prompt.text.removesuffix("Answer:") + "The correct option label is"
    if form == "arrow_key":
        mapping = "\n".join(f"{color} -> {letter}" for letter, color in zip(prompt.labels, prompt.contents))
        return f"{facts}\nColor -> label:\n{mapping}\nQuestion: Which label corresponds to the color of the {obj}?\nLabel:"
    if form == "equals_key":
        mapping = "\n".join(f"{letter} = {color}" for letter, color in zip(prompt.labels, prompt.contents))
        return f"{facts}\nOption labels:\n{mapping}\nQuestion: {question}\nThe correct letter is"
    if form == "object_table":
        table = "\n".join(f"{prompt.objects[i]} | {prompt.facts[i]}" for i in prompt.fact_order)
        mapping = "\n".join(f"{letter} | {color}" for letter, color in zip(prompt.labels, prompt.contents))
        return f"Object | Color\n{table}\n\nLabel | Color\n{mapping}\n\nQuestion: Which label gives the color of the {obj}?\nAnswer:"
    if form == "json_key":
        facts_json = json.dumps({prompt.objects[i]: prompt.facts[i] for i in prompt.fact_order})
        key_json = json.dumps(dict(zip(prompt.contents, prompt.labels)))
        return f"Return only the matching label.\nFacts: {facts_json}\nColor-to-label key: {key_json}\nAsked object: {obj}\nAnswer:"
    raise ValueError(form)


def render(prompt, form):
    if form.startswith("fewshot_"):
        base = "original" if form == "fewshot_original" else "arrow_key"
        examples = "\n\n".join(core(p, base) + " " + p.AT for p in demos())
        return "Answer each question with its matching option label.\n\n" + examples + "\n\n" + core(prompt, base)
    return core(prompt, form)


def panel(which):
    subset = groups()[:2] if which == "screen" else groups()[2:6]
    rows = []
    contents_orders = tuple(permutations(COLORS))
    for gi, group in enumerate(subset):
        contents = contents_orders[int(key([which, group]), 16) % 24]
        order = (0, 1) if gi % 2 == 0 else (1, 0)
        for condition, ci, labels in product(("one_color", "two_colors"), range(4), permutations(LABELS)):
            pair = key([which, group, condition, ci, labels])
            for query in (0, 1):
                if condition == "one_color":
                    facts = ["here", "here"]
                    facts[query] = COLORS[ci]
                else:
                    facts = [COLORS[ci], COLORS[(ci+1+gi % 3) % 4]]
                p = Prompt(tuple(group), tuple(facts), query, order, contents, labels)
                rows.append({"id": p.id, "pair": pair, "prompt": p.record()})
    return sorted(rows, key=lambda row: key(row["id"]))


def source_hashes():
    paths = set(Path(__file__).parent.rglob("*.py"))
    paths.update(Path("experiments/mcqa_c_at_localization").rglob("*.py"))
    paths.update(Path("experiments/mcqa_c_at_prompt_diagnostic").rglob("*.py"))
    paths.add(Path("experiments/mcqa_c_at_query_feasibility/runner.py"))
    return {p.as_posix(): file_hash(p) for p in sorted(paths)}


def schedule(rows, form):
    buckets = defaultdict(list)
    for row in rows:
        buckets[len(row["input_ids"][form])].append(row["id"])
    result = []
    for length, ids in sorted(buckets.items()):
        ids.sort(key=key)
        batch = 8 if length <= 110 else 4
        for start in range(0, len(ids), batch):
            real = ids[start:start+batch]
            result.append({"ids": real, "physical": real + [ids[0]] * (batch-len(real)),
                           "length": length, "batch": batch})
    return result


def prepare(root):
    from transformers import AutoTokenizer
    from experiments.mcqa_c_at_localization.answers import answer_ids
    from experiments.mcqa_c_at_localization.verification import asset_hashes
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    label_ids = answer_ids(tokenizer)
    banks = {name: panel(name) for name in ("screen", "confirm")}
    parent_pairs = read(diagnostic.PARENT / "proposals/train.json")
    banks["bank_probe"] = []
    seen = set()
    probe_pairs = []
    for slot in parent_pairs:
        pair = slot["alternatives"][0]
        probe_pairs.append(pair)
        for side in ("base", "source"):
            p = pair[side]
            if p["id"] not in seen:
                seen.add(p["id"])
                banks["bank_probe"].append({"id": p["id"], "prompt": p})
    assert len(probe_pairs) == 768
    assert len(banks["screen"]) == 768 and len(banks["confirm"]) == 1536
    write(root / "probe_pairs.json", probe_pairs, immutable=True)
    sizes = {}
    for name, rows in banks.items():
        assert len({r["id"] for r in rows}) == len(rows)
        for row in rows:
            prompt = from_record(row["prompt"])
            assert digest(prompt.record()) == digest(row["prompt"])
            row["texts"] = {form: render(prompt, form) for form in FORMS}
            row["input_ids"] = {}
            for form, text in row["texts"].items():
                ids = tokenizer.encode(text, add_special_tokens=True)
                for letter, token in label_ids.items():
                    assert tokenizer.encode(text+" "+letter, add_special_tokens=True) == ids+[token]
                row["input_ids"][form] = ids
        write(root / f"{name}.json", rows, immutable=True)
        sizes[name] = {form: dict(Counter(len(r["input_ids"][form]) for r in rows)) for form in FORMS}
    assets = asset_hashes()
    assert assets == read(diagnostic.PARENT / "registration.json")["model_files"]
    reg = {"protocol": PROTOCOL, "source": source_hashes(), "model_files": assets,
           "parent_registration": file_hash(diagnostic.PARENT / "registration.json"),
           "parent_proposals": file_hash(diagnostic.PARENT / "proposals/train.json"),
           "banks": {name: file_hash(root/f"{name}.json") for name in banks},
           "probe_pairs": file_hash(root / "probe_pairs.json"), "lengths": sizes,
           "label_ids": label_ids, "initial_budget": read(diagnostic.ROOT / "gpu_budget.json")}
    write(root / "registration.json", reg, immutable=True)
    if not (root / "source.tar.gz").exists():
        with tarfile.open(root / "source.tar.gz", "w:gz") as tar:
            for path in reg["source"]:
                tar.add(path, arcname=path)
    print(json.dumps({"status": "prepared", "rows": {k: len(v) for k, v in banks.items()}, "lengths": sizes}), flush=True)


def verify(root):
    reg = read(root / "registration.json")
    assert digest(reg["protocol"]) == digest(PROTOCOL) and reg["source"] == source_hashes()
    assert reg["parent_registration"] == file_hash(diagnostic.PARENT / "registration.json")
    assert reg["parent_proposals"] == file_hash(diagnostic.PARENT / "proposals/train.json")
    assert reg["probe_pairs"] == file_hash(root / "probe_pairs.json")
    banks = {}
    for name, expected in reg["banks"].items():
        assert file_hash(root / f"{name}.json") == expected
        banks[name] = read(root / f"{name}.json")
    return reg, banks


def metrics(rows, predictions):
    groups_by_pair = defaultdict(list)
    cells = defaultdict(lambda: [0, 0])
    letters = Counter()
    prompt_correct = 0
    for row in rows:
        p = row["prompt"]
        correct = predictions[row["id"]]["correct"]
        prompt_correct += int(correct)
        cell = p["condition"]+"/"+"".join(p["labels"])
        cells[cell][0] += int(correct)
        cells[cell][1] += 1
        letters[predictions[row["id"]]["decoded"]] += 1
        if "pair" in row:
            groups_by_pair[row["pair"]].append(row)
    pair_cells = defaultdict(lambda: [0, 0])
    for members in groups_by_pair.values():
        assert len(members) == 2
        p = members[0]["prompt"]
        cell = p["condition"]+"/"+"".join(p["labels"])
        pair_cells[cell][0] += int(all(predictions[r["id"]]["correct"] for r in members))
        pair_cells[cell][1] += 1
    result = {"prompts": len(rows), "correct": prompt_correct, "accuracy": prompt_correct/len(rows),
              "outputs": dict(letters), "condition_order": dict(cells), "pair_cells": dict(pair_cells)}
    if pair_cells:
        result.update(empty_pair_cells=sum(v[0] == 0 for v in pair_cells.values()),
                      worst_pair_accuracy=min(v[0]/v[1] for v in pair_cells.values()),
                      mean_pair_accuracy=sum(v[0]/v[1] for v in pair_cells.values())/len(pair_cells))
    return result


def selection_key(m, form):
    return (m["empty_pair_cells"], -m["worst_pair_accuracy"], -m["mean_pair_accuracy"],
            -m["accuracy"], FORMS.index(form))


def run(root, approval):
    if not approval:
        raise ValueError("Explicit exploration authorization required")
    from experiments.mcqa_c_at_localization.local_run import single_worker
    from experiments.mcqa_c_at_localization.verification import asset_hashes, environment
    from experiments.mcqa_c_at_localization import backend
    import torch
    reg, banks = verify(root)
    assert asset_hashes() == reg["model_files"]
    torch.set_num_threads(8)
    budget = diagnostic.Budget(diagnostic.ROOT / "gpu_budget.json")
    guard = Safety(root, None)
    model, outcome = None, "failed"
    with single_worker(root):
        try:
            guard.start()
            write(root / "authorization.json", {"approval": approval, "no_main_bank_adoption": True})
            budget.start()
            budget.check(50)
            model, tokenizer = backend.load_model()
            versions = {n: p._version for n, p in model.named_parameters()}
            write(root / "environment.json", environment(), immutable=True)
            label_ids = [reg["label_ids"][letter] for letter in LABELS]
            checks, timings = {}, {}
            indexed = {name: {r["id"]: r for r in rows} for name, rows in banks.items()}

            def inputs(ids, name, form):
                return torch.tensor([indexed[name][i]["input_ids"][form] for i in ids], device="cuda")

            def check_shape(name, form, task):
                shape = (task["batch"], task["length"])
                if shape not in checks:
                    guard.guard()
                    ids = inputs(task["physical"], name, form)
                    checks[shape] = diagnostic.numeric_check(model, ids, f"{name}/{form}", budget, torch, backend)
                    write(root / "numerical.json", {"status": "passed", "checks": list(checks.values())})
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    for _ in range(3):
                        logits, _ = backend.forward(model, ids)
                        del logits
                    torch.cuda.synchronize()
                    timings[shape] = (time.monotonic()-start)/3

            def evaluate(name, form):
                rows = banks[name]
                tasks = schedule(rows, form)
                for task in tasks:
                    check_shape(name, form, task)
                pending = [(i, t) for i, t in enumerate(tasks) if not (root / "predictions" / name / form / f"{i:04d}.json").exists()]
                estimate = sum(timings[(t["batch"], t["length"])] for _, t in pending)*1.25
                write(root / "profiles" / f"{name}_{form}.json", {"estimated_remaining_seconds_with_25pct": estimate,
                      "budget_remaining": budget.remaining(), "timings": {str(k): v for k, v in timings.items()}})
                budget.check(20+estimate)
                result = {}
                for i, task in enumerate(tasks):
                    path = root / "predictions" / name / form / f"{i:04d}.json"
                    identity = {"registration": digest(reg), "stage": name, "format": form, "task": task}
                    if path.exists():
                        saved = read(path)
                        assert saved["identity"] == identity
                    else:
                        guard.guard()
                        budget.check(20+4*timings[(task["batch"], task["length"])])
                        logits, _ = backend.forward(model, inputs(task["physical"], name, form))
                        assert torch.isfinite(logits).all()
                        summary = backend.summarize_logits(logits, label_ids)
                        del logits
                        records = []
                        for offset, item in enumerate(task["ids"]):
                            prediction = summary["prediction"][offset]
                            prompt = indexed[name][item]["prompt"]
                            records.append({"id": item, "prediction": prediction,
                                "correct": prediction == reg["label_ids"][prompt["AT"]],
                                "decoded": tokenizer.decode([prediction]), "target": prompt["AT"],
                                "label_logits": summary["label_logits"][offset]})
                        saved = {"identity": identity, "rows": records}
                        write(path, saved, immutable=True)
                    assert task["ids"] == [v["id"] for v in saved["rows"]]
                    result.update({v["id"]: v for v in saved["rows"]})
                assert len(result) == len(rows)
                m = metrics(rows, result)
                write(root / "metrics" / f"{name}_{form}.json", m, immutable=True)
                print(json.dumps({"stage": name, "format": form, "accuracy": m["accuracy"],
                      "empty_pair_cells": m.get("empty_pair_cells"), "worst_pair_accuracy": m.get("worst_pair_accuracy"),
                      "GPU_seconds_remaining": budget.remaining()}), flush=True)
                return m, result

            with torch.no_grad():
                screening = {}
                for form in FORMS:
                    m, _ = evaluate("screen", form)
                    screening[form] = m
                    write(root / "status.json", {"status": "running", "screened_formats": len(screening), "total_formats": len(FORMS)})
                winner = min(FORMS, key=lambda f: selection_key(screening[f], f))
                write(root / "selection_lock.json", {"registration": digest(reg), "winner": winner,
                      "criterion": PROTOCOL["selection"], "screen_metrics_hashes": {f: file_hash(root / "metrics" / f"screen_{f}.json") for f in FORMS}}, immutable=True)
                evaluate("confirm", winner)
                probe_metrics, prediction = evaluate("bank_probe", winner)
                cells = defaultdict(lambda: [0, 0])
                for p in read(root / "probe_pairs.json"):
                    cell = p["family"]+"/"+p["condition"]
                    cells[cell][0] += int(all(prediction[p[side]["id"]]["correct"] for side in ("base", "source")))
                    cells[cell][1] += 1
                write(root / "probe_pair_metrics.json", dict(cells), immutable=True)
                backend.assert_frozen(model, versions)
                assert not model.lm_head._forward_hooks and not any(l._forward_hooks for l in model.model.layers)
                write(root / "completion.json", {"registration": digest(reg), "winner": winner,
                      "frozen_parameters": True, "hooks_clean": True,
                      "prediction_hashes": {p.relative_to(root).as_posix(): file_hash(p) for p in sorted((root / "predictions").rglob("*.json"))}}, immutable=True)
                outcome = "completed"
        except BaseException as error:
            from experiments.mcqa_c_at_localization.runtime import Paused
            outcome = "interrupted" if isinstance(error, (TimeoutError, KeyboardInterrupt, Paused)) else "failed"
            write(root / "error.json", {"status": outcome, "error": str(error), "traceback": traceback.format_exc()})
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
            write(root / "status.json", {"status": outcome, "GPU_seconds_cumulative": budget.state["spent_seconds"]})


def report(root):
    reg, banks = verify(root)
    output = {"scope": PROTOCOL["scope"], "formats": {}, "confirmation": None,
              "probe_pairs": read(root / "probe_pair_metrics.json") if (root / "probe_pair_metrics.json").exists() else None}
    for stage in ("screen", "confirm", "bank_probe"):
        for form in FORMS:
            saved_dir = root / "predictions" / stage / form
            if not saved_dir.exists():
                continue
            indexed = {r["id"]: r for r in banks[stage]}
            predictions = {}
            tasks = schedule(banks[stage], form)
            complete = True
            for i, task in enumerate(tasks):
                path = saved_dir / f"{i:04d}.json"
                if not path.exists():
                    complete = False
                    continue
                saved = read(path)
                assert saved["identity"] == {"registration": digest(reg), "stage": stage, "format": form, "task": task}
                assert [r["id"] for r in saved["rows"]] == task["ids"]
                for record in saved["rows"]:
                    p = indexed[record["id"]]["prompt"]
                    assert record["correct"] == (record["prediction"] == reg["label_ids"][p["AT"]])
                    assert record["id"] not in predictions
                    predictions[record["id"]] = record
            if complete:
                m = metrics(banks[stage], predictions)
                assert m == read(root / "metrics" / f"{stage}_{form}.json")
                if stage == "screen":
                    output["formats"][form] = m
                elif stage == "confirm":
                    output["confirmation"] = {"format": form, **m}
                else:
                    output["bank_probe"] = {"format": form, **m}
    output["complete"] = (root / "completion.json").exists() and read(root / "status.json")["status"] == "completed"
    if output["complete"]:
        receipt = read(root / "completion.json")
        assert receipt["registration"] == digest(reg)
        for name, expected in receipt["prediction_hashes"].items():
            assert file_hash(root / name) == expected
    write(root / ("results.json" if output["complete"] else "partial_results.json"), output, immutable=output["complete"])
    print(json.dumps({"complete": output["complete"], "screening": {k: {f: v[f] for f in ("accuracy", "empty_pair_cells", "worst_pair_accuracy", "mean_pair_accuracy")} for k, v in output["formats"].items()},
                      "confirmation": None if output["confirmation"] is None else {k: output["confirmation"][k] for k in ("format", "accuracy", "empty_pair_cells", "worst_pair_accuracy", "mean_pair_accuracy")},
                      "probe_pairs": output["probe_pairs"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--approval-text")
    args = parser.parse_args()
    if args.command == "run":
        run(args.root, args.approval_text)
    elif args.command == "verify":
        reg, banks = verify(args.root)
        print({"verified": True, "banks": {k: len(v) for k, v in banks.items()}})
    else:
        globals()[args.command](args.root)
