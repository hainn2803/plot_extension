"""Frozen matched panels, strict GPU budget, and descriptive paired reporting."""
import argparse
from collections import Counter, defaultdict
from itertools import permutations, product
import json
from pathlib import Path
import tarfile
import time
import traceback

from experiments.mcqa_c_at_localization.data import Prompt
from experiments.mcqa_c_at_localization.protocol import COLORS, LABELS, MODEL_PATH, allocate_groups
from experiments.mcqa_c_at_localization.storage import digest, file_hash, read, write

NAME = "mcqa_c_at_prompt_diagnostic_20261017_v1"
ROOT = Path("eval") / NAME
PARENT = Path("eval/mcqa_c_at_localization_20261017_local_v8")
INSTRUCTION = ("The option labels below may appear in any order. "
               "Answer with the label printed next to the requested color.")
FORMATS = ("current", "instruction", "explicit_labels")
PROTOCOL = {
    "name": NAME, "formats": list(FORMATS), "instruction": INSTRUCTION,
    "groups": 4, "logical_examples": 6144, "physical_batch": 8,
    "gpu_seconds_cap": 900, "cleanup_reserve_seconds": 15,
    "atol": 2e-4, "rtol": 2e-4, "cpu_threads": 8,
    "scope": "training-group natural-answer diagnostic only; no filtering or interventions",
    "panel": "four hash-first training groups; all C, condition, query, fact order, two content orders, all 24 label orders",
    "irrelevant_color": "offset 1 + group_index % 3 cyclically from queried color; here in one-color condition",
    "scoring": "full-vocabulary argmax equals one space-prefixed A-D token; conditional four-label probabilities supplemental",
    "schedule": "batch-index first, formats current/instruction/explicit_labels within each matched batch",
    "resume": "identical source, protocol, panel, environment, model, batch; outstanding crashed lease consumes remaining budget",
}


def key(value):
    return digest([NAME, value])


def render(prompt, form):
    if form not in FORMATS:
        raise ValueError("Unknown format")
    if form == "current":
        return prompt.text
    text = prompt.text
    if form == "explicit_labels":
        head, _ = text.split("\n", 1)
        options = "\n".join(f"Label {label}: {color}" for label, color in zip(prompt.labels, prompt.contents))
        text = f"{head}\n{options}\nAnswer:"
    return INSTRUCTION + "\n" + text


def build_panel():
    groups = sorted(allocate_groups()["train"], key=key)[:4]
    rows = []
    for gi, group in enumerate(groups):
        for color, condition, query, order, contents, labels in product(
                COLORS, ("one_color", "two_colors"), (0, 1), ((0, 1), (1, 0)),
                (COLORS, tuple(reversed(COLORS))), permutations(LABELS)):
            other = "here" if condition == "one_color" else COLORS[(COLORS.index(color) + 1 + gi % 3) % 4]
            facts = [other, other]
            facts[query] = color
            prompt = Prompt(tuple(group), tuple(facts), query, order, contents, labels)
            record = prompt.record()
            rows.append({"id": digest(record), "prompt": record,
                         "texts": {form: render(prompt, form) for form in FORMATS}})
    return sorted(rows, key=lambda row: key(row["id"]))


def validate_panel(panel):
    if len(panel) != 6144 or len({row["id"] for row in panel}) != 6144:
        raise ValueError("Panel count or uniqueness failed")
    train = {tuple(group) for group in allocate_groups()["train"]}
    groups = {tuple(row["prompt"]["objects"]) for row in panel}
    if groups != set(sorted(train, key=key)[:4]):
        raise ValueError("Panel is not the preregistered training-only groups")
    for row in panel:
        p = row["prompt"]
        prompt = Prompt(**{k: tuple(p[k]) if k in ("objects", "facts", "fact_order", "contents", "labels") else p[k]
                          for k in ("objects", "facts", "query", "fact_order", "contents", "labels", "wording")})
        if digest(prompt.record()) != row["id"] or digest(prompt.record()) != digest(p):
            raise ValueError("Logical state changed")
        if row["texts"] != {form: render(prompt, form) for form in FORMATS}:
            raise ValueError("Rendered format changed logical example")
    cells = Counter((r["prompt"]["condition"], "".join(r["prompt"]["labels"])) for r in panel)
    if len(cells) != 48 or set(cells.values()) != {128}:
        raise ValueError("Label order / condition coverage is not exact")
    for field in ("C", "AT", "AP"):
        counts = Counter((r["prompt"]["condition"], "".join(r["prompt"]["labels"]), r["prompt"][field]) for r in panel)
        if len(counts) != 192 or set(counts.values()) != {32}:
            raise ValueError("Unbalanced panel field: " + field)
    return {"logical_examples": len(panel), "forwards": len(panel) * 3,
            "groups": sorted(groups), "condition_label_order_counts": dict(sorted(("/".join(k), v) for k, v in cells.items())),
            "all_condition_order_C_AT_AP_marginals_exact": True}


def sources():
    own = Path(__file__).parent
    parent = own.parent / "mcqa_c_at_localization"
    paths = list(own.rglob("*.py")) + list(parent.rglob("*.py"))
    return {p.as_posix(): file_hash(p) for p in sorted(paths)}


def prepare(root):
    from transformers import AutoTokenizer
    from experiments.mcqa_c_at_localization.answers import answer_ids
    from experiments.mcqa_c_at_localization.verification import asset_hashes
    panel = build_panel()
    coverage = validate_panel(panel)
    assets = asset_hashes()
    parent = read(PARENT / "registration.json")
    if parent["model_files"] != assets or parent["groups"] != json.loads(json.dumps(allocate_groups())):
        raise ValueError("Parent model/group provenance mismatch")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokens = answer_ids(tok)
    for row in panel:
        row["input_ids"] = {}
        for form in FORMATS:
            text = row["texts"][form]
            ids = tok.encode(text, add_special_tokens=True)
            if tok.encode(text + " " + row["prompt"]["AT"], add_special_tokens=True) != ids + [tokens[row["prompt"]["AT"]]]:
                raise ValueError("Answer continuation changes prompt tokenization")
            row["input_ids"][form] = ids
    lengths = {form: dict(Counter(len(row["input_ids"][form]) for row in panel)) for form in FORMATS}
    if any(len(values) != 1 for values in lengths.values()):
        raise ValueError("Unequal within-format lengths; no padding or substitutions")
    write(root / "panel.json", panel, immutable=True)
    write(root / "coverage.json", coverage, immutable=True)
    registration = {"protocol": PROTOCOL, "parent_registration": file_hash(PARENT / "registration.json"),
                    "model_files": assets, "panel": file_hash(root / "panel.json"), "coverage": coverage,
                    "lengths": lengths, "label_ids": tokens, "source": sources()}
    write(root / "registration.json", registration, immutable=True)
    archive = root / "source.tar.gz"
    if not archive.exists():
        with tarfile.open(archive, "w:gz") as stream:
            for path in registration["source"]:
                stream.add(path, arcname=path)
    print(json.dumps({"status": "prepared", "lengths": lengths, "coverage": coverage}, sort_keys=True), flush=True)


def verify(root):
    reg = read(root / "registration.json")
    if reg["protocol"] != PROTOCOL or reg["source"] != sources():
        raise ValueError("Source/protocol changed; immutable resume refused")
    if reg["parent_registration"] != file_hash(PARENT / "registration.json"):
        raise ValueError("Parent registration changed")
    if reg["panel"] != file_hash(root / "panel.json"):
        raise ValueError("Panel identity changed")
    panel = read(root / "panel.json")
    if digest(validate_panel(panel)) != digest(reg["coverage"]):
        raise ValueError("Coverage changed")
    return reg, panel


class Budget:
    """An unreconciled crashed lease is charged in full, never silently refunded."""
    def __init__(self, path, clock=time.monotonic):
        self.path, self.clock = Path(path), clock
        state = read(self.path) if self.path.exists() else {"spent_seconds": 0, "sessions": []}
        if state.get("active"):
            raise RuntimeError("Prior GPU session did not close; remaining budget is reserved, audit required")
        self.state, self.started = state, None

    def start(self):
        if self.state["spent_seconds"] >= 900 or self.state.get("active"):
            raise RuntimeError("Approved GPU budget exhausted or active")
        self.started = self.clock()
        self.state.update(active=True, started_utc=time.time(), reserved_seconds=900-self.state["spent_seconds"])
        write(self.path, self.state)

    def elapsed(self):
        return self.state["spent_seconds"] + (self.clock() - self.started if self.started is not None else 0)

    def remaining(self):
        return 900 - self.elapsed()

    def check(self, required=0):
        if self.remaining() <= 15 + required:
            raise TimeoutError("Stopping before the 900-second cumulative GPU cap")

    def close(self):
        if self.started is None:
            return
        elapsed = self.clock() - self.started
        self.state["sessions"].append({"started_utc": self.state["started_utc"], "seconds": elapsed})
        self.state.update(active=False, spent_seconds=self.state["spent_seconds"] + elapsed)
        self.started = None
        write(self.path, self.state)


def numeric_check(model, ids, form, budget, torch, backend):
    checks = []
    def check(name, a, b):
        ok = torch.allclose(a, b, atol=2e-4, rtol=2e-4)
        checks.append({"name": name, "passed": ok, "max_abs_difference": float((a.float()-b.float()).abs().max())})
        if not ok:
            raise RuntimeError(f"Numerical mismatch at {form}/{name}; tolerances unchanged")
    budget.check(5)
    logits, _ = backend.forward(model, ids)
    repeated, _ = backend.forward(model, ids)
    check("repeated_forward", logits, repeated)
    raw = {}
    hook = model.lm_head.register_forward_hook(lambda _m, _i, out: raw.update(value=out.detach()))
    try:
        native = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[:, -1].float()
    finally:
        hook.remove()
    cap = model.config.final_logit_softcapping
    check("matched_FP32_softcap_reference", logits, (torch.tanh(raw["value"].float()/cap)*cap)[:, -1])
    check("native_BF16_softcap_reference", native, (torch.tanh(raw["value"]/cap)*cap)[:, -1].float())
    pred, native_pred = logits.argmax(-1), native.argmax(-1)
    if not torch.equal(pred, raw["value"][:, -1].argmax(-1)):
        raise RuntimeError("Registered FP32 softcap unexpectedly changed raw-head argmax")
    differences = pred != native_pred
    if differences.any() and not torch.equal(native.gather(1, pred[:, None])[differences, 0], native.max(-1).values[differences]):
        raise RuntimeError("Native prediction difference is not an exact BF16 softcap tie")
    return {"format": form, "shape": list(ids.shape), "checks": checks,
            "native_BF16_argmax_ties": int(differences.sum()), "atol": 2e-4, "rtol": 2e-4}


def tasks(panel):
    batch = PROTOCOL["physical_batch"]
    return [(start, form) for start in range(0, len(panel), batch) for form in FORMATS]


def run(root):
    reg, panel = verify(root)
    from experiments.mcqa_c_at_localization.verification import asset_hashes
    if asset_hashes() != reg["model_files"]:
        raise ValueError("Model assets changed")
    import torch
    from experiments.mcqa_c_at_localization import backend, runtime
    from experiments.mcqa_c_at_localization.verification import environment
    torch.set_num_threads(PROTOCOL["cpu_threads"])
    budget = Budget(root / "gpu_budget.json")
    model = None
    status = "failed"
    budget.start()
    try:
        runtime.configure(root)
        budget.check(40)
        model, tokenizer = backend.load_model()
        versions = {n: p._version for n, p in model.named_parameters()}
        write(root / "environment.json", environment(), immutable=True)
        torch.cuda.reset_peak_memory_stats()
        ids_for = lambda rows, form: torch.tensor([r["input_ids"][form] for r in rows], device="cuda", dtype=torch.long)
        with torch.no_grad():
            checks = [numeric_check(model, ids_for(panel[:8], form), form, budget, torch, backend) for form in FORMATS]
            write(root / "numerical.json", {"status": "passed", "formats": checks})
            timings = {}
            for form in FORMATS:
                ids = ids_for(panel[:8], form)
                budget.check(5)
                torch.cuda.synchronize()
                start = time.monotonic()
                for _ in range(6):
                    logits, _ = backend.forward(model, ids)
                    del logits
                torch.cuda.synchronize()
                timings[form] = (time.monotonic() - start)/6
            pending = [(s, f) for s, f in tasks(panel) if not (root / "predictions" / f"{s:05d}_{f}.json").exists()]
            estimate = sum(timings[f] for _, f in pending) * 1.25
            profile = {"seconds_per_batch": timings, "pending_batches": len(pending),
                       "estimated_inference_seconds_with_25pct": estimate, "budget_remaining_seconds": budget.remaining(),
                       "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            write(root / "profile.json", profile)
            print(json.dumps({"status": "profiled", **profile}), flush=True)
            budget.check(estimate)
            label_ids = [reg["label_ids"][letter] for letter in LABELS]
            for number, (start, form) in enumerate(pending):
                budget.check(max(timings.values()) * 4)
                runtime.guard()
                rows = panel[start:start+8]
                logits, _ = backend.forward(model, ids_for(rows, form))
                summary = backend.summarize_logits(logits, label_ids)
                results = [{"id": row["id"], "prediction": summary["prediction"][i],
                            "decoded": tokenizer.decode([summary["prediction"][i]]),
                            "label_logits": summary["label_logits"][i],
                            "label_probabilities": summary["label_probabilities"][i],
                            "label_full_vocabulary_probabilities": logits[i].softmax(-1)[label_ids].cpu().tolist()}
                           for i, row in enumerate(rows)]
                del logits
                write(root / "predictions" / f"{start:05d}_{form}.json",
                      {"registration": digest(reg), "format": form, "start": start, "rows": results}, immutable=True)
                if (number+1) % 96 == 0:
                    progress = {"status": "running", "committed_batches": len(tasks(panel))-len(pending)+number+1,
                                "total_batches": len(tasks(panel)), "gpu_elapsed_seconds": budget.elapsed()}
                    write(root / "status.json", progress)
                    print(json.dumps(progress), flush=True)
            backend.assert_frozen(model, versions)
            if model.lm_head._forward_hooks or any(layer._forward_hooks for layer in model.model.layers):
                raise RuntimeError("Forward hooks leaked")
        write(root / "completion.json", {"registration": digest(reg), "batches": len(tasks(panel)),
                                        "frozen_parameters_verified": True, "hooks_clean": True,
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated()}, immutable=True)
        status = "completed"
    except BaseException as error:
        status = "interrupted" if isinstance(error, (TimeoutError, runtime.Paused, KeyboardInterrupt)) else "failed"
        write(root / "error.json", {"status": status, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        runtime.close()
        if model is not None:
            del model
        import gc
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        budget.close()
        write(root / "status.json", {"status": status, "gpu_seconds": budget.state["spent_seconds"]})


def summarize(rows, predictions, label_ids):
    ids_to_labels = {v: k for k, v in label_ids.items()}
    def stats(subset):
        good = sum(predictions[r["id"]]["prediction"] == label_ids[r["prompt"]["AT"]] for r in subset)
        outputs = Counter(ids_to_labels.get(predictions[r["id"]]["prediction"], "other") for r in subset)
        return {"n": len(subset), "correct": good, "accuracy": good/len(subset), "predictions": dict(outputs)}
    result = {"overall": stats(rows)}
    for field in ("condition", "C", "AT", "AP", "query", "labels", "objects", "fact_order", "contents"):
        cells = defaultdict(list)
        for row in rows:
            value = row["prompt"][field]
            cells[json.dumps(value, separators=(",", ":"))].append(row)
        result[field] = {k: stats(v) for k, v in sorted(cells.items())}
    cells = defaultdict(list)
    for row in rows:
        p = row["prompt"]
        cells[p["condition"] + "/" + "".join(p["labels"])].append(row)
    result["condition_order"] = {k: stats(v) for k, v in sorted(cells.items())}
    return result


def report(root):
    reg, panel = verify(root)
    completion = read(root / "completion.json")
    if completion["registration"] != digest(reg) or read(root / "status.json")["status"] != "completed":
        raise ValueError("Run not verified completed")
    by_form = {form: {} for form in FORMATS}
    for start, form in tasks(panel):
        saved = read(root / "predictions" / f"{start:05d}_{form}.json")
        expected = [r["id"] for r in panel[start:start+8]]
        if saved["registration"] != digest(reg) or saved["format"] != form or [r["id"] for r in saved["rows"]] != expected:
            raise ValueError("Prediction provenance mismatch")
        by_form[form].update({r["id"]: r for r in saved["rows"]})
    metrics = {form: summarize(panel, by_form[form], reg["label_ids"]) for form in FORMATS}
    paired = {}
    for left, right in (("current", "instruction"), ("current", "explicit_labels"), ("instruction", "explicit_labels")):
        counts = Counter()
        for row in panel:
            target = reg["label_ids"][row["prompt"]["AT"]]
            a = by_form[left][row["id"]]["prediction"] == target
            b = by_form[right][row["id"]]["prediction"] == target
            counts[f"{int(a)}->{int(b)}"] += 1
        paired[f"{left}->{right}"] = dict(counts)
    budget = read(root / "gpu_budget.json")
    result = {"formats": metrics, "paired_correctness": paired, "gpu_budget": budget,
              "limitations": "Four development object groups, not an independent generalization bank. No bank-construction feasibility, intervention, or DAS claim."}
    write(root / "results.json", result, immutable=True)
    lines = ["# Matched Prompt-Format Diagnostic", "", "Completed training-only natural-answer check; no DAS or final-test use.", "",
             "6,144 matched logical examples per format; all 24 label orders and both color conditions. Four hash-selected training object groups.", "",
             "| Format | Overall | One Color | Two Colors | Worst Order/Condition |", "|---|---:|---:|---:|---:|"]
    for form in FORMATS:
        m = metrics[form]
        lines.append(f"| {form} | {m['overall']['accuracy']:.2%} | {m['condition'][json.dumps('one_color')]['accuracy']:.2%} | {m['condition'][json.dumps('two_colors')]['accuracy']:.2%} | {min(v['accuracy'] for v in m['condition_order'].values()):.2%} |")
    lines += ["", "## Label Orders", "", "| Order | Current | Instruction | Explicit Labels |", "|---|---:|---:|---:|"]
    for labels in permutations(LABELS):
        cell = json.dumps(list(labels), separators=(",", ":"))
        values = [metrics[f]["labels"][cell]["accuracy"] for f in FORMATS]
        lines.append("| " + "".join(labels) + " | " + " | ".join(f"{v:.2%}" for v in values) + " |")
    lines += ["", "## Scope and Verification", "",
              f"GPU allocation wall time: {budget['spent_seconds']:.1f} seconds of the 900-second cap, including loading, numerical checks, profiling, inference, and cleanup.",
              "Full-vocabulary argmax scoring; four-label probabilities are supplemental only. Registered BF16 backbone, FP32 softcap, full-sequence head, SDPA, TF32 disabled. Format-specific repeated-forward and matched-reference checks use atol=rtol=2e-4.",
              "All logical instances and exact format strings were frozen before inference. No examples were excluded or replaced based on predictions.",
              "Only four training object groups; do not interpret the many arrangements as independent subject replications. No format has been substituted into the main experiment, and improved average accuracy does not establish that its exact post-filter quotas are feasible.",
              "", "## Exact Format Changes", "", f"Instruction: {INSTRUCTION}",
              "Current: unchanged prompt. Instruction: prepend the above instruction. Explicit labels: same instruction, with `Label C: red` rows replacing `C. red` rows; row order, facts, and query unchanged."]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "reported", "overall": {f: metrics[f]["overall"] for f in FORMATS}, "gpu_seconds": budget["spent_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "run", "report"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.command == "verify":
        reg, panel = verify(args.root)
        print(json.dumps({"status": "verified", "registration": digest(reg), "examples": len(panel)}))
    else:
        globals()[args.command](args.root)
