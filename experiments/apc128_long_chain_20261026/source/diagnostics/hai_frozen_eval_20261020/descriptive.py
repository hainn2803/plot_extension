"""Explicitly approved unchanged-bank descriptive comparison; certification stays failed."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from diagnostics.hai_frozen_eval_20261020 import run
from experiments.hai_c_at_evaluation import audit, core, schema
from experiments.mcqa_c_at_localization import runtime

ROOT = Path("eval/hai_frozen_descriptive_20261020_v1")
PARENT = run.ROOT


def load_observed(folder):
    identity = run.read(folder / "identity.json")
    batch, count = identity["batch"], len(identity["indices"])
    chunks = []
    for start in range(0, count, batch):
        path = folder / f"chunk_{start:06d}.pt"
        receipt = run.read(path.with_suffix(".json"))
        if schema.file_hash(path) != receipt["sha256"] or receipt["start"] != start or receipt["stop"] != min(start+batch, count):
            raise ValueError("Imported operation chunk failed identity/count verification")
        chunks.append(torch.load(path, map_location="cpu", weights_only=True))
    return core.Observed(torch.cat([p["c"] for p in chunks]), torch.cat([p["at"] for p in chunks]),
        {key: torch.cat([p["predictions"][key] for p in chunks]).numpy() for key in chunks[0]["predictions"]}, 0.)


def wait_for_cooling(root):
    """Reduce physical load on warnings, without clearing any flag or safety timer."""
    start = None
    while True:
        runtime.guard()
        path = root / "engineering/thermal_state.json"
        if not path.exists() or not run.read(path)["active"]:
            break
        if start is None:
            start = time.time()
            print("Thermal warning observed: idling GPU; the unchanged stop guard remains active", flush=True)
        time.sleep(1)
    if start is not None:
        run.write(root / "engineering/cooling" / f"{time.time_ns()}.json",
            dict(start=start, end=time.time(), reason="Monitor reported active thermal flag; inference waited for physical clearance",
                guard_reset=False, guard_changed=False, pause_request_cleared=False))


class CoolingBackend(run.GuardedBackend):
    def __init__(self, *args, root, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root

    def run(self, indices, c_source=None, at_source=None):
        wait_for_cooling(self.root)
        return super().run(indices, c_source, at_source)


def prepare(root):
    parent_identity = run.establish_identity(PARENT)
    profile = run.read(PARENT / "preflight.json")
    if not profile["passed"] or not all(c["passed"] and c["atol"] == c["rtol"] == 2e-4 for c in profile["checks"]):
        raise ValueError("Parent numerical checks did not pass unchanged tolerances")
    lock = run.read(PARENT / "bank_lock.json")
    if schema.file_hash(PARENT / "selected_pairs.json") != lock["pairs_sha256"]:
        raise ValueError("Locked bank changed")
    if (root / "selected_pairs.json").exists():
        if schema.file_hash(root / "selected_pairs.json") != lock["pairs_sha256"]:
            raise ValueError("Descriptive bank differs from the unchanged parent")
    else:
        shutil.copyfile(PARENT / "selected_pairs.json", root / "selected_pairs.json")
    contexts, pairs = schema.import_pairs(run.read(root / "selected_pairs.json"))
    if len(pairs) != 1600 or schema.coverage(contexts, pairs) != lock["coverage"]:
        raise ValueError("Descriptive bank coverage changed")
    natural = load_observed(PARENT / "operations/natural")
    parent_natural = run.read(PARENT / "operations/natural/identity.json")
    if parent_natural["indices"] != list(range(len(contexts))) or parent_natural["targets"] != [c.answer for c in contexts]:
        raise ValueError("Imported natural outputs are not aligned with the bank")
    imports = [PARENT / "bank_lock.json", PARENT / "selected_pairs.json", PARENT / "preflight.json",
        PARENT / "clean_mismatch_diagnosis.json", PARENT / "operations/natural/identity.json"]
    imports += sorted((PARENT / "operations/natural").glob("chunk_*.json"))
    identity = dict(parent_identity=parent_identity, source=schema.file_hash(__file__),
        imports={str(p): schema.file_hash(p) for p in imports},
        authorization="User approved evaluating the unchanged bank descriptively; no pair removed/replaced; failed clean/certification gate remains failed",
        scientific=dict(model=run.read(run.DATA / "protocol.json")["model_revision"],
            physical_batch=32, backbone="frozen BF16", projections="FP32", attention="sdpa", tf32=False,
            atol=2e-4, rtol=2e-4, frozen_handles=True, no_training=True, no_reselection=True),
        execution=dict(baselines="Clean base forwards use exactly the same base indices, physical batches and padding as each intervention",
            restoration="Original AT donor is the matched clean base state, not a differently batched cached state",
            cooling="Idle inference while monitor reports an active flag; existing watchdog and 60-second stop remain unchanged"),
        clean_gate_passed=False, certification_passed=False, no_refiltering=True)
    run.write(root / "identity.json", identity)
    run.write(root / "baseline_gate.json", dict(passed=False, certification_passed=False,
        unique_context_baseline=core.measure(natural.predictions),
        evidence=schema.file_hash(PARENT / "clean_mismatch_diagnosis.json"),
        disposition="All pairs retained under explicit descriptive-only authorization"))
    return contexts, pairs, natural, profile, schema.digest(identity)


@torch.no_grad()
def check_references(root, backend, natural, pairs):
    checks = []
    b = [p.base for p in pairs[:32]]
    s = [p.source for p in pairs[:32]]
    ref = audit.selected_functions(run.SHARED / "mcqa_gradual_das.py", ["answer_logits"])
    ref = audit.selected_functions(run.SHARED / "mcqa_gradual_das_c.py",
        ["run_c_intervention_and_capture", "run_frozen_mediator"], ref)
    clean = backend.run(b)
    run.compare("matched_base_repeat", backend.run(b).logits, clean.logits, checks)
    direct = backend.run(b, c_source=natural.c[s])
    bank = run.padded(backend, b)
    runtime.guard()
    y, _, at = ref["run_c_intervention_and_capture"](backend.model, bank, natural.c[s], backend.basis,
        backend.c_layer, 17, backend.at_layer, None, batch_size=32, require_grad=False)
    run.compare("C_reference_logits", direct.logits, y, checks)
    run.compare("C_reference_AT", direct.at, at, checks)
    replay = backend.run(b, at_source=direct.at)
    y = ref["run_frozen_mediator"](backend.model, bank, direct.at, backend.at_basis, backend.at_layer,
        None, batch_size=32, require_grad=False)
    run.compare("AT_replay_reference", replay.logits, y, checks)
    restoration = backend.run(b, c_source=natural.c[s], at_source=clean.at)
    run.compare("upstream_unchanged_by_AT_restoration", restoration.c, direct.c, checks)
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("Backbone not frozen")
    if any(layer._forward_hooks for layer in backend.model.model.layers):
        raise ValueError("Leaked intervention hooks")
    receipt = dict(passed=True, checks=checks, frozen_weights=True, hook_cleanup=True)
    run.write(root / f"reference_{time.time_ns()}.json", receipt)
    return receipt


def evaluate(root, method, backend, contexts, pairs, natural, *, batch=32, readouts=None):
    b, s = (np.array([getattr(p, key) for p in pairs]) for key in ("base", "source"))
    target = [contexts[bi].label_for(contexts[si].color) for bi, si in zip(b, s)]
    base_target = [contexts[i].answer for i in b]
    source_target = [contexts[i].answer for i in s]
    baseline = run.operation(root, "matched_base", backend, b, base_target, batch)
    direct = run.operation(root, method + "_direct", backend, b, target, batch, c_source=natural.c[s])
    replay = run.operation(root, method + "_replay", backend, b, target, batch, at_source=direct.at)
    restoration = run.operation(root, method + "_restoration", backend, b, base_target, batch,
        c_source=natural.c[s], at_source=baseline.at)
    natural_at = run.operation(root, "natural_AT", backend, b, source_target, batch, at_source=natural.at[s])
    predictions = {name: trace.predictions for name, trace in dict(direct=direct, replay=replay,
        restoration=restoration, natural_at=natural_at).items()}
    changing = np.array([contexts[bi].color != contexts[si].color for bi, si in zip(b, s)])
    joint = predictions["direct"]["correct"] & predictions["replay"]["correct"]
    results = dict(method=method, scope="descriptive only, all locked pairs retained", clean_gate_passed=False,
        certification_passed=False, coverage=schema.coverage(contexts, pairs),
        unique_context_baseline=core.measure(natural.predictions), matched_base_baseline=core.measure(baseline.predictions),
        operations={}, families={}, transitions={}, direct_and_replay=dict(changing=core.rate(joint[changing]), preservation=core.rate(joint[~changing])),
        restored_coordinates_max_error=float(((restoration.at-baseline.at) @ backend.at_basis.cpu()).abs().max()),
        upstream_invariance_max_error=float((restoration.c-direct.c).abs().max()))
    for name, prediction in predictions.items():
        split = changing if name != "natural_at" else np.array(base_target) != np.array(source_target)
        results["operations"][name] = dict(all=core.measure(prediction), changing=core.measure(prediction, split), preservation=core.measure(prediction, ~split))
    for field, values in (("families", np.array([p.family for p in pairs])),
                          ("transitions", np.array([f"{contexts[bi].color}->{contexts[si].color}" for bi, si in zip(b, s)]))):
        for value in sorted(set(values)):
            mask = values == value
            results[field][str(value)] = {name: core.measure(pred, mask) for name, pred in predictions.items()}
            results[field][str(value)]["joint"] = core.rate(joint[mask])
            results[field][str(value)]["matched_base_baseline"] = core.measure(baseline.predictions, mask)
    if readouts:
        backend.readouts = readouts
        core._readouts(backend, natural, direct, [c.answer for c in contexts], target, changing, results)
    arrays = {f"{name}_{key}": value for name, pred in predictions.items() for key, value in pred.items()}
    arrays.update({f"matched_base_{key}": value for key, value in baseline.predictions.items()})
    arrays.update(base=b, source=s, target=np.array(target), base_target=np.array(base_target), source_target=np.array(source_target),
        pair_id=np.array([p.id for p in pairs]), cluster=np.array([p.cluster for p in pairs]),
        family=np.array([p.family for p in pairs]), changing=changing)
    for name, prediction in predictions.items():
        arrays[f"{name}_probability_change"] = prediction["label_probabilities"] - baseline.predictions["label_probabilities"]
    directory = root / method
    directory.mkdir(exist_ok=True)
    path = directory / "predictions.npz"
    if not path.exists():
        np.savez_compressed(path, **arrays)
    else:
        saved = np.load(path, allow_pickle=False)
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise ValueError("Existing descriptive predictions differ")
    run.write(directory / "results.json", results)
    return results, arrays


def report(root, results, arrays, profile, elapsed):
    a, g = arrays["standard"], arrays["gradual"]
    comparison = {family: core.paired_comparison(a, g, mask=a["family"] == family) for family in run.FAMILIES}
    for name, mask in (("C_changing", a["changing"]), ("C_preservation", ~a["changing"])):
        comparison[name] = core.paired_comparison(a, g, mask=mask)
    run.write(root / "paired_comparison.json", comparison)
    n, p = results["standard"]["coverage"]["contexts"], results["standard"]["coverage"]["pairs"]
    run.write(root / "extended_workload.json", dict(contexts=n, pairs=p, clean_donor_cases=n*n,
        cross_recipient_per_method=p*n, independent_assignment_per_method=p*n,
        full_transformer_examples=n*n+4*p*n,
        naive_gpu_hours_including_25_percent=(n*n+4*p*n)*profile["sustained_seconds_per_pair"]*1.25/3600,
        state="NOT run. Requires separately verified exact final-layer-tail optimization or measured compute approval; no silent sampling."))
    lines = ["# Descriptive Frozen-Handle Comparison", "",
        "The clean-bank gate FAILED and remains failed. All 1,600 locked pairs are retained. These are reconstructed shared-code pairs, not certified original pairs or a new heldout test.", "",
        "| Family | Method | Direct C | AT replay | Both correct | Restoration | Natural AT swap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for family in run.FAMILIES:
        for method in ("standard", "gradual"):
            row = results[method]["families"][family]
            lines.append(f"| {family} | {method} | " + " | ".join(f"{row[key]['accuracy']:.2%}" for key in
                ("direct", "replay", "joint", "restoration", "natural_at")) + " |")
    base = results["standard"]
    lines += ["", "## Baseline and Uncertainty", "",
        f"Unique-context strict baseline: {base['unique_context_baseline']['correct']}/{n} ({base['unique_context_baseline']['accuracy']:.2%}); ties count as failures.",
        f"Batch-matched base baseline: {base['matched_base_baseline']['correct']}/{p} ({base['matched_base_baseline']['accuracy']:.2%}). All probability changes and restoration donors use these matching layouts.",
        f"There are {base['coverage']['clusters']} exact color-masked first-line template clusters. Paired intervals use 2,000 cluster resamples. Each family contains 400 pairs.", "",
        "| Family | Direct difference (gradual - standard) | Replay difference | Joint difference |",
        "|---|---:|---:|---:|"]
    for family in (*run.FAMILIES, "C_changing", "C_preservation"):
        values = []
        for endpoint in ("direct", "replay", "joint"):
            e = comparison[family]["endpoints"][endpoint]
            interval = e["percentile_95"]
            values.append(f"{100*e['difference']:+.2f} pp" +
                (f" [{100*interval[0]:+.2f}, {100*interval[1]:+.2f}]" if interval is not None else " [interval unavailable]"))
        lines.append(f"| {family} | " + " | ".join(values) + " |")
    lines += ["", "## Limits", "",
        "One uploaded training seed and different checkpoint-selection rules prevent a controlled general claim about training methods. Shared AT checkpoint/readout identities are verified.",
        "The uploaded AT classifier was a teaching/selection signal for gradual DAS, not an independent readout. Its accuracy cannot independently validate internal faithfulness.",
        "Natural AT swaps use each pair's source; they do not replace matched target-letter clean-donor controls. Exhaustive cross-question recipient and independent-assignment controls are NOT run.",
        "Restoration and continuous AT-coordinate errors are reported separately. BF16 round-trip errors prevent calling a nonzero residual exact vector restoration.",
        "Baseline failures remain in every denominator. No prompt, pair, checkpoint, scoring tolerance or scientific setting was replaced or retuned.", "",
        f"This worker's elapsed wall time: {elapsed/60:.2f} minutes. Thermal waiting and per-operation progress are preserved under engineering/ and operations/."]
    path = root / "RESULTS.md"
    if not path.exists():
        path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    root, start = ROOT, time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    try:
        contexts, pairs, natural, profile, identity = prepare(root)
        runtime.configure(root)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(run.MODEL, local_files_only=True)
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(run.MODEL, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        ro = torch.load(run.SHARED / "results_joint_randomized/at_readout_LogisticRegression.pt", map_location="cpu", weights_only=True)
        results, arrays = {}, {}
        for method in ("standard", "gradual"):
            backend = CoolingBackend(model, tokenizer, contexts, run.checkpoint(method, "C"),
                run.checkpoint(method, "AT"), run.read(run.DATA / "protocol.json")["label_token_ids"], root=root)
            check_references(root, backend, natural, pairs)
            readouts = dict(uploaded_teaching=dict(W=ro["W_AT"], b=ro["b_AT"], role="teaching; NOT independent",
                used_in_training=(method == "gradual"), used_in_selection=(method == "gradual")))
            print(f"Evaluating {method}: all 1600 pairs; failed baseline explicitly retained", flush=True)
            results[method], arrays[method] = evaluate(root, method, backend, contexts, pairs, natural, readouts=readouts)
            print(json.dumps(dict(method=method, changing=results[method]["direct_and_replay"]["changing"])), flush=True)
        report(root, results, arrays, profile, time.perf_counter()-start)
        artifacts = {str(p.relative_to(root)): schema.file_hash(p) for p in root.rglob("*") if p.is_file()
            and p.name in ("results.json", "predictions.npz", "paired_comparison.json", "RESULTS.md", "identity.json", "baseline_gate.json", "selected_pairs.json")}
        run.write(root / "completion.json", dict(identity=identity, seconds=time.perf_counter()-start,
            artifacts=artifacts, descriptive_complete=True, clean_gate_passed=False, certification_passed=False,
            exhaustive_controls_completed=False))
        run.write(root / "status.json", dict(stage="descriptive_complete", pid=os.getpid(), time=time.time()), immutable=False)
        print("Descriptive comparison COMPLETE; certification remains FAILED", flush=True)
    except BaseException as error:
        run.write(root / f"stop_{time.time_ns()}.json", dict(error=str(error), traceback=traceback.format_exc(), time=time.time()))
        run.write(root / "status.json", dict(stage="paused" if isinstance(error, runtime.Paused) else "failed",
            reason=str(error), pid=os.getpid(), time=time.time()), immutable=False)
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
