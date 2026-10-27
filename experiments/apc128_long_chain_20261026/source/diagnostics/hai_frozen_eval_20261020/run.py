"""Guarded, immutable evaluation of Hai's uploaded frozen handles (no training)."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from experiments.hai_c_at_evaluation import audit, backends, core, schema
from experiments.hai_c_at_evaluation.runner import identity as evaluator_identity
from experiments.mcqa_c_at_localization import runtime

ROOT = Path("eval/hai_frozen_evaluation_20261020_v2")
DATA = Path("eval/hai_frozen_reconstruction_20261020_v1")
SHARED = Path(".external/hai_c_at_uploaded_20261020")
MODEL = Path(".cache_mcqa_c_at/hub/models--google--gemma-2-2b/snapshots/c5ebcd40d208330abc697524c919956e692655cf")
FAMILIES = ("AP_only", "AT_only", "C_only", "C_AP_AT")
ATOL = RTOL = 2e-4


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode()
    if path.exists() and immutable:
        if path.read_bytes() != data:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def checkpoint(method, variable):
    folder = "results_joint_randomized" + ("_gradual" if method == "gradual" else "")
    name = f"{folder}/das_{variable.lower()}.pt"
    expected = read(DATA / "manifest.json")["imported_artifacts"][name]
    return backends.load_checkpoint(SHARED / name, expected)


def establish_identity(root):
    evaluator_identity(Path("eval/hai_c_at_evaluation_prep_20261019_v2"))
    manifest, protocol = read(DATA / "manifest.json"), read(DATA / "protocol.json")
    for group, parent in (("artifacts", DATA), ("imported_artifacts", SHARED), ("shared_source", SHARED)):
        for name, expected in manifest[group].items():
            if schema.file_hash(parent / name) != expected:
                raise ValueError(f"Imported identity mismatch: {name}")
    for name, expected in protocol["model_files"].items():
        if schema.file_hash(MODEL / name) != expected:
            raise ValueError(f"Model/tokenizer identity mismatch: {name}")
    sources = [Path(__file__), *Path("experiments/hai_c_at_evaluation").glob("*.py"),
               Path("experiments/mcqa_c_at_localization/runtime.py"),
               Path("experiments/mcqa_c_at_localization/profiling.py"),
               SHARED / "mcqa_gradual_das.py", SHARED / "mcqa_gradual_das_c.py"]
    result = dict(source={str(p): schema.file_hash(p) for p in sources},
        reconstruction_manifest=schema.file_hash(DATA / "manifest.json"),
        evaluation_plan=schema.file_hash(DATA / "evaluation_plan.json"),
        model=protocol, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
        execution=dict(device="cuda:0", backbone="BF16 frozen", projections="FP32", attention="sdpa",
            tf32=False, numerical_atol=ATOL, numerical_rtol=RTOL,
            physical_batch_candidates=[1, 2, 4, 8, 16, 32], no_training=True),
        scope="Reconstructed common-bank frozen-handle comparison, not certified original pairs or fresh heldout generalization")
    result["engineering_parent"] = dict(path="eval/hai_frozen_evaluation_20261020_v1",
        reason="First preflight compared native BF16 softcapping to Hai's FP32 intervention softcapping. Child verifies each separately and reproduces native filtering arithmetic. No tolerance or handle changes.")
    write(root / "identity.json", result)
    return schema.digest(result)


class GuardedBackend(backends.HFBackend):
    def run(self, indices, c_source=None, at_source=None):
        runtime.guard()
        result = super().run(indices, c_source, at_source)
        runtime.guard()
        return result


class NativeFilterBackend(GuardedBackend):
    """Exactly the published predict_correct path, distinct from answer_logits."""
    @torch.no_grad()
    def run(self, indices, c_source=None, at_source=None):
        runtime.guard()
        if c_source is not None or at_source is not None:
            raise ValueError("Natural filtering cannot include an intervention")
        indices = list(map(int, indices))
        bank = padded(self, indices)
        ids, mask = bank["base_input_ids"].cuda(), bank["base_attention_mask"].cuda()
        rows = torch.arange(len(indices), device=self.device)
        c_idx = (mask == 0).sum(1) + bank["base_position_by_id"]["color"].cuda()
        captured = {}
        def capture(name, positions):
            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[name] = hidden[rows, positions].float().cpu()
            return hook
        with ExitStack() as cleanup:
            for layer, name, pos in ((self.c_layer, "c", c_idx), (self.at_layer, "at", mask.shape[1] - 1)):
                handle = self.model.model.layers[layer].register_forward_hook(capture(name, pos))
                cleanup.callback(handle.remove)
            result = self.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp(min=0), use_cache=False, logits_to_keep=1)
        runtime.guard()
        return core.Trace(captured["c"], captured["at"], result.logits[:, -1].float().cpu())


def candidates():
    rows = {family: [json.loads(line) for line in (DATA / f"unfiltered_test_pairs/{family}.jsonl").read_text().splitlines()]
            for family in FAMILIES}
    contexts, by_id, mapping = [], {}, {}
    for family in FAMILIES:
        for i, row in enumerate(rows[family]):
            sides = []
            for side in ("base", "source"):
                context = schema.parse_context(row, side)
                if context.id not in by_id:
                    by_id[context.id] = len(contexts)
                    contexts.append(context)
                sides.append(by_id[context.id])
            mapping[family, i] = sides
    return rows, contexts, mapping


def padded(backend, indices):
    width = max(len(backend.tokens[i]) for i in indices)
    ids = torch.full((len(indices), width), backend.tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, index in enumerate(indices):
        n = len(backend.tokens[index])
        ids[row, -n:] = torch.tensor(backend.tokens[index])
        mask[row, -n:] = 1
    return dict(base_input_ids=ids, base_attention_mask=mask,
        base_position_by_id=dict(color=torch.tensor([backend.color_positions[i] for i in indices]),
            last_token=torch.tensor([len(backend.tokens[i]) - 1 for i in indices])))


def compare(name, actual, expected, checks):
    difference = float((actual.detach().cpu().float() - expected.detach().cpu().float()).abs().max())
    passed = torch.allclose(actual.detach().cpu().float(), expected.detach().cpu().float(), atol=ATOL, rtol=RTOL)
    checks.append(dict(name=name, max_absolute_error=difference, passed=passed, atol=ATOL, rtol=RTOL))
    if not passed:
        raise ValueError(f"Numerical check failed: {name}; max error {difference}")


@torch.no_grad()
def preflight(root, backend):
    checks, profiles, arithmetic_diagnostics = [], [], []
    ref = audit.selected_functions(SHARED / "mcqa_gradual_das.py", ["answer_logits"])
    ref = audit.selected_functions(SHARED / "mcqa_gradual_das_c.py",
        ["run_c_intervention_and_capture", "run_frozen_mediator"], ref)
    hook_counts = [len(layer._forward_hooks) for layer in backend.model.model.layers]
    for batch in (1, 2, 4, 8, 16, 32):
        ids = list(range(batch))
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        clean = backend.run(ids)
        compare(f"batch{batch}/repeat", backend.run(ids).logits, clean.logits, checks)
        compare(f"batch{batch}/identity_C", backend.run(ids, c_source=clean.c).logits, clean.logits, checks)
        compare(f"batch{batch}/identity_AT", backend.run(ids, at_source=clean.at).logits, clean.logits, checks)
        bank = padded(backend, ids)
        x, mask = bank["base_input_ids"].cuda(), bank["base_attention_mask"].cuda()
        runtime.guard()
        capture = {}
        def raw_hook(_module, _args, output):
            capture["raw_logits"] = output.detach()
        def hidden_hook(_module, _args, output):
            capture["hidden"] = output.detach()
        with ExitStack() as cleanup:
            for module, hook in ((backend.model.lm_head, raw_hook), (backend.model.model.norm, hidden_hook)):
                handle = module.register_forward_hook(hook)
                cleanup.callback(handle.remove)
            native = backend.model(input_ids=x, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp(min=0), use_cache=False, logits_to_keep=1).logits[:, -1]
        softcap = backend.model.config.final_logit_softcapping
        expected_native = capture["raw_logits"] / softcap
        expected_native = torch.tanh(expected_native) * softcap
        compare(f"batch{batch}/native_BF16_head", native, expected_native[:, -1], checks)
        reference_head = ref["answer_logits"](backend.model, SimpleNamespace(last_hidden_state=capture["hidden"]), mask)
        compare(f"batch{batch}/intervention_FP32_head", clean.logits, reference_head, checks)
        arithmetic_diagnostics.append(dict(batch=batch,
            native_BF16_vs_intervention_FP32_max_error=float((native.float().cpu() - clean.logits).abs().max()),
            explanation="Published natural filtering and intervention scoring intentionally use different softcap dtypes; not compared as equal-arithmetic checks."))
        donor = backend.run(list(reversed(ids))).c
        direct = backend.run(ids, c_source=donor)
        y, _, at = ref["run_c_intervention_and_capture"](backend.model, bank, donor,
            backend.basis, backend.c_layer, 17, backend.at_layer, None, batch_size=batch, require_grad=False)
        compare(f"batch{batch}/C_reference_logits", direct.logits, y, checks)
        compare(f"batch{batch}/C_reference_AT", direct.at, at, checks)
        replay = backend.run(ids, at_source=direct.at)
        y = ref["run_frozen_mediator"](backend.model, bank, direct.at, backend.at_basis,
            backend.at_layer, None, batch_size=batch, require_grad=False)
        compare(f"batch{batch}/AT_replay_reference", replay.logits, y, checks)
        torch.cuda.synchronize()
        profiles.append(dict(batch=batch, seconds=time.perf_counter() - start,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        write(root / "preflight_progress.json", dict(checks=checks, profiles=profiles), immutable=False)
        print(json.dumps(dict(preflight_batch=batch, last_checks=checks[-3:], profile=profiles[-1])), flush=True)
    if hook_counts != [len(layer._forward_hooks) for layer in backend.model.model.layers]:
        raise ValueError("Forward hooks leaked")
    if any(p.requires_grad for p in backend.model.parameters()):
        raise ValueError("Backbone not frozen")
    start, iterations = time.perf_counter(), 0
    while time.perf_counter() - start < 90:
        backend.run(list(range(32)))
        iterations += 1
    runtime.guard()
    receipt = dict(passed=True, checks=checks, profiles=profiles, physical_batch=32, arithmetic_diagnostics=arithmetic_diagnostics,
        sustained_seconds=time.perf_counter() - start, sustained_batches=iterations,
        sustained_seconds_per_pair=(time.perf_counter() - start) / (iterations * 32),
        hook_cleanup=True, frozen_backbone=True, power=runtime.power_status(),
        note="Every reference uses the same padded input shapes, FP32 projector arithmetic and full-vocabulary head")
    write(root / "preflight.json", receipt)
    return receipt


def tensor_digest(value):
    import hashlib
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def operation(root, name, backend, indices, targets, batch, c_source=None, at_source=None):
    folder = root / "operations" / name
    identity = dict(indices=list(map(int, indices)), targets=list(targets), batch=batch,
        c_source=None if c_source is None else tensor_digest(c_source),
        at_source=None if at_source is None else tensor_digest(at_source),
        c_basis=None if c_source is None else tensor_digest(backend.basis),
        at_basis=tensor_digest(backend.at_basis))
    write(folder / "identity.json", identity)
    pieces, c_states, at_states = [], [], []
    for start in range(0, len(indices), batch):
        stop = min(start + batch, len(indices))
        path = folder / f"chunk_{start:06d}.pt"
        receipt_path = path.with_suffix(".json")
        if receipt_path.exists():
            if schema.file_hash(path) != read(receipt_path)["sha256"]:
                raise ValueError("Committed operation chunk changed")
            saved = torch.load(path, map_location="cpu", weights_only=True)
        else:
            trace = backend.run(indices[start:stop],
                None if c_source is None else c_source[start:stop],
                None if at_source is None else at_source[start:stop])
            predictions = core.score(trace.logits, targets[start:stop], backend)
            saved = dict(c=trace.c, at=trace.at,
                predictions={key: torch.from_numpy(value) for key, value in predictions.items()})
            save_tensor(path, saved)
            write(receipt_path, dict(sha256=schema.file_hash(path), start=start, stop=stop))
        pieces.append({k: v.numpy() for k, v in saved["predictions"].items()})
        c_states.append(saved["c"])
        at_states.append(saved["at"])
        write(root / "status.json", dict(stage=name, committed=stop, total=len(indices), time=time.time(), pid=os.getpid()), immutable=False)
        runtime.guard()
    result = core.Observed(torch.cat(c_states), torch.cat(at_states),
        {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}, 0.)
    write(folder / "completion.json", dict(count=len(indices), identity=schema.digest(identity)))
    return result


def select_bank(root, backend, rows, mapping, batch):
    selected, evidence, natural = [], {}, {}
    plan = read(DATA / "evaluation_plan.json")
    for family in FAMILIES:
        settings = plan["families"][family]
        pool = []
        for start in range(0, len(settings["candidate_row_order"]), settings["finite_chunk_size"]):
            row_ids = settings["candidate_row_order"][start:start + settings["finite_chunk_size"]]
            indices = list(dict.fromkeys(index for i in row_ids for index in mapping[family, i]))
            # Reuse is correctness-only; each family's candidate scan and stopping rule stay fixed.
            missing = [i for i in indices if i not in natural]
            if missing:
                out = operation(root, f"filter_{family}_{start:04d}", backend, missing,
                    [backend.contexts[i].answer for i in missing], batch)
                for j, i in enumerate(missing):
                    natural[i] = {key: value[j].tolist() for key, value in out.predictions.items()}
            for i in row_ids:
                b, s = mapping[family, i]
                if natural[b]["hai_correct"] and natural[s]["hai_correct"]:
                    # The published filter uses exact stripped labels, not uppercase normalization.
                    if backend.decode(natural[b]["token_id"]).strip() == backend.contexts[b].answer and backend.decode(natural[s]["token_id"]).strip() == backend.contexts[s].answer:
                        pool.append(i)
            print(json.dumps(dict(filter_family=family, scanned=min(start + 64, len(settings["candidate_row_order"])), clean=len(pool))), flush=True)
            if len(pool) >= settings["desired_clean_pairs"]:
                break
        if len(pool) < settings["desired_clean_pairs"]:
            raise ValueError(f"Finite clean-correct pool shortfall: {family}: {len(pool)}")
        random.Random(settings["seed"]).shuffle(pool)
        chosen = pool[:settings["desired_clean_pairs"]]
        evidence[family] = dict(scanned=min(start + 64, len(settings["candidate_row_order"])),
            eligible_pool=pool, selected_row_ids=chosen, seed=settings["seed"])
        selected.extend(rows[family][i] for i in chosen)
    contexts, pairs = schema.import_pairs(selected)
    write(root / "selected_pairs.json", selected)
    write(root / "selection_evidence.json", evidence)
    write(root / "filter_natural_predictions.json", natural)
    write(root / "bank_lock.json", dict(pairs_sha256=schema.file_hash(root / "selected_pairs.json"),
        selection_sha256=schema.file_hash(root / "selection_evidence.json"), coverage=schema.coverage(contexts, pairs),
        method_blind=True, caveat=read(DATA / "protocol.json")["caveat"]))
    return contexts, pairs


def primary(root, method, backend, pairs, batch):
    b, s = (np.array([getattr(p, field) for p in pairs]) for field in ("base", "source"))
    contexts = backend.contexts
    targets = [contexts[bi].label_for(contexts[si].color) for bi, si in zip(b, s)]
    base_targets = [contexts[i].answer for i in b]
    natural = operation(root, "natural", backend, np.arange(len(contexts)), [c.answer for c in contexts], batch)
    if not natural.predictions["correct"].all():
        raise ValueError("Selected clean bank no longer reproduces; no refiltering allowed")
    direct = operation(root, method + "_direct", backend, b, targets, batch, c_source=natural.c[s])
    replay = operation(root, method + "_replay", backend, b, targets, batch, at_source=direct.at)
    restoration = operation(root, method + "_restoration", backend, b, base_targets, batch,
        c_source=natural.c[s], at_source=natural.at[b])
    natural_at = operation(root, "natural_AT", backend, b, [contexts[i].answer for i in s], batch, at_source=natural.at[s])
    predictions = {name: value.predictions for name, value in dict(direct=direct, replay=replay,
        restoration=restoration, natural_at=natural_at).items()}
    changing = np.array([contexts[bi].color != contexts[si].color for bi, si in zip(b, s)])
    joint = predictions["direct"]["correct"] & predictions["replay"]["correct"]
    results = dict(method=method, coverage=schema.coverage(contexts, pairs), clean=core.measure(natural.predictions),
        families={}, operations={}, direct_and_replay=dict(changing=core.rate(joint[changing]), preservation=core.rate(joint[~changing])),
        restored_coordinates_max_error=float(((restoration.at - natural.at[b]) @ backend.at_basis.cpu()).abs().max()),
        upstream_invariance_max_error=float((restoration.c - direct.c).abs().max()))
    for name, prediction in predictions.items():
        split = changing if name != "natural_at" else np.array(base_targets) != np.array([contexts[i].answer for i in s])
        results["operations"][name] = dict(all=core.measure(prediction), changing=core.measure(prediction, split), preservation=core.measure(prediction, ~split))
    for family in FAMILIES:
        mask = np.array([p.family == family for p in pairs])
        results["families"][family] = {name: core.measure(prediction, mask) for name, prediction in predictions.items()}
        results["families"][family]["joint"] = core.rate(joint[mask])
    ro = torch.load(SHARED / "results_joint_randomized/at_readout_LogisticRegression.pt", map_location="cpu", weights_only=True)
    backend.readouts = dict(uploaded_teaching=dict(W=ro["W_AT"], b=ro["b_AT"], role="teaching, NOT independent",
        used_in_training=(method == "gradual"), used_in_selection=(method == "gradual")))
    core._readouts(backend, natural, direct, [c.answer for c in contexts], targets, changing, results)
    arrays = {f"{name}_{key}": value for name, prediction in predictions.items() for key, value in prediction.items()}
    arrays.update(base=b, source=s, target=np.array(targets), base_target=np.array(base_targets),
        pair_id=np.array([p.id for p in pairs]), cluster=np.array([p.cluster for p in pairs]), family=np.array([p.family for p in pairs]))
    for name, prediction in predictions.items():
        arrays[f"{name}_probability_change"] = prediction["label_probabilities"] - natural.predictions["label_probabilities"][b]
    output = root / method
    output.mkdir(exist_ok=True)
    np.savez_compressed(output / "predictions.npz", **arrays)
    write(output / "primary_results.json", results)
    return results, arrays


def report(root, results, arrays, profile, elapsed):
    comparison = {family: core.paired_comparison(arrays["standard"], arrays["gradual"],
        mask=arrays["standard"]["family"] == family) for family in FAMILIES}
    write(root / "paired_comparison.json", comparison)
    n, p = results["standard"]["coverage"]["contexts"], results["standard"]["coverage"]["pairs"]
    work = dict(contexts=n, pairs=p, clean_donor_combinations=n*n,
        cross_recipient_per_method=p*n, joint_assignments_per_method=p*n,
        full_transformer_total_additional_examples=n*n + 4*p*n,
        conservative_full_transformer_hours=(n*n + 4*p*n) * profile["sustained_seconds_per_pair"] * 1.25 / 3600,
        status="Not yet executed. Profile exact final-layer tail arithmetic before this exhaustive schedule; never silently subsample controls.")
    write(root / "extended_workload.json", work)
    lines = ["# Frozen C/AT Handle Evaluation", "", "Reconstructed shared-code bank; not Hai's certified original selected pairs. No retraining or checkpoint reselection.", "",
        f"Common bank: {p} pairs, {n} clean contexts, {results['standard']['coverage']['clusters']} color-masked template clusters. All four families have 400 pairs.", "",
        "| Family | Method | Direct C | Same-base AT replay | Joint | Restoration | Natural AT swap |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for family in FAMILIES:
        for method in ("standard", "gradual"):
            row = results[method]["families"][family]
            values = [row[name]["accuracy"] for name in ("direct", "replay", "joint", "restoration", "natural_at")]
            lines.append(f"| {family} | {method} | " + " | ".join(f"{v:.2%}" for v in values) + " |")
    lines.extend(["", "Intervals are paired template-cluster bootstrap (2,000 resamples), saved in paired_comparison.json. One uploaded training seed is not independent replication.", "",
        "The uploaded AT classifier entered gradual training/selection; its scores are NOT independent confirmation. Frozen checkpoint selection rules also differed across methods.", "",
        "Natural AT swaps below use each pair's natural source, not the complete matched clean-donor controls. Cross-recipient and independent-assignment controls remain pending and are not implied to pass.", "",
        "Restoration is tested after C intervention using the original base AT coordinates. Continuous coordinate error is recorded separately; BF16 round-trip error is not exact vector restoration.", "",
        f"Guarded elapsed wall time: {elapsed/60:.1f} minutes. Exhaustive full-transformer controls would require approximately {work['conservative_full_transformer_hours']:.2f} hours including 25% allowance. Inspect extended_workload.json before extending execution."])
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root, start = args.root, time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    try:
        identity = establish_identity(root)
        runtime.configure(root)
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        write(root / "environment.json", dict(transformers=transformers.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name()))
        print("Loading hash-verified frozen BF16 model", flush=True)
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda").eval()
        rows, contexts, mapping = candidates()
        backend = GuardedBackend(model, tokenizer, contexts, checkpoint("standard", "C"), checkpoint("standard", "AT"), read(DATA / "protocol.json")["label_token_ids"])
        if (root / "preflight.json").exists():
            profile = read(root / "preflight.json")
            if not profile["passed"]:
                raise ValueError("Failed saved preflight")
        else:
            profile = preflight(root, backend)
        print("Numerical preflight PASSED; selecting common bank", flush=True)
        filter_backend = NativeFilterBackend(model, tokenizer, contexts, checkpoint("standard", "C"), checkpoint("standard", "AT"), read(DATA / "protocol.json")["label_token_ids"])
        selected_contexts, pairs = select_bank(root, filter_backend, rows, mapping, profile["physical_batch"])
        results, arrays = {}, {}
        for method in ("standard", "gradual"):
            backend = GuardedBackend(model, tokenizer, selected_contexts, checkpoint(method, "C"), checkpoint(method, "AT"), read(DATA / "protocol.json")["label_token_ids"])
            print(f"Evaluating {method} frozen C handle", flush=True)
            results[method], arrays[method] = primary(root, method, backend, pairs, profile["physical_batch"])
        report(root, results, arrays, profile, time.perf_counter() - start)
        artifacts = {str(p.relative_to(root)): schema.file_hash(p) for p in root.rglob("*") if p.is_file() and p.name in
            ("primary_results.json", "predictions.npz", "paired_comparison.json", "RESULTS.md", "bank_lock.json", "preflight.json")}
        write(root / "primary_completion.json", dict(identity=identity, artifacts=artifacts, seconds=time.perf_counter() - start,
            scope="Primary frozen-handle evaluation complete; exhaustive extended controls not run", time=time.time()))
        write(root / "status.json", dict(stage="primary_complete", time=time.time(), pid=os.getpid()), immutable=False)
        print(json.dumps(dict(primary_complete=True, report=str(root / "RESULTS.md"))), flush=True)
    except BaseException as error:
        write(root / f"stop_{time.time_ns()}.json", dict(type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), time=time.time()))
        write(root / "status.json", dict(stage="paused" if isinstance(error, runtime.Paused) else "failed", reason=str(error), time=time.time(), pid=os.getpid()), immutable=False)
        raise
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
