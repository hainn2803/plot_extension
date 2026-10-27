"""Bounded local profiling and a whole-experiment, not training-only, runtime gate."""
import gc
import math
import statistics
import subprocess
import time
from pathlib import Path

import torch

from . import backend
from .data import examples
from .protocol import PROTOCOL, BANKS
from .storage import write, file_hash, read
from .verification import numerical, environment


def thermals():
    fields = "temperature.gpu,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown"
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=15)
    values = [s.strip() for s in result.stdout.splitlines()[0].split(",")]
    if len(values) != 3 or any(value not in ("Active", "Not Active") for value in values[1:]):
        raise RuntimeError("Thermal throttling status is unavailable; do not assume it is safe")
    return {"temperature_C": int(values[0]), "hw_thermal": values[1], "sw_thermal": values[2],
            "active": any(s == "Active" for s in values[1:]), "time": time.time()}


def timed(call, repeats=3):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
    return {"samples": samples, "seconds": statistics.median(samples)}


def profile_batch(model, tokenizer, batch):
    rows = examples()
    base_ids = backend.tokenize(tokenizer, [rows[i % 8]["base"]["text"] for i in range(batch)]).to("cuda")
    source_ids = backend.tokenize(tokenizer, [rows[i % 8]["source"]["text"] for i in range(batch)]).to("cuda")
    from .answers import answer_ids
    labels = answer_ids(tokenizer)
    targets = torch.tensor([labels[rows[i % 8]["targets"]["C"]] for i in range(batch)], device="cuda")
    with torch.no_grad():
        _, source_states = backend.forward(model, source_ids, capture=[0])
        source = source_states[0][:, 6].detach()
    subspace = backend.LearnedSubspace(2304, 512).to("cuda")
    optimizer = torch.optim.Adam(subspace.parameters(), lr=.01)
    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(32 // batch):
            logits, _ = backend.forward(model, base_ids, [backend.Patch(0, 6, source, subspace.basis())])
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum") / 32
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite profiling loss")
            loss.backward()
        optimizer.step()
    def clean():
        with torch.no_grad():
            backend.forward(model, base_ids)
    def swap():
        with torch.no_grad():
            backend.forward(model, base_ids, [backend.Patch(0, 6, source)])
    step()
    clean()
    torch.cuda.reset_peak_memory_stats()
    result = {"batch": batch, "effective_batch": 32, "rank": 512, "layer": 0,
              "effective_update": timed(step), "clean_forward": timed(clean),
              "full_vector_forward": timed(swap)}
    result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    # Accumulation is checked against an identical graph partition and physical shape.
    optimizer.zero_grad(set_to_none=True)
    individual_gradients = []
    for _ in range(2):
        logits, _ = backend.forward(model, base_ids, [backend.Patch(0, 6, source, subspace.basis())])
        loss = torch.nn.functional.cross_entropy(logits, targets) / 2
        individual_gradients.append(torch.autograd.grad(loss, subspace.raw)[0].detach())
    together = sum(individual_gradients)
    optimizer.zero_grad(set_to_none=True)
    del individual_gradients, logits
    for _ in range(2):
        logits, _ = backend.forward(model, base_ids, [backend.Patch(0, 6, source, subspace.basis())])
        (torch.nn.functional.cross_entropy(logits, targets) / 2).backward()
    separate = subspace.raw.grad.detach()
    if not torch.allclose(together, separate, atol=2e-4, rtol=2e-4):
        raise RuntimeError("Matched-shape gradient accumulation check failed")
    result["accumulation_max_abs_difference"] = float((together - separate).abs().max())
    result["accumulation_reference"] = "same two physical microbatches; sum of independent gradients versus accumulated backwards"
    return result


def estimate_work(batch_profile, token_count=38):
    batch = batch_profile["batch"]
    forward = batch_profile["clean_forward"]["seconds"] / batch
    swap = batch_profile["full_vector_forward"]["seconds"] / batch
    update = batch_profile["effective_update"]["seconds"]
    raw_pairs = sum(v["pairs_per_cell"] * 16 // 2 * PROTOCOL["candidate_alternatives_per_slot"] for v in BANKS.values() if "pairs_per_cell" in v)
    readout_proposals = (512 + 256) * PROTOCOL["candidate_alternatives_per_slot"]
    components = {
        "finite_clean_filtering_upper_bound": (2 * raw_pairs + readout_proposals) * forward,
        "full_vector_localization": (token_count - 1) * 26 * 384 * swap,
        "source_and_measurement_caches": (1536 + 768 + 3072 + 512 + 256) * forward,
        "52_trainings_20_epochs": 52 * 20 * (1536 // 32) * update,
        "epoch_calibration": 52 * 20 * 768 * swap,
        "heldout_direct_replay_controls_structural": 300000 * swap,
    }
    # J-lens head timing and CPU readout/runtime overhead are added by profile().
    return components


def profile(root, model=None, tokenizer=None):
    from .lens import load_lens, decode
    root = Path(root)
    if model is None:
        model, tokenizer = backend.load_model()
    versions = {n: p._version for n, p in model.named_parameters()}
    samples, temperatures = [], [thermals()]
    for batch in (1, 2, 4, 8, 16, 32):
        print(f"Profiling physical batch {batch}, early block, rank 512", flush=True)
        try:
            measured = profile_batch(model, tokenizer, batch)
            validation = numerical(model, tokenizer, batch, root)
            measured["numerical_checks"] = len(validation["checks"])
            measured["physical_device_bytes"] = torch.cuda.get_device_properties(0).total_memory
            measured["status"] = ("passed" if measured["peak_reserved_bytes"] <= measured["physical_device_bytes"]
                                  else "numerically_passed_exceeds_dedicated_memory")
        except torch.cuda.OutOfMemoryError:
            measured = {"batch": batch, "status": "out_of_memory_not_validated"}
        samples.append(measured)
        temperatures.append(thermals())
        write(root / "engineering/profile_progress.json", {"batches": samples, "thermals": temperatures})
        gc.collect()
        torch.cuda.empty_cache()
        backend.assert_frozen(model, versions)
        if temperatures[-1]["active"]:
            raise RuntimeError("Thermal throttling during profile; no local run authorized")
    passing = [s for s in samples if s["status"] == "passed"]
    if not passing:
        raise RuntimeError("No physical batch passed the memory and numerical checks")
    selected = max(passing, key=lambda s: s["batch"])
    matrices, lens_receipt = load_lens()
    ids = backend.tokenize(tokenizer, [examples()[0]["base"]["text"]] * selected["batch"]).to("cuda")
    with torch.no_grad():
        _, states = backend.forward(model, ids, capture=[0])
        lens_timing = timed(lambda: decode(model, states[0], 0, matrices))
    components = estimate_work(selected)
    components["J_lens_transport_full_sequence_head"] = 26 * 384 / selected["batch"] * lens_timing["seconds"]
    # Short profiles cannot certify sustained thermals for a long sweep.
    receipt = {"status": "profiled_not_launch_authorized", "batches": samples, "selected_batch": selected["batch"],
               "components_seconds": components, "hours_with_25_percent_allowance": sum(components.values()) * 1.25 / 3600,
               "thermals": temperatures, "sustained_thermal_gate": "pending", "environment": environment(),
               "lens": lens_receipt, "backend_sha256": file_hash(Path(backend.__file__)),
               "includes": "filtering, 962 swap sites, 52 trainings, every epoch calibration, J-lens, heldout evaluations",
               "remaining_timing_gaps": ["actual selected-site timings", "independent classifier fitting", "CPU preparation/reporting"],
               "launch_allowed": False}
    write(root / "engineering/profile.json", receipt)
    print(receipt["hours_with_25_percent_allowance"], "estimated GPU hours with allowance; sustained thermal gate still required", flush=True)
    return receipt


def correct_physical_memory_receipt(root):
    """Preserve the first profile; reject Windows shared-memory spill without rerunning it."""
    root = Path(root)
    original = read(root / "engineering/profile.json")
    write(root / "engineering/profile_before_physical_memory_gate.json", original, immutable=True)
    total = torch.cuda.get_device_properties(0).total_memory
    for sample in original["batches"]:
        if "peak_reserved_bytes" in sample:
            sample["physical_device_bytes"] = total
            if sample["peak_reserved_bytes"] > total:
                sample["status"] = "numerically_passed_exceeds_dedicated_memory"
    selected = max((s for s in original["batches"] if s["status"] == "passed"), key=lambda s: s["batch"])
    old_batch = original["selected_batch"]
    components = estimate_work(selected)
    components["J_lens_transport_full_sequence_head"] = original["components_seconds"]["J_lens_transport_full_sequence_head"] * old_batch / selected["batch"]
    original.update(selected_batch=selected["batch"], components_seconds=components,
                    hours_with_25_percent_allowance=sum(components.values())*1.25/3600,
                    memory_gate="peak reserved bytes must not exceed dedicated device memory",
                    correction="Batch 32 numerical checks pass but Windows allocation exceeds physical VRAM; excluded",
                    lens_timing_note="Conservative scaling of original batch-32 head measurement; reprofile on execution host")
    write(root / "engineering/profile.json", original)
    return original
