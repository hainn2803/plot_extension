"""Matched standard DAS, completed-update recovery, and calibration-only locks."""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from experiments.mcqa_c_at_localization import backend, runtime
from experiments.mcqa_c_at_localization.protocol import LABELS
from experiments.mcqa_c_at_localization.storage import read, write, digest, file_hash, save_torch, checked_torch, encoded
from experiments.mcqa_c_at_localization.training import candidate_id, paired_order
from experiments.mcqa_c_at_localization.localization import equal_cell_score
from experiments.mcqa_c_at_localization.metrics import breakdown
from experiments.mcqa_c_at_localization.matching import select_winner, select_chain
from .protocol import ids_for


def tensor_digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(encoded(["tensor", str(v.dtype), list(v.shape)]))
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            h.update(b"dict")
            for k in sorted(v, key=repr):
                visit(k)
                visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(encoded([type(v).__name__, len(v)]))
            for item in v:
                visit(item)
        else:
            h.update(encoded([type(v).__name__, v]))
    visit(value)
    return h.hexdigest()


def validate_state(state, config, size):
    assert state["config"] == config
    updates = size // 32
    assert 1 <= state["epoch"] <= 21 and 0 <= state["update"] <= updates
    assert state["step"] == (state["epoch"] - 1) * updates + state["update"]
    assert len(state["online"]) == state["update"]
    order = paired_order(size, config["seed"], state["epoch"]).tolist()
    assert sum((u["indices"] for u in state["online"]), []) == order[:state["update"] * 32]
    assert len(state["history"]) == state["epoch"] - 1
    for i, row in enumerate(state["history"]):
        assert row["epoch"] == i + 1
    if "subspace" in state:
        assert all(torch.isfinite(v).all() for v in state["subspace"].values())
        assert torch.isfinite(state["basis"]).all()


def save_state(folder, state):
    path = folder / "checkpoints" / f"s{state['step']:06d}_e{state['epoch']:02d}_u{state['update']:02d}_{time.time_ns()}.pt"
    payload = {"state": state, "payload_digest": tensor_digest(state)}
    save_torch(path, payload)
    receipt = {"sha256": file_hash(path), "config": state["config"], "step": state["step"],
               "epoch": state["epoch"], "update": state["update"], "payload_digest": payload["payload_digest"]}
    write(path.with_suffix(".json"), receipt, immutable=True)
    write(folder / "latest.json", {"path": path.relative_to(folder).as_posix(), **receipt})
    return path, receipt


def recover(folder, config, size):
    candidates = []
    for path in sorted((folder / "checkpoints").glob("*.pt")):
        receipt_path = path.with_suffix(".json")
        if receipt_path.exists():
            receipt = read(receipt_path)
            assert receipt["config"] == config and receipt["sha256"] == file_hash(path), "Checkpoint corruption or identity mismatch"
            candidates.append((receipt["step"], receipt["epoch"], path.name, path, receipt))
        else:
            payload = checked_torch(path)
            state = payload["state"]
            validate_state(state, config, size)
            assert tensor_digest(state) == payload["payload_digest"], "Orphan checkpoint payload changed"
            receipt = {"sha256": file_hash(path), "config": config, "step": state["step"], "epoch": state["epoch"],
                       "update": state["update"], "payload_digest": payload["payload_digest"]}
            write(receipt_path, receipt, immutable=True)
            candidates.append((receipt["step"], receipt["epoch"], path.name, path, receipt))
    if not candidates:
        return None
    _, _, _, path, receipt = max(candidates, key=lambda v: v[:3])
    payload = checked_torch(path, receipt["sha256"])
    state = payload["state"]
    validate_state(state, config, size)
    assert tensor_digest(state) == receipt["payload_digest"] == payload["payload_digest"]
    write(folder / "latest.json", {"path": path.relative_to(folder).as_posix(), **receipt})
    return state, path, receipt


def optimizer_step(model, subspace, optimizer, ids, source, targets, effective, layer, position, batch, session, forward=backend.forward):
    device = subspace.raw.device
    optimizer.zero_grad(set_to_none=True)
    predictions, losses = [], []
    for offset in range(0, len(effective), batch):
        session.check()
        chosen = effective[offset:offset + batch]
        logits, _ = forward(model, ids[chosen].to(device), [backend.Patch(layer, position, source[chosen], subspace.basis())])
        loss = torch.nn.functional.cross_entropy(logits, targets[chosen].to(device), reduction="sum") / len(effective)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training CE; no optimizer update committed")
        loss.backward()
        predictions.extend(logits.detach().argmax(-1).cpu().tolist())
        losses.append(float(loss.detach()))
        del logits, loss
    if subspace.raw.grad is None or not torch.isfinite(subspace.raw.grad).all():
        raise RuntimeError("Missing/nonfinite basis gradient; no optimizer update committed")
    norm = float(subspace.raw.grad.detach().norm())
    if not np.isfinite(norm):
        raise RuntimeError("Nonfinite gradient norm; no clipping or optimizer update")
    optimizer.step()
    if not torch.isfinite(subspace.raw).all():
        raise RuntimeError("Nonfinite optimizer parameters; stop and preserve previous checkpoints")
    return {"indices": effective.tolist(), "predictions": predictions, "ce": sum(losses), "gradient_norm": norm}


def cache_features(model, root, name, batch, sites, identity, session, progress):
    rows = read(root / "banks" / (name + ".json"))
    side = "source" if name in ("train", "calibration") else None
    ids = ids_for(root, name, rows, side=side)
    all_features = {f"L{l:02d}_P{p:02d}": torch.empty((len(rows), 2304), dtype=torch.bfloat16) for l,p in sites}
    for start in range(0, len(rows), batch):
        session.check()
        path = root / "cache" / name / f"b{batch}_{start:04d}.pt"
        ident = {"execution": identity, "bank": name, "start": start, "batch": batch, "sites": [list(s) for s in sites]}
        if path.with_suffix(".json").exists():
            receipt = read(path.with_suffix(".json"))
            assert receipt["identity"] == ident
            values = checked_torch(path, receipt["sha256"])
        else:
            with torch.no_grad():
                logits, hidden = backend.forward(model, ids[start:start + batch].to("cuda"), capture=sorted({l for l,p in sites}))
                assert torch.isfinite(logits).all()
                values = {f"L{l:02d}_P{p:02d}": hidden[l][:,p].detach().cpu().clone() for l,p in sites}
                assert all(torch.isfinite(v).all() for v in values.values())
                label_ids = read(root / "token_map.json")["label_ids"]
                prompts = [r[side] for r in rows[start:start + batch]] if side else rows[start:start + batch]
                expected = [label_ids[LABELS.index(p["AT"])] for p in prompts]
                assert logits.argmax(-1).cpu().tolist() == expected, "Previously clean natural prompt changed under new execution shape; do not refilter"
                del logits, hidden
            save_torch(path, values)
            write(path.with_suffix(".json"), {"identity": ident, "sha256": file_hash(path)}, immutable=True)
        for key, value in values.items():
            all_features[key][start:start + batch] = value
        if (start // batch) % 32 == 0:
            progress(root, "source_features", bank=name, done=start + batch, total=len(rows))
    return all_features


def measure(model, root, rows, ids, source, basis, candidate, batch, session):
    label_ids = read(root / "token_map.json")["label_ids"]
    target = torch.tensor([label_ids[LABELS.index(r["targets"][candidate["variable"]])] for r in rows])
    predictions, losses = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            session.check()
            patch = backend.Patch(candidate["layer"], candidate["position"], source[start:start + batch], basis)
            logits, _ = backend.forward(model, ids[start:start + batch].to("cuda"), [patch])
            if not torch.isfinite(logits).all():
                raise RuntimeError("Nonfinite calibration logits")
            predictions.extend(logits.argmax(-1).cpu().tolist())
            losses.extend(torch.nn.functional.cross_entropy(logits, target[start:start + batch].to("cuda"), reduction="none").cpu().tolist())
            del logits
    correct = np.array(predictions) == target.numpy()
    return {"cal_accuracy": equal_cell_score(correct, rows), "mean_ce": float(np.mean(losses)),
            "predictions": predictions, "breakdown": breakdown(rows, correct, candidate["variable"])}


def fit(model, root, candidate, batch, identity, datasets, session, progress):
    name = candidate_id(**candidate)
    folder = root / "training" / name
    config = {**candidate, "physical_batch": batch, "execution": identity}
    write(folder / "config.json", config, immutable=True)
    if (folder / "completed.json").exists():
        done = read(folder / "completed.json")
        assert done["config"] == config
        for h in done["epochs"]:
            assert file_hash(root / h["checkpoint"]) == h["sha256"]
        return done["epochs"]
    train, cal = datasets["train"], datasets["calibration"]
    rows, ids, features = train
    key = f"L{candidate['layer']:02d}_P{candidate['position']:02d}"
    label_ids = read(root / "token_map.json")["label_ids"]
    targets = torch.tensor([label_ids[LABELS.index(r["targets"][candidate["variable"]])] for r in rows])
    torch.manual_seed(candidate["seed"])
    torch.cuda.manual_seed_all(candidate["seed"])
    subspace = backend.LearnedSubspace(2304, candidate["rank"]).to("cuda")
    optimizer = torch.optim.Adam(subspace.parameters(), lr=.01)
    def snapshot(state):
        state.update(subspace=subspace.state_dict(), basis=subspace.basis().detach().cpu(), optimizer=optimizer.state_dict(),
                     cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        return save_state(folder, state)
    loaded = recover(folder, config, len(rows))
    if loaded is None:
        state = {"config": config, "epoch": 1, "update": 0, "step": 0, "online": [], "history": [], "epoch_seconds": 0.}
        snapshot(state)
    else:
        state = loaded[0]
        subspace.load_state_dict(state["subspace"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    versions = {n:p._version for n,p in model.named_parameters()}
    try:
        while state["epoch"] <= 20:
            epoch = state["epoch"]
            order = paired_order(len(rows), candidate["seed"], epoch)
            begun = time.monotonic()
            for update in range(state["update"], len(rows)//32):
                effective = order[update*32:(update+1)*32]
                measured = optimizer_step(model, subspace, optimizer, ids, features[key], targets, effective,
                                          candidate["layer"], candidate["position"], batch, session)
                state["online"].append(measured)
                state["update"] += 1
                state["step"] += 1
                if state["step"] % 25 == 0:
                    state["epoch_seconds"] += time.monotonic() - begun
                    snapshot(state)
                    begun = time.monotonic()
                    progress(root, "training", candidate=name, epoch=epoch, update=state["update"], step=state["step"])
            cal_rows, cal_ids, cal_features = cal
            cal_result = measure(model, root, cal_rows, cal_ids, cal_features[key], subspace.basis().detach(), candidate, batch, session)
            state["epoch_seconds"] += time.monotonic() - begun
            indices = sum((u["indices"] for u in state["online"]), [])
            predictions = sum((u["predictions"] for u in state["online"]), [])
            correct = np.array(predictions) == targets[indices].numpy()
            record = {**candidate, "epoch": epoch, "cal_accuracy": cal_result["cal_accuracy"], "cal_ce": cal_result["mean_ce"],
                      "cal_breakdown": cal_result["breakdown"], "calibration_predictions": cal_result["predictions"],
                      "train_online_breakdown": breakdown([rows[i] for i in indices], correct, candidate["variable"]),
                      "train_ce": float(np.mean([u["ce"] for u in state["online"]])),
                      "gradient_norm_mean": float(np.mean([u["gradient_norm"] for u in state["online"]])),
                      "seconds": state["epoch_seconds"]}
            state["history"].append(record)
            state.update(epoch=epoch+1, update=0, online=[], epoch_seconds=0.)
            path, receipt = snapshot(state)
            write(folder / "epochs" / f"epoch_{epoch:02d}.json", {"record": {**record, "checkpoint": path.relative_to(root).as_posix(),
                  "sha256": receipt["sha256"]}}, immutable=True)
            backend.assert_frozen(model, versions)
            progress(root, "training", candidate=name, completed_epoch=epoch, total_epochs=20,
                     calibration=record["cal_accuracy"], calibration_changing=record["cal_breakdown"]["changing"],
                     calibration_preservation=record["cal_breakdown"]["preservation"], epoch_seconds=record["seconds"])
    except BaseException as error:
        # A partial microbatch never steps Adam or enters online metrics; its update is replayed.
        if isinstance(error,(runtime.Paused,TimeoutError,KeyboardInterrupt)):
            if state["epoch"]==epoch:
                state["epoch_seconds"] += time.monotonic()-begun
            snapshot(state)
        raise
    history = []
    for epoch in range(1,21):
        path = folder / "epochs" / f"epoch_{epoch:02d}.json"
        if not path.exists():
            matches = []
            for ckpt in (folder / "checkpoints").glob(f"*_e{epoch+1:02d}_u00_*.pt"):
                receipt = read(ckpt.with_suffix(".json"))
                matches.append((ckpt, receipt))
            assert matches, "Missing epoch commit and recoverable state"
            ckpt, receipt = sorted(matches, key=lambda p:p[0].name)[-1]
            saved = checked_torch(ckpt, receipt["sha256"])["state"]
            write(path, {"record": {**saved["history"][epoch-1], "checkpoint": ckpt.relative_to(root).as_posix(), "sha256": receipt["sha256"]}}, immutable=True)
        history.append(read(path)["record"])
    write(folder / "completed.json", {"config":config, "epochs":history}, immutable=True)
    return history


def selection(root, all_candidates):
    shortlists = read(root / "selections/shortlists.json")["shortlists"]
    winners = {}
    for variable in ("C", "AT"):
        winners[variable] = {}
        for method, sites in shortlists[variable].items():
            allowed = {(s["layer"],s["position"]) for s in sites}
            winners[variable][method] = select_winner([c for c in all_candidates if c["variable"]==variable and (c["layer"],c["position"]) in allowed])
    configs = {}
    for c in all_candidates:
        key = tuple(c[k] for k in ("variable","layer","position","rank","seed"))
        configs.setdefault(key,[]).append(c)
    chain = select_chain([select_winner(pool) for pool in configs.values()])
    result = {"method_winners": winners, "chain": chain, "calibration_table": all_candidates}
    write(root / "selections/seed1.json", result, immutable=True)
    return result
