"""Output-only DAS candidates, epoch checkpoints, and calibration-only selection."""
from pathlib import Path
import time
import numpy as np
import torch

from . import backend
from .localization import states, equal_cell_score, boundary
from .matching import select_winner, select_chain
from .metrics import breakdown
from .protocol import LABELS, PROTOCOL
from .storage import read, write, save_torch, checked_torch, file_hash


def candidate_id(variable, layer, position, rank, seed):
    return f"{variable}_L{layer:02d}_P{position:02d}_R{rank:04d}_S{seed}"


def paired_order(size, seed, epoch):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epoch):
        order = torch.randperm(size, generator=generator)
    return order


def measure(model, ids, source, basis, layer, position, target, rows, batch, variable):
    predictions, losses = [], []
    with torch.no_grad():
        for start in range(0, len(ids), batch):
            patch = backend.Patch(layer, position, source[start:start+batch], basis)
            logits, _ = backend.forward(model, ids[start:start+batch].to("cuda"), [patch])
            predictions.extend(logits.argmax(-1).cpu().tolist())
            losses.extend(torch.nn.functional.cross_entropy(logits, target[start:start+batch].to("cuda"), reduction="none").cpu().tolist())
    correct = np.array(predictions) == target.numpy()
    return {"cal_accuracy": equal_cell_score(correct, rows), "predictions": predictions,
            "mean_ce": float(np.mean(losses)), "breakdown": breakdown(rows, correct, variable)}


def recover_epochs(root, folder, config):
    """Recover a fully saved checkpoint even if its JSON/pointer commit was interrupted."""
    root, folder = Path(root), Path(folder)
    history = []
    for epoch in range(1, 21):
        path = folder / f"epoch_{epoch:02d}.pt"
        metadata = path.with_suffix(".json")
        if not path.exists():
            if metadata.exists():
                raise ValueError("Epoch receipt exists without its checkpoint")
            if any((folder / f"epoch_{later:02d}.pt").exists() for later in range(epoch+1, 21)):
                raise ValueError("Checkpoint history has an epoch gap")
            break
        sha = file_hash(path)
        if metadata.exists():
            record = read(metadata)["record"]
            if record["sha256"] != sha or record["epoch"] != epoch:
                raise ValueError("Epoch checkpoint identity differs")
            if any(record.get(key) != value for key, value in config.items()):
                raise ValueError("Epoch receipt configuration differs")
        else:
            checkpoint = checked_torch(path, sha)
            if checkpoint["config"] != config or checkpoint["epoch"] != epoch:
                raise ValueError("Interrupted checkpoint configuration differs")
            if "calibration_predictions" not in checkpoint:
                raise ValueError("Interrupted checkpoint lacks its calibration predictions")
            record = dict(checkpoint["history"][-1], sha256=sha)
            write(metadata, {"record": record, "calibration_predictions": checkpoint["calibration_predictions"]}, immutable=True)
        history.append(record)
    if history:
        last = history[-1]
        write(folder / "latest.json", {"path": last["checkpoint"], "sha256": last["sha256"]})
        write(folder / "history.json", history)
    return history


def fit(model, tokenizer, root, variable, layer, position, rank, seed, batch):
    root = Path(root)
    name = candidate_id(variable, layer, position, rank, seed)
    folder = root / "training" / name
    train_rows, cal_rows = (read(root / "banks" / f"{name}.json") for name in ("train", "calibration"))
    config = {"variable": variable, "layer": layer, "position": position, "rank": rank, "seed": seed,
              "physical_batch": batch, "registration": file_hash(root / "registration.json"),
              "train_sha256": file_hash(root / "banks/train.json"), "cal_sha256": file_hash(root / "banks/calibration.json")}
    write(folder / "config.json", config, immutable=True)
    if (folder / "completed.json").exists():
        receipt = read(folder / "completed.json")
        for row in receipt["epochs"]:
            if file_hash(root / row["checkpoint"]) != row["sha256"]:
                raise ValueError("Completed checkpoint changed")
        return receipt["epochs"]
    datasets = []
    for bank_name, rows in (("train", train_rows), ("calibration", cal_rows)):
        ids = backend.tokenize(tokenizer, [r["base"]["text"] for r in rows])
        source = states(model, tokenizer, [r["source"] for r in rows], [layer], batch,
                        root / "cache" / f"{bank_name}_sources_L{layer:02d}.pt")[layer][:, position]
        from .answers import answer_ids
        labels = answer_ids(tokenizer)
        target = torch.tensor([labels[r["targets"][variable]] for r in rows])
        datasets.append((ids, source, target))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    subspace = backend.LearnedSubspace(2304, rank).to("cuda")
    optimizer = torch.optim.Adam(subspace.parameters(), lr=0.01)
    start_epoch, history = 1, recover_epochs(root, folder, config)
    latest = folder / "latest.json"
    if latest.exists():
        receipt = read(latest)
        checkpoint = checked_torch(root / receipt["path"], receipt["sha256"])
        if checkpoint["config"] != config:
            raise ValueError("Optimizer/RNG resume configuration differs")
        subspace.load_state_dict(checkpoint["subspace"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_epoch = checkpoint["epoch"] + 1
        for row in history:
            if file_hash(root / row["checkpoint"]) != row["sha256"]:
                raise ValueError("Historical epoch changed during resume")
    versions = {n: p._version for n, p in model.named_parameters()}
    ids, source, targets = datasets[0]
    for epoch in range(start_epoch, 21):
        started = time.perf_counter()
        order = paired_order(len(ids), seed, epoch)
        predictions, indices, losses, gradients = [], [], [], []
        for update_start in range(0, len(order), 32):
            effective = order[update_start:update_start+32]
            optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(effective), batch):
                chosen = effective[offset:offset+batch]
                logits, _ = backend.forward(model, ids[chosen].to("cuda"),
                    [backend.Patch(layer, position, source[chosen], subspace.basis())])
                loss = torch.nn.functional.cross_entropy(logits, targets[chosen].to("cuda"), reduction="sum") / len(effective)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; current epoch not committed")
                loss.backward()
                predictions.extend(logits.detach().argmax(-1).cpu().tolist())
                indices.extend(chosen.tolist())
                losses.append(float(loss.detach()))
            gradients.append(float(subspace.raw.grad.norm().detach()))
            if not torch.isfinite(subspace.raw.grad).all():
                raise RuntimeError("Nonfinite basis gradient; current epoch not committed")
            optimizer.step()
        cal_ids, cal_source, cal_targets = datasets[1]
        cal = measure(model, cal_ids, cal_source, subspace.basis().detach(), layer, position, cal_targets, cal_rows, batch, variable)
        correct = np.array(predictions) == targets[indices].numpy()
        changing = np.array([train_rows[i][variable + "_changed"] for i in indices])
        record = {**config, "epoch": epoch, "cal_accuracy": cal["cal_accuracy"], "cal_ce": cal["mean_ce"],
                  "train_online_changing_accuracy": float(correct[changing].mean()),
                  "train_online_preservation_accuracy": float(correct[~changing].mean()),
                  "train_ce": sum(losses) / (len(order) / 32), "gradient_norm_mean": float(np.mean(gradients)),
                  "seconds": time.perf_counter()-started}
        record["train_breakdown"] = breakdown([train_rows[i] for i in indices], correct, variable)
        record["cal_breakdown"] = cal["breakdown"]
        checkpoint_path = folder / f"epoch_{epoch:02d}.pt"
        record["checkpoint"] = checkpoint_path.relative_to(root).as_posix()
        history.append(record)
        checkpoint = {"config": config, "epoch": epoch, "subspace": subspace.state_dict(),
                      "basis": subspace.basis().detach().cpu(), "optimizer": optimizer.state_dict(),
                      "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                      "next_order": paired_order(len(ids), seed, epoch+1), "history": history,
                      "calibration_predictions": cal["predictions"]}
        save_torch(checkpoint_path, checkpoint)
        record["sha256"] = file_hash(checkpoint_path)
        write(folder / f"epoch_{epoch:02d}.json", {"record": record, "calibration_predictions": cal["predictions"]}, immutable=True)
        write(latest, {"path": record["checkpoint"], "sha256": record["sha256"]})
        write(folder / "history.json", history)
        backend.assert_frozen(model, versions)
        print(f"{name} epoch {epoch}/20 calibration {cal['cal_accuracy']:.4f}", flush=True)
        boundary(root)
    write(folder / "completed.json", {"epochs": history}, immutable=True)
    return history


def run_search(model, tokenizer, root, batch):
    root = Path(root)
    shortlist = read(root / "selections/shortlists.json")["shortlists"]
    candidates = []
    for variable in ("C", "AT"):
        sites = sorted({(s["layer"], s["position"]) for sites in shortlist[variable].values() for s in sites})
        for layer, position in sites:
            for rank in PROTOCOL["ranks"]:
                candidates.extend(fit(model, tokenizer, root, variable, layer, position, rank, 1, batch))
    winners = {}
    for variable in ("C", "AT"):
        winners[variable] = {}
        for method, sites in shortlist[variable].items():
            allowed = {(s["layer"], s["position"]) for s in sites}
            pool = [c for c in candidates if c["variable"] == variable and (c["layer"], c["position"]) in allowed]
            winners[variable][method] = select_winner(pool)
    # First choose the epoch for each fixed configuration using the declared AP-style tie rule.
    configurations = {}
    for candidate in candidates:
        key = tuple(candidate[k] for k in ("variable", "layer", "position", "rank", "seed"))
        configurations.setdefault(key, []).append(candidate)
    chain = select_chain([select_winner(pool) for pool in configurations.values()])
    write(root / "selections/seed1.json", {"method_winners": winners, "chain": chain,
          "calibration_table": candidates}, immutable=True)
    repeats = {"1": chain}
    if chain is not None:
        for seed in (0, 2):
            repeats[str(seed)] = {}
            for variable in ("C", "AT"):
                c = chain[variable]
                history = fit(model, tokenizer, root, variable, c["layer"], c["position"], c["rank"], seed, batch)
                repeats[str(seed)][variable] = select_winner(history)
    write(root / "selections/chain_seeds.json", repeats, immutable=True)
    files = {"selections/seed1.json": file_hash(root / "selections/seed1.json"),
             "selections/chain_seeds.json": file_hash(root / "selections/chain_seeds.json")}
    chosen = [c for methods in winners.values() for c in methods.values()]
    chosen += [c for pair in repeats.values() if pair is not None for c in pair.values()]
    files.update({c["checkpoint"]: c["sha256"] for c in chosen})
    # Independent classifiers are selected and locked before final filtering too.
    from .readouts import fit as fit_readout
    unique = {candidate_id(c["variable"], c["layer"], c["position"], c["rank"], c["seed"]): c for c in chosen}
    for name, candidate in unique.items():
        fit_readout(model, tokenizer, root, candidate, batch)
        for suffix in (".pt", ".json"):
            filename = "readouts/" + name + suffix
            files[filename] = file_hash(root / filename)
    write(root / "selections/all_locked.json", {"registration": file_hash(root / "registration.json"),
          "files": files, "no_chain_compatible_pair": chain is None}, immutable=True)
    return {"winners": winners, "chain_seeds": repeats}
