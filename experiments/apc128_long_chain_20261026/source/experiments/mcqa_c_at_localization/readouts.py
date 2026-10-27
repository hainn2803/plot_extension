"""Independent standardized linear measurements; never used to select handles."""
from pathlib import Path
import time
import torch

from .localization import states
from .protocol import COLORS, LABELS
from .storage import read, file_hash, checked_torch, save_torch, write
from .training import candidate_id


def fit(model, tokenizer, root, chosen, batch):
    begun = time.perf_counter()
    root = Path(root)
    variable = chosen["variable"]
    name = candidate_id(variable, chosen["layer"], chosen["position"], chosen["rank"], chosen["seed"])
    path = root / "readouts" / (name + ".pt")
    receipt_path = path.with_suffix(".json")
    identity = {"checkpoint": chosen["sha256"], "fit_bank": file_hash(root / "banks/readout_fit.json"),
                "validation_bank": file_hash(root / "banks/readout_validation.json")}
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["identity"] != identity:
            raise ValueError("Readout identity differs")
        return checked_torch(path, receipt["sha256"]), receipt
    checkpoint = checked_torch(root / chosen["checkpoint"], chosen["sha256"])
    basis = checkpoint["basis"].to("cuda")
    names = COLORS if variable == "C" else LABELS
    values = []
    for bank in ("readout_fit", "readout_validation"):
        prompts = read(root / "banks" / (bank + ".json"))
        hidden = states(model, tokenizer, prompts, [chosen["layer"]], batch,
                        root / "cache" / f"{bank}_L{chosen['layer']:02d}.pt")[chosen["layer"]][:, chosen["position"]]
        x = hidden.to("cuda").float() @ basis
        y = torch.tensor([names.index(p[variable]) for p in prompts], device="cuda")
        values.append((x.detach(), y))
    (x, y), (xc, yc) = values
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    xn, xcn = (x-mean)/std, (xc-mean)/std
    best_key, best, history = None, None, []
    for penalty in (1e-4, 1e-3, 1e-2):
        torch.manual_seed(chosen["seed"])
        torch.cuda.manual_seed_all(chosen["seed"])
        classifier = torch.nn.Linear(basis.shape[1], 4).to("cuda")
        optimizer = torch.optim.Adam(classifier.parameters(), lr=.01)
        for epoch in range(1, 201):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(classifier(xn), y) + penalty * classifier.weight.square().sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                logits = classifier(xcn)
                accuracy = float((logits.argmax(-1) == yc).float().mean())
                ce = float(torch.nn.functional.cross_entropy(logits, yc))
                key = (-accuracy, ce)
                history.append({"penalty": penalty, "epoch": epoch, "val_accuracy": accuracy, "val_ce": ce})
                if best_key is None or key < best_key:
                    best_key = key
                    weight = classifier.weight.detach() / std.unsqueeze(0)
                    bias = classifier.bias.detach() - weight @ mean
                    best = {"weight": weight.cpu().clone(), "bias": bias.cpu().clone(), "variable": variable,
                            "classes": list(names), "penalty": penalty, "epoch": epoch}
    with torch.no_grad():
        weight, bias = best["weight"].to("cuda"), best["bias"].to("cuda")
        predicted = (xc @ weight.T + bias).argmax(-1)
        by_class = {name: float((predicted[yc == i] == i).float().mean()) for i, name in enumerate(names)}
    save_torch(path, best)
    receipt = {"identity": identity, "sha256": file_hash(path), "validation_macro_accuracy": sum(by_class.values())/4,
               "validation_by_class": by_class, "history": history, "used_for_handle_selection": False,
               "elapsed_seconds": time.perf_counter() - begun}
    write(receipt_path, receipt, immutable=True)
    return best, receipt


def predict(hidden, basis, readout):
    features = hidden.float() @ basis.to(hidden.device)
    return (features @ readout["weight"].to(hidden.device).T + readout["bias"].to(hidden.device)).argmax(-1)
