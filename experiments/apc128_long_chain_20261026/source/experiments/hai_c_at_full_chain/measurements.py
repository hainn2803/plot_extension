"""Independent CPU linear measurements; natural validation only selects heads."""
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from diagnostics.hai_frozen_eval_20261020 import run as old
from experiments.hai_c_at_evaluation import core, schema
from . import data
from .worker import cache_checked


def prediction(x, head):
    logits = x @ head["weight"].T + head["bias"]
    top = logits.topk(2, dim=-1)
    return logits.argmax(-1), top.values[:, 0] == top.values[:, 1]


def score(x, y, head):
    predicted, tied = prediction(x, head)
    return core.rate(((predicted == y) & ~tied).numpy())


def fit_head(x, y, xv, yv, plan, classes):
    if set(y.tolist()) != set(range(len(classes))) or set(yv.tolist()) != set(range(len(classes))):
        return None, dict(available=False, reason="Frozen clean-correct fit/validation pool lacks a class",
            fit_counts=torch.bincount(y, minlength=len(classes)).tolist(),
            validation_counts=torch.bincount(yv, minlength=len(classes)).tolist())
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    xn, xvn = (x-mean)/std, (xv-mean)/std
    best, key, history = None, None, []
    for penalty in plan["penalties"]:
        torch.manual_seed(plan["seed"])
        head = torch.nn.Linear(x.shape[1], len(classes))
        opt = torch.optim.Adam(head.parameters(), lr=plan["lr"])
        for epoch in range(1, plan["epochs"]+1):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(xn), y) + penalty*head.weight.square().sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                logits = head(xvn)
                accuracy = float((logits.argmax(-1) == yv).float().mean())
                ce = float(F.cross_entropy(logits, yv))
                history.append(dict(penalty=penalty, epoch=epoch, validation_accuracy=accuracy, validation_ce=ce))
                newkey = (-accuracy, ce)
                if key is None or newkey < key:
                    weight = head.weight.detach()/std[None]
                    best = dict(weight=weight.clone(), bias=(head.bias.detach()-weight @ mean).clone(),
                        penalty=penalty, epoch=epoch, classes=classes)
                    key = newkey
    receipt = dict(available=True, validation=score(xv, yv, best), fitting=score(x, y, best), history=history,
        validation_by_class={name: score(xv[yv == i], yv[yv == i], best) for i, name in enumerate(classes)},
        fit_counts=torch.bincount(y, minlength=len(classes)).tolist(),
        validation_counts=torch.bincount(yv, minlength=len(classes)).tolist(),
        used_in_handle_training=False, used_for_handle_selection=False)
    return best, receipt


def run(root=data.ROOT):
    root = Path(root)
    contexts, pairs, plan, ro = data.load(root)
    if (root / "readouts/results.json").exists():
        existing = old.read(root / "readouts/results.json")
        for name, receipt in existing.items():
            if receipt["available"] and schema.file_hash(root / "readouts" / (name+".pt")) != receipt["checkpoint_sha256"]:
                raise ValueError("Independent readout checkpoint changed")
        return existing
    natural, _, direct = data.caches(plan)
    raw = cache_checked(root / "states/readout_natural.pt")
    clean = raw["correct"].numpy()
    fitting = clean & (np.array(ro["split"]) == "fit")
    validation = clean & (np.array(ro["split"]) == "validation")
    old.write(root / "readouts/clean_selection.json", dict(total=len(clean), correct=int(clean.sum()),
        rejected=int((~clean).sum()), fit_indices=np.flatnonzero(fitting).tolist(),
        validation_indices=np.flatnonzero(validation).tolist(), selection="Natural full-vocabulary correctness, ties fail",
        prompt_templates_disjoint_from_evaluation=True))
    results = {}
    torch.set_num_threads(4)
    for variable, methods in (("C", data.METHODS), ("AT", ("standard",))):
        names = ro["colors"] if variable == "C" else ro["letters"]
        labels = [r["color"] if variable == "C" else r["labels"][r["contents"].index(r["color"])] for r in ro["contexts"]]
        y = torch.tensor([names.index(x) for x in labels])
        for method in methods:
            basis = old.checkpoint(method, variable)["basis"].float()
            feature = "c" if variable == "C" else "at"
            x = raw[feature].float() @ basis
            head, receipt = fit_head(x[fitting], y[fitting], x[validation], y[validation], ro, names)
            key = variable + "_" + method
            if head is not None:
                target = torch.tensor([names.index(contexts[i].color if variable == "C" else contexts[i].answer)
                    for i in plan["valid_contexts"]])
                features = getattr(natural, feature)[plan["valid_contexts"]].float() @ basis
                receipt["natural_evaluation"] = score(features, target, head)
                receipt["induced_pairs"] = {}
                for evaluated_method in (data.METHODS if variable == "AT" else (method,)):
                    indices = plan["original_pair_indices"]
                    states = getattr(direct[evaluated_method], feature)[indices]
                    expected = [contexts[p.source].color if variable == "C" else contexts[p.base].label_for(contexts[p.source].color) for p in pairs]
                    values = states.float() @ basis
                    truth = torch.tensor([names.index(v) for v in expected])
                    changing = np.array([contexts[p.base].color != contexts[p.source].color for p in pairs])
                    receipt["induced_pairs"][evaluated_method] = dict(all=score(values, truth, head),
                        changing=score(values[changing], truth[changing], head), preservation=score(values[~changing], truth[~changing], head))
                path = root / "readouts" / (key + ".pt")
                old.save_tensor(path, head)
                receipt["checkpoint_sha256"] = schema.file_hash(path)
            results[key] = receipt
    old.write(root / "readouts/results.json", results)
    return results
