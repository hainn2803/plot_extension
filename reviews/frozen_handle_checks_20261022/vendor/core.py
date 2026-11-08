"""One evaluation path for the exact toy and imported language-model handles."""
from dataclasses import dataclass
import unicodedata

import numpy as np
import torch


@dataclass
class Trace:
    c: torch.Tensor
    at: torch.Tensor
    logits: torch.Tensor


@dataclass
class Observed:
    c: torch.Tensor | None
    at: torch.Tensor | None
    predictions: dict
    upstream_error: float


@dataclass
class IndexedStates:
    values: torch.Tensor
    indices: np.ndarray

    def __getitem__(self, selection):
        return self.values[torch.as_tensor(self.indices[selection], dtype=torch.long)]


def patch(base, donor, basis):
    return base + ((donor - base) @ basis) @ basis.T


def score(logits, targets, backend):
    logits = logits.detach().cpu().double()
    if logits.ndim != 2 or logits.shape[0] != len(targets) or not torch.isfinite(logits).all():
        raise ValueError("Invalid full-vocabulary logits")
    top = logits.topk(2, dim=1)
    token_ids = logits.argmax(dim=1).numpy()
    ties = (top.values[:, 0] - top.values[:, 1] <= backend.tie_tolerance).numpy()
    decoded = [backend.decode(int(i)).strip() for i in token_ids]
    exact = np.array([text == target for text, target in zip(decoded, targets)]) & ~ties
    hai = np.array([unicodedata.normalize("NFKC", text).upper() == target
                    for text, target in zip(decoded, targets)])
    label_ids = torch.tensor(backend.label_ids, dtype=torch.long)
    label_logits = logits[:, label_ids]
    # Probabilities keep the full-vocabulary denominator; no forced A-Z argmax.
    label_probs = (label_logits - logits.logsumexp(1, keepdim=True)).exp()
    return dict(correct=exact, hai_correct=hai, token_id=token_ids, tied=ties,
                other=np.array([text not in backend.letters for text in decoded]),
                label_logits=label_logits.numpy(), label_probabilities=label_probs.numpy(),
                margin=(top.values[:, 0] - top.values[:, 1]).numpy())


def rate(correct):
    correct = np.asarray(correct, dtype=bool)
    return dict(correct=int(correct.sum()), count=len(correct),
                accuracy=float(correct.mean()) if len(correct) else None)


def measure(predictions, mask=None):
    mask = np.ones(len(predictions["correct"]), dtype=bool) if mask is None else np.asarray(mask)
    return {**rate(predictions["correct"][mask]),
            "hai_accuracy": rate(predictions["hai_correct"][mask])["accuracy"],
            "ties": int(predictions["tied"][mask].sum()),
            "other_tokens": int(predictions["other"][mask].sum())}


def _run(backend, indices, targets, c_source=None, at_source=None, batch=512,
         keep_states=True, upstream_reference=None):
    states, predictions, upstream_error = [], [], 0.
    for start in range(0, len(indices), batch):
        stop = start + batch
        trace = backend.run(indices[start:stop],
            None if c_source is None else c_source[start:stop],
            None if at_source is None else at_source[start:stop])
        predictions.append(score(trace.logits, targets[start:stop], backend))
        if upstream_reference is not None:
            upstream_error = max(upstream_error, float((trace.c.detach().cpu() - upstream_reference[start:stop]).abs().max()))
        if keep_states:
            states.append((trace.c.detach().cpu(), trace.at.detach().cpu()))
    compact = {key: np.concatenate([p[key] for p in predictions]) for key in predictions[0]}
    return Observed(torch.cat([s[0] for s in states]) if keep_states else None,
                    torch.cat([s[1] for s in states]) if keep_states else None,
                    compact, upstream_error)


@torch.no_grad()
def evaluate(backend, pairs, *, extended=True, batch=512):
    contexts = backend.contexts
    natural_targets = [c.answer for c in contexts]
    natural = _run(backend, np.arange(len(contexts)), natural_targets, batch=batch)
    clean = natural.predictions
    # Exact imports are never filtered again. A changed natural prediction stops evaluation.
    if not clean["correct"].all():
        raise ValueError("Imported clean-correct bank is not reproduced; retain evidence before proceeding")
    b = np.array([p.base for p in pairs])
    s = np.array([p.source for p in pairs])
    target = [contexts[bi].label_for(contexts[si].color) for bi, si in zip(b, s)]
    base_target = [contexts[i].answer for i in b]
    source_target = [contexts[i].answer for i in s]
    changing = np.array([contexts[bi].color != contexts[si].color for bi, si in zip(b, s)])
    direct = _run(backend, b, target, natural.c[s], batch=batch)
    replay = _run(backend, b, target, at_source=direct.at, batch=batch, keep_states=False)
    restoration = _run(backend, b, base_target, natural.c[s], natural.at[b], batch=batch)
    natural_at = _run(backend, b, source_target, at_source=natural.at[s], batch=batch, keep_states=False)
    traces = dict(direct=direct, replay=replay, restoration=restoration, natural_at=natural_at)
    predictions = {name: trace.predictions for name, trace in traces.items()}
    results = dict(clean=measure(clean), coverage=dict(pairs=len(pairs), contexts=len(contexts),
                   clusters=len({p.cluster for p in pairs})), operations={}, families={}, transitions={})
    for name, prediction in predictions.items():
        split = changing if name != "natural_at" else np.array(base_target) != np.array(source_target)
        results["operations"][name] = dict(all=measure(prediction), changing=measure(prediction, split),
                                         preservation=measure(prediction, ~split))
    joint = predictions["direct"]["correct"] & predictions["replay"]["correct"]
    results["direct_and_replay"] = dict(changing=rate(joint[changing]), preservation=rate(joint[~changing]))
    results["restored_coordinates_max_error"] = float(((restoration.at - natural.at[b]) @ backend.at_basis.detach().cpu()).abs().max())
    results["upstream_invariance_max_error"] = float((restoration.c - direct.c).abs().max())
    for field in ("family", "transition"):
        values = [p.family for p in pairs] if field == "family" else [
            f"{contexts[bi].color}->{contexts[si].color}" for bi, si in zip(b, s)]
        output = results["families" if field == "family" else "transitions"]
        for value in sorted(set(values)):
            mask = np.array(values) == value
            output[value] = {name: measure(pred, mask) for name, pred in predictions.items()}
            output[value]["joint"] = rate(joint[mask])
    results["coverage"]["preservation_absent"] = not bool((~changing).any())
    arrays = {f"{name}_{key}": value for name, prediction in predictions.items()
              for key, value in prediction.items()}
    arrays.update(base=b, source=s, target=np.array(target), base_target=np.array(base_target),
                  pair_id=np.array([p.id for p in pairs]), cluster=np.array([p.cluster for p in pairs]),
                  original_question=np.array([p.original_question for p in pairs]))
    baseline = {key: value[b] for key, value in natural.predictions.items()}
    for name in ("direct", "replay", "restoration", "natural_at"):
        arrays[f"{name}_probability_change"] = predictions[name]["label_probabilities"] - baseline["label_probabilities"]
    if extended:
        _extended(backend, pairs, natural, direct, target, results, arrays, batch)
    if getattr(backend, "readouts", None):
        _readouts(backend, natural, direct, natural_targets, target, changing, results)
    return results, arrays


def _readouts(backend, natural, direct, natural_targets, targets, changing, results):
    results["readouts"] = {}
    basis = backend.at_basis.detach().cpu().float()
    clean_coordinates = natural.at.float() @ basis
    induced_coordinates = direct.at.float() @ basis
    for name, readout in backend.readouts.items():
        w, bias = readout["W"].detach().cpu().float(), readout["b"].detach().cpu().float()
        if w.shape != (len(backend.letters), basis.shape[1]) or bias.shape != (len(backend.letters),):
            raise ValueError("Readout classes/rank disagree with pinned AT coordinates")
        if not torch.isfinite(w).all() or not torch.isfinite(bias).all():
            raise ValueError("Nonfinite readout weights")
        def decoded(features, expected):
            logits = features @ w.T + bias
            top = logits.topk(2, dim=1)
            predicted = np.array([backend.letters[i] for i in top.indices[:, 0]])
            return (predicted == np.array(expected)) & (top.values[:, 0].numpy() > top.values[:, 1].numpy())
        clean = decoded(clean_coordinates, natural_targets)
        induced = decoded(induced_coordinates, targets)
        results["readouts"][name] = dict(clean=rate(clean), induced_changing=rate(induced[changing]),
            induced_preservation=rate(induced[~changing]), role=readout.get("role", "unverified"),
            used_in_training=readout.get("used_in_training"), used_in_selection=readout.get("used_in_selection"))


def _extended(backend, pairs, natural, direct, targets, results, arrays, batch):
    contexts = backend.contexts
    letters = [c.answer for c in contexts]
    # All natural donor/recipient combinations are evaluated once and reused as
    # controls for every induced state with the same desired letter and recipient.
    recv = np.repeat(np.arange(len(contexts)), len(contexts))
    donor = np.tile(np.arange(len(contexts)), len(contexts))
    control = _run(backend, recv, [letters[i] for i in donor],
                   at_source=IndexedStates(natural.at, donor), batch=batch, keep_states=False)
    cp = control.predictions
    other_question = np.array([contexts[r].question != contexts[d].question for r, d in zip(recv, donor)])
    results["clean_donors"] = dict(all=measure(cp), other_question=measure(cp, other_question))
    lookup = {}
    for r in range(len(contexts)):
        for letter in backend.letters:
            row = slice(r * len(contexts), (r + 1) * len(contexts))
            eligible = (np.array(letters) == letter) & other_question[row]
            lookup[r, letter] = (int(cp["correct"][row][eligible].sum()), int(eligible.sum()))
    arrays.update({f"clean_donor_{k}": v for k, v in cp.items()})
    arrays.update(clean_donor_receiver=recv, clean_donor_source=donor)
    extended = {name: [] for name in ("cross", "assignments")}
    trace_error = 0.
    for start in range(0, len(pairs), max(1, batch // len(contexts))):
        stop = min(len(pairs), start + max(1, batch // len(contexts)))
        case = np.repeat(np.arange(start, stop), len(contexts))
        r = np.tile(np.arange(len(contexts)), stop - start)
        b = np.array([pairs[i].base for i in case])
        s = np.array([pairs[i].source for i in case])
        expected = [targets[i] for i in case]
        cross = _run(backend, r, expected, at_source=direct.at[case], batch=batch, keep_states=False)
        pred = cross.predictions
        valid = np.array([contexts[ri].question not in (contexts[bi].question, contexts[si].question)
                          for ri, bi, si in zip(r, b, s)])
        pred.update(case=case, receiver=r, eligible=valid,
            changing=np.array(expected) != np.array(letters)[r],
            in_options=np.array([t in contexts[ri].labels for ri, t in zip(r, expected)]),
            control_correct=np.array([lookup[ri, t][0] for ri, t in zip(r, expected)]),
            control_count=np.array([lookup[ri, t][1] for ri, t in zip(r, expected)]))
        extended["cross"].append(pred)
        assignment = _run(backend, b, [letters[i] for i in r], natural.c[s], natural.at[r],
                          batch=batch, keep_states=False, upstream_reference=direct.c[case])
        trace_error = max(trace_error, assignment.upstream_error)
        pred = assignment.predictions
        pred.update(case=case, donor=r, conflict=np.array([letters[ri] != targets[i] for ri, i in zip(r, case)]))
        extended["assignments"].append(pred)
    for name, pieces in extended.items():
        combined = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        arrays.update({f"{name}_{key}": value for key, value in combined.items()})
        valid = combined["eligible"] if name == "cross" else np.ones(len(combined["correct"]), dtype=bool)
        results[name] = dict(all=measure(combined, valid))
        if name == "cross":
            for field in ("changing", "in_options"):
                for value in (True, False):
                    results[name][f"{field}_{value}"] = measure(combined, valid & (combined[field] == value))
            count = combined["control_count"]
            matched = valid & (count > 0)
            results[name]["cases_without_clean_donor"] = int((valid & (count == 0)).sum())
            results[name]["matched_clean_donor_mean_accuracy"] = float((
                combined["control_correct"][matched] / count[matched]).mean()) if matched.any() else None
        else:
            results[name]["conflicting"] = measure(combined, combined["conflict"])
            results[name]["consistent"] = measure(combined, ~combined["conflict"])
    results["upstream_invariance_max_error"] = max(trace_error, results["upstream_invariance_max_error"])


def paired_comparison(standard, gradual, mask=None, resamples=2000, seed=20261019):
    """Template cluster resampling; exact identity and target alignment required."""
    for key in ("pair_id", "cluster", "target", "base_target"):
        if not np.array_equal(standard[key], gradual[key]):
            raise ValueError(f"Paired prediction identity differs: {key}")
    count = len(standard["pair_id"])
    mask = np.ones(count, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (count,):
        raise ValueError("Invalid comparison mask")
    groups, inverse = np.unique(standard["cluster"][mask], return_inverse=True)
    sizes = np.bincount(inverse, minlength=len(groups))
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(groups), size=(resamples, len(groups))) if len(groups) > 1 else None
    result = dict(cases=int(mask.sum()), clusters=len(groups), resamples=resamples,
        grouping="Exact first-line template with stated color masked; not guaranteed subject independence",
        endpoints={})
    for name in ("direct", "replay", "joint"):
        s = standard["direct_correct"] & standard["replay_correct"] if name == "joint" else standard[f"{name}_correct"]
        g = gradual["direct_correct"] & gradual["replay_correct"] if name == "joint" else gradual[f"{name}_correct"]
        delta = g[mask].astype(float) - s[mask].astype(float)
        summed = np.bincount(inverse, weights=delta, minlength=len(groups))
        distribution = summed[draws].sum(1) / sizes[draws].sum(1) if draws is not None else None
        result["endpoints"][name] = dict(standard=rate(s[mask]), gradual=rate(g[mask]),
            difference=float(delta.mean()) if len(delta) else None,
            percentile_95=np.quantile(distribution, [.025, .975]).tolist() if distribution is not None else None)
    return result
