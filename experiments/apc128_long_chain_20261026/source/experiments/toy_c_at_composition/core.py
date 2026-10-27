"""The fixed counterexample, differentiable interventions, and exhaustive tests."""
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DTYPE = torch.float64
TIE_TOLERANCE = 1e-10
METHODS = ("standard", "replay", "combined")
STEPS = (0, 10, 50, 100, 250, 500, 1000)


def orthogonal(width, generator):
    q, r = torch.linalg.qr(torch.randn(width, width, generator=generator, dtype=DTYPE))
    return q * torch.where(r.diag() < 0, -1.0, 1.0)


def project(base, source, basis):
    return base + ((source - base) @ basis) @ basis.T


class FrozenNetwork(nn.Module):
    def __init__(self, rotated=True):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(20261019)
        rc, ra = orthogonal(8, generator), orthogonal(12, generator)
        self.rc = nn.Parameter(rc if rotated else torch.eye(8, dtype=DTYPE), requires_grad=False)
        self.ra = nn.Parameter(ra if rotated else torch.eye(12, dtype=DTYPE), requires_grad=False)
        eye = torch.eye(4, dtype=DTYPE)
        self.output_weight = nn.Parameter(4 * torch.cat((eye, eye, -eye), dim=1), requires_grad=False)
        self.register_buffer("colors", eye)
        self.register_buffer("at_basis", self.ra @ torch.cat((eye, torch.zeros(8, 4, dtype=DTYPE))))

    def color_state(self, colors):
        e = self.colors[colors]
        return torch.cat((e, e), dim=-1) @ self.rc.T

    def at_state(self, color_state, mappings):
        u, v = (color_state @ self.rc).split(4, dim=-1)
        routing = F.one_hot(mappings, 4).to(DTYPE)
        mu = torch.bmm(u.unsqueeze(1), routing).squeeze(1)
        mv = torch.bmm(v.unsqueeze(1), routing).squeeze(1)
        return torch.cat((mv, mu, mv), dim=-1) @ self.ra.T

    def output(self, at_state):
        return (at_state @ self.ra) @ self.output_weight.T

    def teach(self, at_state):
        return 4 * (at_state @ self.at_basis)

    def forward_trace(self, colors, mappings, c_source, c_basis, at_source=None):
        h = project(self.color_state(colors), c_source, c_basis)
        z = self.at_state(h, mappings)
        if at_source is not None:
            z = project(z, at_source, self.at_basis)
        return h, z, self.output(z)

    def reference_basis(self, name):
        eye = torch.eye(4, dtype=DTYPE)
        if name == "incompatible":
            q = torch.cat((eye, torch.zeros_like(eye)))
        elif name == "compatible":
            q = torch.cat((eye, eye)) / np.sqrt(2)
        else:
            raise ValueError(name)
        return self.rc @ q


class Subspace(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.raw = nn.Parameter(initial.detach().clone())

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


def bank():
    permutations = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)
    colors = torch.arange(4).repeat_interleave(24)
    mappings = permutations.repeat(4, 1)
    natural = mappings[torch.arange(96), colors]
    base = torch.arange(96).repeat_interleave(96)
    source = torch.arange(96).repeat(96)
    target = mappings[base, colors[source]]
    return dict(colors=colors, mappings=mappings, natural=natural, base=base, source=source,
                target=target, at_target=natural[source], changing=colors[base] != colors[source])


def prepared(model, data):
    h = model.color_state(data["colors"])
    z = model.at_state(h, data["mappings"])
    return {**data, "h": h.detach(), "z": z.detach()}


def initial_basis(model, arm, seed):
    if arm == "repair":
        return model.reference_basis("incompatible")
    if arm != "discovery":
        raise ValueError(arm)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.linalg.qr(torch.randn(8, 4, generator=generator, dtype=DTYPE), mode="reduced").Q


class Order:
    def __init__(self, seed, state=None):
        self.generator = torch.Generator(device="cpu").manual_seed(1_000_003 + seed)
        if state is None:
            self.permutation = torch.randperm(9216, generator=self.generator)
            self.offset, self.epoch = 0, 0
        else:
            self.generator.set_state(state["generator"])
            self.permutation = state["permutation"].clone()
            self.offset, self.epoch = state["offset"], state["epoch"]

    def next(self):
        if self.offset == 9216:
            self.permutation = torch.randperm(9216, generator=self.generator)
            self.offset, self.epoch = 0, self.epoch + 1
        indices = self.permutation[self.offset:self.offset + 128]
        self.offset += 128
        return indices

    def state(self):
        return dict(generator=self.generator.get_state(), permutation=self.permutation.clone(),
                    offset=self.offset, epoch=self.epoch)


def loss_terms(model, data, q, indices):
    base, source = data["base"][indices], data["source"][indices]
    changed = project(data["h"][base], data["h"][source], q)
    induced = model.at_state(changed, data["mappings"][base])
    direct_logits = model.output(induced)
    replayed = project(data["z"][base], induced, model.at_basis)
    target = data["target"][indices]
    return {"direct": F.cross_entropy(direct_logits, target),
            "replay": F.cross_entropy(model.output(replayed), target),
            "teaching": F.cross_entropy(model.teach(induced), target)}


def objective(terms, method):
    if method not in METHODS:
        raise ValueError(method)
    return terms["direct"] + (2 * terms["replay"] if method != "standard" else 0) + (
        2 * terms["teaching"] if method == "combined" else 0)


def metric(logits, target, mask=None):
    if mask is not None:
        logits, target = logits[mask], target[mask]
    margins = logits.gather(1, target[:, None]).squeeze(1) - logits.masked_fill(
        F.one_hot(target, 4).bool(), -torch.inf).max(dim=1).values
    top = logits.topk(2, dim=1).values
    successes = int((margins > TIE_TOLERANCE).sum())
    return dict(correct=successes, count=len(target), accuracy=successes / len(target),
                ties=int(((top[:, 0] - top[:, 1]) <= TIE_TOLERANCE).sum()),
                margin_min=float(margins.min()), margin_mean=float(margins.mean()),
                ce=float(F.cross_entropy(logits, target)))


def compact_predictions(logits, target):
    pred = logits.argmax(dim=-1).numpy().astype(np.int8)
    top = logits.topk(2, dim=-1).values
    tied = ((top[:, 0] - top[:, 1]) <= TIE_TOLERANCE).numpy()
    return dict(prediction=pred, target=target.numpy().astype(np.int8), tied=tied, logits=logits.numpy())


@torch.no_grad()
def evaluate(model, data, q, *, structural=True, predictions=False):
    base, source, target = data["base"], data["source"], data["target"]
    changed = project(data["h"][base], data["h"][source], q)
    induced = model.at_state(changed, data["mappings"][base])
    direct = model.output(induced)
    replay_state = project(data["z"][base], induced, model.at_basis)
    replay = model.output(replay_state)
    teach = model.teach(induced)
    natural_at = model.output(project(data["z"][base], data["z"][source], model.at_basis))
    restored = project(induced, data["z"][base], model.at_basis)
    restoration = model.output(restored)
    changing, at_changing = data["changing"], data["at_target"] != data["natural"][base]
    result = {}
    for name, logits, expected, mask in (
        ("direct_changing", direct, target, changing),
        ("direct_preservation", direct, target, ~changing),
        ("replay_changing", replay, target, changing),
        ("replay_preservation", replay, target, ~changing),
        ("at_changing", natural_at, data["at_target"], at_changing),
        ("at_preservation", natural_at, data["at_target"], ~at_changing),
        ("at_decode_changing", teach, target, changing),
        ("at_decode_preservation", teach, target, ~changing),
        ("restoration_changing", restoration, data["natural"][base], changing),
        ("restoration_preservation", restoration, data["natural"][base], ~changing),
    ):
        result[name] = metric(logits, expected, mask)
    result["direct_by_transition"] = {
        f"{cb}->{cs}": metric(direct, target, (data["colors"][base] == cb) & (data["colors"][source] == cs))
        for cb in range(4) for cs in range(4)}
    result["direct_by_base_mapping"] = {
        str(i): metric(direct, target, (base % 24) == i) for i in range(24)}
    source_at_changes = data["natural"][base] != data["natural"][source]
    result["families"] = {
        f"C_{c}_natural_AT_{a}": metric(direct, target,
            (changing if c == "changes" else ~changing) &
            (source_at_changes if a == "changes" else ~source_at_changes))
        for c in ("changes", "fixed") for a in ("changes", "fixed")}
    error = (induced @ model.at_basis - model.colors[target]).norm(dim=1)
    result["continuous"] = dict(at_error_mean=float(error.mean()), at_error_max=float(error.max()),
        at_error_changing_mean=float(error[changing].mean()),
        restored_at_max_error=float(((restored - data["z"][base]) @ model.at_basis).abs().max()),
        replay_teaching_logits_max_error=float((replay - teach).abs().max()),
        orthogonality_error=float((q.T @ q - torch.eye(4, dtype=DTYPE)).abs().max()))
    result["projector_distance"] = {name: float((q @ q.T - model.reference_basis(name) @
        model.reference_basis(name).T).norm()) for name in ("incompatible", "compatible")}
    arrays = {}
    if predictions:
        for name, logits, expected in (("direct", direct, target), ("replay", replay, target),
            ("teaching", teach, target), ("natural_at", natural_at, data["at_target"]),
            ("restoration", restoration, data["natural"][base])):
            arrays.update({f"{name}_{key}": value for key, value in compact_predictions(logits, expected).items()})
    if structural:
        # Source mappings cannot affect the C-stage state. These 384 donors exactly
        # represent all 9,216 pairs, each with multiplicity 24, without rounding.
        representatives = (torch.arange(96)[:, None] * 96 + torch.arange(4)[None, :] * 24).flatten()
        donor, donor_target = induced[representatives], target[representatives]
        receiver = data["z"].repeat(len(donor), 1)
        replay_donor = donor.repeat_interleave(96, dim=0)
        cross_target = donor_target.repeat_interleave(96)
        cross = model.output(project(receiver, replay_donor, model.at_basis))
        receiver_target = data["natural"].repeat(len(donor))
        result["cross_recipient"] = metric(cross, cross_target)
        for label, mask in (("change", cross_target != receiver_target), ("preserve", cross_target == receiver_target)):
            result[f"cross_recipient_{label}"] = metric(cross, cross_target, mask)
        result["cross_recipient_multiplicity"] = 24
        assign_target = torch.arange(4).repeat(len(donor))
        natural_donor = data["z"][torch.arange(4) * 24].repeat(len(donor), 1)
        joint_base = torch.arange(96).repeat_interleave(16)
        joint_source_color = torch.arange(4).repeat_interleave(4).repeat(96)
        upstream, assigned_state, assigned = model.forward_trace(data["colors"][joint_base],
            data["mappings"][joint_base], data["h"][joint_source_color * 24], q, natural_donor)
        upstream_before = changed[representatives].repeat_interleave(4, dim=0)
        result["joint_assignments"] = metric(assigned, assign_target)
        result["upstream_invariance_max_error"] = float((upstream - upstream_before).abs().max())
        required = [value for value in result.values() if isinstance(value, dict) and "correct" in value]
        result["composition_repaired"] = all(value["correct"] == value["count"] for value in required) and (
            result["upstream_invariance_max_error"] <= 1e-10)
        if predictions:
            for name, logits, expected in (("cross_recipient", cross, cross_target), ("joint", assigned, assign_target)):
                arrays.update({f"{name}_{key}": value for key, value in compact_predictions(logits, expected).items()})
    return result, arrays
