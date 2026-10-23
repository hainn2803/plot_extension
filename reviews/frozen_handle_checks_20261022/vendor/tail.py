from pathlib import Path
import numpy as np
import torch
from . import core
from .safety import cooling

class Tail:
    def __init__(self, backend, root, budget):
        self.backend, self.root, self.budget = backend, Path(root), budget
        self.model, self.device = backend.model, backend.device
        self.q = backend.at_basis
        self.letters, self.label_ids = backend.letters, backend.label_ids
        self.tie_tolerance = backend.tie_tolerance
        if backend.at_layer != len(self.model.model.layers)-1:
            raise ValueError("Final-layer shortcut is invalid for an earlier AT site")

    def decode(self, token):
        return self.backend.decode(token)

    def guard(self):
        self.budget.tick()
        cooling(self.root, self.budget)
        self.budget.tick()

    @torch.no_grad()
    def state(self, base, donor=None, restore=None):
        self.guard()
        h = base.to(self.device).float()
        if donor is not None:
            h = core.patch(h, donor.to(self.device).float(), self.q).to(torch.bfloat16).float()
        if restore is not None:
            h = core.patch(h, restore.to(self.device).float(), self.q).to(torch.bfloat16).float()
        return h

    @torch.no_grad()
    def logits(self, h, head_size=None):
        h = h.to(self.device).to(torch.bfloat16)
        size = len(h) if head_size is None else head_size
        if not 0 < len(h) <= size <= 32:
            raise ValueError("Unregistered physical output-head shape")
        if len(h) < size:
            h = torch.cat((h, h[:1].expand(size-len(h), -1)), dim=0)
        hidden = self.model.model.norm(h[:, None, :])[:, 0]
        y = self.model.lm_head(hidden).float()
        cap = self.model.config.final_logit_softcapping
        if cap is not None:
            y = torch.tanh(y / cap) * cap
        return y[:len(h) if head_size is None else size]

    @torch.no_grad()
    def predict(self, base, targets, donor=None, restore=None, head_sizes=None, baseline=None):
        h = self.state(base, donor, restore)
        head_sizes = np.full(len(base), 32) if head_sizes is None else np.asarray(head_sizes)
        result = None
        for size in sorted(set(head_sizes.tolist())):
            ids = np.flatnonzero(head_sizes == size)
            if len(ids) > size:
                raise ValueError("Tail microbatch exceeds recorded physical shape")
            logits = self.logits(h[torch.as_tensor(ids, device=self.device)], int(size))[:len(ids)]
            pred = core.score(logits, [targets[i] for i in ids], self)
            if result is None:
                result = {k: np.empty((len(base), *v.shape[1:]), dtype=v.dtype) for k, v in pred.items()}
            for key, value in pred.items():
                result[key][ids] = value
        if baseline is not None:
            result["label_probability_change"] = result["label_probabilities"] - baseline
        if restore is not None:
            reference = restore.to(self.device).float()
            error = (h-reference) @ self.q
            result["coordinate_max_error"] = error.abs().amax(-1).cpu().numpy()
            result["coordinate_l2_error"] = error.norm(dim=-1).cpu().numpy()
        return result
