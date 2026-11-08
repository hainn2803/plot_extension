"""CPU toy adapter and a weight-loading-free Hugging Face hook adapter."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch

from .core import Trace, patch
from .schema import Context, Pair, file_hash


def checked_basis(checkpoint, width, expected_variable, dtype=torch.float32):
    if checkpoint.get("variable") != expected_variable:
        raise ValueError("Wrong checkpoint variable")
    q = checkpoint["basis"].detach().cpu().to(dtype)
    if q.ndim != 2 or q.shape[0] != width or not 0 < q.shape[1] <= width or not torch.isfinite(q).all():
        raise ValueError("Invalid saved basis")
    error = (q.T @ q - torch.eye(q.shape[1], dtype=dtype)).abs().max()
    if error > 2e-4:
        raise ValueError("Saved basis is not orthogonal at the unchanged 2e-4 tolerance")
    if int(checkpoint.get("subspace_dim", q.shape[1])) != q.shape[1]:
        raise ValueError("Saved rank disagrees with basis")
    return q


def load_checkpoint(path, expected_sha):
    if file_hash(path) != expected_sha:
        raise ValueError("Checkpoint hash changed")
    return torch.load(path, map_location="cpu", weights_only=True)


class ToyBackend:
    letters = tuple("ABCD")
    label_ids = tuple(range(4))
    tie_tolerance = 1e-10

    def __init__(self, basis, model=None):
        from experiments.toy_c_at_composition import core as toy
        self.model = toy.FrozenNetwork() if model is None else model
        self.data = toy.bank()
        self.basis = basis
        self.at_basis = self.model.at_basis
        self.contexts = []
        colors = ("red", "blue", "green", "yellow")
        for i, (color, mapping) in enumerate(zip(self.data["colors"], self.data["mappings"])):
            self.contexts.append(Context(str(i), f"Exact toy state {i}", colors[int(color)],
                tuple(self.letters[int(x)] for x in mapping), colors, f"toy-question-{i}", (0, 0)))

    def decode(self, token):
        return self.letters[token]

    def pairs(self):
        return [Pair(str(i), int(b), int(s),
            f"C_{self.contexts[int(b)].color != self.contexts[int(s)].color}_AT_"
            f"{self.contexts[int(b)].answer != self.contexts[int(s)].answer}",
            self.contexts[int(b)].question)
            for i, (b, s) in enumerate(zip(self.data["base"], self.data["source"]))]

    def run(self, indices, c_source=None, at_source=None):
        indices = torch.as_tensor(indices, dtype=torch.long)
        c = self.model.color_state(self.data["colors"][indices])
        if c_source is not None:
            c = patch(c, c_source, self.basis)
        at = self.model.at_state(c, self.data["mappings"][indices])
        if at_source is not None:
            at = patch(at, at_source, self.at_basis)
        return Trace(c, at, self.model.output(at))


def toy_saved_backend(root, method):
    from experiments.toy_c_at_composition import core as toy
    root = Path(root)
    model = toy.FrozenNetwork()
    manifest = json.loads((root / "manifest.json").read_text())
    if file_hash(root / "model.pt") != manifest["artifacts"]["model.pt"]:
        raise ValueError("Frozen toy model hash changed")
    model.load_state_dict(torch.load(root / "model.pt", weights_only=True, map_location="cpu"))
    if method in ("incompatible", "compatible"):
        return ToyBackend(model.reference_basis(method), model)
    fit = root / "fits" / f"discovery_S01_{method}"
    receipt = json.loads((fit / "completion.json").read_text())
    checkpoint = fit / receipt["checkpoint"]["path"]
    payload = load_checkpoint(checkpoint, receipt["checkpoint"]["file_hash"])
    raw = payload["subspace"]["raw"]
    return ToyBackend(torch.linalg.qr(raw, mode="reduced").Q, model)


class HFBackend:
    """Uses an already loaded, pinned model; construction never downloads weights."""
    letters = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    tie_tolerance = 0.0

    def __init__(self, model, tokenizer, contexts, c_checkpoint, at_checkpoint, label_ids):
        self.model, self.tokenizer, self.contexts = model, tokenizer, contexts
        self.device = next(model.parameters()).device
        self.c_layer, self.at_layer = int(c_checkpoint["layer"]), int(at_checkpoint["layer"])
        if not 0 <= self.c_layer < self.at_layer < len(model.model.layers):
            raise ValueError("The C capture must precede the AT capture")
        if c_checkpoint.get("token_position") != "color" or at_checkpoint.get("token_position") != "last_token":
            raise ValueError("Unsupported saved token slots; do not silently remap a handle")
        self.basis = checked_basis(c_checkpoint, model.config.hidden_size, "C").to(self.device)
        self.at_basis = checked_basis(at_checkpoint, model.config.hidden_size, "AT").to(self.device)
        if len(label_ids) != 26 or len(set(label_ids)) != 26:
            raise ValueError("Explicit distinct A-Z token IDs are required")
        self.label_ids = tuple(map(int, label_ids))
        if any(self.decode(i).strip() != letter for i, letter in zip(self.label_ids, self.letters)):
            raise ValueError("Pinned label token IDs do not decode to the declared letters")
        self.tokens, self.color_positions = [], []
        for context in contexts:
            encoded = tokenizer(context.prompt, add_special_tokens=True, return_offsets_mapping=True)
            start = context.color_span[0]
            positions = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a <= start < b]
            if len(positions) != 1:
                raise ValueError("Ambiguous literal-color token alignment")
            self.tokens.append(encoded["input_ids"])
            self.color_positions.append(positions[0])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def decode(self, token):
        return self.tokenizer.decode([token])

    @torch.no_grad()
    def run(self, indices, c_source=None, at_source=None):
        indices = list(map(int, indices))
        if not indices:
            raise ValueError("Empty forward batch")
        width = max(len(self.tokens[i]) for i in indices)
        pad_token = self.tokenizer.pad_token_id
        if pad_token is None:
            pad_token = self.tokenizer.eos_token_id
        ids = torch.full((len(indices), width), pad_token, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        positions = []
        for row, i in enumerate(indices):
            length = len(self.tokens[i])
            ids[row, width - length:] = torch.tensor(self.tokens[i], device=self.device)
            mask[row, width - length:] = 1
            positions.append(width - length + self.color_positions[i])
        c_idx = torch.tensor(positions, device=self.device)
        at_idx = torch.full_like(c_idx, width - 1)
        rows = torch.arange(len(indices), device=self.device)
        captured = {}

        def hook_for(name, pos, donor, basis):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                changed = hidden.clone()
                if donor is not None:
                    changed[rows, pos] = patch(hidden[rows, pos].float(),
                        donor.to(self.device).float(), basis).to(hidden.dtype)
                captured[name] = changed[rows, pos].float().detach().cpu()
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            return hook

        with ExitStack() as cleanup:
            for layer, name, pos, donor, basis in (
                    (self.c_layer, "c", c_idx, c_source, self.basis),
                    (self.at_layer, "at", at_idx, at_source, self.at_basis)):
                handle = self.model.model.layers[layer].register_forward_hook(hook_for(name, pos, donor, basis))
                cleanup.callback(handle.remove)
            output = self.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            hidden = output.last_hidden_state[rows, at_idx]
            logits = self.model.lm_head(hidden).float()
            softcap = getattr(self.model.config, "final_logit_softcapping", None)
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
        return Trace(captured["c"], captured["at"], logits.detach().cpu())
