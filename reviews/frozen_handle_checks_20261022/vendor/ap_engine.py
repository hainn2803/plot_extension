from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import schema, core, safety as gpu
from .storage import read
BATCH=32
ROOT=None
AP=None
old=SimpleNamespace(read=read,DATA=Path(__file__).resolve().parents[1]/"data")

class Engine:
    letters = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tie_tolerance = 0.0

    def __init__(self, model, tokenizer, plan, budget):
        self.model, self.tokenizer, self.budget = model, tokenizer, budget
        self.contexts = [schema.Context(**c) for c in plan['contexts']]
        self.device = next(model.parameters()).device
        self.label_ids = old.read(old.DATA / 'protocol.json')['label_token_ids']
        self.q = torch.load(AP, map_location='cpu', weights_only=True)['basis'].float().to(self.device)
        self.encoded = [tokenizer(c.prompt, add_special_tokens=True, return_offsets_mapping=True) for c in self.contexts]
        self.width = max(len(e['input_ids']) for e in self.encoded)
        self.positions, self.color_positions = [], []
        for c, e in zip(self.contexts, self.encoded):
            found = []
            for offset in [*[m.start(1) for m in schema.CHOICE.finditer(c.prompt)], c.color_span[0]]:
                matching = [i for i, (a, b) in enumerate(e['offset_mapping']) if a <= offset < b]
                if len(matching) != 1:
                    raise ValueError('Token offset not unique')
                found.append(self.width-len(e['input_ids'])+matching[0])
            self.positions.append(found[:4])
            self.color_positions.append(found[4])

    def decode(self, token):
        return self.tokenizer.decode([token])

    def inputs(self, indices):
        padded = list(indices) + [indices[0]]*(BATCH-len(indices))
        ids = torch.full((BATCH, self.width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(padded):
            tokens = self.encoded[i]['input_ids']
            ids[row, -len(tokens):] = torch.tensor(tokens, device=self.device)
            mask[row, -len(tokens):] = 1
        return ids, mask, padded

    @torch.no_grad()
    def forward(self, indices, layer=None, donor=None, slots=None, subspace=False, capture=False):
        self.budget.tick()
        gpu.cooling(ROOT, self.budget)
        ids, mask, padded = self.inputs(indices)
        rows = torch.arange(BATCH, device=self.device)
        states, hooks = {}, []
        def hook_for(number):
            def hook(_module, _inputs, out):
                h = out[0] if isinstance(out, tuple) else out
                if number == layer:
                    pos = torch.full((BATCH,), self.width-1, device=self.device, dtype=torch.long) if subspace else torch.tensor(
                        [self.positions[i][slots[j]] for j, i in enumerate(padded)], device=self.device)
                    d = donor.to(self.device).float()
                    if len(d) < BATCH:
                        d = torch.cat((d, d[:1].expand(BATCH-len(d), -1)))
                    changed = h.clone()
                    changed[rows, pos] = (core.patch(h[rows, pos].float(), d, self.q) if subspace else d).to(h.dtype)
                    return (changed,) + out[1:] if isinstance(out, tuple) else changed
                if capture:
                    ps = torch.tensor([self.positions[i] for i in padded], device=self.device)
                    states[number] = h[rows[:, None], ps].detach().cpu()
                    if number == 0:
                        cp = torch.tensor([self.color_positions[i] for i in padded], device=self.device)
                        states['color'] = h[rows, cp].float().cpu()
                    if number == 17:
                        states['last'] = h[:, -1].float().cpu()
                return None
            return hook
        try:
            for number, block in enumerate(self.model.model.layers):
                if capture or number == layer:
                    hooks.append(block.register_forward_hook(hook_for(number)))
            out = self.model.model(input_ids=ids, attention_mask=mask,
                position_ids=(mask.cumsum(-1)-1).clamp(min=0), use_cache=False, return_dict=True)
            logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
            cap = self.model.config.final_logit_softcapping
            if cap is not None:
                logits = torch.tanh(logits/cap)*cap
        finally:
            for h in hooks:
                h.remove()
        self.budget.tick()
        result = dict(logits=logits[:len(indices)].cpu())
        if capture:
            result.update(states=torch.stack([states[i] for i in range(26)], dim=1)[:len(indices)],
                color=states['color'][:len(indices)], last=states['last'][:len(indices)])
        return result
