import random
import string

import numpy as np
import torch
import torch.nn as nn

from mcqa_data_load_all import letter_token_id


ANSWER_LETTERS = tuple(string.ascii_uppercase)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def answer_label_ids(tokenizer):
    return [letter_token_id(tokenizer, letter) for letter in ANSWER_LETTERS]


def answer_logits(model, outputs, attention_mask, label_ids):
    device = outputs.last_hidden_state.device
    rows = torch.arange(attention_mask.shape[0], device=device)
    idx = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask * idx.unsqueeze(0)).max(dim=1).values

    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, device=device, dtype=torch.long)
    W = model.lm_head.weight[ids].to(hidden.dtype)
    logits = hidden @ W.T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[ids]

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits.float()


@torch.no_grad()
def collect_layer_states(model, input_ids, attention_mask, position_by_id, layer, token_position="last_token", batch_size=32):
    device = next(model.parameters()).device
    states = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + position_by_id[token_position][start:end].to(device)
        captured = {}

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["x"] = hidden[rows, pos, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
        finally:
            handle.remove()

        states.append(captured["x"])

    return torch.cat(states, dim=0)


@torch.no_grad()
def collect_all_layer_states(model, input_ids, attention_mask, position_by_id, layers, token_position="last_token", batch_size=32):
    device = next(model.parameters()).device
    collected = {int(layer): [] for layer in layers}

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + position_by_id[token_position][start:end].to(device)
        handles = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                collected[layer].append(hidden[rows, pos, :].detach().float().cpu())
            return hook

        for layer in collected:
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

        try:
            model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

    return {layer: torch.cat(states, dim=0) for layer, states in collected.items()}


class LearnedSubspace(nn.Module):
    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        init = torch.randn(hidden_size, subspace_dim, dtype=torch.float32)
        self.raw = nn.Parameter(torch.linalg.qr(init, mode="reduced").Q)

    def basis(self):
        return torch.linalg.qr(self.raw, mode="reduced").Q


@torch.no_grad()
def run_full_layer_intervention(model, bank, source_states, layer, label_ids, token_position="last_token", batch_size=16):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()
            hidden_new[rows, pos, :] = source.to(hidden.dtype)
            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            all_logits.append(answer_logits(model, outputs, mask, label_ids).cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)


def run_das_intervention(model, bank, source_states, Q, layer, label_ids, token_position="last_token", batch_size=8, require_grad=True):
    device = next(model.parameters()).device
    all_logits = []

    for start in range(0, len(bank["base_input_ids"]), batch_size):
        end = min(start + batch_size, len(bank["base_input_ids"]))
        ids = bank["base_input_ids"][start:end].to(device)
        mask = bank["base_attention_mask"][start:end].to(device)
        source = source_states[start:end].to(device=device, dtype=torch.float32)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        pos = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hidden_new = hidden.clone()

            base = hidden[rows, pos, :].float()
            delta = source - base
            hidden_new[rows, pos, :] = (base + ((delta @ Q) @ Q.T)).to(hidden.dtype)

            return (hidden_new,) + output[1:] if isinstance(output, tuple) else hidden_new

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, label_ids)
            all_logits.append(logits if require_grad else logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)
