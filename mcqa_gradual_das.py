import random
import string
import unicodedata
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn


ANSWER_LETTERS = tuple(string.ascii_uppercase)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def letter_token_id(tokenizer, letter):
    """Return a single-token id for one answer letter."""
    for text in (" " + letter, letter):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"letter {letter!r} is not a single token")


def answer_label_ids(tokenizer):
    label_ids = []
    for letter in ANSWER_LETTERS:
        label_ids.append(letter_token_id(tokenizer, letter))
    return label_ids


def full_vocab_iia(logits, target_class, tokenizer):
    predicted_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    target_classes = target_class.detach().cpu().tolist()

    predicted = []
    for token_id in predicted_ids:
        text = tokenizer.decode([int(token_id)])
        text = unicodedata.normalize("NFKC", text).strip().upper()
        predicted.append(text)

    expected = []
    for target in target_classes:
        expected.append(ANSWER_LETTERS[int(target)])

    if len(expected) == 0:
        return 0.0

    correct = 0
    for prediction, target in zip(predicted, expected):
        if prediction == target:
            correct += 1

    return correct / len(expected)


def output_label_ids(label_ids, output_space):
    """Select the model's output classes independently of the tokenizer argument."""
    if output_space == "full":
        return None
    if output_space == "az":
        return label_ids
    raise ValueError(f"Unknown output_space: {output_space!r}; expected 'full' or 'az'")


def output_targets(target_class, label_ids, output_space, device):
    """Map A-Z class indices to token IDs only for full-vocabulary logits."""
    output_label_ids(label_ids, output_space)
    target_class = target_class.to(device=device, dtype=torch.long)

    if output_space == "full":
        label_ids = torch.as_tensor(label_ids, device=device, dtype=torch.long)
        return label_ids[target_class]

    return target_class


def output_iia(logits, target_class, tokenizer, output_space):
    """Score full-vocabulary or A-Z predictions against A-Z class indices."""
    output_label_ids(None, output_space)

    if output_space == "full":
        if tokenizer is None:
            raise ValueError("tokenizer is required when output_space='full'")
        return full_vocab_iia(logits, target_class, tokenizer)

    if target_class.numel() == 0:
        return 0.0

    predicted = logits.argmax(dim=-1).detach().cpu()
    target = target_class.detach().cpu()
    return float((predicted == target).float().mean())


def answer_logits(model, outputs, attention_mask, label_ids=None):
    device = outputs.last_hidden_state.device
    rows = torch.arange(attention_mask.shape[0], device=device)
    indices = torch.arange(attention_mask.shape[1], device=device)
    positions = (attention_mask * indices.unsqueeze(0)).max(dim=1).values

    hidden = outputs.last_hidden_state[rows, positions, :]

    if label_ids is None:
        logits = model.lm_head(hidden)
    else:
        ids = torch.tensor(label_ids, device=device, dtype=torch.long)
        weights = model.lm_head.weight[ids].to(hidden.dtype)
        logits = hidden @ weights.T

        bias = getattr(model.lm_head, "bias", None)
        if bias is not None:
            logits = logits + bias[ids]

    logits = logits.float()

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits


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
        position = pad_offset + position_by_id[token_position][start:end].to(device)
        captured = {}

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            captured["x"] = hidden[rows, position, :].detach().float().cpu()

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()

        states.append(captured["x"])

    return torch.cat(states, dim=0)


@torch.no_grad()
def collect_all_layer_states(model, input_ids, attention_mask, position_by_id, layers, token_position="last_token", batch_size=32):
    device = next(model.parameters()).device
    collected = {}

    for layer in layers:
        collected[int(layer)] = []

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(len(ids), device=device)

        pad_offset = (mask == 0).sum(dim=1)
        position = pad_offset + position_by_id[token_position][start:end].to(device)
        handles = []

        def make_hook(layer):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                collected[layer].append(hidden[rows, position, :].detach().float().cpu())
            return hook

        for layer in collected:
            handle = model.model.layers[layer].register_forward_hook(make_hook(layer))
            handles.append(handle)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

    result = {}
    for layer, states in collected.items():
        result[layer] = torch.cat(states, dim=0)
    return result


class LearnedSubspace(nn.Module):
    def __init__(self, hidden_size, subspace_dim):
        super().__init__()
        initial = torch.randn(hidden_size, subspace_dim, dtype=torch.float32)
        self.raw = nn.Parameter(torch.linalg.qr(initial, mode="reduced").Q)

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
        position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            new_hidden[rows, position, :] = source.to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, label_ids)
            all_logits.append(logits.cpu())
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
        position = pad_offset + bank["base_position_by_id"][token_position][start:end].to(device)

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            new_hidden = hidden.clone()
            base = hidden[rows, position, :].float()
            delta = source - base
            projected_delta = (delta @ Q) @ Q.T
            new_hidden[rows, position, :] = (base + projected_delta).to(hidden.dtype)

            if isinstance(output, tuple):
                return (new_hidden,) + output[1:]
            return new_hidden

        handle = model.model.layers[layer].register_forward_hook(hook)

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            logits = answer_logits(model, outputs, mask, label_ids)

            if require_grad:
                all_logits.append(logits)
            else:
                all_logits.append(logits.detach().cpu())
        finally:
            handle.remove()

    return torch.cat(all_logits, dim=0)
