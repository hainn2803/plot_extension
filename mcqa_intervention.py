import torch
from sklearn.decomposition import PCA

from mcqa_utils import compute_iia


def basis_dim(mode, n_rows, hidden_size, k=None, max_fit_states=4096):
    if mode == "neuron":
        return int(hidden_size)
    if mode != "pca":
        raise ValueError(f"unknown mode={mode!r}")

    n_fit = 2 * int(n_rows)
    if max_fit_states is not None:
        n_fit = min(n_fit, int(max_fit_states))
    if k is not None:
        n_fit = min(n_fit, int(k))
    return min(n_fit, int(hidden_size))


def fit_bases(sites, base_states, source_states, mode="neuron", k=None, max_fit_states=4096):
    keys = []
    for layer, token_id, _, _ in sites:
        key = (int(layer), token_id)
        if key not in keys:
            keys.append(key)

    bases = {}
    for key in keys:
        if mode == "neuron":
            bases[key] = {"mode": "neuron"}
            continue
        if mode != "pca":
            raise ValueError(f"unknown mode={mode!r}")

        X = torch.cat([base_states[key], source_states[key]], dim=0).float()
        if max_fit_states is not None and len(X) > int(max_fit_states):
            X = X[torch.randperm(len(X))[:int(max_fit_states)]]

        k_eff = min(len(X), X.shape[1])
        if k is not None:
            k_eff = min(k_eff, int(k))

        pca = PCA(n_components=k_eff, whiten=False).fit(X.numpy())
        bases[key] = {"mode": "pca", "components": torch.tensor(pca.components_, dtype=torch.float32)}

    return bases


def last_token_logits(model, outputs, attention_mask, label_ids):
    device = next(model.parameters()).device
    rows = torch.arange(attention_mask.shape[0], device=device)
    cols = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask.to(device) * cols.unsqueeze(0)).max(dim=1).values

    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, dtype=torch.long, device=device)
    logits = hidden @ model.lm_head.weight[ids].to(hidden.dtype).T

    bias = getattr(model.lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[ids]

    softcap = getattr(model.config, "final_logit_softcapping", None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap

    return logits.float()


@torch.no_grad()
def collect_site_activations(model, input_ids, attention_mask, position_by_id, sites, batch_size=32, return_logits=False, label_ids=None):
    device = next(model.parameters()).device
    layer_ids = []
    token_ids = []
    keys = []
    states = {}
    logits = []

    for layer, token_id, _, _ in sites:
        layer = int(layer)
        key = (layer, token_id)

        if layer not in layer_ids:
            layer_ids.append(layer)
        if token_id not in token_ids:
            token_ids.append(token_id)
        if key not in states:
            states[key] = []
            keys.append(key)

    layer_ids = sorted(layer_ids)
    N = input_ids.shape[0]

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)

        pos_by_token = {}
        for token_id in token_ids:
            raw_pos = position_by_id[token_id][start:end].to(device)
            pos_by_token[token_id] = pad_offset + raw_pos

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output

                for key in keys:
                    key_layer, token_id = key
                    if key_layer == layer_id:
                        pos = pos_by_token[token_id]
                        states[key].append(hidden[rows, pos, :].detach().float().cpu())

            return hook

        for layer in layer_ids:
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)

            if return_logits:
                logits.append(last_token_logits(model, outputs, mask, label_ids).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    for key in states:
        states[key] = torch.cat(states[key], dim=0)

    if return_logits:
        return states, torch.cat(logits, dim=0)
    return states


@torch.no_grad()
def run_intervention(model, bank, sites, site_weights, source_states, bases, strength, label_ids, batch_size=32, return_logits=False):
    device = next(model.parameters()).device
    weights = torch.as_tensor(site_weights, dtype=torch.float32, device=device).flatten()

    if weights.numel() != len(sites):
        raise ValueError("site_weights must have one value per site")
    if not torch.isfinite(weights).all() or float(weights.sum().abs().item()) == 0.0:
        raise ValueError("site_weights must be finite with nonzero sum")

    weights = weights / weights.sum()
    input_ids = bank["base_input_ids"]
    attention_mask = bank["base_attention_mask"]
    position_by_id = bank["base_position_by_id"]

    layer_ids = []
    token_ids = []
    outputs_all = []

    for layer, token_id, _, _ in sites:
        layer = int(layer)
        if layer not in layer_ids:
            layer_ids.append(layer)
        if token_id not in token_ids:
            token_ids.append(token_id)

    layer_ids = sorted(layer_ids)

    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])
        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)

        pos_by_token = {}
        for token_id in token_ids:
            pos_by_token[token_id] = pad_offset + position_by_id[token_id][start:end].to(device)

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden
                changed = False

                for site_id, (layer, token_id, a, b) in enumerate(sites):
                    layer = int(layer)
                    a = int(a)
                    b = int(b)

                    if layer != layer_id:
                        continue
                    if not changed:
                        hidden_new = hidden.clone()
                        changed = True

                    key = (layer, token_id)
                    pos = pos_by_token[token_id]
                    base_act = hidden_new[rows, pos, :].float()
                    source_act = source_states[key][start:end].to(device=device, dtype=torch.float32)
                    scale = float(strength) * weights[site_id]

                    if bases[key]["mode"] == "neuron":
                        patched = base_act.clone()
                        patched[:, a:b] = base_act[:, a:b] + scale * (source_act[:, a:b] - base_act[:, a:b])
                    else:
                        comps = bases[key]["components"].to(device=device, dtype=torch.float32)[a:b]
                        diff = source_act - base_act
                        patched = base_act + scale * ((diff @ comps.T) @ comps)

                    hidden_new[rows, pos, :] = patched.to(hidden_new.dtype)

                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (hidden_new,) + output[1:]
                return hidden_new

            return hook

        for layer in layer_ids:
            handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))

        try:
            position_ids = (mask.long().cumsum(dim=-1) - 1).clamp(min=0)
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=position_ids, use_cache=False, return_dict=True)
            batch_logits = last_token_logits(model, outputs, mask, label_ids)

            if return_logits:
                outputs_all.append(batch_logits.detach().cpu())
            else:
                outputs_all.append(torch.softmax(batch_logits, dim=-1).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    return torch.cat(outputs_all, dim=0)


def eval_intervention(model, bank, var_name, sites, weights, source_states, bases, strength, label_ids, batch_size=32):
    outputs = run_intervention(model, bank, sites, weights, source_states, bases, strength, label_ids, batch_size=batch_size)
    return compute_iia(outputs, bank["counterfactual_label_ids"][var_name], var_name)
