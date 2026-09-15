import os
import string
from itertools import combinations

import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from mcqa_data_load_all import build_mcqa_banks, letter_token_id

from mcqa_neural_net import load_gemma_model
from mcqa_utils import set_seed, normalize_rows


FAMILY_ORDER = ("answer_pointer", "answer_token", "both")
TARGETS = ("answer_pointer", "answer_token")
ANSWER_LETTERS = tuple(string.ascii_uppercase)


def answer_label_ids(tokenizer):
    """Get token ids for the A-Z answer labels."""
    ids = []
    for letter in ANSWER_LETTERS:
        ids.append(letter_token_id(tokenizer, letter))
    return ids


def make_signature(X, method="family_mean", pair_source_families=None, family_order=FAMILY_ORDER, eps=1e-8):
    """Turn per-example features into one signature vector."""
    X = torch.as_tensor(X, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError("X must have shape [N, D]")

    # if method == "concat":
    #     X = normalize_rows(X, eps)
    #     return X.reshape(-1)
    if method == "concat":
        signature = X.reshape(-1)
        signature = signature - signature.mean()
        return signature / signature.norm().clamp_min(eps)


    if method == "family_mean":
        if pair_source_families is None or len(pair_source_families) != X.shape[0]:
            raise ValueError("pair_source_families must have length N")

        blocks = []
        for family in family_order:
            mask_values = []
            for current_family in pair_source_families:
                mask_values.append(current_family == family)
            mask = torch.tensor(mask_values, dtype=torch.bool, device=X.device)
            block = X[mask].mean(dim=0) if bool(mask.any()) else torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            block = block - block.mean()
            norm = torch.linalg.vector_norm(block)
            if float(norm.item()) > eps:
                block = block / norm
            blocks.append(block)

        return torch.cat(blocks, dim=0)

    raise ValueError(f"unknown signature method={method!r}")


def variable_signature(bank, num_labels=26, signature_method="family_mean", family_order=FAMILY_ORDER):
    """Build one signature for each causal variable."""
    base = torch.as_tensor(bank["base_answer_label_ids"], dtype=torch.long)
    base_onehot = F.one_hot(base, num_classes=num_labels).float()
    signatures, names = [], []

    for name in TARGETS:
        cf = torch.as_tensor(bank["counterfactual_label_ids"][name], dtype=torch.long)
        delta = F.one_hot(cf, num_classes=num_labels).float() - base_onehot
        signatures.append(make_signature(delta, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order))
        names.append(name)

    return torch.stack(signatures, dim=0), names


def basis_dim(mode, n_rows, hidden_size, k=None, max_fit_states=4096):
    """Get the number of dimensions used to define sites."""
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
    """Prepare the neuron or PCA basis for each layer and token."""
    keys = []
    for L, token_id, _, _ in sites:
        key = (int(L), token_id)
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
    """Get answer-label logits at the last real token."""
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
    """Collect activations at the layers and token positions used by the sites."""
    device = next(model.parameters()).device
    layer_ids, token_ids, keys = [], [], []
    states, logits = {}, []

    for L, token_id, _, _ in sites:
        L = int(L)
        key = (L, token_id)
        if L not in layer_ids:
            layer_ids.append(L)
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
            """Create a hook that records activations from one layer."""
            def hook(_module, _inputs, output):
                """Save the requested token activations."""
                hidden = output[0] if isinstance(output, tuple) else output
                for key in keys:
                    L_key, token_id = key
                    if L_key == layer_id:
                        pos = pos_by_token[token_id]
                        states[key].append(hidden[rows, pos, :].detach().float().cpu())
            return hook

        for L in layer_ids:
            handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
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
    """Patch the selected sites and return the model outputs."""
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
    layer_ids, token_ids, outputs_all = [], [], []

    for L, token_id, _, _ in sites:
        L = int(L)
        if L not in layer_ids:
            layer_ids.append(L)
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
            """Create a hook that applies interventions at one layer."""
            def hook(_module, _inputs, output):
                """Patch the selected sites in this layer."""
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden
                changed = False

                for site_id, (L, token_id, a, b) in enumerate(sites):
                    L, a, b = int(L), int(a), int(b)
                    if L != layer_id:
                        continue
                    if not changed:
                        hidden_new = hidden.clone()
                        changed = True

                    key = (L, token_id)
                    pos = pos_by_token[token_id]
                    base_act = hidden_new[rows, pos, :].float()
                    source_act = source_states[key][start:end].to(device=device, dtype=torch.float32)
                    # scale = float(strength) * weights[site_id]
                    scale = float(strength)

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

        for L in layer_ids:
            handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

        try:
            outputs = model.model(input_ids=ids, attention_mask=mask, position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0), use_cache=False, return_dict=True)
            batch_logits = last_token_logits(model, outputs, mask, label_ids)
            if return_logits:
                outputs_all.append(batch_logits.detach().cpu())
            else:
                outputs_all.append(torch.softmax(batch_logits, dim=-1).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    return torch.cat(outputs_all, dim=0)


def make_sites(layer_id, token_id, total_dim, resolution):
    """Split one layer into contiguous candidate sites."""
    sites = []
    for start in range(0, int(total_dim), int(resolution)):
        end = min(start + int(resolution), int(total_dim))
        sites.append((int(layer_id), token_id, int(start), int(end)))
    return sites


def site_signature(model, bank, sites, label_ids, mode="neuron", k=None, batch_size=32, strength=1.0, max_fit_states=4096, signature_method="family_mean", family_order=FAMILY_ORDER):
    """Build a signature for each neural site from its intervention effect."""
    base_states, base_logits = collect_site_activations(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], sites, batch_size=batch_size, return_logits=True, label_ids=label_ids)
    source_states = collect_site_activations(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], sites, batch_size=batch_size)
    bases = fit_bases(sites, base_states, source_states, mode=mode, k=k, max_fit_states=max_fit_states)
    signatures = []

    for site in sites:
        patched_logits = run_intervention(model, bank, [site], [1.0], source_states, bases, strength, label_ids, batch_size=batch_size, return_logits=True)
        diff = patched_logits.float() - base_logits.float()
        signatures.append(make_signature(diff, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order))

    return {
        "sites": sites,
        "intervention_diff": torch.stack(signatures, dim=0),
        "bases": bases,
        "base_states": base_states,
        "source_states": source_states,
        "base_logits": base_logits,
    }


def cosine_scores(target_signature, site_signatures, eps=1e-8):
    """Cosine similarity from one abstract signature to every neural signature."""
    target = torch.as_tensor(target_signature, dtype=torch.float32).reshape(1, -1)
    neural = torch.as_tensor(site_signatures, dtype=torch.float32)
    if neural.ndim != 2 or neural.shape[1] != target.shape[1]:
        raise ValueError(
            f"signature mismatch: target={tuple(target.shape)}, neural={tuple(neural.shape)}"
        )
    target = F.normalize(target, dim=-1, eps=eps)
    neural = F.normalize(neural, dim=-1, eps=eps)
    return (neural @ target.T).squeeze(-1)


def top_sites_from_scores(scores, sites, top_k):
    """Pick top-k by cosine and normalize their positive cosine scores as weights."""
    scores = torch.as_tensor(scores, dtype=torch.float32).flatten()
    if scores.numel() != len(sites):
        raise ValueError("scores must have one value per site")
    valid = torch.isfinite(scores)
    if not bool(valid.any()):
        raise ValueError("all cosine scores are non-finite")

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    k = min(int(top_k), int(valid_indices.numel()))
    local_scores = scores[valid_indices]
    _, order = torch.topk(local_scores, k=k)
    selected_indices = valid_indices[order].tolist()
    selected_sites = [sites[i] for i in selected_indices]
    selected_scores = scores[selected_indices]
    positive_scores = selected_scores.clamp_min(0.0)
    if float(positive_scores.sum().item()) > 0.0:
        selected_weights = positive_scores / positive_scores.sum()
    else:
        selected_weights = torch.full_like(selected_scores, 1.0 / len(selected_scores))
    return selected_sites, selected_indices, selected_weights


def compute_iia(outputs, labels, var_name, pointer_num_labels=4):
    """Compute IIA and count the number of correct predictions."""
    scores = torch.as_tensor(outputs)
    labels = torch.as_tensor(labels, dtype=torch.long)

    if var_name == "answer_pointer":
        scores = scores[:, :pointer_num_labels]
    elif var_name != "answer_token":
        raise ValueError(f"unknown var_name={var_name!r}")

    pred = scores.argmax(dim=-1).cpu()
    labels = labels.cpu()
    correct = int((pred == labels).sum().item())
    total = int(labels.numel())
    return correct / total, correct


def eval_intervention(model, bank, var_name, sites, weights, source_states, bases, strength, label_ids, batch_size=32):
    """Run an intervention and evaluate its IIA."""
    outputs = run_intervention(model, bank, sites, weights, source_states, bases, strength, label_ids, batch_size=batch_size)
    return compute_iia(outputs, bank["counterfactual_label_ids"][var_name], var_name)


def make_at_handle(sites, site_indices, cosines, weights, strength, bases, bandwidth):
    """Freeze one validated top-k collection of AT bands as a causal handle."""
    return {
        "variable": "answer_token",
        "sites": list(sites),
        "indices": [int(i) for i in site_indices],
        "weights": torch.as_tensor(weights, dtype=torch.float32).cpu(),
        "cosines": torch.as_tensor(cosines, dtype=torch.float32).cpu(),
        "strength": float(strength),
        "bases": bases,
        "bandwidth": int(bandwidth),
    }


def run_at_only_discovery(
    model,
    tokenizer,
    selected_layers=None,
    ft_size=128,
    cal_size=128,
    te_size=256,
    dataset_size=None,
    dataset_split="train",
    bandwidth=128,
    signature_method="concat",
    site_mode="neuron",
    pca_k=None,
    signature_strength=1.0,
    candidate_pool_size=8,
    max_top_k=8,
    strength_values=(0.5, 1, 2, 4, 8, 16, 32, 64),
    at_min_cal_iia=0.7,
    run_full_layer_diagnostic=False,
    full_layer_strength_values=(1.0,),
    chosen_token_position_id="last_token",
    device="cuda",
    seed=0,
    batch_size=32,
    max_fit_states=4096,
):
    """Discover and validate AT only; never continue to AP after a failed AT step."""
    set_seed(seed)
    label_ids = answer_label_ids(tokenizer)

    if selected_layers is None:
        selected_layers = [int(L) for L in range(int(model.config.num_hidden_layers))]
    else:
        if max(selected_layers) >= int(model.config.num_hidden_layers) or min(selected_layers) < 0:
            raise ValueError(f"selected_layers contains invalid layer indices; model has {model.config.num_hidden_layers} layers")

    hidden_size = int(model.config.hidden_size)

    ft_bank, cal_banks, te_banks = build_mcqa_banks(
        model=model,
        tokenizer=tokenizer,
        train_pool_size=ft_size,
        cal_size=cal_size,
        te_size=te_size,
        dataset_size=dataset_size,
        split=dataset_split,
        device=device,
        batch_size=batch_size,
        seed=seed,
    )

    n_ft = int(ft_bank["base_input_ids"].shape[0])
    site_dim = basis_dim(site_mode, n_ft, hidden_size, pca_k, max_fit_states)
    all_sites = []
    for L in selected_layers:
        all_sites.extend(
            make_sites(L, chosen_token_position_id, site_dim, bandwidth)
        )
    print(
        f"[sites] selected_layers={selected_layers} bandwidth={bandwidth} "
        f"site_dim={site_dim} total={len(all_sites)}"
    )

    # FT discovery: cosine-rank every band against the abstract AT signature.
    G_output, names = variable_signature(
        ft_bank,
        num_labels=len(ANSWER_LETTERS),
        signature_method=signature_method,
        family_order=FAMILY_ORDER,
    )
    at_var_id = names.index("answer_token")
    output_sig = site_signature(
        model=model,
        bank=ft_bank,
        sites=all_sites,
        label_ids=label_ids,
        mode=site_mode,
        k=pca_k,
        batch_size=batch_size,
        strength=signature_strength,
        max_fit_states=max_fit_states,
        signature_method=signature_method,
        family_order=FAMILY_ORDER,
    )
    S_to_output = output_sig["intervention_diff"]
    at_cosine_scores = cosine_scores(G_output[at_var_id], S_to_output)

    pool_size = min(int(candidate_pool_size), len(all_sites))
    ranked_indices = torch.topk(at_cosine_scores, k=pool_size).indices.tolist()
    cosine_ranking = []
    print(f"\n===== TOP {pool_size} AT COSINE CANDIDATES =====")
    for rank, site_index in enumerate(ranked_indices, start=1):
        row = {
            "rank": rank,
            "site_index": int(site_index),
            "site": all_sites[site_index],
            "cosine": float(at_cosine_scores[site_index].item()),
        }
        cosine_ranking.append(row)
        print(
            f"[AT cosine] rank={rank:02d} site={row['site']} "
            f"cosine={row['cosine']:.6f}"
        )

    # CAL diagnostics. First verify that the clean model and labels agree.
    at_cal_bank = cal_banks["answer_token"]
    _, cal_base_logits = collect_site_activations(
        model,
        at_cal_bank["base_input_ids"],
        at_cal_bank["base_attention_mask"],
        at_cal_bank["base_position_by_id"],
        [all_sites[0]],
        batch_size=batch_size,
        return_logits=True,
        label_ids=label_ids,
    )
    base_iia, base_correct = compute_iia(
        cal_base_logits,
        at_cal_bank["base_answer_label_ids"],
        "answer_token",
    )
    cal_source_states, cal_source_logits = collect_site_activations(
        model,
        at_cal_bank["source_input_ids"],
        at_cal_bank["source_attention_mask"],
        at_cal_bank["source_position_by_id"],
        all_sites,
        batch_size=batch_size,
        return_logits=True,
        label_ids=label_ids,
    )
    source_target_iia, source_target_correct = compute_iia(
        cal_source_logits,
        at_cal_bank["counterfactual_label_ids"]["answer_token"],
        "answer_token",
    )
    clean_diagnostics = {
        "base_accuracy": float(base_iia),
        "base_correct": int(base_correct),
        "source_target_accuracy": float(source_target_iia),
        "source_target_correct": int(source_target_correct),
        "num_examples": int(len(at_cal_bank["base_input_ids"])),
    }
    print(
        f"[AT clean CAL] base={base_correct}/{clean_diagnostics['num_examples']} "
        f"({base_iia:.4f}) source-vs-target={source_target_correct}/"
        f"{clean_diagnostics['num_examples']} ({source_target_iia:.4f})"
    )

    # Diagnostic only: a full-layer patch separates a bad band ranking from a
    # deeper issue in the bank, labels, hook location, or intervention code.
    full_layer_results = []
    if run_full_layer_diagnostic:
        print("\n===== FULL-LAYER AT DIAGNOSTIC =====")
        for L in selected_layers:
            full_site = (L, chosen_token_position_id, 0, site_dim)
            for strength in full_layer_strength_values:
                iia, correct = eval_intervention(
                    model,
                    at_cal_bank,
                    "answer_token",
                    [full_site],
                    [1.0],
                    cal_source_states,
                    output_sig["bases"],
                    strength,
                    label_ids,
                    batch_size=batch_size,
                )
                row = {
                    "layer": int(L),
                    "site": full_site,
                    "strength": float(strength),
                    "cal_iia": float(iia),
                    "cal_correct": int(correct),
                }
                full_layer_results.append(row)
                print(
                    f"[AT full-layer CAL] layer={L} strength={strength} "
                    f"correct={correct}/{clean_diagnostics['num_examples']} iia={iia:.4f}"
                )

    # Test every proposed band separately. This fixes the old prefix bug where
    # rank 2 was only tested together with a possibly bad rank-1 site.
    singleton_results = []
    print("\n===== INDIVIDUAL AT CANDIDATE CALIBRATION =====")
    for candidate in cosine_ranking:
        site = candidate["site"]
        for strength in strength_values:
            iia, correct = eval_intervention(
                model,
                at_cal_bank,
                "answer_token",
                [site],
                [1.0],
                cal_source_states,
                output_sig["bases"],
                strength,
                label_ids,
                batch_size=batch_size,
            )
            row = {
                **candidate,
                "strength": float(strength),
                "cal_iia": float(iia),
                "cal_correct": int(correct),
            }
            singleton_results.append(row)
            print(
                f"[AT candidate CAL] rank={candidate['rank']:02d} site={site} "
                f"cosine={candidate['cosine']:.6f} strength={strength} "
                f"correct={correct}/{clean_diagnostics['num_examples']} iia={iia:.4f}"
            )

    # Top-k handle search: group the top-8 global cosine candidates by layer,
    # then test the ranked prefixes within each layer. Cross-layer handles are
    # excluded because temporal copies of AT can overwrite one another.
    layer_groups = {}
    for candidate in cosine_ranking:
        layer = int(candidate["site"][0])
        layer_groups.setdefault(layer, []).append(candidate)

    top_k_results = []
    print("\n===== WITHIN-LAYER TOP-K AT CALIBRATION =====")
    for layer, layer_candidates in layer_groups.items():
        layer_max_k = min(int(max_top_k), len(layer_candidates))
        for top_k in range(1, layer_max_k + 1):
            selected = layer_candidates[:top_k]
            sites = [candidate["site"] for candidate in selected]
            site_indices = [candidate["site_index"] for candidate in selected]
            selected_ranks = [candidate["rank"] for candidate in selected]
            selected_cosines = torch.tensor(
                [candidate["cosine"] for candidate in selected],
                dtype=torch.float32,
            )
            positive_cosines = selected_cosines.clamp_min(0.0)
            if float(positive_cosines.sum().item()) > 0.0:
                weights = positive_cosines / positive_cosines.sum()
            else:
                weights = torch.full_like(
                    selected_cosines,
                    1.0 / len(selected_cosines),
                )

            for strength in strength_values:
                iia, correct = eval_intervention(
                    model,
                    at_cal_bank,
                    "answer_token",
                    sites,
                    weights,
                    cal_source_states,
                    output_sig["bases"],
                    strength,
                    label_ids,
                    batch_size=batch_size,
                )
                row = {
                    "selection_type": "top_k",
                    "layer": layer,
                    "top_k": int(top_k),
                    "selected_global_ranks": selected_ranks,
                    "selected_sites": sites,
                    "selected_indices": site_indices,
                    "selected_cosines": selected_cosines,
                    "selected_weights": weights.detach().float().cpu(),
                    "strength": float(strength),
                    "cal_iia": float(iia),
                    "cal_correct": int(correct),
                }
                top_k_results.append(row)
                print(
                    f"[AT top-k CAL] layer={layer} top_k={top_k} "
                    f"global_ranks={selected_ranks} strength={strength} "
                    f"sites={sites} correct={correct}/"
                    f"{clean_diagnostics['num_examples']} iia={iia:.4f}"
                )

    best_at = max(
        top_k_results,
        key=lambda r: (
            r["cal_correct"],
            r["cal_iia"],
            -r["top_k"],
            -abs(r["strength"]),
        ),
    )
    status = (
        "passed_calibration"
        if best_at["cal_iia"] >= float(at_min_cal_iia)
        else "failed_calibration"
    )
    print(
        f"\n[AT BEST] status={status} layer={best_at['layer']} "
        f"top_k={best_at['top_k']} "
        f"global_ranks={best_at['selected_global_ranks']} "
        f"sites={best_at['selected_sites']} "
        f"strength={best_at['strength']} cal_iia={best_at['cal_iia']:.4f}"
    )

    results = {
        "method": "at_only_cosine_within_layer_top_k",
        "status": status,
        "names": names,
        "all_sites": all_sites,
        "G_at_to_output": G_output[at_var_id],
        "S_to_output": S_to_output,
        "at_cosine_scores": at_cosine_scores,
        "cosine_ranking": cosine_ranking,
        "clean_cal_diagnostics": clean_diagnostics,
        "full_layer_cal_results": full_layer_results,
        "singleton_cal_results": singleton_results,
        "top_k_cal_results": top_k_results,
        "best_at": best_at,
        "handle": None,
        "test_results": None,
        "config": {
            "bandwidth": int(bandwidth),
            "selected_layers": tuple(selected_layers),
            "signature_method": signature_method,
            "site_mode": site_mode,
            "pca_k": pca_k,
            "signature_strength": float(signature_strength),
            "candidate_pool_size": int(candidate_pool_size),
            "max_top_k": int(max_top_k),
            "strength_values": tuple(float(s) for s in strength_values),
            "at_min_cal_iia": float(at_min_cal_iia),
            "run_full_layer_diagnostic": bool(run_full_layer_diagnostic),
            "full_layer_strength_values": tuple(
                float(s) for s in full_layer_strength_values
            ),
            "seed": int(seed),
        },
    }

    if status != "passed_calibration":
        print(
            "[STOP] AT did not pass CAL. Saving diagnostics without freezing "
            "AT, evaluating TEST, or searching AP."
        )
        return results

    at_handle = make_at_handle(
        sites=best_at["selected_sites"],
        site_indices=best_at["selected_indices"],
        cosines=best_at["selected_cosines"],
        weights=best_at["selected_weights"],
        strength=best_at["strength"],
        bases=output_sig["bases"],
        bandwidth=bandwidth,
    )
    results["handle"] = at_handle

    # TEST is touched once, only after the AT handle passes CAL and is frozen.
    at_test_bank = te_banks["answer_token"]
    at_test_source_states = collect_site_activations(
        model,
        at_test_bank["source_input_ids"],
        at_test_bank["source_attention_mask"],
        at_test_bank["source_position_by_id"],
        at_handle["sites"],
        batch_size=batch_size,
    )
    test_iia, test_correct = eval_intervention(
        model,
        at_test_bank,
        "answer_token",
        at_handle["sites"],
        at_handle["weights"],
        at_test_source_states,
        at_handle["bases"],
        at_handle["strength"],
        label_ids,
        batch_size=batch_size,
    )
    results["test_results"] = {
        "test_iia": float(test_iia),
        "test_correct": int(test_correct),
        "num_examples": int(len(at_test_bank["base_input_ids"])),
    }
    print(
        f"[AT TEST] correct={test_correct}/{len(at_test_bank['base_input_ids'])} "
        f"iia={test_iia:.4f}"
    )
    return results


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    results = run_at_only_discovery(
        model=model,
        tokenizer=tokenizer,
        # selected_layers=(23, 24, 25),
        ft_size=400,
        cal_size=200,
        te_size=200,
        dataset_size=None,
        dataset_split="train",
        bandwidth=128,
        signature_method="concat",
        site_mode="neuron",
        pca_k=None,
        signature_strength=1.0,
        candidate_pool_size=20,
        max_top_k=10,
        strength_values=(0.5, 1, 2, 4, 8, 16, 32, 64),
        at_min_cal_iia=0.7,
        run_full_layer_diagnostic=False,
        full_layer_strength_values=(1.0,),
        chosen_token_position_id="last_token",
        device=device,
        seed=0,
        batch_size=32,
    )

    os.makedirs("results", exist_ok=True)
    save_path = "results/at_only_cosine_debug.pt"
    torch.save(results, save_path)
    print("saved to:", save_path)