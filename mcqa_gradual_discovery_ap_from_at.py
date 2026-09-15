import os

import torch
import torch.nn.functional as F

from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_utils import set_seed

from mcqa_gradual_discovery_at import (
    answer_label_ids,
    basis_dim,
    collect_site_activations,
    compute_iia,
    fit_bases,
    last_token_logits,
    make_sites,
)


def load_frozen_at_handle(path):
    """Load a validated AT handle from the previous AT-only run."""
    results = torch.load(path, map_location="cpu", weights_only=False)
    handle = results.get("handle")
    assert handle is not None, "AT result file does not contain a frozen handle"
    assert handle.get("variable") == "answer_token", (
        f"expected an answer_token handle, got {handle.get('variable')!r}"
    )
    return results, handle


def handle_features(states, handle):
    """Extract and concatenate the selected coordinates of a frozen handle."""
    features = []

    for L, token_id, a, b in handle["sites"]:
        key = (int(L), token_id)
        X = states[key].float()

        if handle["bases"][key]["mode"] == "neuron":
            features.append(X[:, a:b])
        else:
            comps = handle["bases"][key]["components"][a:b].float()
            features.append(X @ comps.T)

    return torch.cat(features, dim=-1)


def ap_target_mask(bank):
    """Rows where AP changes: answer_pointer or both source families."""
    return torch.tensor(
        [family in ("answer_pointer", "both") for family in bank["pair_source_families"]],
        dtype=torch.float32,
    )


def family_mask(bank, families):
    families = set(families)
    return torch.tensor(
        [family in families for family in bank["pair_source_families"]],
        dtype=torch.bool,
    )


def site_features(states, site, bases):
    """Coordinates of one candidate site in neuron/PCA space."""
    L, token_id, a, b = site
    key = (int(L), token_id)
    X = states[key].float()

    if bases[key]["mode"] == "neuron":
        return X[:, int(a):int(b)]

    comps = bases[key]["components"][int(a):int(b)].float()
    return X @ comps.T


def ap_identity_scores(bank, sites, base_states, source_states, bases, identity_lambda=1.0):
    """Way 3 identity: AP-sensitive but AT-only-invariant.

    Positive rows: answer_pointer or both -> AP changes.
    Negative rows: answer_token -> AT changes while AP stays fixed.
    """
    ap_mask = family_mask(bank, ("answer_pointer", "both"))
    at_only_mask = family_mask(bank, ("answer_token",))

    rows = []
    for site_index, site in enumerate(sites):
        base = site_features(base_states, site, bases)
        source = site_features(source_states, site, bases)
        rms_change = (source - base).pow(2).mean(dim=-1).sqrt()

        d_ap = float(rms_change[ap_mask].mean().item())
        d_at_only = float(rms_change[at_only_mask].mean().item())
        identity_score = d_ap - float(identity_lambda) * d_at_only
        selectivity = d_ap / (d_ap + d_at_only + 1e-8)

        rows.append({
            "site_index": int(site_index),
            "site": site,
            "d_ap": d_ap,
            "d_at_only": d_at_only,
            "selectivity": float(selectivity),
            "identity_score": float(identity_score),
        })

    return rows


@torch.no_grad()
def run_ap_intervention(
    model,
    bank,
    sites,
    site_weights,
    source_states,
    bases,
    strength,
    label_ids,
    batch_size=32,
    return_logits=False,
):
    """Patch one or more AP sites and return output scores.

    For a multi-site handle, each site's effective scale is
        strength * normalized_identity_weight.
    """
    device = next(model.parameters()).device

    weights = torch.as_tensor(site_weights, dtype=torch.float32, device=device).flatten()
    assert weights.numel() == len(sites)
    assert torch.isfinite(weights).all()
    assert float(weights.sum().abs().item()) > 0.0
    weights = weights / weights.sum()

    input_ids = bank["base_input_ids"]
    attention_mask = bank["base_attention_mask"]
    position_by_id = bank["base_position_by_id"]

    layer_ids = sorted({int(site[0]) for site in sites})
    token_ids = list(dict.fromkeys(site[1] for site in sites))
    outputs_all = []

    for start in range(0, input_ids.shape[0], batch_size):
        end = min(start + batch_size, input_ids.shape[0])

        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)

        pos_by_token = {}
        for token_id in token_ids:
            pos_by_token[token_id] = (
                pad_offset + position_by_id[token_id][start:end].to(device)
            )

        handles = []

        def make_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                changed = False

                for site_id, (L, token_id, a, b) in enumerate(sites):
                    L, a, b = int(L), int(a), int(b)
                    if L != layer_id:
                        continue

                    changed = True
                    pos = pos_by_token[token_id]
                    key = (L, token_id)

                    # Use the pre-intervention activation as the base for every
                    # band. This avoids order effects between disjoint bands.
                    base_act = hidden[rows, pos, :].float()
                    source_act = source_states[key][start:end].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    scale = float(strength) * weights[site_id]

                    if bases[key]["mode"] == "neuron":
                        hidden_new[rows, pos, a:b] = (
                            base_act[:, a:b]
                            + scale * (source_act[:, a:b] - base_act[:, a:b])
                        ).to(hidden_new.dtype)
                    else:
                        comps = bases[key]["components"][a:b].to(
                            device=device,
                            dtype=torch.float32,
                        )
                        diff = source_act - base_act
                        delta = scale * ((diff @ comps.T) @ comps)
                        current = hidden_new[rows, pos, :].float()
                        hidden_new[rows, pos, :] = (current + delta).to(hidden_new.dtype)

                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (hidden_new,) + output[1:]
                return hidden_new

            return hook

        for L in layer_ids:
            handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))

        try:
            outputs = model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
            batch_logits = last_token_logits(model, outputs, mask, label_ids)
            if return_logits:
                outputs_all.append(batch_logits.detach().cpu())
            else:
                outputs_all.append(torch.softmax(batch_logits, dim=-1).detach().cpu())
        finally:
            for handle in handles:
                handle.remove()

    return torch.cat(outputs_all, dim=0)


@torch.no_grad()
def run_intervention_collect_at(
    model,
    bank,
    ap_sites,
    site_weights,
    ap_source_states,
    ap_bases,
    at_sites,
    strength=1.0,
    batch_size=32,
):
    """Patch AP sites and collect the resulting downstream AT activations."""
    device = next(model.parameters()).device

    weights = torch.as_tensor(site_weights, dtype=torch.float32, device=device).flatten()
    assert weights.numel() == len(ap_sites)
    assert torch.isfinite(weights).all()
    assert float(weights.sum().abs().item()) > 0.0
    weights = weights / weights.sum()

    ap_layers = sorted({int(site[0]) for site in ap_sites})
    ap_tokens = list(dict.fromkeys(site[1] for site in ap_sites))

    at_keys = []
    at_layers = []
    at_tokens = []
    for L, token_id, _, _ in at_sites:
        key = (int(L), token_id)
        if key not in at_keys:
            at_keys.append(key)
        if int(L) not in at_layers:
            at_layers.append(int(L))
        if token_id not in at_tokens:
            at_tokens.append(token_id)

    states = {key: [] for key in at_keys}

    input_ids = bank["base_input_ids"]
    attention_mask = bank["base_attention_mask"]
    position_by_id = bank["base_position_by_id"]

    for start in range(0, len(input_ids), batch_size):
        end = min(start + batch_size, len(input_ids))

        ids = input_ids[start:end].to(device)
        mask = attention_mask[start:end].to(device)
        rows = torch.arange(ids.shape[0], device=device)
        pad_offset = (mask == 0).sum(dim=1)

        ap_positions = {}
        for token_id in ap_tokens:
            ap_positions[token_id] = (
                pad_offset + position_by_id[token_id][start:end].to(device)
            )

        at_positions = {}
        for token_id in at_tokens:
            at_positions[token_id] = (
                pad_offset + position_by_id[token_id][start:end].to(device)
            )

        handles = []

        def make_ap_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden.clone()
                changed = False

                for site_id, (L, token_id, a, b) in enumerate(ap_sites):
                    L, a, b = int(L), int(a), int(b)
                    if L != layer_id:
                        continue

                    changed = True
                    pos = ap_positions[token_id]
                    key = (L, token_id)

                    base_act = hidden[rows, pos, :].float()
                    source_act = ap_source_states[key][start:end].to(
                        device=device,
                        dtype=torch.float32,
                    )
                    scale = float(strength) * weights[site_id]

                    if ap_bases[key]["mode"] == "neuron":
                        hidden_new[rows, pos, a:b] = (
                            base_act[:, a:b]
                            + scale * (source_act[:, a:b] - base_act[:, a:b])
                        ).to(hidden_new.dtype)
                    else:
                        comps = ap_bases[key]["components"][a:b].to(
                            device=device,
                            dtype=torch.float32,
                        )
                        diff = source_act - base_act
                        delta = scale * ((diff @ comps.T) @ comps)
                        current = hidden_new[rows, pos, :].float()
                        hidden_new[rows, pos, :] = (current + delta).to(hidden_new.dtype)

                if not changed:
                    return None
                if isinstance(output, tuple):
                    return (hidden_new,) + output[1:]
                return hidden_new

            return hook

        def make_at_hook(layer_id):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                for key in at_keys:
                    L, token_id = key
                    if L == layer_id:
                        pos = at_positions[token_id]
                        states[key].append(
                            hidden[rows, pos, :].detach().float().cpu()
                        )

            return hook

        for L in ap_layers:
            handles.append(model.model.layers[L].register_forward_hook(make_ap_hook(L)))
        for L in at_layers:
            handles.append(model.model.layers[L].register_forward_hook(make_at_hook(L)))

        try:
            model.model(
                input_ids=ids,
                attention_mask=mask,
                position_ids=(mask.long().cumsum(dim=-1) - 1).clamp(min=0),
                use_cache=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

    for key in states:
        states[key] = torch.cat(states[key], dim=0)

    return states


def eval_ap_intervention(model, bank, sites, weights, source_states, bases, strength, label_ids, batch_size=32):
    """Evaluate AP IIA after patching the proposed AP handle."""
    outputs = run_ap_intervention(
        model=model,
        bank=bank,
        sites=sites,
        site_weights=weights,
        source_states=source_states,
        bases=bases,
        strength=strength,
        label_ids=label_ids,
        batch_size=batch_size,
        return_logits=False,
    )
    return compute_iia(
        outputs,
        bank["counterfactual_label_ids"]["answer_pointer"],
        "answer_pointer",
    )


def at_recovery_diagnostic(
    model,
    bank,
    ap_sites,
    weights,
    ap_source_states,
    ap_bases,
    at_handle,
    strength,
    batch_size=32,
):
    """Measure how closely an AP patch moves the frozen AT handle to source AT."""
    at_sites = at_handle["sites"]

    at_base_states = collect_site_activations(
        model,
        bank["base_input_ids"],
        bank["base_attention_mask"],
        bank["base_position_by_id"],
        at_sites,
        batch_size=batch_size,
    )
    at_source_states = collect_site_activations(
        model,
        bank["source_input_ids"],
        bank["source_attention_mask"],
        bank["source_position_by_id"],
        at_sites,
        batch_size=batch_size,
    )
    at_patched_states = run_intervention_collect_at(
        model=model,
        bank=bank,
        ap_sites=ap_sites,
        site_weights=weights,
        ap_source_states=ap_source_states,
        ap_bases=ap_bases,
        at_sites=at_sites,
        strength=strength,
        batch_size=batch_size,
    )

    base = handle_features(at_base_states, at_handle)
    source = handle_features(at_source_states, at_handle)
    patched = handle_features(at_patched_states, at_handle)

    desired = source - base
    actual = patched - base

    target_rows = ap_target_mask(bank).bool()
    desired_norm = desired.norm(dim=-1)
    remaining_norm = (source - patched).norm(dim=-1)
    valid = target_rows & (desired_norm > 1e-8)

    if bool(valid.any()):
        recovery_fraction = 1.0 - remaining_norm[valid] / desired_norm[valid]
        mean_recovery_fraction = float(recovery_fraction.mean().item())

        cosine = F.cosine_similarity(actual[valid], desired[valid], dim=-1, eps=1e-8)
        mean_shift_cosine = float(cosine.mean().item())
    else:
        mean_recovery_fraction = 0.0
        mean_shift_cosine = 0.0

    return {
        "mean_recovery_fraction": mean_recovery_fraction,
        "mean_shift_cosine": mean_shift_cosine,
        "num_nonzero_target_examples": int(valid.sum().item()),
    }


def make_ap_handle(sites, site_indices, identity_scores, weights, strength, bases, resolution, at_handle):
    """Freeze an AP handle selected by identity, output gate, and frozen-AT alignment."""
    return {
        "variable": "answer_pointer",
        "discovered_through": "answer_token",
        "sites": list(sites),
        "indices": [int(i) for i in site_indices],
        "weights": torch.as_tensor(weights, dtype=torch.float32).cpu(),
        "identity_scores": torch.as_tensor(identity_scores, dtype=torch.float32).cpu(),
        "strength": float(strength),
        "bases": bases,
        "resolution": int(resolution),
        "downstream_sites": list(at_handle["sites"]),
    }


def run_ap_through_at(
    model,
    tokenizer,
    at_results_path="results/at_only_cosine_debug.pt",
    selected_layers=None,
    ft_size=400,
    cal_size=200,
    te_size=200,
    dataset_size=None,
    dataset_split="train",
    resolutions=(128,),
    site_mode="neuron",
    pca_k=None,
    identity_lambda=1.0,
    identity_min_ap_change=0.0,
    candidate_pool_size=8,
    max_top_k=8,
    strength_values=(0.5, 1, 2, 4, 8, 16, 32, 64),
    ap_min_cal_iia=0.7,
    run_full_layer_diagnostic=False,
    full_layer_strength_values=(1.0,),
    chosen_token_position_id="last_token",
    device="cuda",
    seed=0,
    batch_size=32,
    max_fit_states=4096,
):
    """Way 3: identify AP locally, gate by output IIA, select by frozen-AT alignment."""
    set_seed(seed)

    _, at_frozen_handle = load_frozen_at_handle(at_results_path)
    label_ids = answer_label_ids(tokenizer)

    if selected_layers is None:
        selected_layers = [int(L) for L in range(int(model.config.num_hidden_layers))]
    else:
        if max(selected_layers) >= int(model.config.num_hidden_layers) or min(selected_layers) < 0:
            raise ValueError(
                f"selected_layers contains invalid layer indices; "
                f"model has {model.config.num_hidden_layers} layers"
            )

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

    at_sites = at_frozen_handle["sites"]
    at_layer = min(int(site[0]) for site in at_sites)
    ap_layers = [int(L) for L in selected_layers if int(L) < at_layer]
    assert len(ap_layers) > 0, f"no AP layers are upstream of frozen AT layer {at_layer}"

    n_ft = int(ft_bank["base_input_ids"].shape[0])
    site_dim = basis_dim(site_mode, n_ft, hidden_size, pca_k, max_fit_states)

    if isinstance(resolutions, int):
        resolutions = [int(resolutions)]
    else:
        resolutions = [int(r) for r in resolutions]
    resolutions = list(dict.fromkeys(resolutions))
    assert resolutions and all(r > 0 for r in resolutions)

    ap_candidate_sites = []
    site_resolution = {}
    for L in ap_layers:
        for resolution in resolutions:
            for site in make_sites(L, chosen_token_position_id, site_dim, resolution):
                if site not in site_resolution:
                    ap_candidate_sites.append(site)
                    site_resolution[site] = int(resolution)

    print(
        f"[AP sites] layers={ap_layers} resolutions={resolutions} "
        f"site_dim={site_dim} total={len(ap_candidate_sites)} frozen_AT_layer={at_layer}"
    )

    # ------------------------------------------------------------------
    #   positive: AP-changing families -> candidate should change
    #   negative: AT-only family      -> candidate should stay invariant
    # Output and frozen AT are not used to define candidate identity.
    # ------------------------------------------------------------------
    ap_base_states = collect_site_activations(
        model,
        ft_bank["base_input_ids"],
        ft_bank["base_attention_mask"],
        ft_bank["base_position_by_id"],
        ap_candidate_sites,
        batch_size=batch_size,
    )
    ap_source_states = collect_site_activations(
        model,
        ft_bank["source_input_ids"],
        ft_bank["source_attention_mask"],
        ft_bank["source_position_by_id"],
        ap_candidate_sites,
        batch_size=batch_size,
    )
    ap_bases = fit_bases(
        ap_candidate_sites,
        ap_base_states,
        ap_source_states,
        mode=site_mode,
        k=pca_k,
        max_fit_states=max_fit_states,
    )

    all_identity_rows = ap_identity_scores(
        bank=ft_bank,
        sites=ap_candidate_sites,
        base_states=ap_base_states,
        source_states=ap_source_states,
        bases=ap_bases,
        identity_lambda=identity_lambda,
    )

    identity_candidates = [
        row for row in all_identity_rows
        if row["d_ap"] >= float(identity_min_ap_change)
    ]
    identity_candidates.sort(key=lambda row: row["identity_score"], reverse=True)
    pool_size = min(int(candidate_pool_size), len(identity_candidates))
    identity_ranking = []

    print(f"\n===== TOP {pool_size} AP IDENTITY CANDIDATES =====")
    for rank, row in enumerate(identity_candidates[:pool_size], start=1):
        candidate = dict(row)
        candidate["rank"] = int(rank)
        candidate["resolution"] = int(site_resolution[candidate["site"]])
        identity_ranking.append(candidate)
        print(
            f"[AP identity] rank={rank:02d} site={candidate['site']} "
            f"resolution={candidate['resolution']} "
            f"d_ap={candidate['d_ap']:.6f} "
            f"d_at_only={candidate['d_at_only']:.6f} "
            f"selectivity={candidate['selectivity']:.4f} "
            f"score={candidate['identity_score']:.6f}"
        )

    # ------------------------------------------------------------------
    # CAL:
    #   1) output IIA is only an eligibility gate;
    #   2) among eligible AP-identity handles, frozen-AT alignment chooses best.
    # ------------------------------------------------------------------
    ap_cal_bank = cal_banks["answer_pointer"]

    _, cal_base_logits = collect_site_activations(
        model,
        ap_cal_bank["base_input_ids"],
        ap_cal_bank["base_attention_mask"],
        ap_cal_bank["base_position_by_id"],
        [ap_candidate_sites[0]],
        batch_size=batch_size,
        return_logits=True,
        label_ids=label_ids,
    )

    base_iia, base_correct = compute_iia(
        cal_base_logits,
        ap_cal_bank["base_answer_label_ids"],
        "answer_pointer",
    )

    cal_source_states, cal_source_logits = collect_site_activations(
        model,
        ap_cal_bank["source_input_ids"],
        ap_cal_bank["source_attention_mask"],
        ap_cal_bank["source_position_by_id"],
        ap_candidate_sites,
        batch_size=batch_size,
        return_logits=True,
        label_ids=label_ids,
    )

    source_target_iia, source_target_correct = compute_iia(
        cal_source_logits,
        ap_cal_bank["counterfactual_label_ids"]["answer_pointer"],
        "answer_pointer",
    )

    clean_diagnostics = {
        "base_accuracy": float(base_iia),
        "base_correct": int(base_correct),
        "source_target_accuracy": float(source_target_iia),
        "source_target_correct": int(source_target_correct),
        "num_examples": int(len(ap_cal_bank["base_input_ids"])),
    }

    print(
        f"[AP clean CAL] base={base_correct}/{clean_diagnostics['num_examples']} "
        f"({base_iia:.4f}) source-vs-target={source_target_correct}/"
        f"{clean_diagnostics['num_examples']} ({source_target_iia:.4f})"
    )

    # Optional diagnostic: patch an entire upstream layer.
    full_layer_results = []
    if run_full_layer_diagnostic:
        print("\n===== FULL-LAYER AP DIAGNOSTIC =====")
        for L in ap_layers:
            full_site = (L, chosen_token_position_id, 0, site_dim)
            for strength in full_layer_strength_values:
                iia, correct = eval_ap_intervention(
                    model=model,
                    bank=ap_cal_bank,
                    sites=[full_site],
                    weights=[1.0],
                    source_states=cal_source_states,
                    bases=ap_bases,
                    strength=strength,
                    label_ids=label_ids,
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
                    f"[AP full-layer CAL] layer={L} strength={strength} "
                    f"correct={correct}/{clean_diagnostics['num_examples']} "
                    f"iia={iia:.4f}"
                )

    # Test each discovered AP band separately.
    singleton_results = []
    print("\n===== INDIVIDUAL AP CANDIDATE CALIBRATION =====")
    for candidate in identity_ranking:
        site = candidate["site"]
        for strength in strength_values:
            iia, correct = eval_ap_intervention(
                model=model,
                bank=ap_cal_bank,
                sites=[site],
                weights=[1.0],
                source_states=cal_source_states,
                bases=ap_bases,
                strength=strength,
                label_ids=label_ids,
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
                f"[AP candidate CAL] rank={candidate['rank']:02d} "
                f"site={site} identity={candidate['identity_score']:.6f} "
                f"strength={strength} correct={correct}/"
                f"{clean_diagnostics['num_examples']} iia={iia:.4f}"
            )

    # Same-layer top-k handles. Positive identity scores determine each site's
    # relative intervention scale; global strength is calibrated separately.
    group_candidates = {}
    for candidate in identity_ranking:
        key = (int(candidate["site"][0]), int(candidate["resolution"]))
        group_candidates.setdefault(key, []).append(candidate)

    top_k_results = []
    print("\n===== WITHIN-LAYER / WITHIN-RESOLUTION TOP-K AP CALIBRATION =====")
    for (layer, resolution), candidates in group_candidates.items():
        layer_max_k = min(int(max_top_k), len(candidates))

        for top_k in range(1, layer_max_k + 1):
            selected = candidates[:top_k]
            sites = [candidate["site"] for candidate in selected]
            site_indices = [candidate["site_index"] for candidate in selected]
            selected_ranks = [candidate["rank"] for candidate in selected]
            selected_identity_scores = torch.tensor(
                [candidate["identity_score"] for candidate in selected],
                dtype=torch.float32,
            )

            positive_scores = selected_identity_scores.clamp_min(0.0)
            if float(positive_scores.sum().item()) > 0.0:
                weights = positive_scores / positive_scores.sum()
            else:
                weights = torch.full_like(
                    selected_identity_scores,
                    1.0 / len(selected_identity_scores),
                )

            for strength in strength_values:
                iia, correct = eval_ap_intervention(
                    model=model,
                    bank=ap_cal_bank,
                    sites=sites,
                    weights=weights,
                    source_states=cal_source_states,
                    bases=ap_bases,
                    strength=strength,
                    label_ids=label_ids,
                    batch_size=batch_size,
                )

                at_recovery = None
                if iia >= float(ap_min_cal_iia):
                    at_recovery = at_recovery_diagnostic(
                        model=model,
                        bank=ap_cal_bank,
                        ap_sites=sites,
                        weights=weights,
                        ap_source_states=cal_source_states,
                        ap_bases=ap_bases,
                        at_handle=at_frozen_handle,
                        strength=strength,
                        batch_size=batch_size,
                    )

                row = {
                    "selection_type": "top_k",
                    "layer": int(layer),
                    "resolution": int(resolution),
                    "top_k": int(top_k),
                    "selected_global_ranks": selected_ranks,
                    "selected_sites": sites,
                    "selected_indices": site_indices,
                    "selected_identity_scores": selected_identity_scores,
                    "selected_weights": weights.detach().float().cpu(),
                    "strength": float(strength),
                    "cal_iia": float(iia),
                    "cal_correct": int(correct),
                    "at_recovery": at_recovery,
                }
                top_k_results.append(row)

                extra = ""
                if at_recovery is not None:
                    extra = (
                        f" at_recovery={at_recovery['mean_recovery_fraction']:.4f}"
                        f" at_cos={at_recovery['mean_shift_cosine']:.4f}"
                    )

                print(
                    f"[AP top-k CAL] layer={layer} resolution={resolution} top_k={top_k} "
                    f"global_ranks={selected_ranks} strength={strength} "
                    f"weights={[round(float(x), 4) for x in weights]} "
                    f"correct={correct}/{clean_diagnostics['num_examples']} "
                    f"iia={iia:.4f}{extra}"
                )

    eligible = [
        row for row in top_k_results
        if row["cal_iia"] >= float(ap_min_cal_iia) and row["at_recovery"] is not None
    ]

    if eligible:
        best_ap = max(
            eligible,
            key=lambda r: (
                r["at_recovery"]["mean_shift_cosine"],
                r["at_recovery"]["mean_recovery_fraction"],
                r["cal_iia"],
                -r["top_k"],
                -abs(r["strength"]),
            ),
        )
        status = "passed_calibration"
    else:
        best_ap = max(
            top_k_results,
            key=lambda r: (
                r["cal_iia"],
                r["cal_correct"],
                -r["top_k"],
                -abs(r["strength"]),
            ),
        )
        status = "failed_calibration"

    print(
        f"\n[AP BEST] status={status} layer={best_ap['layer']} "
        f"resolution={best_ap['resolution']} top_k={best_ap['top_k']} "
        f"global_ranks={best_ap['selected_global_ranks']} "
        f"sites={best_ap['selected_sites']} "
        f"weights={[round(float(x), 4) for x in best_ap['selected_weights']]} "
        f"strength={best_ap['strength']} cal_iia={best_ap['cal_iia']:.4f}"
    )

    best_cal_at_recovery = best_ap["at_recovery"]

    if best_cal_at_recovery is not None:
        print(
            f"[AP->AT CAL] recovery={best_cal_at_recovery['mean_recovery_fraction']:.4f} "
            f"shift_cosine={best_cal_at_recovery['mean_shift_cosine']:.4f}"
        )

    results = {
        "method": "ap_identity_then_output_gate_then_frozen_at_selection",
        "status": status,
        "frozen_at_handle": at_frozen_handle,
        "ap_layers": ap_layers,
        "all_sites": ap_candidate_sites,
        "all_identity_rows": all_identity_rows,
        "identity_ranking": identity_ranking,
        "clean_cal_diagnostics": clean_diagnostics,
        "full_layer_cal_results": full_layer_results,
        "singleton_cal_results": singleton_results,
        "top_k_cal_results": top_k_results,
        "best_ap": best_ap,
        "best_cal_at_recovery": best_cal_at_recovery,
        "handle": None,
        "test_results": None,
        "config": {
            "at_results_path": at_results_path,
            "resolutions": tuple(resolutions),
            "selected_layers": tuple(selected_layers),
            "ap_layers": tuple(ap_layers),
            "site_mode": site_mode,
            "pca_k": pca_k,
            "identity_lambda": float(identity_lambda),
            "identity_min_ap_change": float(identity_min_ap_change),
            "candidate_pool_size": int(candidate_pool_size),
            "max_top_k": int(max_top_k),
            "strength_values": tuple(float(s) for s in strength_values),
            "ap_min_cal_iia": float(ap_min_cal_iia),
            "run_full_layer_diagnostic": bool(run_full_layer_diagnostic),
            "full_layer_strength_values": tuple(
                float(s) for s in full_layer_strength_values
            ),
            "seed": int(seed),
        },
    }

    if status != "passed_calibration":
        print(
            "[STOP] AP did not pass CAL. Saving diagnostics without freezing "
            "AP or touching TEST."
        )
        return results

    ap_handle = make_ap_handle(
        sites=best_ap["selected_sites"],
        site_indices=best_ap["selected_indices"],
        identity_scores=best_ap["selected_identity_scores"],
        weights=best_ap["selected_weights"],
        strength=best_ap["strength"],
        bases=ap_bases,
        resolution=best_ap["resolution"],
        at_handle=at_frozen_handle,
    )
    results["handle"] = ap_handle

    # ------------------------------------------------------------------
    # TEST: touched once, only after CAL passes and the AP handle is frozen.
    # ------------------------------------------------------------------
    ap_test_bank = te_banks["answer_pointer"]

    test_source_states = collect_site_activations(
        model,
        ap_test_bank["source_input_ids"],
        ap_test_bank["source_attention_mask"],
        ap_test_bank["source_position_by_id"],
        ap_handle["sites"],
        batch_size=batch_size,
    )

    test_iia, test_correct = eval_ap_intervention(
        model=model,
        bank=ap_test_bank,
        sites=ap_handle["sites"],
        weights=ap_handle["weights"],
        source_states=test_source_states,
        bases=ap_handle["bases"],
        strength=ap_handle["strength"],
        label_ids=label_ids,
        batch_size=batch_size,
    )

    test_at_recovery = at_recovery_diagnostic(
        model=model,
        bank=ap_test_bank,
        ap_sites=ap_handle["sites"],
        weights=ap_handle["weights"],
        ap_source_states=test_source_states,
        ap_bases=ap_handle["bases"],
        at_handle=at_frozen_handle,
        strength=ap_handle["strength"],
        batch_size=batch_size,
    )

    results["test_results"] = {
        "test_iia": float(test_iia),
        "test_correct": int(test_correct),
        "num_examples": int(len(ap_test_bank["base_input_ids"])),
        "at_recovery": test_at_recovery,
    }

    print(
        f"[AP TEST] correct={test_correct}/{len(ap_test_bank['base_input_ids'])} "
        f"iia={test_iia:.4f}"
    )
    print(
        f"[AP->AT TEST diagnostic] recovery="
        f"{test_at_recovery['mean_recovery_fraction']:.4f} "
        f"shift_cosine={test_at_recovery['mean_shift_cosine']:.4f}"
    )

    return results


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device

    results = run_ap_through_at(
        model=model,
        tokenizer=tokenizer,
        at_results_path="results/at_only_cosine_debug.pt",
        selected_layers=(15, 16, 17, 18, 19, 20),
        ft_size=400,
        cal_size=200,
        te_size=200,
        dataset_size=None,
        dataset_split="train",
        resolutions=[128, 256, 288],
        site_mode="neuron",
        pca_k=None,
        identity_lambda=1.0,
        identity_min_ap_change=0.0,
        candidate_pool_size=20,
        max_top_k=10,
        strength_values=(0.5, 1, 2, 4, 8, 16, 32, 64),
        ap_min_cal_iia=0.7,
        run_full_layer_diagnostic=True,
        full_layer_strength_values=(1.0,),
        chosen_token_position_id="last_token",
        device=device,
        seed=0,
        batch_size=32,
        max_fit_states=4096,
    )

    os.makedirs("results", exist_ok=True)
    save_path = "results/ap_through_at_way3.pt"
    torch.save(results, save_path)
    print("saved to:", save_path)