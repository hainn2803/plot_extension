import torch
import torch.nn.functional as F

from mcqa_constants import FAMILY_ORDER, TARGETS
from mcqa_intervention import collect_site_activations, fit_bases, run_intervention
from mcqa_utils import normalize_rows


def make_signature(X, method="family_mean", pair_source_families=None, family_order=FAMILY_ORDER, eps=1e-8):
    X = torch.as_tensor(X, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError("X must have shape [N, D]")

    if method == "concat":
        return normalize_rows(X, eps).reshape(-1)

    if method == "family_mean":
        if pair_source_families is None or len(pair_source_families) != X.shape[0]:
            raise ValueError("pair_source_families must have length N")

        blocks = []
        for family in family_order:
            mask_values = []
            for current_family in pair_source_families:
                mask_values.append(current_family == family)

            mask = torch.tensor(mask_values, dtype=torch.bool, device=X.device)
            if bool(mask.any()):
                block = X[mask].mean(dim=0)
            else:
                block = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)

            block = block - block.mean()
            norm = torch.linalg.vector_norm(block)
            if float(norm.item()) > eps:
                block = block / norm

            blocks.append(block)

        return torch.cat(blocks, dim=0)

    raise ValueError(f"unknown signature method={method!r}")


def variable_signature(bank, num_labels=26, signature_method="family_mean", family_order=FAMILY_ORDER):
    base = torch.as_tensor(bank["base_answer_label_ids"], dtype=torch.long)
    base_onehot = F.one_hot(base, num_classes=num_labels).float()

    signatures = []
    names = []

    for name in TARGETS:
        cf = torch.as_tensor(bank["counterfactual_label_ids"][name], dtype=torch.long)
        delta = F.one_hot(cf, num_classes=num_labels).float() - base_onehot
        signature = make_signature(delta, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order)
        signatures.append(signature)
        names.append(name)

    return torch.stack(signatures, dim=0), names


def site_signature(model, bank, sites, label_ids, mode="neuron", k=None, batch_size=32, strength=1.0, max_fit_states=4096, signature_method="family_mean", family_order=FAMILY_ORDER):
    base_states, base_logits = collect_site_activations(model, bank["base_input_ids"], bank["base_attention_mask"], bank["base_position_by_id"], sites, batch_size=batch_size, return_logits=True, label_ids=label_ids)
    source_states = collect_site_activations(model, bank["source_input_ids"], bank["source_attention_mask"], bank["source_position_by_id"], sites, batch_size=batch_size)
    bases = fit_bases(sites, base_states, source_states, mode=mode, k=k, max_fit_states=max_fit_states)

    signatures = []
    for site in sites:
        patched_logits = run_intervention(model, bank, [site], [1.0], source_states, bases, strength, label_ids, batch_size=batch_size, return_logits=True)
        diff = patched_logits.float() - base_logits.float()
        signature = make_signature(diff, method=signature_method, pair_source_families=bank["pair_source_families"], family_order=family_order)
        signatures.append(signature)

    return {"sites": sites, "intervention_diff": torch.stack(signatures, dim=0), "bases": bases}
