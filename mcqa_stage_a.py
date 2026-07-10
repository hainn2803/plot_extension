import os

import torch

from mcqa_constants import ANSWER_LETTERS, FAMILY_ORDER
from mcqa_data_load_all import answer_label_ids, build_mcqa_banks
from mcqa_intervention import basis_dim, collect_site_activations, eval_intervention
from mcqa_neural_net import load_gemma_model
from mcqa_ot import get_solver, top_sites_from_T
from mcqa_signatures import site_signature, variable_signature
from mcqa_utils import set_seed


def choose_stage_a_layers(model, names, cal_banks, sites, bases, T, label_ids, strength_values, top_k, keep_layers, threshold, batch_size=32):
    top_layers = {}
    top_info = {}
    all_results = []

    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        top_sites, top_indices = top_sites_from_T(T, sites, var_id, top_k, min_mass=0.0)
        candidates = []

        print(f"\n[Stage A variable] {var_id} {var_name}")

        for rank, site in enumerate(top_sites, start=1):
            site_index = top_indices[rank - 1]
            layer = int(site[0])
            mass = float(T[var_id, site_index].detach().cpu().item())
            source_states = collect_site_activations(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], [site], batch_size=batch_size)

            best = None
            for strength in strength_values:
                iia, correct = eval_intervention(model, cal_bank, var_name, [site], [1.0], source_states, bases, float(strength), label_ids, batch_size=batch_size)
                result = {"var_id": var_id, "var_name": var_name, "raw_rank": rank, "site_index": site_index, "site": site, "layer": layer, "coupling_mass": mass, "strength": float(strength), "cal_iia": float(iia), "cal_correct": int(correct)}
                all_results.append(result)

                if best is None or result["cal_correct"] > best["cal_correct"]:
                    best = result

                print(f"[Stage A CAL] var={var_name} layer={layer} strength={strength} correct={correct}/{len(cal_bank['base_input_ids'])} iia={iia:.4f}")

            candidates.append(best)

        passing = []
        for candidate in candidates:
            if candidate["cal_iia"] >= float(threshold):
                passing.append(candidate)

        if not passing:
            passing = candidates
            print(f"[Stage A fallback] var={var_name}: no layer passed threshold={threshold}")

        passing.sort(key=lambda x: x["cal_iia"], reverse=True)
        selected = passing[:int(keep_layers)]

        top_layers[var_id] = []
        for candidate in selected:
            top_layers[var_id].append(candidate["layer"])

        top_info[var_id] = selected
        print(f"[Stage A retained] var={var_name} layers={top_layers[var_id]}")

    return top_layers, top_info, all_results


def stage_a_search(model, ft_bank, cal_banks, layers, label_ids, signature_method="concat", mode="neuron", k=None, eps=1.0, method="uot", top_k=6, keep_layers=2, iia_threshold=0.6, strength_values=(0.5, 1, 2, 4), token_position="last_token", batch_size=32, max_fit_states=4096):
    G, names = variable_signature(ft_bank, num_labels=len(ANSWER_LETTERS), signature_method=signature_method, family_order=FAMILY_ORDER)
    dim = basis_dim(mode, len(ft_bank["base_input_ids"]), model.config.hidden_size, k, max_fit_states)

    sites = []
    for layer in layers:
        sites.append((int(layer), token_position, 0, dim))

    sig = site_signature(model, ft_bank, sites, label_ids, mode=mode, k=k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=signature_method, family_order=FAMILY_ORDER)
    S = sig["intervention_diff"]
    T = get_solver(method)(G, S, eps=eps)

    top_layers, top_info, cal_results = choose_stage_a_layers(model, names, cal_banks, sites, sig["bases"], T, label_ids, strength_values, top_k, keep_layers, iia_threshold, batch_size=batch_size)

    return {
        "names": names,
        "top_layers": top_layers,
        "top_info": top_info,
        "cal_results": cal_results,
        "G_stage_A": G,
        "S_stage_A": S,
        "T_stage_A": T,
        "sites": sites,
        "bases": sig["bases"],
        "signature_method": signature_method,
    }


def log_all_layers(model, names, cal_banks, sites, bases, T, label_ids, strength_values, path, batch_size=32):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    with open(path, "w") as f:
        for var_id, var_name in enumerate(names):
            cal_bank = cal_banks[var_name]
            scores = T[var_id]
            order = torch.argsort(scores, descending=True)
            rank_by_index = {}

            for rank, site_index in enumerate(order.tolist(), start=1):
                rank_by_index[site_index] = rank

            source_states = collect_site_activations(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], sites, batch_size=batch_size)
            header = f"\n===== {var_name} =====\n"
            print(header, end="")
            f.write(header)

            for site_index, site in enumerate(sites):
                layer = int(site[0])
                mass = float(scores[site_index].detach().cpu().item())
                rank = rank_by_index[site_index]

                for strength in strength_values:
                    iia, correct = eval_intervention(model, cal_bank, var_name, [site], [1.0], source_states, bases, float(strength), label_ids, batch_size=batch_size)
                    line = f"layer={layer:2d} strength={float(strength):4.1f} iia={iia:.4f} correct={correct}/{len(cal_bank['base_input_ids'])} uot_rank={rank:2d} mass={mass:.8f}\n"
                    print(line, end="")
                    f.write(line)


def run_stage_a(model, tokenizer, layers, ft_size=200, cal_size=100, te_size=100, dataset_size=None, dataset_split="train", signature_method="concat", mode="neuron", k=None, eps=1.0, method="uot", top_k=6, keep_layers=2, iia_threshold=0.6, strength_values=(0.5, 1, 2, 4), token_position="last_token", device="cuda", seed=0, batch_size=32, max_fit_states=4096, save_path="results/stage_a.pt", log_path=None):
    set_seed(seed)
    label_ids = answer_label_ids(tokenizer)

    ft_bank, cal_banks, _ = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=dataset_size, split=dataset_split, device=device, batch_size=batch_size, seed=seed)
    results = stage_a_search(model, ft_bank, cal_banks, layers, label_ids, signature_method=signature_method, mode=mode, k=k, eps=eps, method=method, top_k=top_k, keep_layers=keep_layers, iia_threshold=iia_threshold, strength_values=strength_values, token_position=token_position, batch_size=batch_size, max_fit_states=max_fit_states)

    if log_path is not None:
        log_all_layers(model, results["names"], cal_banks, results["sites"], results["bases"], results["T_stage_A"], label_ids, strength_values, log_path, batch_size=batch_size)

    results["config"] = {
        "ft_size": ft_size,
        "cal_size": cal_size,
        "te_size": te_size,
        "dataset_size": dataset_size,
        "dataset_split": dataset_split,
        "token_position": token_position,
        "seed": seed,
        "batch_size": batch_size,
    }

    results.pop("bases")
    results.pop("sites")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(results, save_path)
    print("saved Stage A to:", save_path)
    print("selected layers:", results["top_layers"])
    return results


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))

    run_stage_a(model, tokenizer, layers, ft_size=200, cal_size=100, te_size=100, signature_method="concat", mode="neuron", eps=1.0, method="uot", top_k=6, keep_layers=2, iia_threshold=0.6, strength_values=(0.5, 1, 2, 4), token_position="last_token", device=device, seed=0, batch_size=32, save_path="results/stage_a.pt", log_path="results/stage_a_all_layers.txt")
