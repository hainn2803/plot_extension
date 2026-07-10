import os

import torch

from mcqa_constants import ANSWER_LETTERS, FAMILY_ORDER
from mcqa_data_load_all import answer_label_ids, build_mcqa_banks
from mcqa_intervention import basis_dim, collect_site_activations, eval_intervention
from mcqa_neural_net import load_gemma_model
from mcqa_ot import get_solver, top_sites_from_T
from mcqa_signatures import site_signature, variable_signature
from mcqa_stage_a import stage_a_search
from mcqa_utils import make_sites, set_seed


def run_plot_vanilla_multilayer(model, tokenizer, layers, ft_size=128, cal_size=128, te_size=256, dataset_size=None, dataset_split="train", stage_A_signature_method="concat", stage_B_signature_method="family_mean", stage_A_mode="neuron", stage_B_mode="neuron", stage_A_k=None, stage_B_k=None, stage_A_eps=0.001, stage_B_eps=0.001, stage_A_top_layers=6, stage_A_keep_layers=1, stage_A_iia_threshold=0.7, resolutions=(128, 144, 192, 256, 288, 384, 576, 768), top_k_values=(1, 2, 3, 4, 5), strength_values=(1, 2, 4, 8, 16, 32, 64), stage_A_strength_values=None, stage_A_method="uot", stage_B_method="ot", chosen_token_position_id="last_token", device="cuda", seed=0, batch_size=32, max_fit_states=4096):
    set_seed(seed)
    if stage_A_strength_values is None:
        stage_A_strength_values = strength_values

    label_ids = answer_label_ids(tokenizer)
    ft_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=dataset_size, split=dataset_split, device=device, batch_size=batch_size, seed=seed)

    stage_a = stage_a_search(model, ft_bank, cal_banks, layers, label_ids, signature_method=stage_A_signature_method, mode=stage_A_mode, k=stage_A_k, eps=stage_A_eps, method=stage_A_method, top_k=stage_A_top_layers, keep_layers=stage_A_keep_layers, iia_threshold=stage_A_iia_threshold, strength_values=stage_A_strength_values, token_position=chosen_token_position_id, batch_size=batch_size, max_fit_states=max_fit_states)

    G_stage_A = stage_a["G_stage_A"]
    names = stage_a["names"]
    top_layers = stage_a["top_layers"]

    if stage_B_signature_method == stage_A_signature_method:
        G_stage_B = G_stage_A
    else:
        G_stage_B, _ = variable_signature(ft_bank, num_labels=len(ANSWER_LETTERS), signature_method=stage_B_signature_method, family_order=FAMILY_ORDER)

    stage_B_solver = get_solver(stage_B_method)
    fine_dim = basis_dim(stage_B_mode, len(ft_bank["base_input_ids"]), model.config.hidden_size, stage_B_k, max_fit_states)
    top_k_list = []
    for top_k in top_k_values:
        top_k_list.append(int(top_k))

    best_by_var = {}
    stage_B_results = []
    fine_cache = {}
    source_cache = {}

    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        layers_for_var = top_layers[var_id]
        layer_key = tuple(layers_for_var)
        best = []
        best_correct = -1

        print(f"\n[Stage B variable] {var_id} {var_name} layers={layers_for_var}")

        for resolution in resolutions:
            cache_key = (layer_key, int(resolution))

            if cache_key not in fine_cache:
                sites = []
                for layer in layers_for_var:
                    sites.extend(make_sites(layer, chosen_token_position_id, fine_dim, resolution))

                sig = site_signature(model, ft_bank, sites, label_ids, mode=stage_B_mode, k=stage_B_k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=stage_B_signature_method, family_order=FAMILY_ORDER)
                T = stage_B_solver(G_stage_B, sig["intervention_diff"], eps=stage_B_eps)
                fine_cache[cache_key] = {"sites": sites, "S": sig["intervention_diff"], "T": T, "bases": sig["bases"]}

            cached = fine_cache[cache_key]
            sites = cached["sites"]
            T = cached["T"]
            bases = cached["bases"]

            source_key = (var_name, cache_key)
            if source_key not in source_cache:
                source_cache[source_key] = collect_site_activations(model, cal_bank["source_input_ids"], cal_bank["source_attention_mask"], cal_bank["source_position_by_id"], sites, batch_size=batch_size)
            source_states = source_cache[source_key]

            for top_k in top_k_list:
                selected_sites, selected_indices = top_sites_from_T(T, sites, var_id, top_k, min_mass=1e-8)
                selected_weights = T[var_id, selected_indices].detach().float().cpu()

                for strength in strength_values:
                    iia, correct = eval_intervention(model, cal_bank, var_name, selected_sites, selected_weights, source_states, bases, float(strength), label_ids, batch_size=batch_size)
                    result = {"var_id": var_id, "var_name": var_name, "stage_A_layers": layer_key, "resolution": int(resolution), "top_k": len(selected_sites), "strength": float(strength), "cal_iia": float(iia), "cal_correct": int(correct), "selected_indices": selected_indices, "selected_sites": selected_sites, "selected_weights": selected_weights, "cache_key": cache_key, "selection_method": "vanilla_topk"}
                    stage_B_results.append(result)

                    if result["cal_correct"] > best_correct:
                        best_correct = result["cal_correct"]
                        best = [result]
                    elif result["cal_correct"] == best_correct:
                        best.append(result)

                    print(f"[Stage B CAL] var={var_name} layers={layer_key} resolution={resolution} top_k={len(selected_sites)} strength={strength} correct={correct}/{len(cal_bank['base_input_ids'])} iia={iia:.4f}")

        best_by_var[var_id] = best

    test_results = {}
    for var_id, candidates in best_by_var.items():
        current = []

        for best in candidates:
            var_name = best["var_name"]
            te_bank = te_banks[var_name]
            bases = fine_cache[best["cache_key"]]["bases"]
            source_states = collect_site_activations(model, te_bank["source_input_ids"], te_bank["source_attention_mask"], te_bank["source_position_by_id"], best["selected_sites"], batch_size=batch_size)
            test_iia, test_correct = eval_intervention(model, te_bank, var_name, best["selected_sites"], best["selected_weights"], source_states, bases, best["strength"], label_ids, batch_size=batch_size)

            result = dict(best)
            result["test_iia"] = float(test_iia)
            result["test_correct"] = int(test_correct)
            current.append(result)

            print(f"[TEST] var={var_name} layers={best['stage_A_layers']} resolution={best['resolution']} top_k={best['top_k']} strength={best['strength']} test_iia={test_iia:.4f}")

        test_results[var_id] = current

    return {
        "names": names,
        "G_stage_A": G_stage_A,
        "G_stage_B": G_stage_B,
        "S_coarse": stage_a["S_stage_A"],
        "T_coarse": stage_a["T_stage_A"],
        "top_layers_var": top_layers,
        "top_coarse_by_var": stage_a["top_info"],
        "stage_A_cal_results": stage_a["cal_results"],
        "best_by_var": best_by_var,
        "stage_B_cal_results": stage_B_results,
        "test_results": test_results,
        "fine_cache": fine_cache,
        "stage_A_signature_method": stage_A_signature_method,
        "stage_B_signature_method": stage_B_signature_method,
        "stage_A_mode": stage_A_mode,
        "stage_B_mode": stage_B_mode,
        "stage_B_selection": "vanilla_topk",
    }


if __name__ == "__main__":
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))

    results = run_plot_vanilla_multilayer(model, tokenizer, layers, ft_size=200, cal_size=100, te_size=100, stage_A_signature_method="concat", stage_B_signature_method="family_mean", stage_A_mode="neuron", stage_B_mode="neuron", stage_A_eps=2, stage_B_eps=1, stage_A_method="uot", stage_B_method="ot", stage_A_top_layers=6, stage_A_keep_layers=2, stage_A_iia_threshold=0.7, resolutions=(128, 144, 192, 256, 288, 384, 576, 768), top_k_values=(1, 2, 3, 4), strength_values=(0.5, 1, 2, 4), chosen_token_position_id="last_token", device=device, seed=0, batch_size=32)

    os.makedirs("results", exist_ok=True)
    save_path = "results/plot_multilayer_naive.pt"
    torch.save(results, save_path)
    print("saved to:", save_path)

    print("\n===== FINAL TEST RESULTS =====")
    for var_id in results["test_results"]:
        for tie_id, result in enumerate(results["test_results"][var_id]):
            print("var_id=", var_id, "tie_id=", tie_id, "var_name=", result["var_name"], "stage_A_layers=", result["stage_A_layers"], "resolution=", result["resolution"], "top_k=", result["top_k"], "strength=", result["strength"], "cal_iia=", result["cal_iia"], "test_iia=", result["test_iia"])
