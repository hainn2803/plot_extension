import os
import torch
from mcqa_data_load_all import build_mcqa_banks
from mcqa_neural_net import load_gemma_model
from mcqa_stage_a import ANSWER_LETTERS, FAMILY_ORDER, answer_label_ids, basis_dim, collect_site_activations, eval_intervention, get_solver, site_signature, variable_signature, set_seed

def make_sites(layer, token_position, total_dim, resolution):
    sites = []
    for start in range(0, int(total_dim), int(resolution)):
        end = min(start + int(resolution), int(total_dim))
        sites.append((int(layer), token_position, start, end))
    return sites

def greedy_select_sites(model, var_id, var_name, sites, T, bases, cal_bank, source_states, strength, max_k, pool_size, label_ids, batch_size=32, min_mass=1e-08):
    scores = T[var_id]
    valid = []
    for i in range(len(sites)):
        mass = scores[i]
        if torch.isfinite(mass) and float(mass) > min_mass:
            valid.append(i)
    if not valid:
        raise ValueError(f'no positive-mass sites for var_id={var_id}')
    pool_size = min(max(pool_size, max_k), len(valid))
    valid_scores = torch.stack([scores[i] for i in valid])
    top = torch.topk(valid_scores, k=pool_size).indices.tolist()
    remaining = [valid[i] for i in top]
    selected = []
    path = []
    for step in range(min(max_k, len(remaining))):
        best = None
        for candidate in remaining:
            trial_indices = selected + [candidate]
            trial_sites = [sites[i] for i in trial_indices]
            trial_weights = T[var_id, trial_indices].detach().float().cpu()
            iia, correct = eval_intervention(model, cal_bank, var_name, trial_sites, trial_weights, source_states, bases, strength, label_ids, batch_size=batch_size)
            trial = {'added_index': candidate, 'selected_indices': trial_indices, 'selected_sites': trial_sites, 'selected_weights': trial_weights, 'cal_iia': float(iia), 'cal_correct': int(correct)}
            if best is None or trial['cal_correct'] > best['cal_correct']:
                best = trial
            elif trial['cal_correct'] == best['cal_correct']:
                if float(scores[candidate]) > float(scores[best['added_index']]):
                    best = trial
        selected = best['selected_indices']
        remaining.remove(best['added_index'])
        path.append(best)
        print(f"[GREEDY] var={var_name} strength={strength} step={step + 1} site={sites[best['added_index']]} iia={best['cal_iia']:.4f}")
        if not remaining:
            break
    return path

def run_stage_b(model, tokenizer, stage_a_path='results/stage_a.pt', signature_method='family_mean', mode='neuron', k=None, eps=2.0, method='ot', candidate_pool_size=6, resolutions=(128, 144, 192, 256, 288, 384, 576, 768), top_k_values=(1, 2, 3, 4), strength_values=(0.5, 1, 2, 4), device='cuda', batch_size=32, max_fit_states=4096, save_path='results/stage_b.pt'):
    stage_a = torch.load(stage_a_path, map_location='cpu', weights_only=False)
    config = stage_a['config']
    seed = config['seed']
    set_seed(seed)
    ft_bank, cal_banks, te_banks = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=config['ft_size'], cal_size=config['cal_size'], te_size=config['te_size'], dataset_size=config['dataset_size'], split=config['dataset_split'], device=device, batch_size=config['batch_size'], seed=seed)
    names = stage_a['names']
    top_layers = stage_a['top_layers']
    token_position = config['token_position']
    label_ids = answer_label_ids(tokenizer)
    solver = get_solver(method)
    if signature_method == stage_a['signature_method']:
        G = stage_a['G_stage_A']
        print('reusing Stage A variable signature')
    else:
        G, _ = variable_signature(ft_bank, num_labels=len(ANSWER_LETTERS), signature_method=signature_method, family_order=FAMILY_ORDER)
    dim = basis_dim(mode, len(ft_bank['base_input_ids']), model.config.hidden_size, k, max_fit_states)
    max_k = max(top_k_values)
    best_by_var = {}
    cal_results = []
    fine_cache = {}
    source_cache = {}
    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        layers = top_layers[var_id]
        best = []
        best_correct = -1
        print(f'\n[Stage B] var={var_name} layers={layers}')
        for resolution in resolutions:
            cache_key = (tuple(layers), int(resolution))
            if cache_key not in fine_cache:
                sites = []
                for layer in layers:
                    sites.extend(make_sites(layer, token_position, dim, resolution))
                sig = site_signature(model, ft_bank, sites, label_ids, mode=mode, k=k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=signature_method, family_order=FAMILY_ORDER)
                S = sig['intervention_diff']
                T = solver(G, S, eps=eps)
                fine_cache[cache_key] = {'sites': sites, 'T': T, 'bases': sig['bases']}
            cached = fine_cache[cache_key]
            sites = cached['sites']
            T = cached['T']
            bases = cached['bases']
            source_key = (var_name, cache_key)
            if source_key not in source_cache:
                source_cache[source_key] = collect_site_activations(model, cal_bank['source_input_ids'], cal_bank['source_attention_mask'], cal_bank['source_position_by_id'], sites, batch_size=batch_size)
            for strength in strength_values:
                path = greedy_select_sites(model, var_id, var_name, sites, T, bases, cal_bank, source_cache[source_key], float(strength), max_k, candidate_pool_size, label_ids, batch_size=batch_size)
                for top_k in top_k_values:
                    if top_k > len(path):
                        continue
                    step = path[top_k - 1]
                    result = {'var_id': var_id, 'var_name': var_name, 'stage_A_layers': tuple(layers), 'resolution': int(resolution), 'top_k': int(top_k), 'strength': float(strength), 'cal_iia': step['cal_iia'], 'cal_correct': step['cal_correct'], 'selected_indices': step['selected_indices'], 'selected_sites': step['selected_sites'], 'selected_weights': step['selected_weights'], 'cache_key': cache_key}
                    cal_results.append(result)
                    if result['cal_correct'] > best_correct:
                        best_correct = result['cal_correct']
                        best = [result]
                    elif result['cal_correct'] == best_correct:
                        best.append(result)
        best_by_var[var_id] = best
    test_results = {}
    for var_id, candidates in best_by_var.items():
        current = []
        for best in candidates:
            var_name = best['var_name']
            te_bank = te_banks[var_name]
            bases = fine_cache[best['cache_key']]['bases']
            source_states = collect_site_activations(model, te_bank['source_input_ids'], te_bank['source_attention_mask'], te_bank['source_position_by_id'], best['selected_sites'], batch_size=batch_size)
            test_iia, test_correct = eval_intervention(model, te_bank, var_name, best['selected_sites'], best['selected_weights'], source_states, bases, best['strength'], label_ids, batch_size=batch_size)
            result = dict(best)
            result['test_iia'] = float(test_iia)
            result['test_correct'] = int(test_correct)
            current.append(result)
            print(f"[TEST] var={var_name} layers={best['stage_A_layers']} resolution={best['resolution']} top_k={best['top_k']} strength={best['strength']} test_iia={test_iia:.4f}")
        test_results[var_id] = current
    results = {'names': names, 'stage_A_layers': top_layers, 'signature_method': signature_method, 'best_by_var': best_by_var, 'cal_results': cal_results, 'test_results': test_results}
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(results, save_path)
    print('saved Stage B to:', save_path)
    return results
if __name__ == '__main__':
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    results = run_stage_b(model=model, tokenizer=tokenizer, stage_a_path='results/stage_a.pt', signature_method='family_mean', mode='neuron', k=None, eps=2.0, method='ot', candidate_pool_size=6, resolutions=(128, 144, 192, 256, 288, 384, 576, 768), top_k_values=(1, 2, 3, 4), strength_values=(0.5, 1, 2, 4), device=device, batch_size=32, save_path='results/stage_b.pt')
    print('\n===== FINAL TEST RESULTS =====')
    for var_id in results['test_results']:
        for tie_id, result in enumerate(results['test_results'][var_id]):
            print('var_id=', var_id, 'tie_id=', tie_id, 'var_name=', result['var_name'], 'stage_A_layers=', result['stage_A_layers'], 'resolution=', result['resolution'], 'top_k=', result['top_k'], 'strength=', result['strength'], 'cal_iia=', result['cal_iia'], 'test_iia=', result['test_iia'])
