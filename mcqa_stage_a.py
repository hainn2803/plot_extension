import os
import string
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from mcqa_data_load_all import build_mcqa_banks, letter_token_id
from mcqa_neural_net import load_gemma_model
from mcqa_ot import solve_ot, solve_uot
from mcqa_utils import set_seed, normalize_rows
FAMILY_ORDER = ('answer_pointer', 'answer_token', 'both')
TARGETS = ('answer_pointer', 'answer_token')
ANSWER_LETTERS = tuple(string.ascii_uppercase)

def answer_label_ids(tokenizer):
    ids = []
    for letter in ANSWER_LETTERS:
        ids.append(letter_token_id(tokenizer, letter))
    return ids

def get_solver(name):
    if name == 'ot':
        return solve_ot
    if name == 'uot':
        return solve_uot
    raise ValueError(f'unknown solver={name!r}')

def make_signature(X, method='family_mean', pair_source_families=None, family_order=FAMILY_ORDER, eps=1e-08):
    X = torch.as_tensor(X, dtype=torch.float32)
    if X.ndim != 2:
        raise ValueError('X must have shape [N, D]')
    if method == 'concat':
        X = normalize_rows(X, eps)
        return X.reshape(-1)
    if method == 'family_mean':
        if pair_source_families is None or len(pair_source_families) != X.shape[0]:
            raise ValueError('pair_source_families must have length N')
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
    raise ValueError(f'unknown signature method={method!r}')

def variable_signature(bank, num_labels=26, signature_method='family_mean', family_order=FAMILY_ORDER):
    base = torch.as_tensor(bank['base_answer_label_ids'], dtype=torch.long)
    base_onehot = F.one_hot(base, num_classes=num_labels).float()
    signatures, names = ([], [])
    for name in TARGETS:
        cf = torch.as_tensor(bank['counterfactual_label_ids'][name], dtype=torch.long)
        delta = F.one_hot(cf, num_classes=num_labels).float() - base_onehot
        signatures.append(make_signature(delta, method=signature_method, pair_source_families=bank['pair_source_families'], family_order=family_order))
        names.append(name)
    return (torch.stack(signatures, dim=0), names)

def basis_dim(mode, n_rows, hidden_size, k=None, max_fit_states=4096):
    if mode == 'neuron':
        return int(hidden_size)
    if mode != 'pca':
        raise ValueError(f'unknown mode={mode!r}')
    n_fit = 2 * int(n_rows)
    if max_fit_states is not None:
        n_fit = min(n_fit, int(max_fit_states))
    if k is not None:
        n_fit = min(n_fit, int(k))
    return min(n_fit, int(hidden_size))

def fit_bases(sites, base_states, source_states, mode='neuron', k=None, max_fit_states=4096):
    keys = []
    for L, token_id, _, _ in sites:
        key = (int(L), token_id)
        if key not in keys:
            keys.append(key)
    bases = {}
    for key in keys:
        if mode == 'neuron':
            bases[key] = {'mode': 'neuron'}
            continue
        if mode != 'pca':
            raise ValueError(f'unknown mode={mode!r}')
        X = torch.cat([base_states[key], source_states[key]], dim=0).float()
        if max_fit_states is not None and len(X) > int(max_fit_states):
            X = X[torch.randperm(len(X))[:int(max_fit_states)]]
        k_eff = min(len(X), X.shape[1])
        if k is not None:
            k_eff = min(k_eff, int(k))
        pca = PCA(n_components=k_eff, whiten=False).fit(X.numpy())
        bases[key] = {'mode': 'pca', 'components': torch.tensor(pca.components_, dtype=torch.float32)}
    return bases

def last_token_logits(model, outputs, attention_mask, label_ids):
    device = next(model.parameters()).device
    rows = torch.arange(attention_mask.shape[0], device=device)
    cols = torch.arange(attention_mask.shape[1], device=device)
    pos = (attention_mask.to(device) * cols.unsqueeze(0)).max(dim=1).values
    hidden = outputs.last_hidden_state[rows, pos, :]
    ids = torch.tensor(label_ids, dtype=torch.long, device=device)
    logits = hidden @ model.lm_head.weight[ids].to(hidden.dtype).T
    bias = getattr(model.lm_head, 'bias', None)
    if bias is not None:
        logits = logits + bias[ids]
    softcap = getattr(model.config, 'final_logit_softcapping', None)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    return logits.float()

@torch.no_grad()
def collect_site_activations(model, input_ids, attention_mask, position_by_id, sites, batch_size=32, return_logits=False, label_ids=None):
    device = next(model.parameters()).device
    layer_ids, token_ids, keys = ([], [], [])
    states, logits = ({}, [])
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

            def hook(_module, _inputs, output):
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
        return (states, torch.cat(logits, dim=0))
    return states

@torch.no_grad()
def run_intervention(model, bank, sites, site_weights, source_states, bases, strength, label_ids, batch_size=32, return_logits=False):
    device = next(model.parameters()).device
    weights = torch.as_tensor(site_weights, dtype=torch.float32, device=device).flatten()
    if weights.numel() != len(sites):
        raise ValueError('site_weights must have one value per site')
    if not torch.isfinite(weights).all() or float(weights.sum().abs().item()) == 0.0:
        raise ValueError('site_weights must be finite with nonzero sum')
    weights = weights / weights.sum()
    input_ids = bank['base_input_ids']
    attention_mask = bank['base_attention_mask']
    position_by_id = bank['base_position_by_id']
    layer_ids, token_ids, outputs_all = ([], [], [])
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

            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden_new = hidden
                changed = False
                for site_id, (L, token_id, a, b) in enumerate(sites):
                    L, a, b = (int(L), int(a), int(b))
                    if L != layer_id:
                        continue
                    if not changed:
                        hidden_new = hidden.clone()
                        changed = True
                    key = (L, token_id)
                    pos = pos_by_token[token_id]
                    base_act = hidden_new[rows, pos, :].float()
                    source_act = source_states[key][start:end].to(device=device, dtype=torch.float32)
                    scale = float(strength) * weights[site_id]
                    if bases[key]['mode'] == 'neuron':
                        patched = base_act.clone()
                        patched[:, a:b] = base_act[:, a:b] + scale * (source_act[:, a:b] - base_act[:, a:b])
                    else:
                        comps = bases[key]['components'].to(device=device, dtype=torch.float32)[a:b]
                        diff = source_act - base_act
                        patched = base_act + scale * (diff @ comps.T @ comps)
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

def site_signature(model, bank, sites, label_ids, mode='neuron', k=None, batch_size=32, strength=1.0, max_fit_states=4096, signature_method='family_mean', family_order=FAMILY_ORDER):
    base_states, base_logits = collect_site_activations(model, bank['base_input_ids'], bank['base_attention_mask'], bank['base_position_by_id'], sites, batch_size=batch_size, return_logits=True, label_ids=label_ids)
    source_states = collect_site_activations(model, bank['source_input_ids'], bank['source_attention_mask'], bank['source_position_by_id'], sites, batch_size=batch_size)
    bases = fit_bases(sites, base_states, source_states, mode=mode, k=k, max_fit_states=max_fit_states)
    signatures = []
    for site in sites:
        patched_logits = run_intervention(model, bank, [site], [1.0], source_states, bases, strength, label_ids, batch_size=batch_size, return_logits=True)
        diff = patched_logits.float() - base_logits.float()
        signatures.append(make_signature(diff, method=signature_method, pair_source_families=bank['pair_source_families'], family_order=family_order))
    return {'sites': sites, 'intervention_diff': torch.stack(signatures, dim=0), 'bases': bases}

def top_sites_from_T(T, sites, var_id, top_k, min_mass=1e-08):
    valid_indices = []
    for i in range(len(sites)):
        value = T[var_id, i]
        if bool(torch.isfinite(value).item()) and float(value.detach().cpu().item()) > min_mass:
            valid_indices.append(i)
    if not valid_indices:
        raise ValueError(f'no positive-mass sites for var_id={var_id}')
    valid_scores = []
    for i in valid_indices:
        valid_scores.append(T[var_id, i])
    valid_scores = torch.stack(valid_scores)
    k = min(int(top_k), len(valid_indices))
    _, local_indices = torch.topk(valid_scores, k=k)
    selected_sites, selected_indices = ([], [])
    for local_i in local_indices.tolist():
        global_i = valid_indices[int(local_i)]
        selected_sites.append(sites[global_i])
        selected_indices.append(global_i)
    return (selected_sites, selected_indices)

def compute_iia(outputs, labels, var_name, pointer_num_labels=4):
    scores = torch.as_tensor(outputs)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if var_name == 'answer_pointer':
        scores = scores[:, :pointer_num_labels]
    elif var_name != 'answer_token':
        raise ValueError(f'unknown var_name={var_name!r}')
    pred = scores.argmax(dim=-1).cpu()
    labels = labels.cpu()
    correct = int((pred == labels).sum().item())
    total = int(labels.numel())
    return (correct / total, correct)

def eval_intervention(model, bank, var_name, sites, weights, source_states, bases, strength, label_ids, batch_size=32):
    outputs = run_intervention(model, bank, sites, weights, source_states, bases, strength, label_ids, batch_size=batch_size)
    return compute_iia(outputs, bank['counterfactual_label_ids'][var_name], var_name)

def choose_stage_A_layers(model, names, cal_banks, coarse_sites, coarse_bases, T_coarse, label_ids, strength_values, top_k, keep_layers, threshold, batch_size=32):
    top_layers, top_info, all_results = ({}, {}, [])
    for var_id, var_name in enumerate(names):
        cal_bank = cal_banks[var_name]
        top_sites, top_indices = top_sites_from_T(T_coarse, coarse_sites, var_id, top_k, min_mass=0.0)
        candidates = []
        print(f'\n[Stage A variable] {var_id} {var_name}')
        for rank, site in enumerate(top_sites, start=1):
            site_index = top_indices[rank - 1]
            L = int(site[0])
            mass = float(T_coarse[var_id, site_index].detach().cpu())
            source_states = collect_site_activations(model, cal_bank['source_input_ids'], cal_bank['source_attention_mask'], cal_bank['source_position_by_id'], [site], batch_size=batch_size)
            best = None
            for strength in strength_values:
                iia, correct = eval_intervention(model, cal_bank, var_name, [site], [1.0], source_states, coarse_bases, float(strength), label_ids, batch_size=batch_size)
                result = {'var_id': var_id, 'var_name': var_name, 'raw_rank': rank, 'site_index': site_index, 'site': site, 'layer': L, 'coupling_mass': mass, 'strength': float(strength), 'cal_iia': float(iia), 'cal_correct': int(correct)}
                all_results.append(result)
                if best is None or result['cal_correct'] > best['cal_correct']:
                    best = result
                print(f"[Stage A CAL] var={var_name} layer={L} strength={strength} correct={correct}/{len(cal_bank['base_input_ids'])} iia={iia:.4f}")
            candidates.append(best)
        passing = []
        for candidate in candidates:
            if candidate['cal_iia'] >= float(threshold):
                passing.append(candidate)
        if not passing:
            passing = candidates
            print(f'[Stage A fallback] var={var_name}: no layer passed threshold={threshold}')
        passing = sorted(passing, key=lambda x: x['cal_iia'], reverse=True)
        selected = passing[:int(keep_layers)]
        top_layers[var_id] = []
        for candidate in selected:
            top_layers[var_id].append(candidate['layer'])
        top_info[var_id] = selected
        print(f'[Stage A retained] var={var_name} layers={top_layers[var_id]}')
    return (top_layers, top_info, all_results)

def log_all_layers(model, names, cal_banks, coarse_sites, coarse_bases, T_coarse, label_ids, strength_values, path, batch_size=32):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for var_id, var_name in enumerate(names):
            cal_bank = cal_banks[var_name]
            scores = T_coarse[var_id]
            order = torch.argsort(scores, descending=True)
            rank_by_index = {site_index: rank for rank, site_index in enumerate(order.tolist(), start=1)}
            source_states = collect_site_activations(model, cal_bank['source_input_ids'], cal_bank['source_attention_mask'], cal_bank['source_position_by_id'], coarse_sites, batch_size=batch_size)
            header = f'\n===== {var_name} =====\n'
            print(header, end='')
            f.write(header)
            for site_index, site in enumerate(coarse_sites):
                layer = int(site[0])
                mass = float(scores[site_index].detach().cpu())
                rank = rank_by_index[site_index]
                for strength in strength_values:
                    iia, correct = eval_intervention(model, cal_bank, var_name, [site], [1.0], source_states, coarse_bases, float(strength), label_ids, batch_size=batch_size)
                    line = f"layer={layer:2d} strength={float(strength):4.1f} iia={iia:.4f} correct={correct}/{len(cal_bank['base_input_ids'])} uot_rank={rank:2d} mass={mass:.8f}\n"
                    print(line, end='')
                    f.write(line)

def run_stage_a(model, tokenizer, layers, ft_size=200, cal_size=100, te_size=100, dataset_size=None, dataset_split='train', signature_method='concat', mode='neuron', k=None, eps=1.0, method='uot', top_k=6, keep_layers=2, iia_threshold=0.6, strength_values=(0.5, 1, 2, 4), token_position='last_token', device='cuda', seed=0, batch_size=32, max_fit_states=4096, save_path='results/stage_a.pt', log_path=None):
    set_seed(seed)
    label_ids = answer_label_ids(tokenizer)
    solver = get_solver(method)
    ft_bank, cal_banks, _ = build_mcqa_banks(model=model, tokenizer=tokenizer, train_pool_size=ft_size, cal_size=cal_size, te_size=te_size, dataset_size=dataset_size, split=dataset_split, device=device, batch_size=batch_size, seed=seed)
    G, names = variable_signature(ft_bank, num_labels=len(ANSWER_LETTERS), signature_method=signature_method, family_order=FAMILY_ORDER)
    dim = basis_dim(mode, len(ft_bank['base_input_ids']), model.config.hidden_size, k, max_fit_states)
    sites = []
    for layer in layers:
        sites.append((int(layer), token_position, 0, dim))
    sig = site_signature(model, ft_bank, sites, label_ids, mode=mode, k=k, batch_size=batch_size, max_fit_states=max_fit_states, signature_method=signature_method, family_order=FAMILY_ORDER)
    S = sig['intervention_diff']
    T = solver(G, S, eps=eps)
    bases = sig['bases']
    top_layers, top_info, cal_results = choose_stage_A_layers(model, names, cal_banks, sites, bases, T, label_ids, strength_values, top_k, keep_layers, iia_threshold, batch_size=batch_size)
    if log_path is not None:
        log_all_layers(model, names, cal_banks, sites, bases, T, label_ids, strength_values, log_path, batch_size=batch_size)
    results = {'names': names, 'top_layers': top_layers, 'top_info': top_info, 'cal_results': cal_results, 'G_stage_A': G, 'S_stage_A': S, 'T_stage_A': T, 'signature_method': signature_method, 'config': {'ft_size': ft_size, 'cal_size': cal_size, 'te_size': te_size, 'dataset_size': dataset_size, 'dataset_split': dataset_split, 'token_position': token_position, 'seed': seed, 'batch_size': batch_size}}
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(results, save_path)
    print('saved Stage A to:', save_path)
    print('selected layers:', top_layers)
    return results
if __name__ == '__main__':
    model, tokenizer = load_gemma_model()
    device = next(model.parameters()).device
    layers = list(range(model.config.num_hidden_layers))
    run_stage_a(model=model, tokenizer=tokenizer, layers=layers, ft_size=200, cal_size=100, te_size=100, dataset_size=None, dataset_split='train', signature_method='concat', mode='neuron', k=None, eps=1.0, method='uot', top_k=6, keep_layers=2, iia_threshold=0.6, strength_values=(0.5, 1, 2, 4), token_position='last_token', device=device, seed=0, batch_size=32, save_path='results/stage_a.pt', log_path='results/stage_a_all_layers.txt')
