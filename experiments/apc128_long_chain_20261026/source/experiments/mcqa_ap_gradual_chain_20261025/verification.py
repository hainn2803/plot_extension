"""CPU-only audits of saved predictions, paired schedules and bootstraps."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import math
import time
import numpy as np
import psutil
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash, digest
from experiments.mcqa_c_at_it_das.training import tensor_digest
from . import data, banks, training
from .protocol import SEEDS, METHODS, LETTERS, BANK_SIZES, fit_names


def independent_bootstrap(rows, a, b, mask):
    groups = sorted(set(r['metadata'].get('cluster_question', r['base']['question']) for r in rows))
    deltas, weights = [], []
    mask = np.asarray(mask, dtype=bool)
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    for group in groups:
        ix = [i for i, r in enumerate(rows) if r['metadata'].get('cluster_question', r['base']['question']) == group and mask[i]]
        deltas.append(sum(int(b[i])-int(a[i]) for i in ix))
        weights.append(len(ix))
    draws = np.random.default_rng(20261025).integers(0, len(groups), size=(2000, len(groups)))
    values = []
    for draw in draws:
        denominator = sum(weights[int(i)] for i in draw)
        if denominator:
            values.append(sum(deltas[int(i)] for i in draw)/denominator)
    return dict(count=int(mask.sum()), difference=float((b.astype(float)-a.astype(float))[mask].mean()) if mask.any() else None,
                lower=float(np.quantile(values, .025)) if values else None,
                upper=float(np.quantile(values, .975)) if values else None,
                resamples=2000, valid_resamples=len(values), seed=20261025, cluster_count=len(groups))


def compare(a, b, label):
    if a.keys() != b.keys():
        raise ValueError('Audit key mismatch: '+label)
    for key in a:
        if isinstance(a[key], (int, float)) and not isinstance(a[key], bool) and b[key] is not None:
            if not math.isclose(a[key], b[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Independent metric mismatch: '+label+'/'+key)
        elif a[key] != b[key]:
            raise ValueError('Independent metric mismatch: '+label+'/'+key)


def primitive_rate(values):
    values = np.asarray(values, dtype=bool)
    return dict(count=len(values), correct=int(values.sum()), accuracy=float(values.mean()) if len(values) else None)


def audit_metrics(rows, arrays, metrics):
    correct = arrays['correct'].numpy().astype(bool)
    change = np.array([r['metadata']['kind'] == 'changing' for r in rows])
    for label, mask in (('all', np.ones(len(rows), dtype=bool)), ('changing', change), ('preservation', ~change)):
        compare(primitive_rate(correct[mask]), metrics[label], label)
    ce = float(-arrays['target_log_probability'].double().mean())
    if not math.isclose(ce, metrics['CE'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('Saved full-vocabulary CE mismatch')
    selectors = dict(transition=lambda r: f"{r['metadata']['from_AP']+1}->{r['metadata']['to_AP']+1}",
                     parent_transition=lambda r: r['metadata'].get('parent_transition', f"{r['metadata']['from_AP']+1}->{r['metadata']['to_AP']+1}"),
                     target_letter=lambda r: r['target'], family=lambda r: r['metadata']['permutation'],
                     preservation_position=lambda r: str(r['metadata']['from_AP']+1) if r['metadata']['kind'] == 'preservation' else 'changing')
    for field, fn in selectors.items():
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[fn(row)].append(i)
        if set(groups) != set(metrics[field]):
            raise ValueError('Metric strata mismatch')
        for cell, ix in groups.items():
            compare(primitive_rate(correct[ix]), metrics[field][cell], field+'/'+cell)
    for letter in LETTERS:
        mask = change & np.array([r['target'] == letter for r in rows])
        compare(primitive_rate(correct[mask]), metrics['changing_by_letter'][letter], 'changing letter '+letter)


def final(root):
    from transformers import AutoTokenizer
    from .evaluation import inventory
    root = Path(root)
    identity = data.verify(root)
    lock = banks.locked(root)
    torch.set_num_threads(4)
    ledger = read(root/'engineering/gpu_budget.json')
    if ledger['active'] or ledger['seconds'] >= 43200 or ledger['total_limit_seconds'] != 43200:
        raise ValueError('Closed authorized ledger required')
    rows_by_bank = {bank: banks.load_bank(root, bank) for bank in BANK_SIZES}
    for bank, rows in rows_by_bank.items():
        data.audit(rows, bank)
    nonfinal = {r[side]['id'] for bank, rows in rows_by_bank.items() if not bank.endswith('_final')
                for r in rows for side in ('base', 'source')}
    excluded = set(read(root/'historical_exclusions.json'))
    for bank in ('AP_final', 'C_final', 'AT_final'):
        for row in rows_by_bank[bank]:
            if any(row[side]['id'] in excluded or row[side]['id'] in nonfinal for side in ('base', 'source')):
                raise ValueError('Final prompt leakage')
    access = read(root/'final_access.json')
    if access['time'] <= lock['time']:
        raise ValueError('Final opened before lock')
    fits = {}
    for label in fit_names():
        done = read(root/'fits'/label/'completion.json')
        payload = load(done['checkpoint'])
        state = payload['payload']
        if tensor_digest(state) != payload['payload_digest'] or state['step'] != 1000 or len(state['updates']) != 1000:
            raise ValueError('Internal endpoint digest/update mismatch')
        variable = done['identity']['variable']
        seed = done['identity']['seed']
        rows = rows_by_bank[variable+'_train']
        for i, update in enumerate(state['updates']):
            if update['indices'] != training.order(rows, variable, seed, i):
                raise ValueError('Independent saved schedule mismatch')
        initial = training.Subspace(seed, width=9216 if variable == 'AP' else 2304)
        if tensor_digest(initial.state_dict()) != done['initialization']:
            raise ValueError('Initial projector mismatch under original four CPU threads')
        if not state['optimizer']['state'] or any(int(v['step']) != 1000 for v in state['optimizer']['state'].values()):
            raise ValueError('Optimizer endpoint mismatch')
        if not state['python_rng'] or not state['numpy_rng'] or not len(state['cpu_rng']) or not state['cuda_rng']:
            raise ValueError('Missing optimizer/RNG preservation')
        q = state['basis']
        if not torch.allclose(q.T@q, torch.eye(128), atol=2e-4, rtol=2e-4):
            raise ValueError('Endpoint basis lost orthogonality')
        fits[label] = done
    for variable in ('AP', 'C'):
        for seed in SEEDS:
            a, b = (fits[f'{variable}_{m}_S{seed}'] for m in METHODS)
            for field in ('initialization', 'order_digest', 'preservation_digest'):
                if a[field] != b[field]:
                    raise ValueError('Paired endpoint mismatch: '+field)
    tokenizer = AutoTokenizer.from_pretrained(identity['model_root'], local_files_only=True)
    decoded = {}
    counts = 0
    actual = {p.stem for p in (root/'evaluation').glob('*.pt')}
    if actual != set(inventory()):
        raise ValueError('Missing/extra registered final artifacts')
    audit_outputs = {}
    for label in inventory():
        path = root/'evaluation'/(label+'.pt')
        arrays = load(path)
        rows = read(root/'evaluation_rows'/(label+'.json'))['rows']
        if arrays['rows_digest'] != digest(rows) or arrays['pair_ids'] != [r['id'] for r in rows] or arrays['targets'] != [r['target'] for r in rows]:
            raise ValueError('Saved row/target alignment failed: '+label)
        if rows:
            for token in arrays['token_id'].tolist():
                if token not in decoded:
                    decoded[token] = tokenizer.decode([token]).strip()
            correct = torch.tensor([decoded[int(token)] == row['target'] and not bool(tied)
                                    for row, token, tied in zip(rows, arrays['token_id'], arrays['tied'])])
            if not torch.equal(correct, arrays['correct']):
                raise ValueError('Independent saved-token scoring failed: '+label)
            if not torch.equal(arrays['top_two_token_ids'][:, 0], arrays['token_id']):
                raise ValueError('Top-two ID mismatch')
            if not torch.equal(arrays['top_two_logits'][:, 0] == arrays['top_two_logits'][:, 1], arrays['tied']):
                raise ValueError('Tie evidence mismatch')
            label_logp = arrays['label_log_probabilities']
            if not torch.isfinite(label_logp).all() or not torch.allclose(label_logp.exp(), arrays['label_probabilities'], atol=2e-4, rtol=2e-4):
                raise ValueError('Saved full-vocabulary probability evidence mismatch')
            target_index = torch.tensor([LETTERS.index(r['target']) for r in rows])
            if not torch.equal(label_logp.gather(1, target_index[:, None])[:, 0], arrays['target_log_probability']):
                raise ValueError('Saved target log-probability mismatch')
            audit_metrics(rows, arrays, read(path.with_suffix('.json')))
        counts += len(rows)
        audit_outputs[label] = dict(count=len(rows), sha256=file_hash(path), rows_sha256=file_hash(root/'evaluation_rows'/(label+'.json')))
    result = read(root/'results.json')
    bootstrap_checks = 0
    for seed in SEEDS:
        values, masks = {}, {}
        for variable in ('AP', 'C'):
            rows = rows_by_bank[variable+'_final']
            masks[variable] = np.array([r['metadata']['kind'] == 'changing' for r in rows])
            for method in METHODS:
                prefix = f'{variable}_{method}_S{seed}'
                direct = load(root/'evaluation'/(prefix+'_direct.pt'))['correct'].numpy()
                values[variable, method] = direct
                if variable == 'AP':
                    replay = load(root/'evaluation'/(prefix+'_AT_replay.pt'))['correct'].numpy()
                    values['AP_joint', method] = direct & replay
                else:
                    for ap_method in METHODS:
                        p = prefix+'_via_'+ap_method
                        ap_rep = load(root/'evaluation'/(p+'_AP_replay.pt'))['correct'].numpy()
                        at_rep = load(root/'evaluation'/(p+'_AT_chain.pt'))['correct'].numpy()
                        values['C_joint', method, ap_method] = direct & ap_rep & at_rep
            for group, mask in (('changing', masks[variable]), ('preservation', ~masks[variable])):
                key = variable+'_direct_'+group
                value = independent_bootstrap(rows, values[variable, 'standard'], values[variable, 'gradual'], mask)
                compare(value, result['comparisons'][str(seed)][key], key)
                bootstrap_checks += 1
        expected = independent_bootstrap(rows_by_bank['AP_final'], values['AP_joint', 'standard'], values['AP_joint', 'gradual'], masks['AP'])
        compare(expected, result['comparisons'][str(seed)]['AP_primary'], 'AP primary')
        bootstrap_checks += 1
        for ap_method in METHODS:
            for conditional in (False, True):
                mask = masks['C'].copy()
                if conditional:
                    mask &= values['C', 'standard'] & values['C', 'gradual']
                expected = independent_bootstrap(rows_by_bank['C_final'], values['C_joint', 'standard', ap_method], values['C_joint', 'gradual', ap_method], mask)
                key = 'C_via_'+ap_method+('_both_direct_correct' if conditional else '')
                compare(expected, result['comparisons'][str(seed)][key], key)
                bootstrap_checks += 1
                if ap_method == 'standard' and not conditional:
                    compare(expected, result['comparisons'][str(seed)]['C_primary'], 'C primary')
                    bootstrap_checks += 1
    readouts = readout_diagnostics(root, rows_by_bank)
    receipt = dict(passed=True, time=time.time(), completed_fits=15, verified_update_orders=15000, evaluation_artifacts=len(audit_outputs),
                   prediction_rows=counts, bootstrap_checks=bootstrap_checks, resamples_per_check=2000,
                   independently_decoded_saved_tokens=True, output_inventory=audit_outputs, readouts=readouts,
                   closed_ledger=ledger, final_opened_after_all_locks=True, controls_complete=result['controls_complete'])
    write(root/'independent_verification.json', receipt)
    return receipt


def readout_diagnostics(root, rows_by_bank):
    out = {}
    for seed in SEEDS:
        atq = training.basis(root, 'AT', 'standard', seed, 'cpu')
        athead = load(root/'readouts'/f'AT_S{seed}.pt')
        for variable in ('AP', 'C'):
            rows = rows_by_bank[variable+'_final']
            target = torch.tensor([LETTERS.index(r['target']) for r in rows])
            for method in METHODS:
                label = f'{variable}_{method}_S{seed}'
                arr = load(root/'evaluation'/(label+'_direct.pt'))
                z = (arr['AT_state'].float()@atq)@athead['weight'].T+athead['bias']
                top = z.topk(2, dim=-1)
                correct = (z.argmax(-1) == target) & (top.values[:, 0] > top.values[:, 1])
                out[label] = dict(AT_readout=primitive_rate(correct.numpy()), AP_readouts={})
                for ap_method in METHODS:
                    pq = training.basis(root, 'AP', ap_method, seed, 'cpu')
                    head = load(root/'readouts'/f'AP_{ap_method}_S{seed}.pt')
                    z = (arr['AP_state'].float()@pq)@head['weight'].T+head['bias']
                    positions = torch.tensor([r['metadata']['to_AP'] for r in rows])
                    top = z.topk(2, dim=-1)
                    correct = (z.argmax(-1) == positions) & (top.values[:, 0] > top.values[:, 1])
                    out[label]['AP_readouts'][ap_method] = primitive_rate(correct.numpy())
    return out


def exited(root):
    root = Path(root)
    value = read(root/'gpu_completion.json')
    original = value['process']
    if read(root/'engineering/gpu_budget.json')['active']:
        raise ValueError('GPU ledger is still active')
    live = None
    try:
        proc = psutil.Process(original['pid'])
        live = dict(created=proc.create_time(), cmdline=proc.cmdline(), executable=proc.exe())
        if abs(live['created']-original['created']) < .01:
            raise RuntimeError('Original worker has not exited')
    except psutil.NoSuchProcess:
        pass
    if not read(root/'independent_verification.json')['passed']:
        raise ValueError('Independent audit is not complete')
    write(root/'completion.json', dict(completed=True, worker_exit_verified=True, original_process=original,
                                       current_pid_owner=live, ledger=read(root/'engineering/gpu_budget.json'), time=time.time()))
