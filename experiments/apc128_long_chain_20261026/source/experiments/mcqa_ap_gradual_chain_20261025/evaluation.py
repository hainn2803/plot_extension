"""Locked final predictions and the complete C-method x AP-evaluator matrix."""
from pathlib import Path
import time
import numpy as np
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, load, status, digest
from . import data, banks, controls, training
from .engine import projection
from .protocol import SEEDS, METHODS, PROTOCOL


def inventory():
    out = {}
    for seed in SEEDS:
        for suffix in ('direct', 'identity', 'restore'):
            out[f'AT_S{seed}_{suffix}'] = 'AT_final'
        for bank in ('AP', 'C'):
            for role in ('AT_donor0', 'AT_donor1'):
                for suffix in ('', '_restore'):
                    out[f'clean_{bank}_S{seed}_{role}{suffix}'] = bank+'_final'
        for method in METHODS:
            ap_name = f'AP_{method}_S{seed}'
            for suffix in ('direct', 'identity', 'restore', 'AT_replay', 'AT_restore', 'conflict0', 'conflict1',
                           'AP_donor0', 'AP_donor1', 'recipient0', 'recipient1', 'recipient2', 'recipient3'):
                out[ap_name+'_'+suffix] = 'AP_final'
            c_name = f'C_{method}_S{seed}'
            for suffix in ('direct', 'identity', 'restore', 'AT_parallel'):
                out[c_name+'_'+suffix] = 'C_final'
            for ap_method in METHODS:
                for suffix in ('AP_replay', 'AT_chain', 'AT_restore', 'conflict0', 'conflict1',
                               'recipient0', 'recipient1', 'recipient2', 'recipient3'):
                    out[c_name+'_via_'+ap_method+'_'+suffix] = 'C_final'
    return out


def originals(rows):
    return [dict(r, target=data.answer(r['base'])) for r in rows]


def write_rows(root, label, rows, indices=None):
    write(Path(root)/'evaluation_rows'/(label+'.json'), dict(rows=rows, parent_indices=indices))


def put(root, label, rows, fn, indices=None):
    write_rows(root, label, rows, indices)
    if not rows:
        value = dict(pair_ids=[], targets=[], target_token_ids=[], label_ids=[], correct=torch.zeros(0, dtype=torch.bool),
                     rows_digest=digest(rows))
        from experiments.mcqa_at_anchor_comparison.storage import save
        path = Path(root)/'evaluation'/(label+'.pt')
        if path.exists():
            return read(path.with_suffix('.json')), load(path)
        save(path, value)
        metric = dict(all=training.rate([]), missing_all=True)
        write(path.with_suffix('.json'), metric)
        return metric, value
    metric, arrays = training.saved(Path(root)/'evaluation'/(label+'.pt'), fn)
    if arrays['pair_ids'] != [r['id'] for r in rows] or arrays['rows_digest'] != digest(rows):
        raise ValueError('Saved evaluation identity mismatch: '+label)
    status(root, 'evaluating', operation=label, count=len(rows))
    return metric, arrays


@torch.no_grad()
def tail_evaluate(engine, cache, rows, induced, q, restore=False):
    out = {}
    for start in range(0, len(rows), engine.batch):
        rr = rows[start:start+engine.batch]
        n = len(rr)
        baseline = cache.field([r['base'] for r in rr], 'AT').float().to(engine.device)
        state = induced[start:start+n].float().to(engine.device)
        if restore:
            changed = projection(state, baseline, q).to(torch.bfloat16).float()
        else:
            changed = projection(baseline, state, q).to(torch.bfloat16).float()
        y = engine.tail(changed)[:n]
        arrays = training.records(engine, y, rr)
        arrays['AT_state'] = changed.cpu().to(torch.bfloat16)
        if restore:
            delta = changed-baseline
            arrays['restoration_l2'] = delta.norm(dim=-1).cpu()
            arrays['restoration_projected_l2'] = (delta@q).norm(dim=-1).cpu()
            arrays['restoration_max_absolute_error'] = delta.abs().max(dim=-1).values.cpu()
        for k, v in arrays.items():
            out.setdefault(k, []).append(v)
    out = {k: torch.cat(v) for k, v in out.items()}
    out.update(pair_ids=[r['id'] for r in rows], targets=[r['target'] for r in rows], rows_digest=digest(rows),
               label_ids=list(engine.label_ids), target_token_ids=[engine.label_ids[engine.letters.index(r['target'])] for r in rows])
    return training.summarize(rows, out), out


@torch.no_grad()
def conflict(engine, cache, rows, parent_states, indices, q):
    induced = parent_states[indices].float().to(engine.device)
    donated = cache.field([r['source'] for r in rows], 'AT').float().to(engine.device)
    changed = projection(induced, donated, q).to(torch.bfloat16).cpu()
    # These are already the post-conflict AT states; do not project them again.
    out = {}
    for start in range(0, len(rows), engine.batch):
        rr = rows[start:start+engine.batch]
        y = engine.tail(changed[start:start+len(rr)])[:len(rr)]
        for k, v in training.records(engine, y, rr).items():
            out.setdefault(k, []).append(v)
    arrays = {k: torch.cat(v) for k, v in out.items()}
    arrays.update(pair_ids=[r['id'] for r in rows], targets=[r['target'] for r in rows], rows_digest=digest(rows),
                  label_ids=list(engine.label_ids), target_token_ids=[engine.label_ids[engine.letters.index(r['target'])] for r in rows])
    return training.summarize(rows, arrays), arrays


def downstream(root, engine, cache, rows, control, trace, q, prefix, replay_suffix):
    rep = put(root, prefix+'_'+replay_suffix, rows,
              lambda: tail_evaluate(engine, cache, rows, trace['AT_state'], q))
    original = originals(rows)
    put(root, prefix+'_AT_restore', original,
        lambda: tail_evaluate(engine, cache, original, trace['AT_state'], q, restore=True))
    for role in ('conflict0', 'conflict1'):
        rr, ix = controls.rows_for(rows, control, role)
        put(root, prefix+'_'+role, rr, lambda rr=rr, ix=ix: conflict(engine, cache, rr, trace['AT_state'], ix, q), ix)
    return rep


def bootstrap(rows, a, b, mask):
    clusters = sorted({r['metadata'].get('cluster_question', r['base']['question']) for r in rows})
    lookup = {c: i for i, c in enumerate(clusters)}
    mask = np.asarray(mask, dtype=bool)
    diff = np.asarray(b, dtype=float)-np.asarray(a, dtype=float)
    sums, counts = np.zeros(len(clusters)), np.zeros(len(clusters))
    for i, row in enumerate(rows):
        if mask[i]:
            j = lookup[row['metadata'].get('cluster_question', row['base']['question'])]
            sums[j] += diff[i]
            counts[j] += 1
    rng = np.random.default_rng(20261025)
    draws = rng.integers(0, len(clusters), (2000, len(clusters)))
    den = counts[draws].sum(1)
    values = sums[draws].sum(1)[den > 0]/den[den > 0]
    return dict(count=int(mask.sum()), difference=float(diff[mask].mean()) if mask.any() else None,
                lower=float(np.quantile(values, .025)) if len(values) else None,
                upper=float(np.quantile(values, .975)) if len(values) else None, resamples=2000,
                valid_resamples=len(values), seed=20261025, cluster_count=len(clusters))


def results_from_saved(root):
    root = Path(root)
    result = dict(seeds={}, comparisons={}, controls_complete=True, native_switching_proven=False,
                  support_overlap=dict(C_AP=False, AP_AT=False, C_AT=False, distinct_layers=[0, 14, 25]))
    for bank in ('AP', 'C'):
        result['controls_complete'] &= read(root/'controls'/bank/'completion.json')['complete']
    for seed in SEEDS:
        arrays, rr = {}, {}
        seed_result = {}
        comparisons = {}
        for variable in ('AP', 'C'):
            rows = banks.load_bank(root, variable+'_final')
            rr[variable] = rows
            mask = np.array([r['metadata']['kind'] == 'changing' for r in rows])
            for method in METHODS:
                prefix = f'{variable}_{method}_S{seed}'
                arr = load(root/'evaluation'/(prefix+'_direct.pt'))
                arrays[variable, method] = arr
                entry = dict(direct=read(root/'evaluation'/(prefix+'_direct.json')))
                if variable == 'AP':
                    rep = load(root/'evaluation'/(prefix+'_AT_replay.pt'))
                    joint = arr['correct'] & rep['correct']
                    arrays['AP_joint', method] = joint
                    entry['direct_and_AT_replay'] = dict(changing=training.rate(joint[mask]), preservation=training.rate(joint[~mask]))
                else:
                    entry['via_AP'] = {}
                    for ap_method in METHODS:
                        p = prefix+'_via_'+ap_method
                        ap_rep = load(root/'evaluation'/(p+'_AP_replay.pt'))
                        at_rep = load(root/'evaluation'/(p+'_AT_chain.pt'))
                        joint = arr['correct'] & ap_rep['correct'] & at_rep['correct']
                        arrays['C_joint', method, ap_method] = joint
                        entry['via_AP'][ap_method] = dict(changing=training.rate(joint[mask]), preservation=training.rate(joint[~mask]),
                                                         AP_replay=training.rate(ap_rep['correct']), AT_chain=training.rate(at_rep['correct']))
                seed_result[variable+'_'+method] = entry
            for group, select in (('changing', mask), ('preservation', ~mask)):
                comparisons[variable+'_direct_'+group] = bootstrap(rows, arrays[variable, 'standard']['correct'], arrays[variable, 'gradual']['correct'], select)
        ap_mask = np.array([r['metadata']['kind'] == 'changing' for r in rr['AP']])
        c_mask = np.array([r['metadata']['kind'] == 'changing' for r in rr['C']])
        comparisons['AP_primary'] = bootstrap(rr['AP'], arrays['AP_joint', 'standard'], arrays['AP_joint', 'gradual'], ap_mask)
        both_correct = arrays['C', 'standard']['correct'] & arrays['C', 'gradual']['correct']
        for ap_method in METHODS:
            comparisons['C_via_'+ap_method] = bootstrap(rr['C'], arrays['C_joint', 'standard', ap_method], arrays['C_joint', 'gradual', ap_method], c_mask)
            comparisons['C_via_'+ap_method+'_both_direct_correct'] = bootstrap(rr['C'], arrays['C_joint', 'standard', ap_method],
                                                                            arrays['C_joint', 'gradual', ap_method], c_mask & both_correct.numpy())
        comparisons['C_primary'] = comparisons['C_via_standard']
        result['seeds'][str(seed)] = seed_result
        result['comparisons'][str(seed)] = comparisons
    result['strong_practical_superiority'] = result['controls_complete'] and all(
        result['comparisons'][str(seed)][primary]['difference'] >= .05 and result['comparisons'][str(seed)][primary]['lower'] > 0
        for seed in SEEDS for primary in ('AP_primary', 'C_primary')) and all(
        result['comparisons'][str(seed)][v+'_direct_'+g]['difference'] >= -.02
        for seed in SEEDS for v in ('AP', 'C') for g in ('changing', 'preservation'))
    return result


def evaluate_all(root, engine, cache):
    root = Path(root)
    banks.locked(root)
    write(root/'final_access.json', dict(time=time.time(), after_all_endpoints_and_readouts=True))
    rows = {v: banks.filter_bank(root, v+'_final', cache) for v in ('AT', 'AP', 'C')}
    control = {v: controls.filter_controls(root, rows[v], cache, v) for v in ('AP', 'C')}
    for seed in SEEDS:
        teacher = training.teachers(root, engine, seed)
        aq = teacher['AT_q']
        at_rows, at_states = rows['AT'], cache.states(rows['AT'])
        prefix = f'AT_S{seed}'
        put(root, prefix+'_direct', at_rows, lambda: training.evaluate(engine, at_rows, at_states, aq, 'AT', keep=True))
        original = originals(at_rows)
        base = cache.field([r['base'] for r in at_rows], 'AT').float()
        identity = put(root, prefix+'_identity', original, lambda: training.evaluate(engine, original, at_states, aq, 'AT', override=base))
        if not identity[1]['correct'].all():
            raise ValueError('AT identity changed a clean answer')
        put(root, prefix+'_restore', original, lambda: training.evaluate(engine, original, at_states, aq, 'AT', restore=True))
        for variable in ('AP', 'C'):
            rr = rows[variable]
            for role in ('AT_donor0', 'AT_donor1'):
                cr, ix = controls.rows_for(rr, control[variable], role)
                ss = cache.states(cr) if cr else None
                label = f'clean_{variable}_S{seed}_{role}'
                put(root, label, cr, lambda cr=cr, ss=ss: training.evaluate(engine, cr, ss, aq, 'AT'), ix)
                orig = originals(cr)
                put(root, label+'_restore', orig, lambda orig=orig, ss=ss: training.evaluate(engine, orig, ss, aq, 'AT', restore=True), ix)
        for method in METHODS:
            pq = training.basis(root, 'AP', method, seed, engine.device)
            rr, ss = rows['AP'], cache.states(rows['AP'])
            prefix = f'AP_{method}_S{seed}'
            _, trace = put(root, prefix+'_direct', rr, lambda: training.evaluate(engine, rr, ss, pq, 'AP', keep=True))
            original = originals(rr)
            own = cache.field([r['base'] for r in rr], 'AP').float()
            identity = put(root, prefix+'_identity', original, lambda: training.evaluate(engine, original, ss, pq, 'AP', override=own))
            if not identity[1]['correct'].all():
                raise ValueError('AP identity changed a clean answer')
            put(root, prefix+'_restore', original, lambda: training.evaluate(engine, original, ss, pq, 'AP', restore=True))
            downstream(root, engine, cache, rr, control['AP'], trace, aq, prefix, 'AT_replay')
            for role in ('AP_donor0', 'AP_donor1', 'recipient0', 'recipient1', 'recipient2', 'recipient3'):
                cr, ix = controls.rows_for(rr, control['AP'], role)
                cs = cache.states(cr) if cr else None
                donated = trace['AP_state'][ix].float() if role.startswith('recipient') else None
                put(root, prefix+'_'+role, cr, lambda cr=cr, cs=cs, donated=donated: training.evaluate(engine, cr, cs, pq, 'AP', override=donated), ix)
        for method in METHODS:
            cq = training.basis(root, 'C', method, seed, engine.device)
            rr, ss = rows['C'], cache.states(rows['C'])
            prefix = f'C_{method}_S{seed}'
            _, trace = put(root, prefix+'_direct', rr, lambda: training.evaluate(engine, rr, ss, cq, 'C', keep=True))
            original = originals(rr)
            own = cache.field([r['base'] for r in rr], 'C').float()
            identity = put(root, prefix+'_identity', original, lambda: training.evaluate(engine, original, ss, cq, 'C', override=own))
            if not identity[1]['correct'].all():
                raise ValueError('C identity changed a clean answer')
            put(root, prefix+'_restore', original, lambda: training.evaluate(engine, original, ss, cq, 'C', restore=True))
            put(root, prefix+'_AT_parallel', rr, lambda: tail_evaluate(engine, cache, rr, trace['AT_state'], aq))
            for ap_method in METHODS:
                pq = training.basis(root, 'AP', ap_method, seed, engine.device)
                label = prefix+'_via_'+ap_method
                _, rep = put(root, label+'_AP_replay', rr,
                             lambda: training.evaluate(engine, rr, ss, pq, 'AP', keep=True, override=trace['AP_state'].float()))
                downstream(root, engine, cache, rr, control['C'], rep, aq, label, 'AT_chain')
                for role in ('recipient0', 'recipient1', 'recipient2', 'recipient3'):
                    cr, ix = controls.rows_for(rr, control['C'], role)
                    cs = cache.states(cr) if cr else None
                    put(root, label+'_'+role, cr, lambda cr=cr, cs=cs, ix=ix: training.evaluate(engine, cr, cs, pq, 'AP', override=trace['AP_state'][ix].float()), ix)
    actual = {p.stem for p in (root/'evaluation').glob('*.pt')}
    if actual != set(inventory()):
        raise ValueError('Registered evaluation inventory mismatch')
    result = results_from_saved(root)
    write(root/'results.json', result)
    return result
