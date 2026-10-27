"""Fixed-update fits, paired schedules, frozen readouts, and endpoint locks."""
from collections import defaultdict
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F

from experiments.mcqa_at_anchor_comparison.training import checkpoint, recover, restore_rng
from experiments.mcqa_at_anchor_comparison.storage import read, write, save, load, file_hash, digest, status
from experiments.mcqa_c_at_it_das.training import tensor_digest
from experiments.mcqa_ap_localization_sweep.banks import score
from .engine import Subspace, projection, AP_SITE
from .protocol import SEEDS, METHODS, STEPS, LETTERS, fit_names
from . import data, banks


def name(variable, method, seed):
    return f'AT_S{seed}' if variable == 'AT' else f'{variable}_{method}_S{seed}'


def order(rows, variable, seed, step):
    if variable != 'AT':
        cells = defaultdict(list)
        for i, row in enumerate(rows):
            cells[row['metadata']['from_AP'], row['metadata']['to_AP']].append(i)
        indices = []
        for cell, values in sorted(cells.items()):
            epoch, offset = divmod(step, len(values)//2)
            g = torch.Generator().manual_seed(int(data.key('schedule', variable, seed, cell, epoch)[:15], 16))
            indices.extend(values[j] for j in torch.randperm(len(values), generator=g)[2*offset:2*offset+2].tolist())
    else:
        pools = {kind: defaultdict(list) for kind in ('changing', 'preservation')}
        for i, row in enumerate(rows):
            pools[row['metadata']['kind']][row['target']].append(i)
        indices = []
        for kind, count in (('changing', 24), ('preservation', 8)):
            for offset in range(count):
                occurrence = step*count+offset
                letter = LETTERS[occurrence % 26]
                values = pools[kind][letter]
                cycle, index = divmod(occurrence//26, len(values))
                g = torch.Generator().manual_seed(int(data.key('schedule', variable, seed, kind, letter, cycle)[:15], 16))
                indices.append(values[int(torch.randperm(len(values), generator=g)[index])])
    if len(indices) != 32 or len(set(indices)) != 32:
        raise ValueError('Invalid paired batch')
    if sum(rows[i]['metadata']['kind'] == 'changing' for i in indices) != 24:
        raise ValueError('Changing/preservation quota changed')
    return indices


def rate(values):
    values = torch.as_tensor(values, dtype=torch.bool)
    return dict(count=len(values), correct=int(values.sum()), accuracy=float(values.float().mean()) if len(values) else None)


def summarize(rows, arrays):
    correct = arrays['correct'].bool()
    change = torch.tensor([r['metadata']['kind'] == 'changing' for r in rows])
    out = dict(all=rate(correct), changing=rate(correct[change]), preservation=rate(correct[~change]),
               CE=float(-arrays['target_log_probability'].double().mean()))
    for field, fn in (
        ('transition', lambda r: f"{r['metadata']['from_AP']+1}->{r['metadata']['to_AP']+1}"),
        ('parent_transition', lambda r: r['metadata'].get('parent_transition', f"{r['metadata']['from_AP']+1}->{r['metadata']['to_AP']+1}")),
        ('target_letter', lambda r: r['target']), ('family', lambda r: r['metadata']['permutation']),
        ('preservation_position', lambda r: str(r['metadata']['from_AP']+1) if r['metadata']['kind'] == 'preservation' else 'changing')):
        groups = defaultdict(list)
        for i, row in enumerate(rows):
            groups[fn(row)].append(i)
        out[field] = {k: rate(correct[ix]) for k, ix in groups.items()}
    out['changing_by_letter'] = {letter: rate(correct[change & torch.tensor([r['target'] == letter for r in rows])]) for letter in LETTERS}
    return out


def records(engine, y, rows, trace=None, keep=False):
    targets = [r['target'] for r in rows]
    arrays = score(engine, y, targets)
    ids = torch.tensor([engine.label_ids[LETTERS.index(t)] for t in targets], device=engine.device)
    logp = y.float() - y.float().logsumexp(-1, keepdim=True)
    arrays['target_log_probability'] = logp.gather(1, ids[:, None])[:, 0].cpu()
    arrays['label_log_probabilities'] = logp[:, list(engine.label_ids)].cpu()
    if trace is not None and keep:
        for key, value in (('AP_state', trace['site']), ('AT_state', trace['at_after_override']), ('C_state', trace['c'])):
            arrays[key] = value[:len(rows)].cpu().to(torch.bfloat16)
    return arrays


@torch.no_grad()
def evaluate(engine, rows, states, q, variable, keep=False, override=None, restore=False):
    out = {}
    for start in range(0, len(rows), engine.batch):
        rr = rows[start:start+engine.batch]
        n = len(rr)
        local = {k: v[start:start+n] for k, v in states.items()}
        trace = None
        if variable == 'AT':
            b, s = engine.pad(local['base_AT']).float(), engine.pad(local['source_AT']).float()
            if override is not None:
                s = engine.pad(override[start:start+n]).float()
            induced = projection(b, s, q).to(torch.bfloat16).float()
            if restore:
                induced = projection(induced, b, q).to(torch.bfloat16).float()
            y = engine.tail(induced)[:n]
        else:
            args = {}
            if variable == 'AP':
                source = local['source_AP'] if override is None else override[start:start+n]
                args.update(donor=dict(values=source, available=torch.ones(n, dtype=torch.bool)), q=q, restore=restore)
            else:
                source = local['source_C'] if override is None else override[start:start+n]
                args.update(c_donor=source, c_q=q, c_restore=restore)
            y, trace = engine.forward([r['base'] for r in rr], **args)
            y = y[:n]
        arrays = records(engine, y, rr, trace, keep)
        if variable == 'AT' and keep:
            arrays['AT_state'] = induced[:n].cpu().to(torch.bfloat16)
        if restore:
            actual = induced[:n] if variable == 'AT' else trace['site'][:n] if variable == 'AP' else trace['c'][:n]
            original = b[:n] if variable == 'AT' else trace['original_site'][:n] if variable == 'AP' else trace['original_c'][:n]
            delta = actual.float()-original.float()
            arrays['restoration_l2'] = delta.norm(dim=-1).cpu()
            arrays['restoration_projected_l2'] = (delta @ q).norm(dim=-1).cpu()
            arrays['restoration_max_absolute_error'] = delta.abs().max(dim=-1).values.cpu()
        for key, value in arrays.items():
            out.setdefault(key, []).append(value)
    out = {k: torch.cat(v) for k, v in out.items()}
    out.update(pair_ids=[r['id'] for r in rows], targets=[r['target'] for r in rows],
               target_token_ids=[engine.label_ids[LETTERS.index(r['target'])] for r in rows],
               label_ids=list(engine.label_ids), rows_digest=digest(rows))
    return summarize(rows, out), out


def saved(path, fn):
    path = Path(path)
    if path.exists():
        value = load(path)
        return read(path.with_suffix('.json')), value
    metrics, arrays = fn()
    save(path, arrays)
    write(path.with_suffix('.json'), metrics)
    return metrics, arrays


def fit(root, engine, cache, variable, method, seed, teachers=None):
    root = Path(root)
    label = name(variable, method, seed)
    folder = root / 'fits' / label
    rows, cal = banks.load_bank(root, variable+'_train'), banks.load_bank(root, variable+'_calibration')
    identity = dict(run=file_hash(root/'identity.json'), execution=file_hash(root/'execution.json'), variable=variable,
                    method=method, seed=seed, maximum=1000, train=digest(rows), calibration=digest(cal), teachers=tensor_digest(teachers))
    if (folder/'completion.json').exists():
        done = read(folder/'completion.json')
        if done['identity'] != identity or file_hash(done['checkpoint']) != done['sha256']:
            raise ValueError('Completed fit changed')
        return done
    states, cstates = cache.states(rows), cache.states(cal)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sub = Subspace(seed, width=9216 if variable == 'AP' else 2304).to(engine.device)
    opt = torch.optim.Adam(sub.parameters(), lr=.01)
    initial = tensor_digest(sub.state_dict())
    old = recover(folder, identity)
    if old:
        payload, _ = old
        sub.load_state_dict(payload['subspace'])
        opt.load_state_dict(payload['optimizer'])
        restore_rng(payload)
        state = {k: payload[k] for k in ('identity', 'step', 'updates', 'initialization', 'evaluations')}
        if state['initialization'] != initial:
            raise ValueError('Original initialization changed')
        for i, update in enumerate(state['updates']):
            if update['indices'] != order(rows, variable, seed, i):
                raise ValueError('Saved order changed')
    else:
        state = dict(identity=identity, step=0, updates=[], initialization=initial, evaluations={})
        checkpoint(folder, state, sub, opt)
    safe = True
    try:
        while True:
            step = state['step']
            if step in STEPS and str(step) not in state['evaluations']:
                record = {}
                for split, rr, ss in (('train', rows, states), ('calibration', cal, cstates)):
                    record[split], _ = saved(folder/'evaluations'/f'{step:06d}_{split}.pt',
                                             lambda rr=rr, ss=ss: evaluate(engine, rr, ss, sub.basis(), variable))
                state['evaluations'][str(step)] = record
                checkpoint(folder, state, sub, opt)
            if step == 1000:
                break
            ix = order(rows, variable, seed, step)
            safe = False
            tick = time.perf_counter()
            value = engine.update(variable, method, [rows[i] for i in ix], {k: v[ix] for k, v in states.items()}, sub, opt, teachers)
            state['step'] += 1
            state['updates'].append(dict(step=state['step'], indices=ix, seconds=time.perf_counter()-tick, **value))
            safe = True
            if state['step'] % 25 == 0:
                checkpoint(folder, state, sub, opt)
                status(root, 'training', configuration=label, step=state['step'], **value)
        path = checkpoint(folder, state, sub, opt)
    except BaseException:
        if safe and len(state['updates']) == state['step'] and torch.isfinite(sub.raw).all():
            checkpoint(folder, state, sub, opt)
        raise
    preservation = [[rows[i]['id'] for i in u['indices'] if rows[i]['metadata']['kind'] == 'preservation'] for u in state['updates']]
    done = dict(identity=identity, checkpoint=str(path), sha256=file_hash(path), steps=1000, initialization=initial,
                order_digest=digest([u['indices'] for u in state['updates']]), preservation_digest=digest(preservation),
                endpoint=state['evaluations']['1000'])
    write(folder/'completion.json', done)
    return done


def basis(root, variable, method, seed, device):
    done = read(Path(root)/'fits'/name(variable, method, seed)/'completion.json')
    if file_hash(done['checkpoint']) != done['sha256']:
        raise ValueError('Fit checkpoint changed')
    return load(done['checkpoint'])['payload']['basis'].to(device).detach()


def readout(root, engine, cache, variable, method, seed):
    root = Path(root)
    label = name(variable, method, seed)
    path = root/'readouts'/(label+'.pt')
    q = basis(root, variable, method, seed, engine.device)
    if path.exists():
        return {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in load(path).items()}
    rows, cal = (banks.load_bank(root, f'readout_{variable}_{split}') for split in ('train', 'calibration'))
    x, xv = (cache.field([r['base'] for r in rr], variable).float().to(engine.device) @ q for rr in (rows, cal))
    classes = list(LETTERS) if variable == 'AT' else list(range(4))
    y, yv = (torch.tensor([classes.index(data.answer(r['base']) if variable == 'AT' else data.ap(r['base'])) for r in rr], device=engine.device) for rr in (rows, cal))
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    xn, xvn = (x-mean)/std, (xv-mean)/std
    torch.manual_seed(seed)
    best_accuracy, best, history = -1., None, []
    for penalty in (.0001, .001, .01):
        head = torch.nn.Linear(128, len(classes)).to(engine.device)
        opt = torch.optim.Adam(head.parameters(), lr=.01)
        for epoch in range(1, 201):
            engine.session.check()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(xn), y) + penalty*head.weight.square().sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite readout loss')
            loss.backward()
            opt.step()
            with torch.no_grad():
                z = head(xvn)
                accuracy = float((z.argmax(-1) == yv).float().mean())
                history.append(dict(penalty=penalty, epoch=epoch, accuracy=accuracy))
                if accuracy > best_accuracy:
                    w = head.weight.detach()/std[None, :]
                    best = dict(weight=w.cpu().clone(), bias=(head.bias.detach()-w@mean).cpu().clone(),
                                classes=classes, penalty=penalty, epoch=epoch)
                    best_accuracy = accuracy
    z = xv@best['weight'].to(engine.device).T+best['bias'].to(engine.device)
    top = z.topk(2, dim=-1)
    correct = (z.argmax(-1) == yv) & (top.values[:, 0] > top.values[:, 1])
    save(path, best)
    write(path.with_suffix('.json'), dict(accuracy=rate(correct.cpu()), by_class={str(c): rate(correct[yv == i].cpu()) for i, c in enumerate(classes)},
                                         history=history, train=digest(rows), calibration=digest(cal), q_digest=tensor_digest(q)))
    return {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in best.items()}


def teachers(root, engine, seed, ap_method=None):
    out = dict(AT_q=basis(root, 'AT', 'standard', seed, engine.device))
    out['AT_head'] = {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in load(Path(root)/'readouts'/f'AT_S{seed}.pt').items()}
    if ap_method is not None:
        out['AP_q'] = basis(root, 'AP', ap_method, seed, engine.device)
        out['AP_head'] = {k: v.to(engine.device) if isinstance(v, torch.Tensor) else v for k, v in load(Path(root)/'readouts'/f'AP_{ap_method}_S{seed}.pt').items()}
    return out


def gate_at(root, seeds=SEEDS):
    result = {}
    for seed in seeds:
        done = read(Path(root)/'fits'/f'AT_S{seed}'/'completion.json')
        cal = done['endpoint']['calibration']
        ro = read(Path(root)/'readouts'/f'AT_S{seed}.json')
        checks = dict(changing_each_letter=all(v['count'] == 16 and v['accuracy'] >= .90 for v in cal['changing_by_letter'].values()),
                      preservation=cal['preservation']['accuracy'] >= .95,
                      readout_overall=ro['accuracy']['accuracy'] >= .95,
                      readout_each_letter=all(v['count'] == 20 and v['accuracy'] >= .90 for v in ro['by_class'].values()))
        result[str(seed)] = dict(checks=checks, calibration=cal, readout=ro['accuracy'], passed=all(checks.values()))
    passed = all(v['passed'] for v in result.values())
    filename = 'AT_gate.json' if tuple(seeds) == SEEDS else f'AT_gate_S{seeds[0]}.json'
    write(Path(root)/filename, dict(passed=passed, seeds=result, no_automatic_replacement=True))
    if not passed:
        raise RuntimeError('Fresh AT teacher calibration gate failed; no AP/C training permitted')


def lock(root):
    root = Path(root)
    endpoints, readouts = {}, {}
    for label in fit_names():
        done = read(root/'fits'/label/'completion.json')
        if done['steps'] != 1000 or file_hash(done['checkpoint']) != done['sha256']:
            raise ValueError('Incomplete fixed endpoint')
        endpoints[label] = dict(path=done['checkpoint'], sha256=done['sha256'])
    for variable in ('AP', 'C'):
        for seed in SEEDS:
            a, b = (read(root/'fits'/f'{variable}_{m}_S{seed}'/'completion.json') for m in METHODS)
            for field in ('initialization', 'order_digest', 'preservation_digest'):
                if a[field] != b[field]:
                    raise ValueError('Paired initialization/order mismatch: ' + field)
    for label in [f'AT_S{s}' for s in SEEDS] + [f'AP_{m}_S{s}' for s in SEEDS for m in METHODS]:
        path = root/'readouts'/(label+'.pt')
        load(path)
        readouts[label] = dict(path=str(path), sha256=file_hash(path))
    write(root/'locks/all_final.json', dict(endpoints=endpoints, readouts=readouts, time=time.time(), paired=True,
                                           final_opened_before_lock=False))
