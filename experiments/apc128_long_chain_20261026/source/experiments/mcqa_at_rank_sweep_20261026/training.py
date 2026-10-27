"""Same fixed-update Standard AT fit; only rank and readout width vary."""
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F

from experiments.mcqa_at_anchor_comparison.engine import Subspace
from experiments.mcqa_at_anchor_comparison.training import checkpoint, recover, restore_rng
from experiments.mcqa_at_anchor_comparison.storage import read, write, save, load, file_hash, digest, status
from experiments.mcqa_c_at_it_das.training import tensor_digest
from experiments.mcqa_ap_gradual_chain_20261025 import training as original, banks, data
from .protocol import PARENT, ROOT, STEPS, label


def locations(rank, seed, root=ROOT):
    if (rank, seed) == (128, 1):
        return PARENT/'fits/AT_S1', PARENT/'readouts/AT_S1.pt'
    return Path(root)/'fits'/label(rank, seed), Path(root)/'readouts'/(label(rank, seed)+'.pt')


def fit(root, engine, cache, rank, seed):
    folder, _ = locations(rank, seed, root)
    if folder.is_relative_to(PARENT):
        raise ValueError('Never modify an imported parent fit')
    rows, cal = (banks.load_bank(root, 'AT_'+s) for s in ('train', 'calibration'))
    identity = dict(run=file_hash(root/'identity.json'), execution=file_hash(root/'execution.json'),
                    variable='AT', method='standard', rank=rank, seed=seed, maximum=1000,
                    train=digest(rows), calibration=digest(cal), teachers=tensor_digest(None))
    if (folder/'completion.json').exists():
        done = read(folder/'completion.json')
        if done['identity'] != identity or file_hash(done['checkpoint']) != done['sha256']:
            raise ValueError('Completed rank fit changed')
        return done
    states, cstates = cache.states(rows), cache.states(cal)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    sub = Subspace(seed, rank=rank).to(engine.device)
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
            raise ValueError('Original rank initialization changed')
        for i, update in enumerate(state['updates']):
            if update['indices'] != original.order(rows, 'AT', seed, i):
                raise ValueError('Saved paired order changed')
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
                    record[split], _ = original.saved(folder/'evaluations'/f'{step:06d}_{split}.pt',
                        lambda rr=rr, ss=ss: original.evaluate(engine, rr, ss, sub.basis(), 'AT'))
                state['evaluations'][str(step)] = record
                checkpoint(folder, state, sub, opt)
            if step == 1000:
                break
            indices = original.order(rows, 'AT', seed, step)
            safe = False
            tick = time.perf_counter()
            value = engine.update('AT', 'standard', [rows[i] for i in indices],
                                  {k: v[indices] for k, v in states.items()}, sub, opt)
            state['step'] += 1
            state['updates'].append(dict(step=state['step'], indices=indices, seconds=time.perf_counter()-tick, **value))
            safe = True
            if state['step'] % 25 == 0:
                checkpoint(folder, state, sub, opt)
                status(root, 'training', rank=rank, seed=seed, step=state['step'], **value)
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


def rate(values):
    values = torch.as_tensor(values, dtype=torch.bool)
    correct, count = int(values.sum()), len(values)
    return dict(correct=correct, count=count, accuracy=correct/count if count else None)


def qualifies(value, count=None):
    n, k = value['count'], value['correct']
    if not isinstance(n, int) or not isinstance(k, int) or not 0 <= k <= n or n == 0:
        raise ValueError('Invalid qualification counts')
    return (count is None or n == count) and 100*k >= 90*n


def gate(cal, ro):
    checks = dict(changing_each_letter=all(qualifies(v, 16) for v in cal['changing_by_letter'].values()),
                  preservation=qualifies(cal['preservation'], 104), readout_overall=qualifies(ro['accuracy'], 520),
                  readout_each_letter=all(qualifies(v, 20) for v in ro['by_class'].values()))
    if set(cal['changing_by_letter']) != set(data.LETTERS) or set(ro['by_class']) != set(data.LETTERS):
        raise ValueError('Incomplete per-letter qualification coverage')
    return dict(checks=checks, passed=all(checks.values()))


def choose(cells, ranks=(128, 256, 512), seeds=(1, 0, 2)):
    if set(cells) != {label(r, s) for r in ranks for s in seeds}:
        raise ValueError('All registered cells must precede selection')
    return next((rank for rank in ranks if all(cells[label(rank, s)]['gate']['passed'] for s in seeds)), None)


def readout(root, engine, cache, rank, seed):
    folder, path = locations(rank, seed, root)
    if path.exists():
        return read(path.with_suffix('.json'))
    if folder.is_relative_to(PARENT):
        raise ValueError('Missing imported readout; never refit it')
    done = read(folder/'completion.json')
    if file_hash(done['checkpoint']) != done['sha256']:
        raise ValueError('Readout teacher checkpoint changed')
    q = load(done['checkpoint'])['payload']['basis'].to(engine.device)
    if q.shape != (2304, rank):
        raise ValueError('Rank/readout width mismatch')
    rows, cal = (banks.load_bank(root, 'readout_AT_'+s) for s in ('train', 'calibration'))
    x, xv = (cache.field([r['base'] for r in rr], 'AT').float().to(engine.device)@q for rr in (rows, cal))
    y, yv = (torch.tensor([data.LETTERS.index(data.answer(r['base'])) for r in rr], device=engine.device) for rr in (rows, cal))
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    xn, xvn = (x-mean)/std, (xv-mean)/std
    torch.manual_seed(seed)
    best_count, best, history = -1, None, []
    for penalty in (.0001, .001, .01):
        head = torch.nn.Linear(rank, 26).to(engine.device)
        opt = torch.optim.Adam(head.parameters(), lr=.01)
        for epoch in range(1, 201):
            engine.session.check()
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(xn), y)+penalty*head.weight.square().sum()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite factual readout loss')
            loss.backward()
            opt.step()
            with torch.no_grad():
                z = head(xvn)
                count = int((z.argmax(-1) == yv).sum())
                history.append(dict(penalty=penalty, epoch=epoch, correct=count, count=len(cal), accuracy=count/len(cal)))
                if count > best_count:
                    weight = head.weight.detach()/std[None, :]
                    best = dict(weight=weight.cpu().clone(), bias=(head.bias.detach()-weight@mean).cpu().clone(),
                                classes=list(data.LETTERS), penalty=penalty, epoch=epoch)
                    best_count = count
    z = xv@best['weight'].to(engine.device).T+best['bias'].to(engine.device)
    top = z.topk(2, dim=-1)
    correct = (z.argmax(-1) == yv) & (top.values[:, 0] > top.values[:, 1])
    save(path, best)
    evidence = dict(accuracy=rate(correct.cpu()), by_class={letter: rate(correct[yv == i].cpu()) for i, letter in enumerate(data.LETTERS)},
                    history=history, train=digest(rows), calibration=digest(cal), q_digest=tensor_digest(q), rank=rank)
    write(path.with_suffix('.json'), evidence)
    return evidence
