"""Actual full-reference loss/gradient checks and complete-work timing gates."""
from pathlib import Path
import gc
import statistics
import time
import numpy as np
import torch

from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from experiments.mcqa_at_anchor_comparison.storage import read, write, status
from diagnostics.ap_localization_sweep_20261024.scoped_counter import verify_live_counter
from . import data, banks, training
from .engine import Subspace, checked, total_loss
from .protocol import SEEDS, METHODS, BANK_SIZES

MODES = [('AT', 'standard'), ('AP', 'standard'), ('AP', 'gradual'), ('C', 'standard'), ('C', 'gradual')]


def live_states(engine, rows):
    out = {key: [] for key in ('base_AT', 'source_AT', 'source_AP', 'source_C')}
    for start in range(0, len(rows), engine.batch):
        rr = rows[start:start+engine.batch]
        with torch.no_grad():
            _, base = engine.forward([r['base'] for r in rr])
            _, source = engine.forward([r['source'] for r in rr])
        for key, value in (('base_AT', base['at']), ('source_AT', source['at']),
                           ('source_AP', source['site']), ('source_C', source['c'])):
            out[key].append(value[:len(rr)].cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def scratch_teachers(engine):
    def head(classes, seed):
        g = torch.Generator().manual_seed(seed)
        return dict(weight=(torch.randn(classes, 128, generator=g)*.01).to(engine.device),
                    bias=torch.zeros(classes, device=engine.device))
    return dict(AT_q=Subspace(37).to(engine.device).basis().detach(), AT_head=head(26, 37),
                AP_q=Subspace(41, width=9216).to(engine.device).basis().detach(), AP_head=head(4, 41))


def sample(root, variable):
    if variable == 'AP':
        rows = banks.load_bank(root, 'AP_train')
    else:
        groups = read(Path(root)/'groups.json')
        rows = [r for slot in data.slots(variable+'_train') for r in data.proposal(slot, 0, groups)]
    return [rows[i] for i in training.order(rows, variable, 1, 0)]


def numerical(engine, rows, states, teachers):
    checks = []
    n = min(engine.batch, 32)
    with torch.no_grad():
        cs = [r['base'] for r in rows['AP'][:n]]
        y, h = engine.forward(cs)
        yr, hr = engine.forward(cs, full=True, reference=True)
        checks.extend([checked('clean_full', y, yr), checked('clean_native', h['native'], hr['native']),
                       checked('clean_repeat', engine.forward(cs)[0], y), checked('cached_AT_tail', engine.tail(h['at']), y)])
        identity, _ = engine.forward(cs, donor=dict(values=h['site'], available=h['available']), q=teachers['AP_q'])
        checks.append(checked('AP_identity', identity, y))
        identity, _ = engine.forward(cs, c_donor=h['c'], c_q=teachers['AT_q'])
        checks.append(checked('C_identity', identity, y))
        identity, _ = engine.forward(cs, at_donor=h['at'], at_q=teachers['AT_q'])
        checks.append(checked('AT_identity', identity, y))
    for variable, method in MODES:
        rr = rows[variable][:n]
        ss = {k: v[:n] for k, v in states[variable].items()}
        sub = Subspace(1, width=9216 if variable == 'AP' else 2304).to(engine.device)
        terms, y = engine.losses(variable, method, rr, ss, sub, teachers)
        loss = total_loss(terms, method)
        grad = torch.autograd.grad(loss, sub.raw)[0]
        reference, yr = engine.losses(variable, method, rr, ss, sub, teachers, reference=True)
        ref_loss = total_loss(reference, method)
        ref_grad = torch.autograd.grad(ref_loss, sub.raw)[0]
        prefix = variable+'_'+method
        checks.extend([checked(prefix+'_logits', y, yr), checked(prefix+'_total_loss', loss, ref_loss),
                       checked(prefix+'_combined_gradient', grad, ref_grad)])
        checks.extend(checked(prefix+'_'+k, terms[k], reference[k]) for k in terms)
        del sub, terms, reference, loss, grad, ref_loss, ref_grad, y, yr
    if n > 1:
        rr = rows['AP'][:n-1]
        ss = {k: v[:n-1] for k, v in states['AP'].items()}
        sub = Subspace(1, width=9216).to(engine.device)
        terms, y = engine.losses('AP', 'gradual', rr, ss, sub, teachers)
        reference, yr = engine.losses('AP', 'gradual', rr, ss, sub, teachers, reference=True)
        checks.extend([checked('filler_excluded_logits', y, yr),
                       checked('filler_excluded_total_CE', total_loss(terms, 'gradual'), total_loss(reference, 'gradual'))])
    if any(p.requires_grad or p.grad is not None for p in engine.model.parameters()):
        raise ValueError('Backbone gradients present')
    if any(v.requires_grad for k, v in teachers.items() if isinstance(v, torch.Tensor)):
        raise ValueError('Teacher basis not frozen')
    return checks


def profile(root, engine, cache):
    root = Path(root)
    rows = {variable: sample(root, variable) for variable in ('AT', 'AP', 'C')}
    teachers = scratch_teachers(engine)
    if (root/'profile.json').exists():
        engine.batch = 32
        states = {v: cache.states(rr) for v, rr in rows.items()}
        with MemoryWatch() as watch:
            checks = numerical(engine, rows, states, teachers)
            saved_checks = validate_saved(root, engine, cache)
        memory = watch.result()
        require_memory(memory)
        write(root/'engineering/resume_validations'/f'{time.time_ns()}.json',
              dict(passed=True, checks=checks, saved_projectors=saved_checks, memory=memory, live_counter=verify_live_counter()))
        return read(root/'profile.json')
    trials = []
    for batch in (1, 2, 4, 8, 16, 32):
        engine.batch = batch
        gc.collect()
        torch.cuda.empty_cache()
        with MemoryWatch() as watch:
            states = {v: live_states(engine, rr) for v, rr in rows.items()}
            checks = numerical(engine, rows, states, teachers)
            timing = {}
            for variable, method in MODES:
                sub = Subspace(1, width=9216 if variable == 'AP' else 2304).to(engine.device)
                opt = torch.optim.Adam(sub.parameters(), lr=.01)
                durations = []
                for _ in range(3):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    engine.update(variable, method, rows[variable], states[variable], sub, opt, teachers)
                    torch.cuda.synchronize()
                    durations.append(time.perf_counter()-tick)
                timing[variable+'_'+method] = dict(seconds=statistics.median(durations), samples=durations)
                del sub, opt
            tick = time.perf_counter()
            with torch.no_grad():
                for start in range(0, 32, batch):
                    engine.forward([r['base'] for r in rows['AP'][start:start+batch]])
            torch.cuda.synchronize()
            natural = (time.perf_counter()-tick)/32
        memory = watch.result()
        require_memory(memory)
        trials.append(dict(batch=batch, passed=True, checks=checks, memory=memory, timing=timing, natural_seconds=natural))
        write(root/'profile_progress.json', trials, immutable=False)
        status(root, 'profiling', batch=batch, passed=True)
    engine.batch = 32
    chosen = trials[-1]
    sub = Subspace(1).to(engine.device)
    opt = torch.optim.Adam(sub.parameters(), lr=.01)
    sustained = []
    tick = time.perf_counter()
    with MemoryWatch() as watch:
        while time.perf_counter()-tick < 180:
            torch.cuda.synchronize()
            start = time.perf_counter()
            engine.update('C', 'gradual', rows['C'], states['C'], sub, opt, teachers)
            torch.cuda.synchronize()
            sustained.append(time.perf_counter()-start)
    memory = watch.result()
    require_memory(memory)
    factor = max(1., float(np.quantile(sustained, .9))/chosen['timing']['C_gradual']['seconds'])
    result = dict(batch=32, trials=trials, timing=chosen['timing'], natural_seconds=chosen['natural_seconds'],
                  sustained=sustained, sustained_factor=factor, memory=memory, live_counter=verify_live_counter(), passed=True)
    write(root/'execution.json', dict(batch=32, width=37, backbone='BF16 SDPA', projection='FP32 QR', tf32=False,
                                      full_vocabulary=True, ties_fail=True, atol=2e-4, rtol=2e-4))
    write(root/'profile.json', result)
    return result


def validate_saved(root, engine, cache):
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_at_anchor_comparison.storage import load
    checks = []
    for path in sorted((Path(root)/'fits').glob('*/latest.json')):
        latest = read(path)
        wrapper = load(latest['path'])
        payload = wrapper['payload']
        if tensor_digest(payload) != wrapper['payload_digest'] or len(payload['updates']) != payload['step']:
            raise ValueError('Saved projector internal digest failed')
        config = payload['identity']
        variable, method, seed = config['variable'], config['method'], config['seed']
        bank = banks.load_bank(root, variable+'_train')
        for i, update in enumerate(payload['updates']):
            if update['indices'] != training.order(bank, variable, seed, i):
                raise ValueError('Resume paired schedule mismatch')
        sub = Subspace(seed, width=9216 if variable == 'AP' else 2304).to(engine.device)
        sub.load_state_dict(payload['subspace'])
        local = [bank[i] for i in training.order(bank, variable, seed, 0)]
        states = cache.states(local)
        live = live_states(engine, local)
        record = [checked('saved_basis', sub.basis(), payload['basis'].to(engine.device))]
        record.extend(checked('saved_natural_'+k, states[k], live[k]) for k in states)
        teacher = training.teachers(root, engine, seed, 'gradual' if variable == 'C' else None) if method == 'gradual' else None
        terms, y = engine.losses(variable, method, local, states, sub, teacher)
        loss = total_loss(terms, method)
        gradient = torch.autograd.grad(loss, sub.raw)[0]
        reference, yr = engine.losses(variable, method, local, live, sub, teacher, reference=True)
        ref_loss = total_loss(reference, method)
        ref_gradient = torch.autograd.grad(ref_loss, sub.raw)[0]
        record.extend([checked('saved_logits', y, yr), checked('saved_CE', loss, ref_loss), checked('saved_gradient', gradient, ref_gradient)])
        checks.append(dict(fit=path.parent.name, step=payload['step'], checks=record))
    for variable in ('AT', 'AP', 'C'):
        for split in ('train', 'calibration'):
            bank = variable+'_'+split
            if (Path(root)/'banks'/bank/'completion.json').exists():
                cache.layout(banks.load_bank(root, bank), 'resume_'+bank)
    return checks


def require_memory(value):
    if not value['no_added_shared_memory'] or not value['all_process_device_usage_within_capacity']:
        raise RuntimeError('Original memory audit failed')


def estimate(root, p, label):
    root = Path(root)
    natural = p['natural_seconds']
    factor = p['sustained_factor']
    fits = 0.
    components = {}
    for variable, method in MODES:
        for seed in SEEDS:
            folder = root/'fits'/training.name(variable, method, seed)
            if (folder/'completion.json').exists():
                continue
            step = read(folder/'latest.json')['step'] if (folder/'latest.json').exists() else 0
            fits += (1000-step)*p['timing'][variable+'_'+method]['seconds']*factor
            eval_remaining = sum(v >= step for v in (0, 100, 250, 500, 750, 1000))
            fits += eval_remaining*(BANK_SIZES[variable+'_train']+BANK_SIZES[variable+'_calibration'])*natural
    attempts = []
    for page in (root/'banks').glob('*/page_*.json'):
        value = read(page)
        retained = [d for d in value['dispositions'] if d.get('retained')]
        attempts.extend(d['rank']+1 for d in retained)
    # Same conservative natural-filter forecasting rule as the prior sweep.
    attempts_per_slot = max(5., 2*sum(attempts)/max(1, len(attempts)))
    filtering = sum(len(data.slots(bank))*2*attempts_per_slot*natural for bank in BANK_SIZES
                    if not (root/'banks'/bank/'completion.json').exists())
    filtering += sum(1792*10*attempts_per_slot*natural for bank in ('AP', 'C')
                     if not (root/'controls'/bank/'completion.json').exists())
    from .evaluation import inventory
    final_eval = sum(BANK_SIZES[bank] for bank in inventory().values())*natural
    if (root/'results.json').exists():
        final_eval = 0.
    readouts = sum(not (root/'readouts'/(training.name(v, m, s)+'.pt')).exists()
                   for v, methods in (('AT', ('standard',)), ('AP', METHODS)) for m in methods for s in SEEDS)*90
    components.update(fits_and_learning_curves=fits, natural_filtering=filtering, final_evaluation=final_eval,
                      readouts=readouts, checkpoint_cache_IO=1200.)
    result = dict(label=label, components=components, total_seconds=sum(components.values()),
                  filter_attempts=attempts_per_slot, allowance=1.25, full_coverage=True)
    write(root/'engineering/estimates'/f'{time.time_ns()}.json', result)
    return result
