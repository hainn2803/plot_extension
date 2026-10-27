"""CPU-only continuation regression, including a partly saved evaluation."""
from pathlib import Path
from unittest.mock import patch
import uuid


def cpu(output):
    import random
    import time
    import numpy as np
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load, save, file_hash, digest
    from experiments.mcqa_at_anchor_comparison.training import checkpoint
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_ap_gradual_chain_20261025 import training
    from experiments.mcqa_ap_gradual_chain_20261025.execution import Session
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    root = Path('.tmp')/('apcr_'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'identity.json', dict(synthetic_only=True))
    write(root/'execution.json', dict(synthetic_only=True))
    rows, cal = (training.banks.load_bank(ROOT, 'AP_'+s) for s in ('train', 'calibration'))
    class Subspace(torch.nn.Module):
        def __init__(self, seed, width=2304):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.eye(4, 2))
        def basis(self):
            return torch.linalg.qr(self.raw, mode='reduced').Q
    class Cache:
        def states(self, rr):
            return dict(source_AP=torch.zeros(len(rr), 4))
    class Engine:
        device = torch.device('cpu')
        def __init__(self):
            self.updates = 0
        def update(self, variable, method, rr, states, sub, opt, teachers):
            assert variable == 'AP' and method == 'gradual' and teachers == {'synthetic': True}
            opt.zero_grad(set_to_none=True)
            sub.raw.square().sum().backward()
            opt.step()
            self.updates += 1
            return dict(losses=dict(direct=0., AT_replay=0., AT_readout=0.), gradient_norm=0.)
    teacher = dict(synthetic=True)
    identity = dict(run=file_hash(root/'identity.json'), execution=file_hash(root/'execution.json'),
                    variable='AP', method='gradual', seed=2, maximum=1000,
                    train=digest(rows), calibration=digest(cal), teachers=tensor_digest(teacher))
    torch.manual_seed(2)
    np.random.seed(2)
    random.seed(2)
    sub = Subspace(2, width=9216)
    initial = tensor_digest(sub.state_dict())
    opt = torch.optim.Adam(sub.parameters(), lr=.01)
    sub.raw.square().sum().backward()
    opt.step()
    opt.state[sub.raw]['step'].fill_(100)
    updates = [dict(step=i+1, indices=training.order(rows, 'AP', 2, i), seconds=0.,
                    losses=dict(direct=0., AT_replay=0., AT_readout=0.), gradient_norm=0.) for i in range(100)]
    state = dict(identity=identity, step=100, updates=updates, initialization=initial,
                 evaluations={'0': dict(train=dict(synthetic=True), calibration=dict(synthetic=True))})
    folder = root/'fits/AP_gradual_S2'
    old_checkpoint = checkpoint(folder, state, sub, opt)
    partial_eval = folder/'evaluations/000100_train.pt'
    save(partial_eval, dict(synthetic=True))
    write(partial_eval.with_suffix('.json'), dict(synthetic=True))
    before = file_hash(partial_eval)
    prefix = tensor_digest(updates)
    engine = Engine()
    evaluated = []
    def evaluate(_engine, rr, ss, q, variable):
        evaluated.append(len(rr))
        return dict(synthetic=True), dict(synthetic=True)
    with patch.object(training, 'Subspace', Subspace), \
         patch.object(training.banks, 'load_bank', side_effect=lambda r,b: rows if b.endswith('train') else cal), \
         patch.object(training, 'evaluate', side_effect=evaluate), patch.object(training, 'status'):
        done = training.fit(root, engine, Cache(), 'AP', 'gradual', 2, teacher)
    payload = load(done['checkpoint'])['payload']
    assert engine.updates == 900 and payload['step'] == len(payload['updates']) == 1000
    assert tensor_digest(payload['updates'][:100]) == prefix
    assert file_hash(partial_eval) == before and len(evaluated) == 9
    assert all(int(v['step']) == 1000 for v in payload['optimizer']['state'].values())
    assert load(old_checkpoint)['payload']['step'] == 100
    session = object.__new__(Session)
    session.previous, session.profile_previous = 12988.62537730002, 853.1099996000121
    session.started = session.stage_started = time.perf_counter()
    session.stage, session.active = 'resume_validation', True
    ledger = session.ledger()
    assert ledger['seconds'] >= session.previous and ledger['profile_seconds'] == session.profile_previous
    assert ledger['total_limit_seconds'] == 43200 and not torch.cuda.is_initialized()
    result = dict(passed=True, synthetic_only=True, fixture=str(root), resumed_step=100,
                  new_updates=900, original_100_update_prefix_exact=True, optimizer_restored=True,
                  completed_partial_evaluation_unchanged=True, only_missing_evaluations_executed=True,
                  cumulative_budget_and_profile_preserved=True, cuda_initialized=False)
    write(output/'CPU_regressions.json', result)
    return result
