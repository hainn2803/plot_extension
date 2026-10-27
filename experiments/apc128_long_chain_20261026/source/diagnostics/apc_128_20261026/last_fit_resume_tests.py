"""CPU-only regressions for the exact C step-525 continuation and stop scope."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import uuid


def cpu(output, validate_stop, process, ledger, terminal, pause):
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load, save, file_hash, digest
    from experiments.mcqa_at_anchor_comparison.training import checkpoint
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_ap_gradual_chain_20261025 import training
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    from .ac_resume_tests import cpu as prior_regression
    prior_regression(output/'prior_C50_regression',validate_stop,process,ledger,terminal,pause)
    rejected = 0
    for part, key, value in ((3,'reason','thermal stop'), (3,'time',pause['time']+1),
                             (1,'active',True), (1,'seconds',0.),
                             (1,'total_limit_seconds',86400), (2,'pid',0)):
        data = deepcopy([process, ledger, terminal, pause])
        data[part][key] = value
        try:
            validate_stop(*data)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('A different stop or budget was accepted')
    root = Path('.tmp')/('apclast_'+uuid.uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'identity.json', dict(synthetic_only=True))
    write(root/'execution.json', dict(synthetic_only=True))
    rows, cal = (training.banks.load_bank(ROOT, 'C_'+s) for s in ('train','calibration'))
    class Subspace(torch.nn.Module):
        def __init__(self, seed, width=2304):
            super().__init__()
            self.raw = torch.nn.Parameter(torch.eye(4,2))
        def basis(self):
            return torch.linalg.qr(self.raw, mode='reduced').Q
    class Cache:
        def states(self, rr):
            return dict(source_C=torch.zeros(len(rr),4))
    class Engine:
        device = torch.device('cpu')
        def __init__(self):
            self.updates = 0
        def update(self, variable, method, rr, states, sub, opt, teachers):
            assert variable=='C' and method=='gradual' and teachers=={'synthetic':True}
            opt.zero_grad(set_to_none=True)
            sub.raw.square().sum().backward()
            opt.step()
            self.updates += 1
            return dict(losses=dict(direct=0.,AT_replay=0.,AT_readout=0.,AP_replay=0.,AP_readout=0.),gradient_norm=0.)
    teacher = dict(synthetic=True)
    identity = dict(run=file_hash(root/'identity.json'),execution=file_hash(root/'execution.json'),
                    variable='C',method='gradual',seed=2,maximum=1000,
                    train=digest(rows),calibration=digest(cal),teachers=tensor_digest(teacher))
    sub = Subspace(2)
    initial = tensor_digest(sub.state_dict())
    opt = torch.optim.Adam(sub.parameters(),lr=.01)
    sub.raw.square().sum().backward()
    opt.step()
    opt.state[sub.raw]['step'].fill_(525)
    updates = [dict(step=i+1,indices=training.order(rows,'C',2,i),seconds=0.) for i in range(525)]
    state = dict(identity=identity,step=525,updates=updates,initialization=initial,
                 evaluations={str(s):dict(train=dict(synthetic=True),calibration=dict(synthetic=True)) for s in (0,100,250,500)})
    folder = root/'fits/C_gradual_S2'
    old = checkpoint(folder,state,sub,opt)
    preserved = folder/'evaluations/000000_train.pt'
    save(preserved,dict(synthetic=True))
    write(preserved.with_suffix('.json'),dict(synthetic=True))
    before, prefix = file_hash(preserved), tensor_digest(updates)
    engine, evaluated = Engine(), []
    def evaluate(*args, **kwargs):
        evaluated.append(True)
        return dict(synthetic=True),dict(synthetic=True)
    with patch.object(training,'Subspace',Subspace), \
         patch.object(training.banks,'load_bank',side_effect=lambda r,b: rows if b.endswith('train') else cal), \
         patch.object(training,'evaluate',side_effect=evaluate), patch.object(training,'status'):
        done = training.fit(root,engine,Cache(),'C','gradual',2,teacher)
    payload = load(done['checkpoint'])['payload']
    assert engine.updates==475 and payload['step']==len(payload['updates'])==1000
    assert tensor_digest(payload['updates'][:525])==prefix
    assert all(int(v['step'])==1000 for v in payload['optimizer']['state'].values())
    assert len(evaluated)==4 and file_hash(preserved)==before and load(old)['payload']['step']==525
    assert rejected==6 and not torch.cuda.is_initialized()
    result = dict(passed=True,synthetic_only=True,fixture=str(root),resumed_step=525,new_updates=475,
                  original_525_update_prefix_exact=True,optimizer_restored=True,old_evaluation_preserved=True,
                  expected_remaining_evaluations=4,wrong_stops_or_budgets_rejected=rejected,
                  prior_partial_AP_evaluation_regression_passed=True,prior_C50_continuation_regression_passed=True,cuda_initialized=False)
    write(output/'CPU_regressions.json',result)
    return result
