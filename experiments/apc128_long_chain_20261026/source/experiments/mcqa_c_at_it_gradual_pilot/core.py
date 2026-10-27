"""Differentiable same-base replay and exact completed-update checkpointing."""
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mcqa_c_at_localization import backend, runtime
from experiments.mcqa_c_at_localization.storage import read, write, file_hash, digest, save_torch, checked_torch
from experiments.mcqa_c_at_localization.training import paired_order
from experiments.mcqa_c_at_it_das.training import tensor_digest
from .protocol import score


def loss_terms(model, ids, source, subspace, position, at_basis, head, targets, class_targets,
               gradual, forward=backend.forward):
    q = subspace.basis()
    direct, hidden = forward(model, ids, [backend.Patch(19, position, source, q)], capture=[24] if gradual else [])
    terms = {'direct': F.cross_entropy(direct, targets)}
    if gradual:
        # Never detach this state: both auxiliary paths must differentiate into C.
        induced = hidden[24][:, 81]
        replay, _ = forward(model, ids, [backend.Patch(24, 81, induced, at_basis)])
        terms['replay'] = F.cross_entropy(replay, targets)
        coordinates = induced.float() @ at_basis
        scores = coordinates @ head['weight'].T + head['bias']
        terms['readout'] = F.cross_entropy(scores, class_targets)
    return terms, direct


def update(model, subspace, opt, ids, sources, targets, classes, indices, position, at_basis,
           head, method, batch, session, forward=backend.forward):
    assert method in ('standard', 'gradual') and len(indices) == 32
    opt.zero_grad(set_to_none=True)
    values, predictions = {'direct': 0., 'replay': 0., 'readout': 0.}, []
    device = subspace.raw.device
    for start in range(0, len(indices), batch):
        session.check()
        chosen = indices[start:start+batch]
        terms, logits = loss_terms(model, ids[chosen].to(device), sources[chosen].to(device), subspace,
                                    position, at_basis, head, targets[chosen].to(device), classes[chosen].to(device),
                                    method == 'gradual', forward)
        total = terms['direct'] + (2*terms['replay'] + 2*terms['readout'] if method == 'gradual' else 0)
        if not torch.isfinite(total):
            raise RuntimeError('Nonfinite objective; no optimizer update committed')
        (total * len(chosen) / len(indices)).backward()
        predictions.extend(logits.detach().argmax(-1).cpu().tolist())
        for name, value in terms.items():
            values[name] += float(value.detach()) * len(chosen) / len(indices)
        del total, logits, terms
    grad = subspace.raw.grad
    if grad is None or not torch.isfinite(grad).all():
        raise RuntimeError('Missing or nonfinite C-basis gradient')
    norm = float(grad.norm())
    opt.step()
    if not torch.isfinite(subspace.raw).all():
        raise RuntimeError('Nonfinite Adam state; retain previous committed checkpoint')
    return {'indices': indices.tolist(), 'predictions': predictions, 'losses': values, 'gradient_norm': norm}


def save(folder, state, subspace, opt):
    payload = {**state, 'subspace': subspace.state_dict(), 'basis': subspace.basis().detach().cpu(),
               'optimizer': opt.state_dict(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
    sha = tensor_digest(payload)
    path = folder / 'checkpoints' / f"step_{state['step']:06d}_{time.time_ns()}.pt"
    save_torch(path, {'state': payload, 'payload_digest': sha})
    receipt = {'identity': state['identity'], 'step': state['step'], 'sha256': file_hash(path), 'payload_digest': sha,
               'path': path.relative_to(folder).as_posix()}
    write(path.with_suffix('.json'), receipt, immutable=True)
    write(folder / 'latest.json', receipt)
    return receipt


def recover(folder, identity):
    receipts = []
    for path in sorted((folder / 'checkpoints').glob('*.json')):
        row = read(path)
        assert row['identity'] == identity, 'Resume identity mismatch'
        assert file_hash(folder / row['path']) == row['sha256'], 'Checkpoint corruption'
        receipts.append((row['step'], path.name, row))
    if not receipts:
        return None
    row = max(receipts, key=lambda r:r[:2])[2]
    data = checked_torch(folder / row['path'], row['sha256'])
    state = data['state']
    assert tensor_digest(state) == data['payload_digest'] == row['payload_digest']
    assert state['identity'] == identity and state['step'] == row['step']
    expected = list(range(1, state['step']+1))
    assert [r['step'] for r in state['updates']] == expected
    return state, row


def batch_order(size, seed, step):
    epoch, offset = divmod(step, size//32)
    return paired_order(size, seed, epoch+1)[offset*32:(offset+1)*32]


def evaluate(model, data, q, position, at_basis, teacher, measurement, batch, session):
    rows, ids, source, target, classes = data
    predictions, replay, taught, measured, ce = [], [], [], [], []
    with torch.no_grad():
        for start in range(0,len(rows),batch):
            session.check()
            sl = slice(start,start+batch)
            logits, hidden = backend.forward(model,ids[sl].to('cuda'),
                [backend.Patch(19,position,source[position][sl],q)],capture=[24])
            induced = hidden[24][:,81].detach().clone()
            predictions.extend(logits.argmax(-1).cpu().tolist())
            ce.extend(F.cross_entropy(logits,target[sl].to('cuda'),reduction='none').cpu().tolist())
            z = induced.float() @ at_basis
            taught.extend((z@teacher['weight'].T+teacher['bias']).argmax(-1).cpu().tolist())
            measured.extend((z@measurement['weight'].T+measurement['bias']).argmax(-1).cpu().tolist())
            del logits,hidden,z
            logits,_ = backend.forward(model,ids[sl].to('cuda'),[backend.Patch(24,81,induced,at_basis)])
            replay.extend(logits.argmax(-1).cpu().tolist())
            del logits,induced
    correct = np.array(predictions) == target.numpy()
    result = score(rows,correct)
    result.update(predictions=predictions, replay_predictions=replay, teacher_predictions=taught,
                  measurement_predictions=measured, mean_ce=float(np.mean(ce)),
                  replay=score(rows,np.array(replay)==target.numpy()),
                  joint=score(rows,correct & (np.array(replay)==target.numpy())),
                  teaching_readout=score(rows,np.array(taught)==classes.numpy()),
                  measurement_readout=score(rows,np.array(measured)==classes.numpy()))
    for fact in (1,2):
        idx = [i for i,r in enumerate(rows) if r['C_changed'] and r['family'] != 'requested_object'
               and r['base']['fact_order'].index(r['base']['query'])+1 == fact]
        result[f'fact{fact}'] = {'n':len(idx),'accuracy':float(correct[idx].mean()) if idx else None}
    idx=[i for i,r in enumerate(rows) if r['C_changed'] and r['family']=='requested_object']
    result['query_only']={'n':len(idx),'accuracy':float(correct[idx].mean()) if idx else None}
    return result


def fit(model, root, config, datasets, at_basis, teacher, measurement, batch, execution, session, progress):
    name = f"{config.get('stage','pilot')}_P{config['position']:02d}_{config['method']}_S{config['seed']}"
    folder = root / 'training' / name
    identity = {'config':config, 'execution':execution, 'teacher':tensor_digest(teacher), 'AT':tensor_digest(at_basis),
                'measurement':tensor_digest(measurement), 'train':digest(datasets['train'][0]), 'calibration':digest(datasets['calibration'][0])}
    write(folder / 'config.json',identity,immutable=True)
    if (folder / 'completed.json').exists():
        done=read(folder/'completed.json')
        assert done['identity']==identity
        assert file_hash(folder/done['checkpoint']['path'])==done['checkpoint']['sha256']
        for step,sha in done['evaluations'].items():
            assert file_hash(folder/'evaluations'/f'step_{int(step):06d}.json')==sha, 'Completed evaluation changed'
        return done
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    subspace=backend.LearnedSubspace(2304,128).to('cuda')
    opt=torch.optim.Adam(subspace.parameters(),lr=.01)
    loaded=recover(folder,identity)
    if loaded:
        state,_=loaded
        subspace.load_state_dict(state['subspace'])
        opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        state={k:v for k,v in state.items() if k not in ('subspace','basis','optimizer','cpu_rng','cuda_rng')}
    else:
        state={'identity':identity,'step':0,'updates':[],'evaluations':{},'initialization':tensor_digest(subspace.state_dict())}
        save(folder,state,subspace,opt)
    versions={n:p._version for n,p in model.named_parameters()}
    schedule=config.get('evaluations',[0,50,100,250,500])
    try:
        while True:
            if state['step'] in schedule:
                path=folder/'evaluations'/f"step_{state['step']:06d}.json"
                if path.exists():
                    record=read(path)
                    if str(state['step']) in state['evaluations']:
                        assert file_hash(path)==state['evaluations'][str(state['step'])], 'Saved evaluation hash changed'
                    assert record['identity']==identity and record['basis']==tensor_digest(subspace.basis().detach())
                else:
                    record={'identity':identity,'step':state['step'],'basis':tensor_digest(subspace.basis().detach())}
                    for bank,data in datasets.items():
                        record[bank]=evaluate(model,data,subspace.basis().detach(),config['position'],at_basis,teacher,measurement,batch,session)
                    write(path,record,immutable=True)
                state['evaluations'][str(state['step'])]=file_hash(path)
                save(folder,state,subspace,opt)
                progress(root,'fit_evaluation',configuration=name,step=state['step'],
                         train_changing=record['train']['changing'],train_preservation=record['train']['preservation'],
                         calibration_changing=record['calibration']['changing'],calibration_preservation=record['calibration']['preservation'])
            if state['step']==config['steps']:
                break
            indices=batch_order(len(datasets['train'][0]),config['seed'],state['step'])
            _,ids,source,targets,classes=datasets['train']
            start=time.monotonic()
            measured=update(model,subspace,opt,ids,source[config['position']],targets,classes,indices,
                            config['position'],at_basis,teacher,config['method'],batch,session)
            state['step']+=1
            state['updates'].append({'step':state['step'],'seconds':time.monotonic()-start,**measured})
            if state['step']%25==0:
                save(folder,state,subspace,opt)
                backend.assert_frozen(model,versions)
                progress(root,'training',configuration=name,step=state['step'],total_steps=config['steps'],losses=measured['losses'],gradient_norm=measured['gradient_norm'])
    except (runtime.Paused,TimeoutError):
        # Cooperative guards run before a forward, never halfway through Adam.
        save(folder,state,subspace,opt)
        raise
    # An uncontrolled interruption resumes a previous committed checkpoint, not a partial Adam update.
    final=read(folder/'evaluations'/f"step_{config['steps']:06d}.json")
    receipt=save(folder,state,subspace,opt)
    done={'identity':identity,'position':config['position'],'method':config['method'],'seed':config['seed'],
          'train':final['train'],'calibration':final['calibration'],'checkpoint':receipt,'evaluations':state['evaluations'],
          'initialization':state['initialization'],'steps':config['steps'], 'order_digest':digest([u['indices'] for u in state['updates']])}
    write(folder/'completed.json',done,immutable=True)
    backend.assert_frozen(model,versions)
    return done
