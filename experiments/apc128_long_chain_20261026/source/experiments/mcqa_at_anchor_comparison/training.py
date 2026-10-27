"""Paired fixed-update fitting, complete optimizer recovery and AT qualification."""
from collections import defaultdict
import hashlib
import random
import time

import numpy as np
import torch

from experiments.hai_c_at_evaluation.core import rate
from experiments.mcqa_c_at_it_das.training import tensor_digest
from . import data
from .engine import Subspace,projection
from .storage import read,write,save,load,file_hash,digest,status


def row_cell(row,variable):
    m=row['metadata']
    return (m['category']+'/'+str(m['same_color'])) if variable=='AT' else m['family']


def order(rows,variable,seed,step):
    cells=defaultdict(list)
    for i,r in enumerate(rows): cells[row_cell(r,variable)].append(i)
    epoch,offset=divmod(step,len(rows)//32);out=[]
    for cell,indices in sorted(cells.items()):
        take=8 if variable=='C' or cell.startswith('preserve/') else 4
        g=torch.Generator().manual_seed(int(digest([data.SALT,'order',seed,cell,epoch])[:15],16))
        perm=torch.randperm(len(indices),generator=g)[offset*take:(offset+1)*take].tolist()
        out.extend(indices[i] for i in perm)
    assert len(out)==len(set(out))==32
    return out


def summarize_at(rows,correct):
    correct=np.asarray(correct,dtype=bool)
    cells={}
    categories={}
    letters={}
    for category in ('preserve','in_options','outside_options'):
        mask=np.array([r['metadata']['category']==category for r in rows])
        categories[category]=rate(correct[mask])
        for same in (True,False):
            sub=mask & np.array([r['metadata']['same_color']==same for r in rows])
            cells[category+'/'+str(same)]=rate(correct[sub])
        letters[category]={}
        for letter in data.LETTERS:
            sub=mask & np.array([r['target']==letter for r in rows])
            letters[category][letter]={**rate(correct[sub]),'templates':len({r['base']['question'] for r,v in zip(rows,sub) if v})}
    return dict(all=rate(correct),cells=cells,categories=categories,by_letter=letters)


def summarize_c(rows,direct,replay=None,readout=None,detailed=False):
    change=np.array([r['base']['color']!=r['source']['color'] for r in rows])
    result={}
    outcomes={'direct':np.asarray(direct,dtype=bool)}
    if replay is not None:
        outcomes.update(replay=np.asarray(replay,dtype=bool),joint=np.asarray(direct)&np.asarray(replay))
    if readout is not None: outcomes['readout']=np.asarray(readout,dtype=bool)
    for name,correct in outcomes.items():
        result[name]=dict(changing=rate(correct[change]),preservation=rate(correct[~change]),all=rate(correct))
    result['families']={family:{name:rate(values[np.array([r['metadata']['family']==family for r in rows])]) for name,values in outcomes.items()} for family in data.FAMILIES}
    result['strata']={}
    selectors={'donor_context':lambda r:'same_question' if r['base']['question']==r['source']['question'] else 'different_question',
               'target_letter':lambda r:r['target']}
    if detailed:selectors['label_order']=lambda r:','.join(r['base']['labels'])
    for label,fn in selectors.items():
        groups=defaultdict(list);result['strata'][label]={}
        for i,r in enumerate(rows):groups[fn(r)].append(i)
        for key,indices in sorted(groups.items()):
            result['strata'][label][key]={name:rate(values[indices]) for name,values in outcomes.items()}
    return result


@torch.no_grad()
def evaluate(engine,rows,states,q,variable,anchor=None,teacher=None,keep=False):
    correct=[];rep=[];taught=[];restore=[];pred=[];reppred=[];induced_states=[]
    ce=0.;coord=0.
    for start in range(0,len(rows),engine.batch):
        engine.session.check();small=rows[start:start+engine.batch];n=len(small)
        b=engine.pad(states['base_at'][start:start+n]).float()
        targets,_=engine.targets(small)
        if variable=='AT':
            source=engine.pad(states['source_at'][start:start+n]).float()
            induced=projection(b,source,q).to(torch.bfloat16).float()
            logits=engine.tail(induced)
            restored=projection(induced,b,q).to(torch.bfloat16).float()
            rs=engine.score(engine.tail(restored)[:n],[data.answer(r['base']) for r in small])
            restore.extend(rs['correct'].tolist())
            coord=max(coord,float(((restored-b)@q).abs().max()))
        else:
            logits,_,induced=engine.forward([r['base'] for r in small],c_donor=states['source_c'][start:start+n],c_q=q)
            replay=engine.tail(projection(b,induced,anchor).to(torch.bfloat16))
            scores=engine.score(replay[:n],[r['target'] for r in small]);rep.extend(scores['correct'].tolist());reppred.extend(scores['token_id'].tolist())
            if teacher is not None:
                y=(induced[:n]@anchor)@teacher['weight'].T+teacher['bias']
                top=y.topk(2,dim=1);classes=[data.LETTERS.index(r['target']) for r in small]
                taught.extend(((y.argmax(-1)==torch.tensor(classes,device=engine.device))&(top.values[:,0]>top.values[:,1])).cpu().tolist())
            if keep: induced_states.append(induced[:n].cpu())
        scores=engine.score(logits[:n],[r['target'] for r in small]);correct.extend(scores['correct'].tolist());pred.extend(scores['token_id'].tolist())
        ce+=float(torch.nn.functional.cross_entropy(logits[:n],targets,reduction='sum'))
    if variable=='AT':
        result=summarize_at(rows,correct);result.update(restoration=rate(restore),coordinate_max_error=coord)
    else: result=summarize_c(rows,correct,rep,taught if teacher is not None else None,detailed=keep)
    result['cross_entropy']=ce/len(rows)
    arrays=dict(correct=torch.tensor(correct),token_id=torch.tensor(pred))
    if variable=='AT': arrays['restoration']=torch.tensor(restore)
    else:
        arrays.update(replay=torch.tensor(rep),replay_token_id=torch.tensor(reppred))
        if teacher is not None: arrays['teaching']=torch.tensor(taught)
        if keep: arrays['induced_at']=torch.cat(induced_states)
    return result,arrays


def checkpoint(folder,state,sub,opt):
    numpy_rng=np.random.get_state()
    payload=dict(**state,subspace=sub.state_dict(),basis=sub.basis().detach().cpu(),optimizer=opt.state_dict(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        python_rng=random.getstate(),numpy_rng=[numpy_rng[0],numpy_rng[1].tolist(),numpy_rng[2],numpy_rng[3],numpy_rng[4]])
    path=folder/'checkpoints'/f'step_{state["step"]:06d}_{time.time_ns()}.pt'
    receipt=save(path,dict(payload=payload,payload_digest=tensor_digest(payload)))
    write(folder/'latest.json',dict(path=str(path),step=state['step'],identity=state['identity'],**receipt),immutable=False)
    return path


def recover(folder,identity):
    candidates=sorted((folder/'checkpoints').glob('step_*.receipt.json'))
    if not candidates:return None
    path=candidates[-1].with_name(candidates[-1].name.replace('.receipt.json','.pt'))
    wrapper=load(path);state=wrapper['payload']
    if state['identity']!=identity or tensor_digest(state)!=wrapper['payload_digest']:
        raise ValueError('Checkpoint identity/state mismatch')
    if not torch.isfinite(state['subspace']['raw']).all() or not torch.isfinite(state['basis']).all():
        raise ValueError('Nonfinite saved projector')
    q=state['basis']
    if not torch.allclose(q.T@q,torch.eye(q.shape[1],dtype=q.dtype),atol=2e-4,rtol=2e-4):
        raise ValueError('Saved basis is not orthonormal')
    assert len(state['updates'])==state['step']
    return state,path


def restore_rng(state):
    torch.set_rng_state(state['cpu_rng']);random.setstate(state['python_rng'])
    n=state['numpy_rng'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])


def fit(root,engine,variable,method,seed,train,calibration,anchor=None,teacher=None):
    rows,states=train;cal_rows,cal_states=calibration
    name=f'AT_S{seed}' if variable=='AT' else f'C_{method}_S{seed}'
    folder=root/'fits'/name;folder.mkdir(parents=True,exist_ok=True)
    maximum=2000 if variable=='AT' else 1000
    identity=dict(run=data.verify_identity(root),execution=file_hash(root/'execution.json'),variable=variable,method=method,seed=seed,
        train=digest(rows),calibration=digest(cal_rows),maximum=maximum,anchor=tensor_digest(anchor),teacher=tensor_digest(teacher))
    if (folder/'completion.json').exists():
        complete=read(folder/'completion.json')
        if complete['identity']!=identity:raise ValueError('Completed fit configuration changed')
        load(complete['checkpoint'])
        return complete
    sub=Subspace(seed).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01)
    initialization=tensor_digest(sub.state_dict())
    resumed=recover(folder,identity)
    if resumed:
        previous,_=resumed;sub.load_state_dict(previous['subspace']);opt.load_state_dict(previous['optimizer']);restore_rng(previous)
        state={k:previous[k] for k in ('identity','step','updates','initialization','evaluations')}
        assert state['initialization']==initialization
        for i,record in enumerate(state['updates']):
            assert record['indices']==order(rows,variable,seed,i)
    else:
        state=dict(identity=identity,step=0,updates=[],initialization=initialization,evaluations={})
        checkpoint(folder,state,sub,opt)
    eval_steps=([0]+data.PROTOCOL['at_selection_steps']) if variable=='AT' else data.PROTOCOL['c_evaluation_steps']
    try:
        while True:
            step=state['step']
            if step in eval_steps and str(step) not in state['evaluations']:
                record={}
                for label,rr,ss in [('train',rows,states),('calibration',cal_rows,cal_states)]:
                    if variable=='AT' and label=='train':continue
                    metrics,arrays=evaluate(engine,rr,ss,sub.basis().detach(),variable,anchor,teacher)
                    record[label]=metrics
                    save(folder/'evaluations'/f'{step:06d}_{label}_{time.time_ns()}.pt',arrays)
                write(folder/'evaluations'/f'{step:06d}.json',record)
                state['evaluations'][str(step)]=record
                checkpoint(folder,state,sub,opt)
            if step==maximum:break
            indices=order(rows,variable,seed,step)
            batch=[rows[i] for i in indices];selected={k:v[indices] for k,v in states.items()}
            result=engine.update('AT' if variable=='AT' else method,batch,sub,opt,selected,anchor,teacher)
            state['step']+=1;state['updates'].append(dict(step=state['step'],indices=indices,**result))
            if state['step']%25==0:
                checkpoint(folder,state,sub,opt)
                status(root,'training',configuration=name,step=state['step'],total=maximum,**result)
        path=checkpoint(folder,state,sub,opt)
    except BaseException:
        # Only a successfully completed optimizer update is committed.
        if len(state['updates'])==state['step'] and torch.isfinite(sub.raw).all():
            checkpoint(folder,state,sub,opt)
        raise
    result=dict(identity=identity,checkpoint=str(path),checkpoint_sha256=file_hash(path),steps=maximum,
        initialization=initialization,order_digest=digest([r['indices'] for r in state['updates']]),evaluations=state['evaluations'])
    write(folder/'completion.json',result)
    return result


def select_at(root):
    selected={}
    for seed in data.PROTOCOL['seeds']:
        folder=root/'fits'/f'AT_S{seed}';done=read(folder/'completion.json')
        def key(step):
            cells=done['evaluations'][str(step)]['calibration']['cells']
            scores=[x['accuracy'] for x in cells.values()]
            return (-min(scores),-sum(scores)/len(scores),step)
        step=min(data.PROTOCOL['at_selection_steps'],key=key)
        candidates=sorted((folder/'checkpoints').glob(f'step_{step:06d}_*.receipt.json'))
        path=candidates[-1].with_name(candidates[-1].name.replace('.receipt.json','.pt'))
        payload=load(path)['payload']
        selected[str(seed)]=dict(step=step,checkpoint=str(path),sha256=file_hash(path),basis_digest=tensor_digest(payload['basis']),
                                 calibration=done['evaluations'][str(step)]['calibration'])
    write(root/'locks/AT.json',dict(seeds=selected,selection='worst six cells, then equal-cell mean, then earlier step; all three seeds retained'))
    return selected


def anchor(root,seed,device='cpu'):
    entry=read(root/'locks/AT.json')['seeds'][str(seed)]
    if file_hash(entry['checkpoint'])!=entry['sha256']:raise ValueError('AT lock changed')
    value=load(entry['checkpoint'])['payload']['basis']
    if tensor_digest(value)!=entry['basis_digest']:raise ValueError('AT basis lock mismatch')
    return value.to(device)


def interval(rows,correct,mask=None,seed=20261020):
    correct=np.asarray(correct,dtype=float);mask=np.ones(len(rows),dtype=bool) if mask is None else np.asarray(mask)
    groups=sorted({r['base']['question'] for r,v in zip(rows,mask) if v})
    totals=[];counts=[]
    keys=np.array([r['base']['question'] for r in rows])
    for key in groups:
        m=mask&(keys==key);totals.append(correct[m].sum());counts.append(m.sum())
    g=np.random.default_rng(seed);indices=g.integers(0,len(groups),(2000,len(groups)))
    values=np.array(totals)[indices].sum(1)/np.array(counts)[indices].sum(1)
    lo,hi=np.quantile(values,[.025,.975])
    return dict(low=float(lo),high=float(hi),resamples=2000,clusters=len(groups),seed=seed)


def qualify(root,engine,bank):
    if (root/'at_qualification.json').exists():return read(root/'at_qualification.json')
    rows,states=bank;results={}
    references=[(str(s),anchor(root,s,engine.device)) for s in data.PROTOCOL['seeds']]
    references.append(('historical',data.parent.checkpoint('standard','AT')['basis'].to(engine.device)))
    for name,q in references:
        metrics,arrays=evaluate(engine,rows,states,q,'AT')
        lower={category:interval(rows,arrays['correct'].numpy(),[r['metadata']['category']==category for r in rows]) for category in ('preserve','in_options','outside_options')}
        gates=qualification_gates(metrics,lower)
        results[name]=dict(metrics=metrics,intervals=lower,gates=gates,passed=all(gates.values()))
        save(root/'qualification'/f'{name}.pt',arrays)
        status(root,'AT_qualification',seed=name,passed=all(gates.values()),gates=gates)
    result=dict(results=results,all_passed=all(results[str(s)]['passed'] for s in data.PROTOCOL['seeds']),
                no_checkpoint_reselection=True,all_seeds_required=True)
    write(root/'at_qualification.json',result)
    return result


def qualification_gates(metrics,lower):
    return dict(cells=len(metrics['cells'])==6 and all(x['accuracy']>=.95 for x in metrics['cells'].values()),
        letters=len(metrics['by_letter'])==3 and all(len(dd)==26 and all(v['accuracy']>=.90 and v['templates']>=25 for v in dd.values()) for dd in metrics['by_letter'].values()),
        intervals=len(lower)==3 and all(v['low']>.90 for v in lower.values()),restoration=metrics['restoration']['accuracy']>=.99)
