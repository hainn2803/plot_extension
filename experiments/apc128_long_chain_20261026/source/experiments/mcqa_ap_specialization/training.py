"""Fixed-update paired fits, exact data schedules and optimizer/RNG recovery."""
from collections import defaultdict
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments.mcqa_at_anchor_comparison.engine import Subspace
from experiments.mcqa_at_anchor_comparison.training import checkpoint,recover,restore_rng
from experiments.mcqa_c_at_it_das.training import tensor_digest
from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,digest,file_hash,status
from . import data
from .protocol import ARMS,SEEDS,STEPS

def order(rows,seed,step):
    cells=defaultdict(list)
    for i,r in enumerate(rows):
        m=r['metadata'];cells[(m['from_AP'],m['to_AP'])].append(i)
    epoch,offset=divmod(step,26);changed=[];preserved=[]
    for (b,s),ix in sorted(cells.items()):
        take=2 if b==s or len(ix)==52 else 4
        g=torch.Generator().manual_seed(int(data.keyed('order',seed,b,s,epoch)[:15],16))
        chosen=torch.randperm(len(ix),generator=g)[offset*take:(offset+1)*take].tolist()
        (preserved if b==s else changed).extend(ix[i] for i in chosen)
    indices=changed+preserved
    assert len(indices)==len(set(indices))==32
    return indices

def summarize(rows,arrays):
    correct=arrays['correct'].numpy();retain=arrays['retention'].numpy()
    changing=np.array([r['metadata']['kind']=='changing' for r in rows]);out={}
    def metric(mask):
        return dict(count=int(mask.sum()),accuracy=float(correct[mask].mean()),base_retention=float(retain[mask].mean()),
            ties=int(arrays['tied'].numpy()[mask].sum()),output_counts=np.bincount(arrays['output_position'].numpy()[mask]+1,minlength=6).tolist())
    out['all']=metric(np.ones(len(rows),bool));out['changing']=metric(changing);out['preservation']=metric(~changing)
    out['transitions']={}
    for b in range(4):
        for s in range(4):
            mask=np.array([(r['metadata']['from_AP'],r['metadata']['to_AP'])==(b,s) for r in rows])
            if mask.any():out['transitions'][f'{b+1}->{s+1}']=metric(mask)
    out['destinations']={str(s+1):metric(np.array([r['metadata']['to_AP']==s for r in rows])&changing) for s in range(4)
        if any(r['metadata']['to_AP']==s and r['metadata']['kind']=='changing' for r in rows)}
    for arm,dests in [('edge',(0,3)),('middle',(1,2))]:
        mask=changing&np.array([r['metadata']['to_AP'] in dests for r in rows])
        if mask.any():out[arm]=metric(mask)
    out['starting_positions']={str(b+1):metric(changing&np.array([r['metadata']['from_AP']==b for r in rows]))
        for b in range(4) if any(r['metadata']['kind']=='changing' and r['metadata']['from_AP']==b for r in rows)}
    out['target_letters']={letter:metric(np.array([r['target']==letter for r in rows])) for letter in data.LETTERS
        if any(r['target']==letter for r in rows)}
    out['group_transfers']={}
    for name,orig,dest in [('edge_to_edge',(0,3),(0,3)),('edge_to_middle',(0,3),(1,2)),
            ('middle_to_edge',(1,2),(0,3)),('middle_to_middle',(1,2),(1,2))]:
        mask=changing&np.array([r['metadata']['from_AP'] in orig and r['metadata']['to_AP'] in dest for r in rows])
        if mask.any():out['group_transfers'][name]=metric(mask)
    return out

@torch.no_grad()
def evaluate(engine,rows,source,q,cache=None):
    scores=[];ret=[];positions=[];texts=[];ce=0.
    for start in range(0,len(rows),engine.batch):
        small=rows[start:start+engine.batch];n=len(small)
        logits,_=engine.forward([r['base'] for r in small],source[start:start+n],q)
        score=engine.score(logits[:n],[r['target'] for r in small]);scores.append(score)
        decoded=[engine.decode(i).strip() for i in score['token_id']]
        texts.extend(decoded)
        ret.extend(text==data.answer(r['base']) and not tie for text,r,tie in zip(decoded,small,score['tied']))
        # -1 = tied/invalid; 4 = valid letter not present among base options.
        positions.extend(-1 if tie or text not in data.LETTERS else
            r['base']['labels'].index(text) if text in r['base']['labels'] else 4
            for text,r,tie in zip(decoded,small,score['tied']))
        target,_=engine.targets(small);ce+=float(F.cross_entropy(logits[:n],target,reduction='sum'))
    arrays={key:torch.cat([torch.as_tensor(s[key]) for s in scores]) for key in scores[0]}
    arrays.update(retention=torch.tensor(ret),output_position=torch.tensor(positions),
        pair_ids=[r['id'] for r in rows],prediction_text=texts)
    if cache is not None:
        baseline=cache.field([r['base'] for r in rows],'label_probabilities')
        arrays.update(clean_label_probabilities=baseline,probability_change=arrays['label_probabilities']-baseline,
            clean_token_ids=cache.field([r['base'] for r in rows],'token_id'))
    metrics=summarize(rows,arrays);metrics['cross_entropy']=ce/len(rows)
    return metrics,arrays

def fit(root,engine,arm,seed,train,calibration,cache):
    root=Path(root);rows,source=train;cal_rows,cal_source=calibration;name=f'{arm}_S{seed}';folder=root/'fits'/name
    identity=dict(run=data.verify_identity(root),execution=file_hash(root/'execution.json'),arm=arm,seed=seed,
        train=digest(rows),calibration=digest(cal_rows),updates=1000,objective='output CE only')
    if (folder/'completion.json').exists():
        done=read(folder/'completion.json')
        if done['identity']!=identity:raise ValueError('Completed training identity changed')
        load(done['checkpoint']);return done
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    sub=Subspace(seed).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01)
    initial=tensor_digest(sub.state_dict());old=recover(folder,identity)
    if old:
        previous,_=old;sub.load_state_dict(previous['subspace']);opt.load_state_dict(previous['optimizer']);restore_rng(previous)
        state={k:previous[k] for k in ('identity','step','updates','initialization','evaluations')}
        if initial!=state['initialization']:raise ValueError('Initialization changed')
        for i,x in enumerate(state['updates']):
            if x['indices']!=order(rows,seed,i):raise ValueError('Saved example order changed')
    else:
        state=dict(identity=identity,step=0,updates=[],initialization=initial,evaluations={});checkpoint(folder,state,sub,opt)
    try:
        while True:
            step=state['step']
            if step in STEPS and str(step) not in state['evaluations']:
                record={}
                for label,rr,ss in [('train',rows,source),('calibration',cal_rows,cal_source)]:
                    metrics,arrays=evaluate(engine,rr,ss,sub.basis().detach(),cache)
                    save(folder/'evaluations'/f'{step:06d}_{label}_{time.time_ns()}.pt',arrays);record[label]=metrics
                write(folder/'evaluations'/f'{step:06d}.json',record)
                state['evaluations'][str(step)]=record;checkpoint(folder,state,sub,opt)
                status(root,'training',configuration=name,step=step,calibration=record['calibration']['changing'])
            if step==1000:break
            ix=order(rows,seed,step);result=engine.update([rows[i] for i in ix],source[ix],sub,opt)
            state['step']+=1;state['updates'].append(dict(step=state['step'],indices=ix,**result))
            if state['step']%25==0:
                checkpoint(folder,state,sub,opt);status(root,'training',configuration=name,step=state['step'],total=1000,**result)
        path=checkpoint(folder,state,sub,opt)
    except BaseException:
        if len(state['updates'])==state['step'] and torch.isfinite(sub.raw).all():checkpoint(folder,state,sub,opt)
        raise
    preservation=[[rows[i]['id'] for i in x['indices'] if rows[i]['metadata']['kind']=='preservation'] for x in state['updates']]
    result=dict(identity=identity,checkpoint=str(path),sha256=file_hash(path),steps=1000,initialization=initial,
        order_digest=digest([x['indices'] for x in state['updates']]),preservation_order_digest=digest(preservation),
        final_evaluation=state['evaluations']['1000'])
    write(folder/'completion.json',result);return result

def lock(root):
    runs={}
    for seed in SEEDS:
        records=[]
        for arm in ARMS:
            done=read(Path(root)/'fits'/f'{arm}_S{seed}'/'completion.json')
            if done['steps']!=1000 or file_hash(done['checkpoint'])!=done['sha256']:raise ValueError('Final fit incomplete/corrupt')
            load(done['checkpoint']);records.append(done);runs[f'{arm}_S{seed}']=done
        assert len({r['initialization'] for r in records})==1
        assert len({r['preservation_order_digest'] for r in records})==1
    write(Path(root)/'locks/all_final.json',dict(runs=runs,primary_step=1000,no_selection=True))
    return runs
