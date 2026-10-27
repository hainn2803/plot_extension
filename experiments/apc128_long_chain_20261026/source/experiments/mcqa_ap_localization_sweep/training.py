"""Uniform screens, fixed endpoints and calibration-only finalist selection."""
from collections import defaultdict
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from experiments.mcqa_at_anchor_comparison.engine import Subspace
from experiments.mcqa_at_anchor_comparison.training import checkpoint,recover,restore_rng
from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,file_hash,digest,status
from experiments.mcqa_c_at_it_das.training import tensor_digest
from . import data,banks
from .sites import Site
from .protocol import domain,SEEDS,STEPS,PROTOCOL

def view(rows,arm):
    return [r for r in rows if r['metadata']['kind']=='preservation' or domain(arm,r['metadata']['to_AP'])]

def order(rows,seed,step):
    cells=defaultdict(list)
    for i,r in enumerate(rows):cells[(r['metadata']['from_AP'],r['metadata']['to_AP'])].append(i)
    changing=sum(b!=s for b,s in cells);take=24//changing;out=[]
    for (b,s),indices in sorted(cells.items()):
        count=2 if b==s else take
        if len(indices)%count:raise ValueError('Schedule cells must divide exactly')
        epoch,offset=divmod(step,len(indices)//count)
        generator=torch.Generator().manual_seed(int(data.key('schedule',seed,b,s,epoch)[:15],16))
        selected=torch.randperm(len(indices),generator=generator)[offset*count:(offset+1)*count].tolist()
        out.extend(indices[i] for i in selected)
    assert len(out)==len(set(out))==32
    return out

def subset(rows):
    cells=defaultdict(list)
    for r in rows:cells[(r['metadata']['from_AP'],r['metadata']['to_AP'])].append(r)
    selected=[]
    for cell,rr in sorted(cells.items()):
        if cell[0]!=cell[1]:
            for typ in ('transposition','cycle'):
                selected.extend(sorted([r for r in rr if r['metadata']['permutation']==typ],key=lambda r:data.key('map',r['id']))[:16])
        else:selected.extend(sorted(rr,key=lambda r:data.key('map',r['id']))[:32])
    assert len(selected)==512
    return selected

def rate(values):
    return dict(count=len(values),correct=int(sum(values)),accuracy=float(np.mean(values)) if len(values) else None)

def summarize(rows,arrays):
    correct=arrays['correct'].numpy().astype(bool);masks={};groups=defaultdict(dict)
    for i,r in enumerate(rows):
        m=r['metadata'];b,s=m['from_AP'],m['to_AP'];change=b!=s
        keys=dict(transition=f'{b+1}->{s+1}',parent_transition=m.get('parent_transition',f'{b+1}->{s+1}'),permutation=m.get('permutation','control'),
            target_letter=r['target'],color=r['base']['color'],origin=str(b+1),destination=str(s+1),
            group='preservation' if not change else 'edge' if s in (0,3) else 'middle',
            transfer=('edge' if b in (0,3) else 'middle')+'->'+('edge' if s in (0,3) else 'middle'))
        for field,key in keys.items():groups[field].setdefault(key,[]).append(i)
        masks.setdefault('changing' if change else 'preservation',[]).append(i)
    out={name:rate(correct[ix]) for name,ix in masks.items()};out['all']=rate(correct)
    for field,entries in groups.items():
        out[field]={key:dict(**rate(correct[ix]),base_retention=float(arrays['retention'][ix].float().mean()),
            available=float(arrays['available'][ix].float().mean()),ties=int(arrays['tied'][ix].sum()),
            output_positions={str(p):int((arrays['output_position'][ix]==p).sum()) for p in (-1,0,1,2,3,4)}) for key,ix in entries.items()}
    return out

@torch.no_grad()
def evaluate(engine,rows,donor,q,site,natural,restore=False,keep_states=False,c_donor=None,c_q=None,at_donor=None,at_q=None,at_restore=None):
    output={};loss=0.;errors=[]
    for start in range(0,len(rows),engine.batch):
        small=rows[start:start+engine.batch];n=len(small);cs=[r['base'] for r in small]
        src=None if donor is None else {k:v[start:start+n] for k,v in donor.items()}
        y,h=engine.forward(cs,src,q,site,restore=restore,
            c_donor=None if c_donor is None else c_donor[start:start+n],c_q=c_q,
            at_donor=None if at_donor is None else at_donor[start:start+n],at_q=at_q,
            at_restore=None if at_restore is None else at_restore[start:start+n])
        targets=[data.answer(c) if restore else r['target'] for c,r in zip(cs,small)]
        scores=banks.score(engine,y[:n],targets);decoded=[engine.decode(i).strip() for i in scores['token_id']]
        scores['retention']=torch.tensor([not tied and text==data.answer(r['base']) for text,tied,r in zip(decoded,scores['tied'],small)])
        scores['output_position']=torch.tensor([-1 if tied or text not in engine.letters else r['base']['labels'].index(text) if text in r['base']['labels'] else 4 for text,tied,r in zip(decoded,scores['tied'],small)])
        scores['available']=h['available'][:n].cpu()
        pp,_=engine.positions(cs,site);_,_,cp=engine.encode(cs)
        scores['C_support_overlap']=((pp==cp[:,None]).any(1)&(site.layer==0))[:n].cpu()
        tid=torch.tensor([engine.label_ids[engine.letters.index(t)] for t in targets],device=engine.device)
        loss+=float(F.cross_entropy(y[:n],tid,reduction='sum'))
        if restore:errors.extend((h['site'][:n]-h['original_site'][:n]).norm(dim=-1).cpu().tolist())
        elif at_restore is not None:
            errors.extend((h['at_after_override'][:n].cpu()-at_restore[start:start+n].cpu()).norm(dim=-1).tolist())
        if keep_states:
            scores['site_state']=h['site'][:n].cpu().to(torch.bfloat16);scores['at_state']=h['at'][:n].cpu().to(torch.bfloat16)
        for k,v in scores.items():output.setdefault(k,[]).append(v)
    arrays={k:torch.cat(v) for k,v in output.items()};arrays.update(pair_ids=[r['id'] for r in rows],
        targets=[data.answer(r['base']) if restore else r['target'] for r in rows],
        clean_token_id=natural.field([r['base'] for r in rows],'token_id'))
    if errors:arrays['restoration_l2']=torch.tensor(errors)
    metrics=summarize(rows,arrays);metrics['CE']=loss/len(rows)
    metrics['C_support_overlap_count']=int(arrays['C_support_overlap'].sum())
    if errors:metrics['restoration_l2']=dict(mean=float(np.mean(errors)),maximum=max(errors))
    return metrics,arrays

def saved_evaluate(path,fn):
    path=Path(path)
    if path.exists():return read(path.with_suffix('.json')),load(path)
    metrics,arrays=fn();save(path,arrays);write(path.with_suffix('.json'),metrics);return metrics,arrays

def fit(root,engine,site,rank,arm,seed,maximum,master,cal,natural,states,screen=False):
    root=Path(root);name=site.id+f'_R{rank:04d}_{arm}_S{seed}'
    folder=root/('screens' if screen else 'fits')/name;rows=view(master,arm)
    identity=dict(run=file_hash(root/'identity.json'),execution=file_hash(root/'execution.json'),site=site.record(),
        rank=rank,arm=arm,seed=seed,maximum=maximum,train=digest(rows),cal=digest(cal))
    if (folder/'completion.json').exists():
        done=read(folder/'completion.json')
        if done['identity']!=identity:raise ValueError('Completed fit identity changed')
        load(done['checkpoint']);return done
    donors=states.get([r['source'] for r in rows],site);cal_source=states.get([r['source'] for r in cal],site)
    train_eval=subset(master) if screen else rows;train_source=states.get([r['source'] for r in train_eval],site)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    sub=Subspace(seed,width=2304*site.count,rank=rank).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01)
    initial=tensor_digest(sub.state_dict());old=recover(folder,identity)
    if old:
        previous,_=old;sub.load_state_dict(previous['subspace']);opt.load_state_dict(previous['optimizer']);restore_rng(previous)
        state={k:previous[k] for k in ('identity','step','updates','initialization','evaluations')}
        assert state['initialization']==initial
        for i,u in enumerate(state['updates']):assert u['indices']==order(rows,seed,i)
    else:
        state=dict(identity=identity,step=0,updates=[],initialization=initial,evaluations={});checkpoint(folder,state,sub,opt)
    eval_steps=(0,50) if screen else STEPS
    safe_checkpoint=True
    try:
        while True:
            step=state['step']
            if step in eval_steps and str(step) not in state['evaluations']:
                record={}
                for label,rr,src in [('calibration',cal,cal_source),('train',train_eval,train_source)]:
                    if screen and step==0 and label=='train':continue
                    path=folder/'evaluations'/f'{step:06d}_{label}.pt'
                    record[label],_=saved_evaluate(path,lambda rr=rr,src=src:evaluate(engine,rr,src,sub.basis(),site,natural))
                write(folder/'evaluations'/f'{step:06d}.json',record);state['evaluations'][str(step)]=record
                checkpoint(folder,state,sub,opt)
            if step==maximum:break
            ix=order(rows,seed,step);safe_checkpoint=False
            value=engine.update([rows[i] for i in ix],{k:v[ix] for k,v in donors.items()},sub,opt,site)
            state['step']+=1;state['updates'].append(dict(step=state['step'],indices=ix,**value))
            safe_checkpoint=True
            if state['step']%25==0:
                checkpoint(folder,state,sub,opt);status(root,'screening' if screen else 'training',configuration=name,step=state['step'],**value)
        path=checkpoint(folder,state,sub,opt)
    except BaseException:
        if safe_checkpoint and len(state['updates'])==state['step'] and torch.isfinite(sub.raw).all():checkpoint(folder,state,sub,opt)
        raise
    preservation=[[rows[i]['id'] for i in u['indices'] if rows[i]['metadata']['kind']=='preservation'] for u in state['updates']]
    done=dict(name=name,identity=identity,checkpoint=str(path),sha256=file_hash(path),steps=maximum,
        initialization=initial,order_digest=digest([u['indices'] for u in state['updates']]),preservation_digest=digest(preservation),
        endpoint=state['evaluations'][str(maximum)])
    write(folder/'completion.json',done);return done

def rank_key(done,arm):
    metric=done['endpoint']['calibration'];cells=metric['transition'];changes=[];preserves=[]
    for key,value in cells.items():
        b,s=[int(x)-1 for x in key.split('->')]
        if b==s:preserves.append(value['accuracy'])
        elif domain(arm,s):changes.append(value['accuracy'])
    return (-min(changes+preserves),-float(np.mean(changes)),-float(np.mean(preserves)),metric['CE'],Site(**done['identity']['site']).id)

def select(root):
    root=Path(root);catalogue=read(root/'sites.json');done={}
    for s in catalogue['single']+catalogue['distributed']:
        site=Site(**s);name=site.id+'_R0128_shared_S1';value=read(root/'screens'/name/'completion.json')
        if value['steps']!=50 or file_hash(value['checkpoint'])!=value['sha256']:raise ValueError('Incomplete screening')
        done[site.id]=value
    single=[done[Site(**s).id] for s in catalogue['single']];multi=[done[Site(**s).id] for s in catalogue['distributed']]
    winners={arm:min(single,key=lambda x:rank_key(x,arm)) for arm in ('shared','edge','middle')}
    winners['distributed']=min(multi,key=lambda x:rank_key(x,'shared'))
    slots={k:dict(site=v['identity']['site'],rank=128,arm=k if k in ('edge','middle') else 'shared') for k,v in winners.items()}
    slots['capacity']=dict(site=slots['shared']['site'],rank=512,arm='shared')
    slots['baseline']=dict(site=Site(17,'offset',0).record(),rank=128,arm='shared')
    configurations={};aliases={}
    for slot,config in slots.items():
        key=digest(config)[:16];configurations[key]=config;aliases[slot]=key
    receipt=dict(slots=slots,aliases=aliases,unique_configurations=configurations,selection_final_data_used=False,
        rankings={arm:[dict(name=d['name'],score=rank_key(d,arm)) for d in sorted(single,key=lambda x:rank_key(x,arm))] for arm in ('shared','edge','middle')},
        distributed_ranking=[dict(name=d['name'],score=rank_key(d,'shared')) for d in sorted(multi,key=lambda x:rank_key(x,'shared'))])
    write(root/'selection.json',receipt);return receipt

def lock(root):
    root=Path(root);selected=read(root/'selection.json');endpoints={};initial={};preservation={}
    for key,config in selected['unique_configurations'].items():
        site=Site(**config['site'])
        for seed in SEEDS:
            name=site.id+f"_R{config['rank']:04d}_{config['arm']}_S{seed}";done=read(root/'fits'/name/'completion.json')
            if done['steps']!=1000 or file_hash(done['checkpoint'])!=done['sha256']:raise ValueError('Final endpoint incomplete')
            load(done['checkpoint']);endpoints[key+f'_S{seed}']=done
            initial.setdefault((seed,site.count,config['rank']),set()).add(done['initialization'])
            preservation.setdefault(seed,set()).add(done['preservation_digest'])
    if any(len(x)!=1 for x in [*initial.values(),*preservation.values()]):raise ValueError('Paired initialization/preservation mismatch')
    write(root/'locks/all_final.json',dict(selected,endpoints=endpoints,paired_initializations=True,paired_preservation=True))
