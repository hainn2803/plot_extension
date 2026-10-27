"""CPU algebra/protocol tests and independent saved-evidence verification."""
from pathlib import Path
import random
import tempfile
from contextlib import nullcontext
import time
import numpy as np
import torch
from experiments.mcqa_at_anchor_comparison.engine import Subspace,projection
from experiments.mcqa_at_anchor_comparison.training import checkpoint,recover,restore_rng
from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,digest,file_hash
from experiments.mcqa_c_at_it_das.training import tensor_digest
from . import data,training,banks,controls
from .sites import Site,catalog,gather,scatter,structural_positions

def cpu(root,prepared=True):
    torch.set_num_threads(2)
    if prepared:data.verify(root)
    groups=data.inventory();checks=[]
    assert [len(groups[k]) for k in ('train','calibration','final')]==[96,48,108]
    assert len({o['id'] for group in groups.values() for o in group})==252
    checks.append('Disjoint object split')
    master=None
    for bank in data.SIZES:
        rows=[r for slot in data.slots(bank) for r in data.proposal(slot,0,groups)]
        assert len(rows)==dict(train=1792,calibration=512,final=1792,c_diagnostic=256)[bank]
        if bank=='train':master=rows
        for r in rows:
            b,s=r['base'],r['source'];m=r['metadata']
            requested=b['contents'].index(s['color']) if bank=='c_diagnostic' else s['contents'].index(s['color'])
            assert r['target']==b['labels'][requested]
            if bank!='c_diagnostic':
                reverse=s['labels'][b['contents'].index(b['color'])]
                assert reverse in s['labels']
                assert data.answer(b)==data.answer(s) and b['color']==s['color']
                if m['permutation']=='cycle':assert s['labels'][m['from_AP']]!=r['target']
        checks.append(bank+' target/count/reciprocal/cycle checks')
    assert len(training.subset(master))==512
    for seed in (1,0,2):
        paired=[]
        for arm in ('shared','edge','middle'):
            rows=training.view(master,arm);sequence=[]
            for step in (0,1,31,32,49,63,64,999):
                ix=training.order(rows,seed,step);batch=[rows[i] for i in ix]
                assert sum(r['metadata']['kind']=='changing' for r in batch)==24
                assert all(sum(r['metadata']['from_AP']==r['metadata']['to_AP']==p for r in batch)==2 for p in range(4))
                sequence.append([r['id'] for r in batch if r['metadata']['kind']=='preservation'])
            paired.append(digest(sequence))
        assert len(set(paired))==1
    checks.append('Balanced schedules and paired preservation order')
    assert len(catalog(37))==26*45 and len(catalog(37,True))==52
    assert len({s.id for s in catalog(37)+catalog(37,True)})==1222
    g=torch.Generator().manual_seed(73);b=torch.randn(5,7,8,generator=g,dtype=torch.float64)
    s=torch.randn(5,32,generator=g,dtype=torch.float64);pos=torch.tensor([[0,2,4,6]]*5);valid=torch.tensor([1,1,0,1,1],dtype=torch.bool)
    sub=Subspace(3,width=32,rank=4,dtype=torch.float64);q=sub.basis();base=gather(b,pos)
    changed=projection(base,s,q);out=scatter(b,pos,changed,valid)
    assert torch.equal(out[2],b[2]) and torch.equal(out[:,[1,3,5]],b[:,[1,3,5]])
    assert torch.allclose(q.T@q,torch.eye(4,dtype=torch.float64),atol=1e-10,rtol=1e-10)
    assert torch.allclose((changed-base)@(torch.eye(32,dtype=torch.float64)-q@q.T),torch.zeros_like(base),atol=1e-10,rtol=1e-10)
    loss=out.square().sum();grad=torch.autograd.grad(loss,sub.raw)[0];original=sub.raw.detach().clone();eps=1e-6
    for i,j in ((0,0),(13,1),(31,3)):
        with torch.no_grad():sub.raw.copy_(original);sub.raw[i,j]+=eps
        plus=scatter(b,pos,projection(base,s,sub.basis()),valid).square().sum().item()
        with torch.no_grad():sub.raw.copy_(original);sub.raw[i,j]-=eps
        minus=scatter(b,pos,projection(base,s,sub.basis()),valid).square().sum().item()
        assert abs((plus-minus)/(2*eps)-float(grad[i,j]))<2e-4
    with torch.no_grad():sub.raw.copy_(original)
    checks.append('Independent four-token scatter, padding identity, complement and finite differences')
    def advance(model,opt):
        opt.zero_grad();z=projection(base,s,model.basis());z.square().mean().backward();opt.step()
        torch.rand(());random.random();np.random.rand()
    with nullcontext(str(Path(root)/'checkpoint_tests'/str(time.time_ns()))) as tmp:
        folder=Path(tmp);opt=torch.optim.Adam(sub.parameters(),lr=.01);advance(sub,opt)
        checkpoint(folder,dict(identity={'test':True},step=1,updates=[dict(step=1)]),sub,opt)
        saved,_=recover(folder,{'test':True});advance(sub,opt)
        expected=tensor_digest([sub.state_dict(),opt.state_dict(),torch.get_rng_state(),random.getstate(),np.random.get_state()[1].tolist()])
        other=Subspace(3,width=32,rank=4,dtype=torch.float64);other.load_state_dict(saved['subspace'])
        oo=torch.optim.Adam(other.parameters(),lr=.01);oo.load_state_dict(saved['optimizer']);restore_rng(saved);advance(other,oo)
        actual=tensor_digest([other.state_dict(),oo.state_dict(),torch.get_rng_state(),random.getstate(),np.random.get_state()[1].tolist()])
        assert actual==expected
        try:recover(folder,{'test':False})
        except ValueError:pass
        else:raise AssertionError('Changed resume identity accepted')
        try:banks.locked(folder)
        except FileNotFoundError:pass
        else:raise AssertionError('Final seal opened without endpoints')
    checks.append('Exact optimizer/RNG resume, identity rejection and final seal')
    from experiments.toy_c_at_composition import core as toy
    model=toy.FrozenNetwork();bank=toy.prepared(model,toy.bank());toy_results={}
    for name in ('compatible','incompatible'):
        metrics=toy.evaluate(model,bank,model.reference_basis(name),structural=True)
        if isinstance(metrics,tuple):metrics=metrics[0]
        assert metrics['direct_changing']['accuracy']==1 and metrics['at_changing']['accuracy']==1
        assert metrics['replay_changing']['accuracy']==(1 if name=='compatible' else 0)
        toy_results[name]=metrics
    checks.append('Executed exhaustive compatible/incompatible composition counterexamples')
    from transformers import AutoTokenizer
    from diagnostics.hai_frozen_eval_20261020 import run as historical
    tokenizer=AutoTokenizer.from_pretrained(historical.MODEL,local_files_only=True)
    for r in master[::31]:
        for c in (r['base'],r['source']):
            t=tokenizer(c['prompt'],return_offsets_mapping=True);p=structural_positions(c,t['offset_mapping'])
            assert len(set(p['label']+p['color']))==8
    for r in master[::47]:
        for role in controls.ROLES:
            c=controls.proposal(r,role,0,groups['final'])
            assert c['question']!=r['base']['question'] and c['color']!=r['base']['color']
            if role.startswith('AP'):assert data.ap(c)==r['metadata']['to_AP'] and data.answer(c)!=r['target']
    checks.append('Pinned-tokenizer structural spans and independent donor target construction')
    result=dict(passed=True,checks=checks,toy_results=toy_results,cuda_initialized=torch.cuda.is_initialized())
    assert not result['cuda_initialized'];write(Path(root)/'cpu_verification.json',result);return result

def final(root):
    root=Path(root);data.verify(root,model=True);banks.locked(root)
    from transformers import AutoTokenizer
    identity=read(root/'identity.json');tokenizer=AutoTokenizer.from_pretrained(identity['model_root'],local_files_only=True)
    known={};primary={}
    for label,bank in [('final','final'),('C','c_diagnostic')]:
        rr=banks.load_bank(root,bank)
        for r in rr:
            base,source=r['base'],r['source']
            requested=base['contents'].index(source['color']) if bank=='c_diagnostic' else source['contents'].index(source['color'])
            assert r['target']==base['labels'][requested]
            known[r['id']]=(r['target'],data.answer(base));primary[r['id']]=r
        selected={}
        for page in sorted((root/'controls'/label).glob('page_*.json')):selected.update(read(page)['accepted'])
        for role in controls.ROLES:
            control_rows,_=controls.rows_for(rr,selected,role)
            for r in control_rows:
                parent=primary[r['metadata']['parent_id']]
                dest=(parent['base']['contents'].index(parent['source']['color']) if bank=='c_diagnostic'
                      else parent['source']['contents'].index(parent['source']['color']))
                expected=data.answer(r['source']) if role.startswith('conflict') else r['base']['labels'][dest]
                assert r['target']==expected
                known[r['id']]=(expected,data.answer(r['base']))
    count=0;files=[]
    for p in sorted((root/'evaluation').glob('*.pt')):
        arr=load(p)
        if not all(k in arr for k in ('token_id','targets','tied','correct')):raise ValueError('Missing prediction evidence: '+str(p))
        restore=p.stem.endswith(('_restore','_identity'))
        expected_targets=[known[ident][1 if restore else 0] for ident in arr['pair_ids']]
        if arr['targets']!=expected_targets:raise ValueError('Saved target differs from independent pair semantics: '+str(p))
        decoded=[tokenizer.decode([int(i)]).strip() for i in arr['token_id']]
        expected=torch.tensor([not bool(t) and s==target for s,target,t in zip(decoded,arr['targets'],arr['tied'])])
        if not torch.equal(expected,arr['correct']):raise ValueError('Independent scoring disagreement: '+str(p))
        if 'top_two_logits' in arr:
            if not torch.equal(arr['tied'],arr['top_two_logits'][:,0]==arr['top_two_logits'][:,1]):raise ValueError('Tie mismatch')
            if not torch.equal(arr['token_id'],arr['top_two_token_ids'][:,0]):raise ValueError('Winning token mismatch')
        count+=len(expected);files.append(dict(path=str(p),sha256=file_hash(p),count=len(expected)))
    groups=read(root/'groups.json');all_objects=[set(o['id'] for o in groups[g]) for g in ('train','calibration','final')]
    assert not any(all_objects[i]&all_objects[j] for i in range(3) for j in range(i))
    rows=banks.load_bank(root,'final');coverage=data.audit(rows,'final')
    results=read(root/'results.json');locks=read(root/'locks/all_final.json');crows=banks.load_bank(root,'c_diagnostic')
    def independent_interval(rr,first,second,mask):
        group_names=sorted({r['base']['question'] for r in rr});lookup={q:i for i,q in enumerate(group_names)}
        index=np.array([lookup[r['base']['question']] for r in rr]);difference=second.numpy().astype(float)-first.numpy().astype(float)
        sums=np.bincount(index[mask],weights=difference[mask],minlength=len(group_names))
        counts=np.bincount(index[mask],minlength=len(group_names));rng=np.random.default_rng(20261024)
        draw=rng.integers(0,len(group_names),(2000,len(group_names)));den=counts[draw].sum(1)
        values=sums[draw].sum(1)[den>0]/den[den>0]
        return dict(count=int(mask.sum()),difference=float(difference[mask].mean()),lower=float(np.quantile(values,.025)),upper=float(np.quantile(values,.975)),resamples=2000)
    paired_checks=[]
    for seed in (1,0,2):
        aliases={slot:key+f'_S{seed}' for slot,key in locks['aliases'].items()}
        primary_arrays={slot:load(root/'evaluation'/f'{key}_direct.pt') for slot,key in aliases.items()}
        for arr in primary_arrays.values():
            assert arr['pair_ids']==[r['id'] for r in rows] and len(arr['correct'])==1792
        for arm,other in [('shared','baseline'),('edge','baseline'),('middle','baseline'),('distributed','baseline'),('distributed','shared'),('distributed','capacity')]:
            mask=np.array([r['metadata']['from_AP']!=r['metadata']['to_AP'] and
                (arm not in ('edge','middle') or r['metadata']['to_AP'] in ((0,3) if arm=='edge' else (1,2))) for r in rows])
            value=independent_interval(rows,primary_arrays[other]['correct'],primary_arrays[arm]['correct'],mask)
            saved=results['comparisons'][str(seed)][arm+'_minus_'+other]
            assert all(abs(saved[k]-value[k])<1e-12 for k in value)
            paired_checks.append(dict(seed=seed,comparison=arm+'_minus_'+other,**value))
        for endpoint in set(aliases.values()):
            site=Site(**locks['endpoints'][endpoint]['identity']['site']);joint=[]
            for method in ('standard','gradual2.0'):
                direct=load(root/'evaluation'/f'C_{method}_{site.id}_direct.pt')['correct']
                replay=load(root/'evaluation'/f'{endpoint}_C_{method}_AP_replay.pt')['correct'];joint.append(direct&replay)
            mask=np.array([r['metadata']['from_AP']!=r['metadata']['to_AP'] for r in crows])
            value=independent_interval(crows,*joint,mask);saved=results['comparisons'][str(seed)][endpoint+'_C_gradual_minus_standard']
            assert all(abs(saved[k]-value[k])<1e-12 for k in value)
            paired_checks.append(dict(seed=seed,comparison=endpoint+'_C_gradual_minus_standard',**value))
    receipt=dict(passed=True,predictions=count,files=files,object_separation=True,coverage=coverage,
        all_primary_pairs_retained=True,targets_independently_recomputed=True,paired_bootstrap_recomputed=paired_checks,final_data_used_for_selection=False)
    write(root/'independent_verification.json',receipt);return receipt
