"""Independent final AP and frozen C/AT composition diagnostics."""
from pathlib import Path
import numpy as np
import torch
from experiments.hai_c_at_evaluation.backends import checked_basis
from experiments.mcqa_at_anchor_comparison.storage import read,write,load,file_hash,status
from . import data,banks,controls,training
from .sites import Site
from .protocol import FROZEN,SEEDS,PROTOCOL

def uploaded():
    result={}
    for method in ('standard','gradual2.0'):
        path=FROZEN/f'results_joint_randomized_{method}'/'das_c.pt'
        result[method]=checked_basis(torch.load(path,map_location='cpu',weights_only=True),2304,'C')
    path=FROZEN/'results_joint_randomized_standard/das_at.pt'
    result['AT']=checked_basis(torch.load(path,map_location='cpu',weights_only=True),2304,'AT')
    return result

def bootstrap(rows,a,b,mask):
    clusters=sorted({r['metadata'].get('cluster_question',r['base']['question']) for r in rows});index={x:i for i,x in enumerate(clusters)}
    diff=np.asarray(b,dtype=float)-np.asarray(a,dtype=float);mask=np.asarray(mask,dtype=bool)
    sums=np.zeros(len(clusters));counts=np.zeros(len(clusters))
    for i,r in enumerate(rows):
        if mask[i]:
            j=index[r['metadata'].get('cluster_question',r['base']['question'])];sums[j]+=diff[i];counts[j]+=1
    rng=np.random.default_rng(PROTOCOL['bootstrap_seed']);draw=rng.integers(0,len(clusters),(2000,len(clusters)))
    den=counts[draw].sum(1);values=sums[draw].sum(1)[den>0]/den[den>0]
    return dict(count=int(mask.sum()),difference=float(diff[mask].mean()) if mask.any() else None,
        lower=float(np.quantile(values,.025)) if len(values) else None,upper=float(np.quantile(values,.975)) if len(values) else None,resamples=2000)

def targets(rows,original=False):
    return [dict(r,target=data.answer(r['base']) if original else r['target']) for r in rows]

def put(root,name,fn):
    result=training.saved_evaluate(Path(root)/'evaluation'/f'{name}.pt',fn)
    status(root,'evaluating',operation=name,count=len(result[1]['pair_ids']));return result

def clean_at(root,engine,rows,control,natural,states,at_q,label):
    result={};success=[]
    for role in ('AT_donor0','AT_donor1'):
        rr,ix=controls.rows_for(rows,control,role)
        mask=torch.zeros(len(rows),dtype=torch.bool)
        if rr:
            donor=states.get([r['source'] for r in rr],Site(25,'offset',0))['values']
            metric,arr=put(root,f'{label}_{role}',lambda:training.evaluate(engine,rr,None,None,Site(25,'offset',0),natural,at_donor=donor,at_q=at_q))
            mask[ix]=arr['correct'];result[role]=dict(metrics=metric,indices=ix)
            base=states.get([r['base'] for r in rr],Site(25,'offset',0))['values']
            restored,_=put(root,f'{label}_{role}_roundtrip_restore',lambda:training.evaluate(engine,targets(rr,True),None,None,
                Site(25,'offset',0),natural,at_donor=donor,at_q=at_q,at_restore=base))
            result[role]['natural_roundtrip']=restored
        else:result[role]=dict(metrics=None,indices=[])
        success.append(mask)
    return result,success[0]&success[1]

def downstream(root,engine,rows,control,trace,natural,states,at_q,name):
    replay,arr=put(root,name+'_AT_replay',lambda:training.evaluate(engine,rows,None,None,Site(25,'offset',0),natural,
        at_donor=trace['at_state'].float(),at_q=at_q))
    result=dict(replay=replay)
    for role in ('conflict0','conflict1'):
        rr,ix=controls.rows_for(rows,control,role)
        if not rr:result[role]=None;continue
        donor=states.get([r['source'] for r in rr],Site(25,'offset',0))['values']
        # The inducing AP run is already recorded. Apply AT to that induced final state,
        # then run the unchanged final normalization/head through Engine.tail.
        from experiments.mcqa_at_anchor_comparison.engine import projection
        outputs={}
        for start in range(0,len(rr),engine.batch):
            small=rr[start:start+engine.batch];ii=ix[start:start+len(small)]
            with torch.no_grad():
                h=projection(trace['at_state'][ii].to(engine.device).float(),donor[start:start+len(small)].to(engine.device),at_q).to(torch.bfloat16)
                y=engine.tail(h)[:len(small)];scores=banks.score(engine,y,[r['target'] for r in small])
            for k,v in scores.items():outputs.setdefault(k,[]).append(v)
        values={k:torch.cat(v) for k,v in outputs.items()};values.update(pair_ids=[r['id'] for r in rr],targets=[r['target'] for r in rr])
        from experiments.mcqa_at_anchor_comparison.storage import save
        path=Path(root)/'evaluation'/f'{name}_{role}.pt'
        if path.exists():values=load(path)
        else:save(path,values)
        result[role]=training.rate(values['correct'].numpy())
    original=states.get([r['base'] for r in rows],Site(25,'offset',0))['values']
    from experiments.mcqa_at_anchor_comparison.engine import projection
    restored=[];errors=[];probabilities=[];ids=[];ties=[]
    for start in range(0,len(rows),engine.batch):
        rr=rows[start:start+engine.batch];n=len(rr)
        induced=trace['at_state'][start:start+n].to(engine.device).float();base=original[start:start+n].to(engine.device)
        with torch.no_grad():
            state=projection(induced,base,at_q).to(torch.bfloat16);y=engine.tail(state)[:n]
            score=banks.score(engine,y,[data.answer(r['base']) for r in rr])
        restored.extend(score['correct'].tolist());errors.extend((state.float()-base).norm(dim=-1).cpu().tolist())
        ids.extend(score['token_id'].tolist());ties.extend(score['tied'].tolist());probabilities.append(score['label_probabilities'])
    from experiments.mcqa_at_anchor_comparison.storage import save
    path=Path(root)/'evaluation'/f'{name}_AT_restore.pt'
    if not path.exists():save(path,dict(pair_ids=[r['id'] for r in rows],targets=[data.answer(r['base']) for r in rows],
        correct=torch.tensor(restored),token_id=torch.tensor(ids),tied=torch.tensor(ties),label_probabilities=torch.cat(probabilities),restoration_l2=torch.tensor(errors)))
    result['restoration']=dict(**training.rate(restored),mean_state_l2=float(np.mean(errors)),max_state_l2=max(errors))
    return result,arr

def evaluate_all(root,engine,natural,states):
    root=Path(root);banks.locked(root);locked=read(root/'locks/all_final.json')
    rows=banks.filter_bank(root,'final',natural);crows=banks.filter_bank(root,'c_diagnostic',natural)
    control=controls.filter_controls(root,rows,natural,'final');ccontrol=controls.filter_controls(root,crows,natural,'C')
    refs=uploaded();at_q=refs['AT'].to(engine.device)
    natural.ensure([c for groups in (control,ccontrol) for roles in groups.values() for c in roles.values()])
    for label,rr in [('final',rows),('C',crows)]:natural.layout(rr,label+'_evaluation')
    clean,clean_ok=clean_at(root,engine,rows,control,natural,states,at_q,'clean_final')
    cclean,cclean_ok=clean_at(root,engine,crows,ccontrol,natural,states,at_q,'clean_C')
    results={};primary={};ctraces={};direct_C={}
    for endpoint,done in locked['endpoints'].items():
        config=done['identity'];site=Site(**config['site']);q=load(done['checkpoint'])['payload']['basis'].to(engine.device)
        donor=states.get([r['source'] for r in rows],site)
        direct,arr=put(root,endpoint+'_direct',lambda:training.evaluate(engine,rows,donor,q,site,natural,keep_states=True))
        primary[endpoint]=arr;result=dict(direct=direct,site=config['site'],rank=config['rank'],controls={})
        for role in ('AP_donor0','AP_donor1','recipient0','recipient1','recipient2','recipient3'):
            rr,ix=controls.rows_for(rows,control,role)
            if not rr:result['controls'][role]=None;continue
            source=(dict(values=arr['site_state'][ix].float(),available=arr['available'][ix]) if role.startswith('recipient')
                    else states.get([r['source'] for r in rr],site))
            metric,_=put(root,endpoint+'_'+role,lambda rr=rr,source=source:training.evaluate(engine,rr,source,q,site,natural))
            result['controls'][role]=dict(metrics=metric,parent_indices=ix)
        own=states.get([r['base'] for r in rows],site)
        result['identity'],identity_arr=put(root,endpoint+'_identity',lambda:training.evaluate(engine,targets(rows,True),own,q,site,natural))
        if not identity_arr['correct'].all():raise ValueError('Identity intervention changed a clean answer')
        result['restoration'],_=put(root,endpoint+'_restore',lambda:training.evaluate(engine,rows,donor,q,site,natural,restore=True))
        result['downstream'],at_arr=downstream(root,engine,rows,control,arr,natural,states,at_q,endpoint)
        result['direct_and_AT_replay']=training.rate((arr['correct']&at_arr['correct']).numpy())
        result['AT_replay_given_both_clean_donors_succeed']=training.rate(at_arr['correct'][clean_ok].numpy())
        result['AT_support_overlap']=bool(site.layer==25 and (site.kind=='offset' and site.index==0))
        result['C']={}
        for method in ('standard','gradual2.0'):
            c_source=natural.field([r['source'] for r in crows],'c').float();cq=refs[method].to(engine.device)
            name=f'C_{method}_{site.id}'
            metric,trace=put(root,name+'_direct',lambda:training.evaluate(engine,crows,None,None,site,natural,
                c_donor=c_source,c_q=cq,keep_states=True))
            if method in direct_C and not torch.equal(direct_C[method],trace['token_id']):raise ValueError('Frozen C direct outputs changed across AP sites')
            direct_C[method]=trace['token_id']
            induced=dict(values=trace['site_state'].float(),available=trace['available'])
            rep,replayed=put(root,endpoint+f'_C_{method}_AP_replay',lambda:training.evaluate(engine,crows,induced,q,site,natural,keep_states=True))
            joint=trace['correct']&replayed['correct'];changing=torch.tensor([r['metadata']['from_AP']!=r['metadata']['to_AP'] for r in crows])
            chain,chain_arr=downstream(root,engine,crows,ccontrol,replayed,natural,states,at_q,endpoint+f'_C_{method}')
            result['C'][method]=dict(direct=metric,AP_replay=rep,joint_changing=training.rate(joint[changing].numpy()),
                joint_preservation=training.rate(joint[~changing].numpy()),AT=chain,
                joint_chain=training.rate((joint&chain_arr['correct']).numpy()),
                AT_replay_given_both_clean_donors_succeed=training.rate(chain_arr['correct'][cclean_ok].numpy()))
            ctraces[(endpoint,method)]=joint
        results[endpoint]=result
        write(root/'evaluation'/f'{endpoint}_summary.json',result)
    comparisons={};interpretation={};change=np.array([r['metadata']['from_AP']!=r['metadata']['to_AP'] for r in rows])
    for seed in SEEDS:
        aliases={k:v+f'_S{seed}' for k,v in locked['aliases'].items()};comp={}
        for arm,other in [('shared','baseline'),('edge','baseline'),('middle','baseline'),('distributed','baseline'),('distributed','shared'),('distributed','capacity')]:
            mask=change&np.array([arm not in ('edge','middle') or r['metadata']['to_AP'] in ((0,3) if arm=='edge' else (1,2)) for r in rows])
            comp[arm+'_minus_'+other]=bootstrap(rows,primary[aliases[other]]['correct'],primary[aliases[arm]]['correct'],mask)
        comparisons[str(seed)]=comp
        for arm,endpoint in aliases.items():
            res=results[endpoint];cells=res['direct']['transition']
            passed=all(v['accuracy'] >= (.95 if k.split('->')[0]==k.split('->')[1] else .90) for k,v in cells.items())
            for role in ('AP_donor0','AP_donor1','recipient0','recipient1','recipient2','recipient3'):
                entry=res['controls'][role]
                if entry is None or len(entry['parent_indices'])!=len(rows):passed=False;continue
                for key,v in entry['metrics']['parent_transition'].items():
                    if key.split('->')[0]!=key.split('->')[1] and v['accuracy']<.90:passed=False
            interpretation[arm+f'_S{seed}']=dict(reliable_AP=passed,full_chain_certified=False,native_switching_proven=False)
        for endpoint in aliases.values():
            comparisons[str(seed)][endpoint+'_C_gradual_minus_standard']=bootstrap(crows,ctraces[(endpoint,'standard')],ctraces[(endpoint,'gradual2.0')],
                np.array([r['metadata']['from_AP']!=r['metadata']['to_AP'] for r in crows]))
    useful={}
    for arm,other in [('shared','baseline'),('edge','baseline'),('middle','baseline'),('distributed','baseline'),('distributed','shared'),('distributed','capacity')]:
        useful[arm+'_minus_'+other]=all(
            comparisons[str(seed)][arm+'_minus_'+other]['difference']>=.05 and
            comparisons[str(seed)][arm+'_minus_'+other]['lower']>0 and
            results[locked['aliases'][arm]+f'_S{seed}']['direct']['preservation']['accuracy']>=
            results[locked['aliases'][other]+f'_S{seed}']['direct']['preservation']['accuracy']-.02 for seed in SEEDS)
    result=dict(endpoints=results,comparisons=comparisons,interpretation=interpretation,clean_AT=clean,clean_C_AT=cclean,
        aliases=locked['aliases'],useful_improvement=useful,AP_pairs=len(rows),C_pairs=len(crows),C_training_replicates=1,all_intervention_failures_retained=True)
    write(root/'results.json',result);return result
