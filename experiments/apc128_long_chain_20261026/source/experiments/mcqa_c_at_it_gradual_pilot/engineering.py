"""Unchanged numerical checks, frozen teachers, and measured launch gates."""
from collections import Counter
import gc
import json
import os
import subprocess
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.mcqa_c_at_localization import backend
from experiments.mcqa_c_at_localization.reference import jonathan_patch
from experiments.mcqa_c_at_localization.storage import read,write,file_hash,digest,save_torch,checked_torch
from experiments.mcqa_c_at_localization.protocol import COLORS,LABELS
from experiments.mcqa_c_at_prompt_diagnostic.runner import numeric_check
from experiments.mcqa_c_at_it_das.training import tensor_digest
from experiments.mcqa_c_at_it_das.engineering import check
from . import protocol,core


def encode(tokenizer,prompts):
    it=protocol.parent.parent.it
    ids=[tokenizer.encode(it.render(tokenizer,p['text'],'chat_explicit'),add_special_tokens=False) for p in prompts]
    assert all(len(v)==82 and v[0]==2 and v.count(2)==1 for v in ids)
    return torch.tensor(ids,dtype=torch.long)


def device_memory():
    text=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,check=True,timeout=15).stdout
    used,total=[int(s.strip())*1024*1024 for s in text.splitlines()[0].split(',')]
    command=("$x=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples; "
             f"$s=($x | Where-Object {{$_.InstanceName -like 'pid_{os.getpid()}_*'}} | Measure-Object -Property CookedValue -Sum).Sum; "
             "[long]$s")
    shared=int(subprocess.run(['powershell.exe','-NoProfile','-Command',command],capture_output=True,text=True,
                               check=True,timeout=15).stdout.strip())
    return {'time':time.time(),'used_device_bytes_all_processes':used,'physical_device_bytes':total,
            'shared_bytes_this_process':shared,'reserved_bytes':torch.cuda.memory_reserved()}


class MemoryWatch:
    def __init__(self): self.samples=[];self.stop=threading.Event();self.error=None
    def __enter__(self):
        self.samples.append(device_memory())
        def run():
            try:
                while not self.stop.wait(1): self.samples.append(device_memory())
            except BaseException as e: self.error=str(e)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=20)
        self.samples.append(device_memory())
        if self.error: raise RuntimeError('Memory audit unavailable: '+self.error)
    def result(self):
        return {'samples':self.samples,'baseline_shared_bytes':self.samples[0]['shared_bytes_this_process'],
                'peak_shared_bytes':max(s['shared_bytes_this_process'] for s in self.samples),
                'peak_total_device_used_bytes':max(s['used_device_bytes_all_processes'] for s in self.samples),
                'no_added_shared_memory':max(s['shared_bytes_this_process'] for s in self.samples)<=self.samples[0]['shared_bytes_this_process'],
                'all_process_device_usage_within_capacity':all(s['used_device_bytes_all_processes']<=s['physical_device_bytes'] for s in self.samples)}


def clean_features(model,tokenizer,root,prompts,session,label_ids,namespace):
    path=root/'cache'/(namespace+'.pt')
    identity={'registration':file_hash(root/'registration.json'),'ids':[p['id'] for p in prompts], 'batch':8,'sites':[[19,p] for p in (57,58,81)]+[[24,81]]}
    if path.with_suffix('.json').exists():
        receipt=read(path.with_suffix('.json'))
        assert receipt['identity']==identity
        return checked_torch(path,receipt['sha256'])
    out={p:[] for p in (57,58,81,'AT')}
    with torch.no_grad():
        for start in range(0,len(prompts),8):
            session.check()
            chunk=prompts[start:start+8]
            physical=chunk+[chunk[-1]]*(8-len(chunk))
            logits,h=backend.forward(model,encode(tokenizer,physical).to('cuda'),capture=[19,24])
            expected=[label_ids[LABELS.index(p['AT'])] for p in chunk]
            assert logits[:len(chunk)].argmax(-1).cpu().tolist()==expected, 'Locked clean prompt changed; do not refilter'
            for p in (57,58,81): out[p].append(h[19][:len(chunk),p].detach().cpu().clone())
            out['AT'].append(h[24][:len(chunk),81].detach().cpu().clone())
            del logits,h
    result={k:torch.cat(v) for k,v in out.items()}
    save_torch(path,result)
    write(path.with_suffix('.json'),{'identity':identity,'sha256':file_hash(path)},immutable=True)
    return result


def teachers(model,tokenizer,root,anchors,session,progress):
    # Only the frozen historical training proposals and natural predictions are eligible.
    pool=read(root/'teacher/proposals.json')
    known={}
    imported={}
    expected_reg=digest(read(protocol.NATURAL/'registration.json'))
    for path in sorted((protocol.NATURAL/'predictions/train').glob('*/*.json')):
        saved=read(path)
        assert saved['identity']['registration']==expected_reg
        imported[path.as_posix()]=file_hash(path)
        for row in saved['rows']:
            assert row['id'] in saved['identity']['task']['ids']
            if row['id'] in known: assert known[row['id']]==row['accepted']
            known[row['id']]=row['accepted']
    selection=root/'teacher/selection.json'
    labels=read(root/'token_map.json')['label_ids']
    if selection.exists():
        chosen=read(selection)
        assert chosen['pool']==file_hash(root/'teacher/proposals.json')
    else:
        chosen={'pool':file_hash(root/'teacher/proposals.json'),'historical_cache':imported,'banks':{},'disposition':{}}
        with torch.no_grad():
            for name,count in (('fit',512),('validation',256)):
                selected=[]
                counts=Counter()
                rejected=[]
                newly_scored={}
                for p in pool[name]:
                    key=(p['condition'],p['C'],p['AT'])
                    if counts[key]>=count//32: continue
                    if p['id'] not in known:
                        session.check()
                        logits,_=backend.forward(model,encode(tokenizer,[p]*8).to('cuda'))
                        pred=int(logits[0].argmax())
                        known[p['id']]=pred==labels[LABELS.index(p['AT'])]
                        newly_scored[p['id']]={'prediction':pred,'correct':known[p['id']]}
                        del logits
                    if not known[p['id']]:
                        rejected.append(p['id'])
                        continue
                    counts[key]+=1
                    selected.append(p)
                    if len(selected)==count: break
                assert len(selected)==count, 'Finite teacher pool shortfall; do not expand or relax'
                assert len(counts)==32 and set(counts.values())=={count//32}
                chosen['banks'][name]=selected
                selected_ids={p['id'] for p in selected}
                chosen['disposition'][name]={'rejected':rejected,'new_predictions':newly_scored,
                    'unevaluated':[p['id'] for p in pool[name] if p['id'] not in known],
                    'unused_clean':[p['id'] for p in pool[name] if known.get(p['id']) and p['id'] not in selected_ids]}
        assert not {p['id'] for p in chosen['banks']['fit']} & {p['id'] for p in chosen['banks']['validation']}
        write(selection,chosen,immutable=True)
    features={}
    for name,rows in chosen['banks'].items():
        features[name]=clean_features(model,tokenizer,root,rows,session,labels,'teacher_'+name)['AT'].to('cuda').float()
    results={}
    for seed,at_basis in anchors.items():
        path=root/'teacher'/('seed'+seed+'.pt')
        identity={'selection':file_hash(selection),'anchor':tensor_digest(at_basis),'seed':int(seed),'measurement':False}
        if path.with_suffix('.json').exists():
            receipt=read(path.with_suffix('.json'))
            assert receipt['identity']==identity
            best=checked_torch(path,receipt['sha256'])
        else:
            x,xc=(features[n]@at_basis for n in ('fit','validation'))
            y,yc=(torch.tensor([LABELS.index(p['AT']) for p in chosen['banks'][n]],device='cuda') for n in ('fit','validation'))
            mean,std=x.mean(0),x.std(0,unbiased=False).clamp_min(1e-6)
            xn,xcn=(x-mean)/std,(xc-mean)/std
            best,key,history=None,None,[]
            for penalty in (1e-4,1e-3,1e-2):
                torch.manual_seed(int(seed)); torch.cuda.manual_seed_all(int(seed))
                head=torch.nn.Linear(8,4).to('cuda')
                opt=torch.optim.Adam(head.parameters(),lr=.01)
                for epoch in range(1,201):
                    session.check()
                    opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(head(xn),y)+penalty*head.weight.square().sum()
                    assert torch.isfinite(loss)
                    loss.backward();opt.step()
                    with torch.no_grad():
                        z=head(xcn);acc=float((z.argmax(-1)==yc).float().mean());ce=float(F.cross_entropy(z,yc))
                        history.append({'penalty':penalty,'epoch':epoch,'validation_accuracy':acc,'ce':ce})
                        if key is None or (-acc,ce)<key:
                            key=(-acc,ce)
                            w=head.weight.detach()/std.unsqueeze(0)
                            best={'weight':w.cpu().clone(),'bias':(head.bias.detach()-w@mean).cpu().clone(),
                                  'penalty':penalty,'epoch':epoch,'classes':list(LABELS)}
            save_torch(path,best)
            pred=(xc@best['weight'].to('cuda').T+best['bias'].to('cuda')).argmax(-1)
            byclass={label:float((pred[yc==i]==i).float().mean()) for i,label in enumerate(LABELS)}
            receipt={'identity':identity,'sha256':file_hash(path),'history':history,'validation_by_letter':byclass,
                     'validation_accuracy':sum(byclass.values())/4,'training_object_overlap_disclosed':True}
            write(path.with_suffix('.json'),receipt,immutable=True)
        results[seed]={k:best[k].to('cuda').detach() for k in ('weight','bias')}
        assert not any(v.requires_grad for v in results[seed].values())
        progress(root,'teacher_ready',seed=seed,validation_accuracy=receipt['validation_accuracy'])
    return results


def reference_forward(model,ids,patches=(),capture=()):
    replacements=[]
    for patch in patches:
        if patch.basis is None:
            replacements.append(patch)
        else:
            with torch.no_grad():
                _,h=backend.forward(model,ids,capture=[patch.layer])
                base=h[patch.layer][:,patch.position].detach().clone()
                del h
            value=backend.same_shape_reference(base,patch.source,patch.basis,jonathan_patch)
            replacements.append(backend.Patch(patch.layer,patch.position,value))
    return backend.forward(model,ids,replacements,capture)


def numerical(model,data,at_basis,head,batch,session,root):
    rows,ids,sources,targets,classes=data
    checks=[]
    with torch.no_grad():
        native=numeric_check(model,ids[:batch].to('cuda'),'postquery_pilot_physical_shape',session,torch,backend)
    for position in (57,58,81):
        torch.manual_seed(1)
        sub=backend.LearnedSubspace(2304,128).to('cuda')
        args=(model,ids[:batch].to('cuda'),sources[position][:batch].to('cuda'),sub,position,at_basis,head,
              targets[:batch].to('cuda'),classes[:batch].to('cuda'),True)
        terms,logits=core.loss_terms(*args)
        gradients={k:torch.autograd.grad(v,sub.raw,retain_graph=True)[0].detach() for k,v in terms.items()}
        for name,g in gradients.items():
            assert torch.isfinite(g).all() and g.norm()>0, f'Missing {name} gradient at P{position}'
        total=terms['direct']+2*terms['replay']+2*terms['readout']
        coefficients=torch.stack(torch.autograd.grad(total,tuple(terms.values()),retain_graph=True))
        checks.append(check(f'P{position}_loss_coefficients_FP32',coefficients,torch.tensor([1.,2.,2.],device='cuda')))
        g=torch.autograd.grad(total,sub.raw)[0].detach()
        separate=gradients['direct']+2*gradients['replay']+2*gradients['readout']
        # Separate BF16 reverse passes round adjoints before the FP32 sum. They are
        # not a same-arithmetic reference for the one combined backward we train.
        path=root/'engineering/adjoint_rounding'/f'P{position}.pt'
        save_torch(path,{'combined':g.cpu(),'separate':{k:v.cpu() for k,v in gradients.items()}})
        write(path.with_suffix('.json'),{'sha256':file_hash(path),'atol':2e-4,'rtol':2e-4,
            'separate_sum_close':bool(torch.allclose(g,separate,atol=2e-4,rtol=2e-4)),
            'max_abs_difference':float((g-separate).abs().max()),
            'scope':'Different BF16 reverse-accumulation order; informational, not a replacement for the trusted combined-gradient check'},immutable=True)
        direct=logits.detach().clone()
        del terms,logits,total
        trusted,trusted_logits=core.loss_terms(*args,forward=reference_forward)
        checks.append(check(f'P{position}_trusted_direct_logits',direct,trusted_logits))
        ref=torch.autograd.grad(trusted['direct']+2*trusted['replay']+2*trusted['readout'],sub.raw)[0]
        checks.append(check(f'P{position}_trusted_combined_gradient',g,ref))
        del trusted,trusted_logits,args,sub,ref,g,direct,gradients
        gc.collect();torch.cuda.empty_cache()
    # Compare effective-32 accumulation to an explicit sum of the same physical-shaped graphs.
    torch.manual_seed(1)
    sub=backend.LearnedSubspace(2304,128).to('cuda')
    expected=torch.zeros_like(sub.raw)
    for start in range(0,32,batch):
        session.check()
        sl=slice(start,start+batch)
        terms,_=core.loss_terms(model,ids[sl].to('cuda'),sources[81][sl].to('cuda'),sub,81,at_basis,head,
                                targets[sl].to('cuda'),classes[sl].to('cuda'),True)
        loss=(terms['direct']+2*terms['replay']+2*terms['readout'])*batch/32
        expected+=torch.autograd.grad(loss,sub.raw)[0].detach()
        del loss,terms
    class NoStep:
        def zero_grad(self,set_to_none=True): sub.zero_grad(set_to_none=True)
        def step(self): pass
    core.update(model,sub,NoStep(),ids,sources[81],targets,classes,torch.arange(32),81,at_basis,head,'gradual',batch,session)
    checks.append(check('effective32_matched_shape_accumulation',sub.raw.grad,expected))
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    assert not at_basis.requires_grad and all(not v.requires_grad for v in head.values())
    assert not any(layer._forward_hooks for layer in model.model.layers)
    result={'passed':True,'atol':2e-4,'rtol':2e-4,'checks':checks,'native':native,'batch':batch}
    write(root/'engineering/numerical.json',result,immutable=True)
    return result


def profile(model,data,at_basis,head,session,root,progress):
    rows,ids,sources,targets,classes=data
    total=torch.cuda.get_device_properties(0).total_memory
    trials=[]
    selected=None
    for batch in (8,4,2,1):
        try:
            session.check();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(1)
            sub=backend.LearnedSubspace(2304,128).to('cuda');opt=torch.optim.Adam(sub.parameters(),lr=.01)
            with MemoryWatch() as watcher:
                core.update(model,sub,opt,ids,sources[81],targets,classes,torch.arange(32),81,at_basis,head,'gradual',batch,session)
            peak=torch.cuda.max_memory_reserved()
            memory=watcher.result()
            fits=peak<=total and memory['no_added_shared_memory'] and memory['all_process_device_usage_within_capacity']
            trials.append({'batch':batch,'peak_reserved_bytes':peak,'device_bytes':total,'memory_audit':memory,
                           'status':'passed' if fits else 'memory_audit_not_passed'})
            if fits: selected=batch
            del sub,opt
        except torch.cuda.OutOfMemoryError:
            trials.append({'batch':batch,'status':'CUDA_OOM_not_validated'})
        gc.collect();torch.cuda.empty_cache()
        write(root/'engineering/profile_trials.json',trials)
        if selected is not None: break
    assert selected is not None,'No fitting microbatch within dedicated memory'
    numerical(model,data,at_basis,head,selected,session,root)
    times=[]
    for position in (57,58,81):
        for method in ('standard','gradual'):
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(1)
            sub=backend.LearnedSubspace(2304,128).to('cuda');opt=torch.optim.Adam(sub.parameters(),lr=.01)
            seconds=[]
            for _ in range(3):
                torch.cuda.synchronize();start=time.monotonic()
                core.update(model,sub,opt,ids,sources[position],targets,classes,torch.arange(32),position,at_basis,head,method,selected,session)
                torch.cuda.synchronize();seconds.append(time.monotonic()-start)
            times.append({'position':position,'method':method,'update_seconds':float(np.median(seconds)),
                          'samples':seconds,'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            assert times[-1]['peak_reserved_bytes']<=total
            del sub,opt
    # Use the normal safety monitor for three minutes of representative sustained work.
    start=time.monotonic();torch.manual_seed(1)
    sub=backend.LearnedSubspace(2304,128).to('cuda');opt=torch.optim.Adam(sub.parameters(),lr=.01)
    sustained=[]
    with MemoryWatch() as watcher:
        while time.monotonic()-start<180:
            torch.cuda.synchronize();tick=time.monotonic()
            core.update(model,sub,opt,ids,sources[81],targets,classes,torch.arange(32),81,at_basis,head,'gradual',selected,session)
            torch.cuda.synchronize();sustained.append(time.monotonic()-tick)
    sustained_memory=watcher.result()
    assert sustained_memory['no_added_shared_memory'] and sustained_memory['all_process_device_usage_within_capacity'], 'Sustained memory audit failed'
    factor=max(1.,float(np.quantile(sustained,.9))/next(t['update_seconds'] for t in times if t['position']==81 and t['method']=='gradual'))
    update_seconds=sum(t['update_seconds'] for t in times)*500*factor
    with torch.no_grad():
        torch.cuda.synchronize();tick=time.monotonic()
        for _ in range(5):
            logits,_=backend.forward(model,ids[:selected].to('cuda'))
            del logits
        torch.cuda.synchronize();forward_seconds=(time.monotonic()-tick)/5
    evaluation=6*5*(384+768)/selected*2*forward_seconds
    snapshot_state={'identity':{'profile_scratch':True},'step':0,'updates':[]}
    tick=time.monotonic()
    for _ in range(3): core.save(root/'engineering/scratch_checkpoints',snapshot_state,sub,opt)
    checkpoint_seconds=(time.monotonic()-tick)/3
    preparation_elapsed=time.monotonic()-session.started
    pilot=(update_seconds+evaluation+preparation_elapsed+6*32*checkpoint_seconds)*1.25
    continuation=(3*960*sum(max(t['update_seconds'] for t in times if t['method']==m) for m in ('standard','gradual'))*factor
                 +6*21*(1536+768)/selected*2*forward_seconds+1536/8*forward_seconds+6*80*checkpoint_seconds)*1.25
    # Include the finite worst-case fresh filter, six selected handles, independent
    # heads and structural/replay checks in the complete-local-work gate.
    confirmation=(393216/selected*forward_seconds+300000/selected*forward_seconds+6*768/selected*forward_seconds)*1.25
    result={'batch':selected,'trials':trials,'times':times,'sustained_seconds':time.monotonic()-start,
            'sustained_samples':sustained,'sustained_factor':factor,'forward_seconds':forward_seconds,
            'pilot_hours_with_allowance':pilot/3600,'stage2_hours_with_allowance':continuation/3600,
            'confirmation_upper_hours_with_allowance':confirmation/3600,
            'combined_hours_with_allowance':(pilot+continuation+confirmation)/3600,'pilot_local_allowed':pilot<=12*3600,
            'conditional_stage2_local_allowed':pilot+continuation+confirmation<=12*3600,'evaluation_included':True,
            'measured_preparation_seconds':preparation_elapsed,'measured_checkpoint_seconds':checkpoint_seconds,
            'sustained_memory_audit':sustained_memory,'confirmation_included_in_full_work_gate':True,
            'numerical_sha256':file_hash(root/'engineering/numerical.json')}
    write(root/'engineering/profile.json',result,immutable=True)
    progress(root,'profile_complete',**{k:v for k,v in result.items() if k in ('batch','pilot_hours_with_allowance','stage2_hours_with_allowance','combined_hours_with_allowance')})
    return result
