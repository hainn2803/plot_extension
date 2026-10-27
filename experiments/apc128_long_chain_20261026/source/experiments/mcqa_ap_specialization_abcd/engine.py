"""Final-token AP projection only: no AT handle, classifier or auxiliary path."""
from contextlib import ExitStack
import gc
import statistics
import time

import torch
from torch.nn import functional as F

from experiments.mcqa_at_anchor_comparison.engine import Engine as BaseEngine,Subspace,projection,checked
from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from experiments.mcqa_at_anchor_comparison.storage import read,write,status
from . import data
from .protocol import PROTOCOL,ARMS

class Engine(BaseEngine):
    def forward(self,contexts,donor=None,q=None,reference=False,full=False):
        self.session.check();ids,mask,_=self.encode(contexts)
        captures={};rr=torch.arange(self.batch,device=self.device)
        before=[len(x._forward_hooks) for x in self.model.model.layers]
        def hook(_m,_i,out):
            h=out[0] if isinstance(out,tuple) else out
            if donor is not None:
                h=h.clone();base=h[rr,-1].float();source=self.pad(donor).float()
                change=(base+(source-base)@q@q.T) if reference else projection(base,source,q)
                h[rr,-1]=change.to(h.dtype)
            captures['ap']=h[rr,-1].float()
            return (h,)+out[1:] if isinstance(out,tuple) else h
        with ExitStack() as stack:
            handle=self.model.model.layers[17].register_forward_hook(hook);stack.callback(handle.remove)
            args=dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False,return_dict=True)
            if full:
                handle=self.model.lm_head.register_forward_pre_hook(lambda _m,args:(args[0].contiguous(),))
                stack.callback(handle.remove)
                handle=self.model.lm_head.register_forward_hook(lambda _m,_i,out:captures.update(raw=out))
                stack.callback(handle.remove);self.model(**args,logits_to_keep=1)
                logits=self.softcap(captures['raw'][:,0])
            else:
                out=self.model.model(**args)
                logits=self.softcap(self.model.lm_head(out.last_hidden_state[:,-1].contiguous()))
        assert before==[len(x._forward_hooks) for x in self.model.model.layers]
        return logits,captures['ap']

    def loss(self,rows,donors,sub,reference=False,full=False):
        logits,_=self.forward([r['base'] for r in rows],donors,sub.basis(),reference,full)
        targets,_=self.targets(rows)
        return F.cross_entropy(logits[:len(rows)],targets),logits[:len(rows)]

    def update(self,rows,donors,sub,opt):
        if len(rows)!=32:raise ValueError('Effective batch must equal 32')
        opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,32,self.batch):
            small=rows[start:start+self.batch]
            loss,_=self.loss(small,donors[start:start+self.batch],sub)
            if not torch.isfinite(loss):raise ValueError('Nonfinite CE')
            (loss*len(small)/32).backward();total+=float(loss.detach())*len(small)/32
        if sub.raw.grad is None or not torch.isfinite(sub.raw.grad).all():raise ValueError('Nonfinite/missing basis gradient')
        norm=float(sub.raw.grad.norm());opt.step()
        if not torch.isfinite(sub.raw).all():raise ValueError('Nonfinite projector')
        return dict(cross_entropy=total,gradient_norm=norm)

@torch.no_grad()
def donors(engine,rows):
    values=[]
    for start in range(0,len(rows),engine.batch):
        small=rows[start:start+engine.batch]
        _,x=engine.forward([r['source'] for r in small]);values.append(x[:len(small)].cpu())
    return torch.cat(values)

def numerical(engine,rows,source):
    n=engine.batch;small=rows[:n];sub=Subspace(17).to(engine.device);q=sub.basis().detach();checks=[]
    with torch.no_grad():
        clean,state=engine.forward([r['base'] for r in small])
        native,ns=engine.forward([r['base'] for r in small],full=True)
        checks.extend([checked('clean_vs_full',clean,native),checked('capture_vs_full',state,ns),
            checked('repeated_clean',engine.forward([r['base'] for r in small])[0],clean),
            checked('identity_patch',engine.forward([r['base'] for r in small],state,q)[0],clean)])
        reverse=list(reversed([r['base'] for r in small]))
        checks.append(checked('clean_reordered_batch',engine.forward(reverse)[0].flip(0),clean))
    a,logits=engine.loss(small,source[:n],sub)
    ga=torch.autograd.grad(a,sub.raw)[0]
    b,ref=engine.loss(small,source[:n],sub,reference=True,full=True)
    gb=torch.autograd.grad(b,sub.raw)[0]
    checks.extend([checked('patch_logits_reference',logits,ref),checked('CE_reference',a,b),checked('basis_gradient_reference',ga,gb)])
    # Compare independent sums at exactly the registered microbatch boundaries.
    gradients=[]
    for start in range(0,32,n):
        loss,_=engine.loss(rows[start:start+n],source[start:start+n],sub)
        gradients.append(torch.autograd.grad(loss*n/32,sub.raw)[0])
    expected=torch.stack(gradients).sum(0)
    opt=torch.optim.Adam(sub.parameters(),lr=.01)
    engine.update(rows,source,sub,opt)
    checks.append(checked('effective_batch_accumulation',sub.raw.grad,expected))
    if n>1:
        short=small[:-1];loss,logits=engine.loss(short,source[:n-1],sub)
        targets,_=engine.targets(short)
        checks.append(checked('filler_excluded_CE',loss,F.cross_entropy(logits,targets)))
        changed=source[:n].clone();changed[-1].fill_(1000)
        ls,_=engine.loss(short,changed[:n-1],sub)
        checks.append(checked('filler_no_effect',ls,loss))
    assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
    return checks

def profile(root,engine):
    if (root/'profile.json').exists():
        value=read(root/'profile.json');engine.batch=value['batch'];return value
    groups=read(root/'groups.json');ss=read(root/'proposals/train_master.slots.json')
    rows=[data.proposal(ss[i*31],data.alternative_order(ss[i*31])[0],groups) for i in range(32)]
    trials=[];valid=[]
    for batch in PROTOCOL['batch_candidates']:
        engine.batch=batch;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            with MemoryWatch() as memory:
                source=donors(engine,rows);checks=numerical(engine,rows,source)
                timings={}
                for arm in ARMS:
                    sub=Subspace(1).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01);times=[]
                    for _ in range(3):
                        torch.cuda.synchronize();start=time.perf_counter();engine.update(rows,source,sub,opt)
                        torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    timings[arm]=statistics.median(times);del sub,opt
                start=time.perf_counter();donors(engine,rows);torch.cuda.synchronize()
                clean_seconds=(time.perf_counter()-start)/32
                q=Subspace(2).to(engine.device).basis().detach()
                start=time.perf_counter()
                with torch.no_grad():
                    for i in range(0,32,batch):
                        yy,_=engine.forward([r['base'] for r in rows[i:i+batch]],source[i:i+batch],q)
                        engine.score(yy,[r['target'] for r in rows[i:i+batch]])
                torch.cuda.synchronize();eval_seconds=(time.perf_counter()-start)/32
            mem=memory.result()
            passed=mem['peak_shared_bytes']==0 and mem['all_process_device_usage_within_capacity']
            trial=dict(batch=batch,passed=passed,checks=checks,updates_seconds=timings,
                clean_seconds_per_prompt=clean_seconds,evaluation_seconds_per_row=eval_seconds,
                memory=mem,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            trials.append(trial)
            if passed:valid.append(trial)
        except torch.OutOfMemoryError as e:
            trials.append(dict(batch=batch,passed=False,reason='CUDA OOM',detail=str(e)))
        write(root/'profile_progress.json',dict(trials=trials),immutable=False)
        status(root,'profiling',batch=batch,passed=trials[-1]['passed'])
        gc.collect();torch.cuda.empty_cache()
    if not valid:raise RuntimeError('No fitting numerically validated physical batch')
    best=max(valid,key=lambda x:x['batch']);engine.batch=best['batch'];source=donors(engine,rows)
    sub=Subspace(1).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01)
    start=time.perf_counter();count=0
    while time.perf_counter()-start<180:
        engine.update(rows,source,sub,opt);count+=1
    sustained=(time.perf_counter()-start)/count
    factor=max(1.,sustained/max(best['updates_seconds'].values()))
    result=dict(**best,trials=trials,sustained_seconds=time.perf_counter()-start,sustained_updates=count,
        sustained_update_seconds=sustained,sustained_factor=factor,allowance=1.25)
    write(root/'execution.json',dict(batch=engine.batch,width=engine.width,projection='FP32 QR',backbone='BF16',
        head='contiguous last-token BF16 GEMM, FP32 softcap',padding='fixed left padding; filler excluded',
        tf32=False,attention='sdpa',full_vocabulary=True,ties_fail=True))
    write(root/'profile.json',result)
    return result
