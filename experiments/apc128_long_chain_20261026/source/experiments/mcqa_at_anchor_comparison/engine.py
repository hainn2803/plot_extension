"""Matched-layout differentiable interventions and real-model profiling."""
from contextlib import ExitStack
import gc
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.hai_c_at_evaluation import core
from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from . import data
from .storage import read, write, file_hash, status


class Subspace(nn.Module):
    def __init__(self, seed, width=2304, rank=128, dtype=torch.float32):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        raw = torch.linalg.qr(torch.randn(width,rank,generator=g,dtype=dtype),mode='reduced').Q
        self.raw = nn.Parameter(raw)

    def basis(self):
        return torch.linalg.qr(self.raw,mode='reduced').Q


def projection(base,source,q):
    return base + ((source-base) @ q) @ q.T


def answer_label_ids(tokenizer):
    # Match the existing Hai letter_token_id convention, including whitespace.
    ids=[]
    for letter in data.LETTERS:
        for text in (' '+letter,letter):
            tokens=tokenizer.encode(text,add_special_tokens=False)
            if len(tokens)==1:
                ids.append(tokens[0]);break
        else:raise ValueError('Answer letter is not a single token: '+letter)
    return tuple(ids)


def checked(name, actual, expected):
    a,b = actual.detach().float(),expected.detach().float()
    passed = torch.allclose(a,b,atol=2e-4,rtol=2e-4)
    result = dict(name=name,passed=passed,max_absolute_error=float((a-b).abs().max()),atol=2e-4,rtol=2e-4)
    if not passed:
        raise ValueError('Numerical check failed: '+str(result))
    return result


class Engine:
    letters=data.LETTERS
    tie_tolerance=0.0

    def __init__(self, root, session):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        self.root,self.session = root,session
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        self.device=torch.device('cuda:0')
        self.tokenizer=AutoTokenizer.from_pretrained(data.parent.MODEL,local_files_only=True)
        self.label_ids=answer_label_ids(self.tokenizer)
        assert len(set(self.label_ids))==26 and all(self.decode(i).strip()==x for i,x in zip(self.label_ids,self.letters))
        self.model=AutoModelForCausalLM.from_pretrained(data.parent.MODEL,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').to(self.device).eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        groups=read(root/'groups.json')
        prefix=max(len(self.tokenizer.encode(g['template'].format(color=c)))
            for collection in groups.values() for g in collection for c in data.COLORS)
        row=max(len(self.tokenizer.encode('\n'+a+'. '+c,add_special_tokens=False)) for a in self.letters for c in data.COLORS)
        suffix=len(self.tokenizer.encode('\nAnswer:',add_special_tokens=False))
        self.width=prefix+4*row+suffix
        self.batch=1
        self.tokens={}
        self.session.check()
        if (root/'execution.json').exists():
            execution=read(root/'execution.json')
            if execution['width']!=self.width: raise ValueError('Frozen padding width changed')
            self.batch=execution['batch']

    def decode(self,i):
        return self.tokenizer.decode([int(i)])

    def encode(self,contexts):
        for c in contexts:
            if c['id'] not in self.tokens:
                t=self.tokenizer(c['prompt'],return_offsets_mapping=True,add_special_tokens=True)
                positions=[i for i,(a,b) in enumerate(t['offset_mapping']) if a<=c['color_span'][0]<b]
                if len(positions)!=1 or len(t['input_ids'])>self.width:
                    raise ValueError('Literal color alignment/padding contract failed')
                self.tokens[c['id']]=(t['input_ids'],positions[0])
        ids=torch.full((self.batch,self.width),self.tokenizer.pad_token_id,dtype=torch.long,device=self.device)
        mask=torch.zeros_like(ids); cp=[]
        if not 0<len(contexts)<=self.batch: raise ValueError('Unexpected physical batch')
        padded=list(contexts)+[contexts[0]]*(self.batch-len(contexts))
        for i,c in enumerate(padded):
            toks,pos=self.tokens[c['id']]; n=len(toks)
            ids[i,-n:]=torch.tensor(toks,device=self.device); mask[i,-n:]=1
            cp.append(self.width-n+pos)
        return ids,mask,torch.tensor(cp,device=self.device)

    def pad(self,x):
        x=x.to(self.device)
        return x if len(x)==self.batch else torch.cat([x,x[:1].expand(self.batch-len(x),*x.shape[1:])])

    def softcap(self,raw):
        raw=raw.float(); cap=self.model.config.final_logit_softcapping
        return torch.tanh(raw/cap)*cap if cap is not None else raw

    def tail(self,state):
        state=self.pad(state).to(torch.bfloat16)
        hidden=self.model.model.norm(state[:,None,:])[:,0]
        return self.softcap(self.model.lm_head(hidden))

    def forward(self, contexts, c_donor=None,c_q=None,at_donor=None,at_q=None,reference=False):
        self.session.check()
        ids,mask,cp=self.encode(contexts)
        rr=torch.arange(self.batch,device=self.device)
        captures={}; before=[len(x._forward_hooks) for x in self.model.model.layers]
        def hook(layer,pos,donor,q,name):
            def apply(_m,_i,output):
                h=output[0] if isinstance(output,tuple) else output
                h=h.clone()
                if donor is not None:
                    b=h[rr,pos].float(); s=self.pad(donor).float()
                    if reference:
                        replacement=b+(s-b)@q@q.T
                    else:
                        replacement=projection(b,s,q)
                    h[rr,pos]=replacement.to(h.dtype)
                captures[name]=h[rr,pos].float()
                return (h,)+output[1:] if isinstance(output,tuple) else h
            return apply
        with ExitStack() as stack:
            for layer,pos,donor,q,name in [(0,cp,c_donor,c_q,'c'),(25,self.width-1,at_donor,at_q,'at')]:
                handle=self.model.model.layers[layer].register_forward_hook(hook(layer,pos,donor,q,name))
                stack.callback(handle.remove)
            arguments=dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False,return_dict=True)
            if reference:
                # A strided Bx1xH head selects a different BF16 batched GEMM.
                # Contiguous Bx1xH matches the registered BxH head arithmetic.
                pre=self.model.lm_head.register_forward_pre_hook(lambda _m,args:(args[0].contiguous(),))
                stack.callback(pre.remove)
                def capture_raw(_m,_i,out): captures['raw']=out
                handle=self.model.lm_head.register_forward_hook(capture_raw);stack.callback(handle.remove)
                self.model(**arguments,logits_to_keep=1)
                logits=self.softcap(captures['raw'][:,0])
            else:
                out=self.model.model(**arguments)
                logits=self.softcap(self.model.lm_head(out.last_hidden_state[:,-1]))
        assert before==[len(x._forward_hooks) for x in self.model.model.layers]
        return logits,captures['c'],captures['at']

    def targets(self,rows):
        classes=torch.tensor([self.letters.index(r['target']) for r in rows],device=self.device)
        targets=torch.tensor([self.label_ids[i] for i in classes.tolist()],device=self.device)
        return targets,classes

    def losses(self,kind,rows,sub,states,anchor=None,teacher=None,reference=False):
        n=len(rows); q=sub.basis(); targets,classes=self.targets(rows)
        if kind=='AT':
            b=self.pad(states['base_at']).float(); s=self.pad(states['source_at']).float()
            changed=projection(b,s,q).to(torch.bfloat16)
            logits=self.tail(changed)
            return {'direct':F.cross_entropy(logits[:n],targets)},logits[:n]
        logits,_,induced=self.forward([r['base'] for r in rows],c_donor=states['source_c'],c_q=q,reference=reference)
        terms={'direct':F.cross_entropy(logits[:n],targets)}
        if kind=='gradual':
            b=self.pad(states['base_at']).float()
            if reference:
                replay,_,_=self.forward([r['base'] for r in rows],at_donor=induced,at_q=anchor,reference=True)
            else:
                replay=self.tail(projection(b,induced,anchor).to(torch.bfloat16))
            terms['replay']=F.cross_entropy(replay[:n],targets)
            scores=(induced[:n]@anchor)@teacher['weight'].T+teacher['bias']
            terms['readout']=F.cross_entropy(scores,classes)
        return terms,logits[:n]

    def update(self,kind,rows,sub,opt,states,anchor=None,teacher=None):
        if len(rows)!=32: raise ValueError('Effective batch must equal 32')
        opt.zero_grad(set_to_none=True); totals={}
        for start in range(0,32,self.batch):
            self.session.check()
            chunk=rows[start:start+self.batch]
            terms,_=self.losses(kind,chunk,sub,{k:v[start:start+self.batch] for k,v in states.items()},anchor,teacher)
            total=sum(terms.values())*len(chunk)/32
            if not torch.isfinite(total): raise ValueError('Nonfinite loss')
            total.backward()
            for key,v in terms.items(): totals[key]=totals.get(key,0.)+float(v.detach())*len(chunk)/32
        if sub.raw.grad is None or not torch.isfinite(sub.raw.grad).all(): raise ValueError('Invalid projector gradient')
        norm=float(sub.raw.grad.norm());opt.step()
        if not torch.isfinite(sub.raw).all(): raise ValueError('Nonfinite projector')
        return dict(losses=totals,gradient_norm=norm)

    def score(self,logits,targets):
        return core.score(logits,targets,self)


def sample_rows(root,n=32):
    groups=read(root/'groups.json'); slots=read(root/'proposals/c_train.slots.json')
    selected=[slots[(i%4)*832+i//4] for i in range(n)]
    return [data.proposal(s,0,groups) for s in selected]


def sample_states(engine,rows):
    output={k:[] for k in ('base_at','source_at','source_c')}
    with torch.no_grad():
        for start in range(0,len(rows),engine.batch):
            chunk=rows[start:start+engine.batch]; n=len(chunk)
            _,_,ba=engine.forward([r['base'] for r in chunk])
            _,sc,sa=engine.forward([r['source'] for r in chunk])
            for key,value in [('base_at',ba),('source_at',sa),('source_c',sc)]: output[key].append(value[:n].detach().cpu())
    return {k:torch.cat(v) for k,v in output.items()}


def numerical(engine,rows,states):
    n=engine.batch; chunk=rows[:n]; checks=[]
    anchor=Subspace(37).to(engine.device).basis().detach()
    g=torch.Generator().manual_seed(5)
    teacher=dict(weight=torch.randn(26,128,generator=g).to(engine.device)*.01,bias=torch.zeros(26,device=engine.device))
    with torch.no_grad():
        clean,c,at=engine.forward([r['base'] for r in chunk])
        ref,cr,ar=engine.forward([r['base'] for r in chunk],reference=True)
        for name,a,b in [('clean',clean,ref),('C_capture',c,cr),('AT_capture',at,ar),('cached_tail',engine.tail(at),clean)]: checks.append(checked(name,a,b))
        checks.append(checked('repeat',engine.forward([r['base'] for r in chunk])[0],clean))
        checks.append(checked('identity_C',engine.forward([r['base'] for r in chunk],c_donor=c,c_q=anchor)[0],clean))
        checks.append(checked('identity_AT',engine.forward([r['base'] for r in chunk],at_donor=at,at_q=anchor)[0],clean))
        donated=states['source_at'][:n].to(engine.device)
        full=engine.forward([r['base'] for r in chunk],at_donor=donated,at_q=anchor,reference=True)[0]
        checks.append(checked('natural_AT_full_vs_tail',engine.tail(projection(at,donated,anchor).to(torch.bfloat16)),full))
    for kind in ('standard','gradual'):
        sub=Subspace(17).to(engine.device)
        local={k:v[:n] for k,v in states.items()}
        terms,y=engine.losses(kind,chunk,sub,local,anchor,teacher)
        loss=sum(terms.values()); grad=torch.autograd.grad(loss,sub.raw)[0]
        terms2,y2=engine.losses(kind,chunk,sub,local,anchor,teacher,reference=True)
        grad2=torch.autograd.grad(sum(terms2.values()),sub.raw)[0]
        checks.extend([checked(kind+'_logits',y,y2),checked(kind+'_loss',loss,sum(terms2.values())),checked(kind+'_gradient',grad,grad2)])
        del terms,terms2,y,y2,grad,grad2,sub
    sub=Subspace(11).to(engine.device)
    b=states['base_at'][:n].to(engine.device); s=states['source_at'][:n].to(engine.device)
    y=engine.tail(projection(b,s,sub.basis()).to(torch.bfloat16)); targets,_=engine.targets(chunk)
    gr=torch.autograd.grad(F.cross_entropy(y,targets),sub.raw)[0]
    yf=engine.forward([r['base'] for r in chunk],at_donor=s,at_q=sub.basis(),reference=True)[0]
    gf=torch.autograd.grad(F.cross_entropy(yf,targets),sub.raw)[0]
    checks.append(checked('AT_gradient_cached_vs_full',gr,gf))
    # Match microbatch graph arithmetic exactly, including filler exclusion.
    sub=Subspace(29).to(engine.device);local={k:v[:n] for k,v in states.items()}
    terms,_=engine.losses('gradual',chunk,sub,local,anchor,teacher)
    first=torch.autograd.grad(sum(terms.values())*.5,sub.raw)[0]
    terms,_=engine.losses('gradual',chunk,sub,local,anchor,teacher)
    second=torch.autograd.grad(sum(terms.values())*.5,sub.raw)[0]
    terms,_=engine.losses('gradual',chunk,sub,local,anchor,teacher)
    combined=torch.autograd.grad(sum(terms.values()),sub.raw)[0]
    checks.append(checked('two_microbatch_accumulation',first+second,combined))
    if n>1:
        short=chunk[:-1];ss={k:v[:-1] for k,v in local.items()}
        terms,logits=engine.losses('standard',short,sub,ss)
        target,_=engine.targets(short)
        checks.append(checked('filler_excluded_CE',terms['direct'],F.cross_entropy(logits,target)))
    assert not any(p.requires_grad for p in engine.model.parameters())
    return checks


def profile(root,engine):
    if (root/'profile.json').exists():
        result=read(root/'profile.json');engine.batch=result['batch'];return result
    rows=sample_rows(root); trials=[]; passed=[]
    for batch in data.PROTOCOL['batch_candidates']:
        engine.batch=batch;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        try:
            with MemoryWatch() as watch:
                states=sample_states(engine,rows)
                checks=numerical(engine,rows,states)
                anchor=Subspace(37).to('cuda').basis().detach()
                teacher=dict(weight=torch.zeros(26,128,device='cuda'),bias=torch.zeros(26,device='cuda'))
                timing={}
                for kind in ('AT','standard','gradual'):
                    sub=Subspace(1).to('cuda');opt=torch.optim.Adam(sub.parameters(),lr=.01)
                    durations=[]
                    for _ in range(3):
                        torch.cuda.synchronize();start=time.perf_counter()
                        engine.update(kind,rows,sub,opt,states,anchor,teacher)
                        torch.cuda.synchronize();durations.append(time.perf_counter()-start)
                    timing[kind]=statistics.median(durations)
                    del sub,opt
                start=time.perf_counter()
                sample_states(engine,rows)
                torch.cuda.synchronize();clean_seconds=(time.perf_counter()-start)/64
                tail_times=[]
                with torch.no_grad():
                    for _ in range(10):
                        start=time.perf_counter();engine.tail(states['base_at'][:batch]);torch.cuda.synchronize()
                        tail_times.append((time.perf_counter()-start)/batch)
            memory=watch.result()
            if memory['peak_shared_bytes']>0 or not memory['all_process_device_usage_within_capacity']:
                trials.append(dict(batch=batch,passed=False,reason='Physical batch spilled outside dedicated GPU memory',memory=memory))
            else:
                result=dict(batch=batch,checks=checks,updates_seconds=timing,clean_seconds_per_prompt=clean_seconds,
                            tail_seconds_per_row=statistics.median(tail_times),memory=memory,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),passed=True)
                passed.append(result);trials.append(result)
        except torch.OutOfMemoryError as error:
            trials.append(dict(batch=batch,passed=False,reason='CUDA out of memory',detail=str(error)))
        write(root/'profile_progress.json',dict(trials=trials),immutable=False)
        status(root,'profiling',batch=batch,passed=trials[-1]['passed'])
        gc.collect();torch.cuda.empty_cache()
    if not passed: raise RuntimeError('No validated physical batch fits')
    best=max(passed,key=lambda x:x['batch']);engine.batch=best['batch']
    # Sustained scratch updates exercise the slowest backward path and unchanged guard.
    states=sample_states(engine,rows);sub=Subspace(1).to('cuda');opt=torch.optim.Adam(sub.parameters(),lr=.01)
    anchor=Subspace(37).to('cuda').basis().detach();teacher=dict(weight=torch.zeros(26,128,device='cuda'),bias=torch.zeros(26,device='cuda'))
    start=time.perf_counter();updates=0
    while time.perf_counter()-start<180:
        engine.update('gradual',rows,sub,opt,states,anchor,teacher);updates+=1
    factor=max(1.,(time.perf_counter()-start)/updates/best['updates_seconds']['gradual'])
    # Estimate all raw filtering conservatively using two proposal alternatives per slot.
    filtering=sum(data.SIZES.values())*4*best['clean_seconds_per_prompt']
    at_training=6000*best['updates_seconds']['AT']*factor
    at_eval=3*5*3328*best['tail_seconds_per_row']+4*6656*2*best['tail_seconds_per_row']
    c_training=3000*(best['updates_seconds']['standard']+best['updates_seconds']['gradual'])*factor
    c_eval=6*(6*(3328+1664)+26312)*best['clean_seconds_per_prompt']
    structural=(6*26312*7+159744)*best['tail_seconds_per_row']
    result=dict(**best,trials=trials,sustained_seconds=time.perf_counter()-start,sustained_updates=updates,sustained_factor=factor,
        at_remaining_seconds=at_training+at_eval+filtering*.35+180,
        total_remaining_seconds=filtering+at_training+at_eval+c_training+c_eval+structural+600,
        estimate_components=dict(filtering=filtering,AT_training=at_training,AT_evaluation=at_eval,C_training=c_training,C_evaluation=c_eval,structural=structural),
        allowance=1.25,scope='Timing estimate, not a reduced proposal pool; re-estimate after measured filtering')
    write(root/'execution.json',dict(batch=engine.batch,width=engine.width,backbone='BF16',projection='FP32',softcap='FP32',attention='sdpa',tf32=False,
                                  full_vocabulary=True,ties_fail=True,padding='fixed left padding; duplicate filler rows excluded'))
    write(root/'profile.json',result)
    return result
