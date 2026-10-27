"""Matched-arithmetic single-token and four-token DAS interventions."""
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path
import gc
import time
import torch
from torch.nn import functional as F
from experiments.mcqa_at_anchor_comparison.engine import Engine as BaseEngine,Subspace,projection,checked
from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,digest,status
from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from .sites import Site,catalog,structural_positions,gather,scatter
from .protocol import PROTOCOL
from . import data

class Engine(BaseEngine):
    def __init__(self,root,session):
        super().__init__(root,session);self.structures={}

    def positions(self,contexts,site):
        positions=[];valid=[]
        cs=list(contexts)+[contexts[0]]*(self.batch-len(contexts))
        for c in cs:
            if c['id'] not in self.structures:
                tokenized=self.tokenizer(c['prompt'],return_offsets_mapping=True,add_special_tokens=True)
                self.structures[c['id']]=(len(tokenized['input_ids']),structural_positions(c,tokenized['offset_mapping']))
            n,struct=self.structures[c['id']]
            if site.kind=='offset':p=[self.width-1-site.index];ok=site.index<n
            else:
                kind=site.kind.removeprefix('joint_');values=struct[kind]
                p=[self.width-n+v for v in (values if site.count==4 else [values[site.index]])];ok=True
            positions.append(p);valid.append(ok)
        return (torch.tensor(positions,device=self.device),torch.tensor(valid,device=self.device))

    def tail(self,state):
        self.session.check();state=self.pad(state).to(torch.bfloat16)
        carrier=torch.zeros(self.batch,self.width,2304,dtype=torch.bfloat16,device=self.device)
        carrier[:,-1]=state
        hidden=self.model.model.norm(carrier)
        return self.softcap(self.model.lm_head(hidden[:,-1:,:])[:,0])

    def forward(self,contexts,donor=None,q=None,site=None,restore=False,c_donor=None,c_q=None,
                at_donor=None,at_q=None,at_restore=None,capture_layers=(),full=False,reference=False):
        self.session.check();ids,mask,cp=self.encode(contexts);rr=torch.arange(self.batch,device=self.device)
        site=site or Site(17,'offset',0);pos,available=self.positions(contexts,site)
        captures={};before=[len(l._forward_hooks) for l in self.model.model.layers]
        source=None
        if donor is not None:
            source=self.pad(donor['values']).float();available=available&self.pad(donor['available']).bool()
        def swap(b,s,basis):
            return b+((s-b)@basis)@basis.T if reference else projection(b,s,basis)
        def hook(layer):
            def apply(_m,_i,out):
                h=out[0] if isinstance(out,tuple) else out
                if layer==0:
                    captures['c']=h[rr,cp].float()
                    if c_donor is not None:
                        h=h.clone();h[rr,cp]=swap(h[rr,cp].float(),self.pad(c_donor).float(),c_q).to(h.dtype)
                if layer==site.layer:
                    original=gather(h,pos).float();captures['original_site']=original
                    if source is not None:
                        changed=swap(original,source,q) if q is not None else source
                        h=scatter(h,pos,changed,available)
                        if restore:
                            restored=swap(gather(h,pos).float(),original,q)
                            h=scatter(h,pos,restored,available)
                    captures['site']=gather(h,pos).float();captures['available']=available
                if layer==17:captures['ap']=h[rr,-1].float()
                if layer==25:
                    captures['at']=h[rr,-1].float()
                    if at_donor is not None:
                        h=h.clone();h[rr,-1]=swap(h[rr,-1].float(),self.pad(at_donor).float(),at_q).to(h.dtype)
                    if at_restore is not None:
                        h=h.clone();h[rr,-1]=swap(h[rr,-1].float(),self.pad(at_restore).float(),at_q).to(h.dtype)
                    captures['at_after_override']=h[rr,-1].float()
                if layer in capture_layers:captures.setdefault('layers',{})[layer]=h.detach().cpu().to(torch.bfloat16)
                return (h,)+out[1:] if isinstance(out,tuple) else h
            return apply
        with ExitStack() as stack:
            for layer in sorted({0,17,25,site.layer,*capture_layers}):
                handle=self.model.model.layers[layer].register_forward_hook(hook(layer));stack.callback(handle.remove)
            args=dict(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False,return_dict=True)
            if full:
                def head(_m,_i,out):captures['raw']=out[:,0]
                handle=self.model.lm_head.register_forward_hook(head);stack.callback(handle.remove)
                native=self.model(**args,logits_to_keep=1).logits[:,-1];raw=captures['raw']
            else:
                hidden=self.model.model(**args).last_hidden_state
                raw=self.model.lm_head(hidden[:,-1:,:])[:,0];cap=self.model.config.final_logit_softcapping
                native=torch.tanh(raw/cap)*cap
        assert before==[len(l._forward_hooks) for l in self.model.model.layers]
        captures['native']=native
        return self.softcap(raw),captures

    def update(self,rows,donor,sub,opt,site):
        if len(rows)!=32:raise ValueError('Effective batch must equal 32')
        opt.zero_grad(set_to_none=True);total=0.
        for start in range(0,32,self.batch):
            small=rows[start:start+self.batch];n=len(small)
            states={k:v[start:start+n] for k,v in donor.items()}
            logits,_=self.forward([r['base'] for r in small],states,sub.basis(),site)
            ids=torch.tensor([self.label_ids[self.letters.index(r['target'])] for r in small],device=self.device)
            loss=F.cross_entropy(logits[:n],ids)*n/32
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();total+=float(loss.detach())
        if sub.raw.grad is None or not torch.isfinite(sub.raw.grad).all():raise ValueError('Invalid projector gradient')
        norm=float(sub.raw.grad.norm());opt.step()
        if not torch.isfinite(sub.raw).all():raise ValueError('Invalid projector state')
        return dict(cross_entropy=total,gradient_norm=norm)

class StateCache:
    """Layer-chunked natural states; never cache intervened states as natural."""
    def __init__(self,root,engine):
        self.root=Path(root);self.engine=engine;self.folder=self.root/'states';self.index={}
        for p in sorted(self.folder.glob('*.ids.json')):
            value=read(p)
            for ident,i in value['index'].items():
                if ident in self.index:raise ValueError('Duplicate natural state cache ID')
                self.index[ident]=(value['chunk'],i)
        self.next=len(list(self.folder.glob('*.ids.json')))

    @lru_cache(maxsize=4)
    def chunk(self,layer,chunk):
        return load(self.folder/f'L{layer:02d}_{chunk:06d}.pt')

    def ensure(self,contexts):
        todo=list({c['id']:c for c in contexts if c['id'] not in self.index}.values())
        for start in range(0,len(todo),self.engine.batch):
            cs=todo[start:start+self.engine.batch];n=len(cs)
            with torch.no_grad():_,states=self.engine.forward(cs,capture_layers=tuple(range(26)))
            chunk=self.next
            for layer,h in states['layers'].items():
                path=self.folder/f'L{layer:02d}_{chunk:06d}.pt'
                if path.exists():
                    if not torch.equal(load(path),h[:n]):raise ValueError('Partial natural cache differs on resume')
                else:save(path,h[:n])
            index={c['id']:i for i,c in enumerate(cs)}
            write(self.folder/f'{chunk:06d}.ids.json',dict(chunk=chunk,index=index))
            self.index.update({ident:(chunk,i) for ident,i in index.items()});self.next+=1

    def get(self,contexts,site):
        self.ensure(contexts);values=[];valid=[]
        for start in range(0,len(contexts),self.engine.batch):
            cs=contexts[start:start+self.engine.batch]
            pos,ok=self.engine.positions(cs,site);pos=pos.cpu()
            for j,c in enumerate(cs):
                chunk,i=self.index[c['id']];h=self.chunk(site.layer,chunk)[i]
                values.append(h[pos[j]].flatten());valid.append(bool(ok[j]))
        return dict(values=torch.stack(values).float(),available=torch.tensor(valid))

def numerical(engine,rows,site,rank):
    cs=[r['base'] for r in rows];ss=[r['source'] for r in rows];n=len(rows);checks=[]
    with torch.no_grad():
        y,b=engine.forward(cs,site=site);ref,full=engine.forward(cs,site=site,full=True)
        _,s=engine.forward(ss,site=site)
        checks += [checked('clean_full',y,ref),checked('native_full',b['native'],full['native']),
                   checked('repeat',engine.forward(cs,site=site)[0],y),checked('cached_tail',engine.tail(b['at']),y)]
    donor=dict(values=s['site'][:n].detach().cpu(),available=s['available'][:n].cpu())
    sub=Subspace(1,width=site.count*2304,rank=rank).to(engine.device)
    ids=torch.tensor([engine.label_ids[engine.letters.index(r['target'])] for r in rows],device=engine.device)
    ya,_=engine.forward(cs,donor,sub.basis(),site);la=F.cross_entropy(ya[:n],ids);ga=torch.autograd.grad(la,sub.raw)[0]
    yr,_=engine.forward(cs,donor,sub.basis(),site,full=True,reference=True);lr=F.cross_entropy(yr[:n],ids);gr=torch.autograd.grad(lr,sub.raw)[0]
    checks += [checked('logits',ya,yr),checked('loss',la,lr),checked('gradient',ga,gr)]
    _,tail_trace=engine.forward(cs,donor,sub.basis(),site)
    tail_loss=F.cross_entropy(engine.tail(tail_trace['at'])[:n],ids)
    tail_grad=torch.autograd.grad(tail_loss,sub.raw)[0]
    checks += [checked('cached_tail_loss',tail_loss,la),checked('cached_tail_gradient',tail_grad,ga)]
    with torch.no_grad():
        identity=dict(values=b['site'][:n].cpu(),available=b['available'][:n].cpu())
        checks.append(checked('identity',engine.forward(cs,identity,sub.basis(),site)[0],y))
    assert all(not p.requires_grad and p.grad is None for p in engine.model.parameters())
    return checks,donor

def profile(root,engine):
    root=Path(root)
    if (root/'profile.json').exists():return read(root/'profile.json')
    groups=read(root/'groups.json');slots=data.slots('train');rows=[data.proposal(s,0,groups)[0] for s in slots[::max(1,len(slots)//32)][:32]]
    trials=[];valid=[]
    for batch in PROTOCOL['batches']:
        engine.batch=batch;checks=[];times=[];gc.collect();torch.cuda.empty_cache()
        try:
            with MemoryWatch() as memory:
                for layer in (0,13,25):
                    for kind,rank in [('offset',128),('offset',512),('joint_label',128),('joint_color',128)]:
                        site=Site(layer,kind,0);c,_=numerical(engine,rows[:batch],site,rank)
                        checks.append(dict(site=site.record(),rank=rank,checks=c))
                        donor_parts=[];t=time.perf_counter()
                        with torch.no_grad():
                            for start in range(0,32,batch):
                                rr=rows[start:start+batch];_,h=engine.forward([r['source'] for r in rr],site=site)
                                donor_parts.append(dict(values=h['site'][:len(rr)].cpu(),available=h['available'][:len(rr)].cpu()))
                        torch.cuda.synchronize();natural=(time.perf_counter()-t)/32
                        donor={k:torch.cat([v[k] for v in donor_parts]) for k in ('values','available')}
                        sub=Subspace(1,width=site.count*2304,rank=rank).to(engine.device);opt=torch.optim.Adam(sub.parameters(),lr=.01)
                        engine.update(rows,donor,sub,opt,site);torch.cuda.synchronize();t=time.perf_counter()
                        for _ in range(3):engine.update(rows,donor,sub,opt,site)
                        torch.cuda.synchronize();update=(time.perf_counter()-t)/3;t=time.perf_counter()
                        with torch.no_grad():
                            for start in range(0,32,batch):
                                engine.forward([r['base'] for r in rows[start:start+batch]],
                                    {k:v[start:start+batch] for k,v in donor.items()},sub.basis(),site)
                        torch.cuda.synchronize();inference=(time.perf_counter()-t)/32
                        times.append(dict(layer=layer,kind=kind,rank=rank,update=update,inference=inference,natural=natural))
                        del opt,sub
            evidence=memory.result();passed=evidence['no_added_shared_memory'] and evidence['all_process_device_usage_within_capacity']
            trial=dict(batch=batch,passed=passed,times=times,checks=checks,memory=evidence)
            if passed:valid.append(trial)
        except torch.cuda.OutOfMemoryError as error:
            trial=dict(batch=batch,passed=False,reason=str(error));gc.collect();torch.cuda.empty_cache()
        trials.append(trial);write(root/'profile_progress.json',trials,immutable=False)
        status(root,'profiling',batch=batch,passed=trial['passed'])
    if not valid:raise RuntimeError('No common numerically validated physical batch')
    chosen=max(valid,key=lambda x:x['batch']);engine.batch=chosen['batch']
    execution=dict(batch=engine.batch,width=engine.width,padding='fixed left padding',position_ids='mask cumulative -1',
        backbone='BF16 SDPA',projection='FP32 QR',tf32=False,head='Native rank-three BF16 GEMM; FP32 softcap; native correctness also required',tolerances=[2e-4,2e-4])
    write(root/'execution.json',execution)
    write(root/'sites.json',dict(single=[s.record() for s in catalog(engine.width)],distributed=[s.record() for s in catalog(engine.width,True)]))
    result=dict(chosen,trials=trials);write(root/'profile.json',result);return result
