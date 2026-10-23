"""Portable, guarded GPU replay of the exact exported frozen-handle cases."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
import review
from vendor import core, schema, backends, runtime, references, ap_engine
from vendor.storage import write
from vendor.safety import Budget, cooling
from vendor.tail import Tail

ROOT=Path(__file__).resolve().parent


def save_npz(path,a):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**a)
    os.replace(temporary,path)


def join(parts):
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def compare(a,b,name,checks):
    a,b=a.detach().cpu().float(),b.detach().cpu().float()
    error=float((a-b).abs().max())
    passed=torch.allclose(a,b,atol=2e-4,rtol=2e-4)
    checks.append(dict(name=name,max_error=error,passed=passed,atol=2e-4,rtol=2e-4))
    if not passed:
        raise ValueError('Numerical reference failed: '+name)


def checkpoint(folder,path,protocol):
    expected=protocol['checkpoint_files'][path]
    p=folder/path
    if schema.file_hash(p)!=expected:
        raise ValueError('Checkpoint differs from pinned upload: '+path)
    return torch.load(p,map_location='cpu',weights_only=True)


def projected_seconds(budget,seconds):
    estimate=seconds*1.25
    if budget.elapsed()+estimate>=budget.limit-15:
        raise runtime.Paused('Measured remaining workload plus 25% exceeds GPU-session budget')


@torch.no_grad()
def run_ap(model,tokenizer,args,protocol,budget,checks):
    plan=review.read(ROOT/'data/ap_plan.json')
    ap_engine.ROOT=args.out
    ap_engine.AP=args.checkpoints_root/'results_joint_randomized_standard/das_ap.pt'
    checkpoint(args.checkpoints_root,'results_joint_randomized_standard/das_ap.pt',protocol)
    engine=ap_engine.Engine(model,tokenizer,plan,budget)
    pieces,pred=[],[]
    for start in range(0,len(engine.contexts),32):
        ii=list(range(start,min(start+32,len(engine.contexts))))
        a=engine.forward(ii,capture=True)
        pieces.append(a)
        pred.append(core.score(a['logits'],[engine.contexts[i].answer for i in ii],engine))
    natural={k:torch.cat([p[k] for p in pieces]) for k in ('states','last','color')}
    save_npz(args.out/'ap_natural.npz',join(pred))
    # Independent pinned source functions use the identical 32-row input layout.
    ref=references.selected_functions(ROOT/'vendor/source_references/mcqa_gradual_das.py',['answer_logits','run_full_layer_intervention'])
    ref=references.selected_functions(ROOT/'vendor/source_references/mcqa_gradual_das_c.py',['run_frozen_mediator'],ref)
    ii=list(range(32))
    ids,mask,_=engine.inputs(ii)
    bank=dict(base_input_ids=ids.cpu(),base_attention_mask=mask.cpu(),base_position_by_id={
        'label':torch.tensor([engine.positions[i][0]-(engine.width-len(engine.encoded[i]['input_ids'])) for i in ii]),
        'last_token':torch.tensor([len(engine.encoded[i]['input_ids'])-1 for i in ii])})
    clean=engine.forward(ii)['logits']
    y=model.model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False,return_dict=True)
    compare(clean,ref['answer_logits'](model,y,mask,None),'AP/clean',checks)
    compare(clean,engine.forward(ii)['logits'],'AP/repeat',checks)
    for layer in (0,17,25):
        donor=natural['states'][np.roll(ii,1),layer,0].float()
        a=engine.forward(ii,layer=layer,donor=donor,slots=[0]*32)['logits']
        b=ref['run_full_layer_intervention'](model,bank,donor,layer,None,token_position='label',batch_size=32)
        compare(a,b,'AP/full_L'+str(layer),checks)
    donor=natural['last'][np.roll(ii,1)]
    a=engine.forward(ii,layer=17,donor=donor,subspace=True)['logits']
    b=ref['run_frozen_mediator'](model,bank,donor,engine.q,17,None,batch_size=32,require_grad=False)
    compare(a,b,'AP/frozen_projector',checks)
    rows=plan['rows']
    def operation(name,layer,subspace=False):
        parts=[]
        for start in range(0,len(rows),32):
            batch=rows[start:start+32]
            ii,slot,source=([r[k] for r in batch] for k in ('base','start','source'))
            donor=natural['last'][source] if subspace else natural['states'][source,layer,slot].float()
            out=engine.forward(ii,layer=layer,donor=donor,slots=slot+[slot[0]]*(32-len(slot)),subspace=subspace)
            parts.append(core.score(out['logits'],[r['target'] for r in batch],engine))
        save_npz(args.out/'ap'/(name+'.npz'),join(parts))
        print(json.dumps(dict(task='AP',operation=name,GPU_seconds=budget.elapsed())),flush=True)
    t=time.perf_counter()
    operation('full_L00',0)
    per_layer=time.perf_counter()-t
    projected_seconds(budget,per_layer*26)
    for layer in range(1,26):
        operation('full_L'+str(layer).zfill(2),layer)
    operation('learned_AP_L17',17,True)


class Guarded(backends.HFBackend):
    def __init__(self,*args,root,budget,**kwargs):
        self.root,self.budget=root,budget
        super().__init__(*args,**kwargs)

    def run(self,*args,**kwargs):
        self.budget.tick()
        cooling(self.root,self.budget)
        out=super().run(*args,**kwargs)
        self.budget.tick()
        return out


def observed(backend,indices,targets,source=None,kind=None):
    cc,aa,pred=[],[],[]
    for start in range(0,len(indices),32):
        stop=min(start+32,len(indices))
        kwargs={} if source is None else {kind+'_source':source[start:stop]}
        a=backend.run(indices[start:stop],**kwargs)
        cc.append(a.c)
        aa.append(a.at)
        pred.append(core.score(a.logits,targets[start:stop],backend))
    return dict(c=torch.cat(cc),at=torch.cat(aa),predictions=join(pred))


@torch.no_grad()
def run_c_at(model,tokenizer,args,protocol,budget,checks,validate_only=False):
    contexts=[schema.Context(**c) for c in review.read(ROOT/'data/contexts.json')]
    schedule=review.read(ROOT/'data/c_at_schedule.json')
    ck={m:{v:checkpoint(args.checkpoints_root,folder+'/das_'+v.lower()+'.pt',protocol) for v in ('C','AT')}
        for m,folder in (('standard','results_joint_randomized_standard'),('gradual','results_joint_randomized_gradual2.0'))}
    back={m:Guarded(model,tokenizer,contexts,ck[m]['C'],ck[m]['AT'],protocol['label_token_ids'],root=args.out,budget=budget) for m in ck}
    ii=schedule['base'][800:832]
    ss=schedule['source'][800:832]
    backend=back['standard']
    source=backend.run(ss)
    clean=backend.run(ii)
    compare(clean.logits,backend.run(ii).logits,'C/repeat',checks)
    width=max(len(backend.tokens[i]) for i in ii)
    ids=torch.full((32,width),tokenizer.pad_token_id,dtype=torch.long)
    mask=torch.zeros_like(ids)
    for row,i in enumerate(ii):
        ids[row,-len(backend.tokens[i]):]=torch.tensor(backend.tokens[i])
        mask[row,-len(backend.tokens[i]):]=1
    bank=dict(base_input_ids=ids,base_attention_mask=mask,base_position_by_id={
        'color':torch.tensor([backend.color_positions[i] for i in ii]),
        'last_token':torch.tensor([len(backend.tokens[i])-1 for i in ii])})
    ref=references.selected_functions(ROOT/'vendor/source_references/mcqa_gradual_das.py',['answer_logits'])
    ref=references.selected_functions(ROOT/'vendor/source_references/mcqa_gradual_das_c.py',['run_c_intervention_and_capture','run_frozen_mediator'],ref)
    for method,backend in back.items():
        direct=backend.run(ii,c_source=source.c)
        y,_,at=ref['run_c_intervention_and_capture'](model,bank,source.c,backend.basis,0,17,25,None,batch_size=32,require_grad=False)
        compare(direct.logits,y,method+'/C_reference',checks)
        compare(direct.at,at,method+'/AT_capture',checks)
        replay=backend.run(ii,at_source=direct.at)
        y=ref['run_frozen_mediator'](model,bank,direct.at,backend.at_basis,25,None,batch_size=32,require_grad=False)
        compare(replay.logits,y,method+'/replay_reference',checks)
        tail=Tail(backend,args.out,budget)
        compare(tail.logits(tail.state(clean.at,direct.at),32),replay.logits,method+'/tail_reference',checks)
    if validate_only:
        return
    start=time.perf_counter()
    for _ in range(3):
        backend.run(ii)
    forward_seconds=(time.perf_counter()-start)/(3*32)
    tail=Tail(back['standard'],args.out,budget)
    start=time.perf_counter()
    for _ in range(8):
        tail.predict(clean.at,['A']*32,source.at)
    tail_seconds=(time.perf_counter()-start)/(8*32)
    remaining=forward_seconds*(len(contexts)+1600*5)+tail_seconds*(55380+2952)
    write(args.out/'profile_c_at.json',dict(forward_seconds_per_row=forward_seconds,tail_seconds_per_row=tail_seconds,
        estimated_remaining_with_allowance=remaining*1.25))
    projected_seconds(budget,remaining)
    natural=observed(back['standard'],list(range(len(contexts))),[c.answer for c in contexts])
    save_npz(args.out/'natural.npz',natural['predictions'])
    b,s=np.asarray(schedule['base']),np.asarray(schedule['source'])
    targets=[contexts[i].label_for(contexts[j].color) for i,j in zip(b,s)]
    base=observed(back['standard'],b.tolist(),[contexts[i].answer for i in b])
    save_npz(args.out/'matched_base.npz',base['predictions'])
    for method,backend in back.items():
        direct=observed(backend,b.tolist(),targets,natural['c'][s],'c')
        replay=observed(backend,b.tolist(),targets,direct['at'],'at')
        save_npz(args.out/('c_at_'+method+'.npz'),{**{'direct_'+k:v for k,v in direct['predictions'].items()},
            **{'replay_'+k:v for k,v in replay['predictions'].items()}})
        print(json.dumps(dict(task='C_AT',method=method,GPU_seconds=budget.elapsed())),flush=True)
    def controls(name,base_indices,donor_indices,targets):
        parts=[]
        tail=Tail(back['standard'],args.out,budget)
        for start in range(0,len(targets),32):
            stop=min(start+32,len(targets))
            parts.append(tail.predict(base['at'][base_indices[start:stop]],targets[start:stop],natural['at'][donor_indices[start:stop]]))
        save_npz(args.out/(name+'.npz'),join(parts))
    retained=np.asarray(schedule['original_pair_indices'])
    controls('pair_controls',np.repeat(retained,2),np.asarray(schedule['pair_donors']).reshape(-1),
        np.repeat(np.asarray(targets)[retained],2).tolist())
    qualification=review.read(ROOT/'data/at_qualification.json')
    controls('clean_AT',np.repeat(qualification['base_occurrences'],52),np.asarray(qualification['donors']).reshape(-1),
        np.tile(np.repeat(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),2),len(qualification['bases'])).tolist())


def compare_saved(out):
    comparisons={}
    for path in sorted(out.rglob('*.npz')):
        reference=ROOT/'predictions'/path.relative_to(out)
        if not reference.exists():
            continue
        a,b=review.arrays(path),review.arrays(reference)
        fields={}
        for name in a.keys()&b.keys():
            if name.endswith(('correct','token_id','tied')):
                fields[name]=dict(equal=int((a[name]==b[name]).sum()),count=a[name].size,all_equal=bool(np.array_equal(a[name],b[name])))
        comparisons[path.relative_to(out).as_posix()]=fields
    write(out/'saved_prediction_agreement.json',comparisons)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['ap','c-at','all','validate'],default='ap')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoints-root',type=Path,default=ROOT.parents[1])
    parser.add_argument('--model-cache',type=Path)
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--budget-minutes',type=float,default=15)
    args=parser.parse_args()
    if not 1<=args.budget_minutes<=90:
        raise ValueError('Budget must be between 1 and 90 GPU-session minutes')
    args.out=args.out.resolve()
    args.checkpoints_root=args.checkpoints_root.resolve()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Use a fresh output directory; prior evidence will not be overwritten')
    review.verify_manifest()
    args.out.mkdir(parents=True,exist_ok=True)
    protocol=review.read(ROOT/'data/protocol.json')
    model=budget=None
    checks=[]
    try:
        runtime.configure(args.out)
        budget=Budget(args.out,args.budget_minutes*60)
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
        from transformers import AutoTokenizer,AutoModelForCausalLM
        import transformers
        location=str(args.model_cache) if args.model_cache else protocol['model_id']
        options=dict(local_files_only=args.local_files_only)
        if args.model_cache:
            for name,expected in protocol['model_files'].items():
                if schema.file_hash(args.model_cache/name)!=expected:
                    raise ValueError('Pinned model/tokenizer file differs: '+name)
        else:
            options['revision']=protocol['model_revision']
        tokenizer=AutoTokenizer.from_pretrained(location,**options)
        tokenizer.padding_side='left'
        model=AutoModelForCausalLM.from_pretrained(location,dtype=torch.bfloat16,attn_implementation='sdpa',**options).to('cuda').eval()
        for p in model.parameters():
            p.requires_grad_(False)
        versions=[p._version for p in model.parameters()]
        write(args.out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,
            numpy=np.__version__,transformers=transformers.__version__,GPU=torch.cuda.get_device_name(),
            revision=protocol['model_revision'],task=args.task,atol=2e-4,rtol=2e-4))
        if args.task in ('ap','all'):
            run_ap(model,tokenizer,args,protocol,budget,checks)
        if args.task in ('c-at','all','validate'):
            run_c_at(model,tokenizer,args,protocol,budget,checks,args.task=='validate')
        if versions!=[p._version for p in model.parameters()] or any(b._forward_hooks for b in model.model.layers):
            raise ValueError('Frozen-backbone or hook-cleanup failure')
        compare_saved(args.out)
        winning_tokens=set()
        for path in args.out.rglob('*.npz'):
            for key,value in review.arrays(path).items():
                if key.endswith('token_id'):
                    winning_tokens.update(map(int,np.unique(value)))
        write(args.out/'token_decodings.json',{str(i):tokenizer.decode([i]).strip() for i in sorted(winning_tokens)})
        write(args.out/'completion.json',dict(complete=True,no_training=True,checks=checks,GPU_seconds=budget.elapsed(),
            interpretation='Fresh predictions retain the frozen masks; inspect numerical and saved-prediction agreement before making reproduction claims'))
    except BaseException as error:
        write(args.out/'stop.json',dict(complete=False,type=type(error).__name__,reason=str(error),checks=checks))
        raise
    finally:
        model=None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if budget is not None:
            budget.close()
        runtime.close()


if __name__=='__main__':
    main()
