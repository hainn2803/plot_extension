"""CPU oracles, independent projector checks, and immutable reference import."""
from collections import Counter
import hashlib
from pathlib import Path
import tempfile
import urllib.request

import numpy as np
import torch
from torch.nn import functional as F

from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
from experiments.mcqa_at_anchor_comparison.engine import Subspace,projection
from experiments.mcqa_at_anchor_comparison.training import checkpoint,recover,restore_rng
from experiments.mcqa_c_at_it_das.training import tensor_digest
from . import data,training
from .protocol import PROTOCOL,ARMS,SEEDS,BANKS

def reference(root):
    folder=Path(root)/'reference';folder.mkdir(parents=True,exist_ok=True);path=folder/'hai_ap.pt'
    if not path.exists():
        url='https://raw.githubusercontent.com/hainn2803/plot_extension/'+PROTOCOL['reference_commit']+'/results_joint_randomized_standard/das_ap.pt'
        with urllib.request.urlopen(url,timeout=60) as response:
            body=response.read(8*1024*1024)
        if len(body)>=8*1024*1024:raise ValueError('Oversized AP reference')
        blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
        if blob!=PROTOCOL['reference_blob']:raise ValueError('Downloaded AP reference blob mismatch')
        path.write_bytes(body)
    body=path.read_bytes();blob=hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest()
    if blob!=PROTOCOL['reference_blob']:raise ValueError('AP reference file changed')
    weights=torch.load(path,map_location='cpu',weights_only=True)
    if weights['basis'].shape!=(2304,128) or weights['layer']!=17 or weights['token_position']!='last_token':
        raise ValueError('AP reference location/rank differ from registered design')
    write(folder/'receipt.json',dict(checkpoint=str(path),sha256=file_hash(path),blob=blob,
        commit=PROTOCOL['reference_commit'],layer=17,rank=128,position='last_token',role='Descriptive reference, not matched training baseline'))

def examples():
    obj=dict(id='sky',template='The sky is {color}. What color is the sky?')
    b=data.context(obj,'blue',list('ABCD'),['red','blue','green','yellow'])
    examples=[(list('ACBD'),['red','green','blue','yellow'],'C'),
        (list('ADCB'),['red','yellow','green','blue'],'D'),
        (list('CBAD'),['green','blue','red','yellow'],'B')]
    for labels,contents,expected in examples:
        s=data.context(obj,'blue',labels,contents)
        assert data.target(b,s)==expected and data.answer(b)==data.answer(s)=='B'
        assert data.target(s,b)==s['labels'][data.ap(b)]
    return dict(passed=True,count=3,AP_positions='Internal zero-based; displayed one-based')

def algebra():
    g=torch.Generator().manual_seed(21)
    raw=torch.randn(8,4,generator=g,dtype=torch.float64,requires_grad=True)
    b=torch.randn(5,8,generator=g,dtype=torch.float64);s=torch.randn(5,8,generator=g,dtype=torch.float64)
    q=torch.linalg.qr(raw,mode='reduced').Q;new=projection(b,s,q)
    assert torch.allclose(q.T@q,torch.eye(4,dtype=q.dtype),atol=1e-10,rtol=1e-10)
    assert torch.allclose((new-b)@(torch.eye(8,dtype=q.dtype)-q@q.T),torch.zeros_like(b),atol=1e-10,rtol=1e-10)
    assert torch.equal(projection(b,b,q),b)
    reference=b+(s-b)@(q@q.T)
    assert torch.allclose(new,reference,atol=1e-10,rtol=1e-10)
    frozen=torch.randn(8,5,generator=g,dtype=torch.float64)
    targets=torch.tensor([0,1,2,3,4]);loss=F.cross_entropy(new@frozen,targets)
    grad=torch.autograd.grad(loss,raw)[0]
    oracle=torch.autograd.grad(F.cross_entropy((b+(s-b)@(torch.linalg.qr(raw,mode='reduced').Q@torch.linalg.qr(raw,mode='reduced').Q.T))@frozen,targets),raw)[0]
    assert torch.allclose(grad,oracle,atol=1e-10,rtol=1e-10)
    eps=1e-6;errors=[]
    for i,j in [(0,0),(3,1),(7,3)]:
        plus=raw.detach().clone();minus=plus.clone();plus[i,j]+=eps;minus[i,j]-=eps
        def f(x):return F.cross_entropy(projection(b,s,torch.linalg.qr(x,mode='reduced').Q)@frozen,targets)
        fd=(f(plus)-f(minus))/(2*eps);errors.append(float(abs(fd-grad[i,j])))
        assert torch.allclose(fd,grad[i,j],atol=2e-5,rtol=2e-5)
    assert not frozen.requires_grad and frozen.grad is None
    return dict(passed=True,algebra_tolerance=1e-10,gradient_tolerance=2e-5,finite_difference_max_error=max(errors))

def schedule(groups):
    master=[dict(id=str(s['slot']),metadata=s) for s in data.slots('train_master',groups)]
    result={}
    for arm in ARMS:
        view=data.view(master,arm);assert len(view)==832
        assert sum(r['metadata']['kind']=='changing' for r in view)==624
        assert len({r['metadata']['base_group'] for r in view})==96
        changed=[r for r in view if r['metadata']['kind']=='changing']
        for transition in set((r['metadata']['from_AP'],r['metadata']['to_AP']) for r in changed):
            cell=[r for r in changed if (r['metadata']['from_AP'],r['metadata']['to_AP'])==transition]
            for field in ('target','base_letter'):assert len(set(Counter(r['metadata'][field] for r in cell).values()))==1
        result[arm]=view
    for seed in SEEDS:
        for step in (0,1,25,26,100,999):
            preserved=[]
            for arm,rows in result.items():
                ix=training.order(rows,seed,step);batch=[rows[i] for i in ix]
                same=[r for r in batch if r['metadata']['kind']=='preservation']
                changing=[r for r in batch if r['metadata']['kind']=='changing']
                assert len(same)==8 and len(changing)==24
                assert Counter(r['metadata']['to_AP'] for r in same)==Counter({p:2 for p in range(4)})
                counts=Counter((r['metadata']['from_AP'],r['metadata']['to_AP']) for r in changing)
                assert len(counts)==(12 if arm=='shared' else 6)
                assert set(counts.values())==({2} if arm=='shared' else {4})
                preserved.append([r['id'] for r in same])
            assert preserved[0]==preserved[1]==preserved[2]
    return dict(passed=True,training_sizes={a:len(r) for a,r in result.items()},paired_preservation=True)

def resume_check(folder):
    torch.manual_seed(9);np.random.seed(9)
    x=torch.randn(16,8);y=torch.arange(16)%4;identity={'fixture':'AP resume'}
    sub=Subspace(1,width=8,rank=4);opt=torch.optim.Adam(sub.parameters(),lr=.01)
    def step(model,optimizer):
        optimizer.zero_grad();loss=F.cross_entropy(x@model.basis(),y);loss.backward();optimizer.step()
    for _ in range(2):step(sub,opt)
    state=dict(identity=identity,step=2,updates=[{},{}],initialization='fixture',evaluations={})
    checkpoint(folder,state,sub,opt)
    for _ in range(3):step(sub,opt)
    expected=tensor_digest(dict(model=sub.state_dict(),optimizer=opt.state_dict()))
    restored,_=recover(folder,identity);other=Subspace(1,width=8,rank=4);oo=torch.optim.Adam(other.parameters(),lr=.01)
    other.load_state_dict(restored['subspace']);oo.load_state_dict(restored['optimizer']);restore_rng(restored)
    for _ in range(3):step(other,oo)
    assert expected==tensor_digest(dict(model=other.state_dict(),optimizer=oo.state_dict()))
    try:recover(folder,{'fixture':'changed'})
    except ValueError:pass
    else:raise AssertionError('Changed resume identity was accepted')
    return dict(passed=True,optimizer_and_projector_exact=True,changed_identity_rejected=True)

def verify(root):
    root=Path(root);identity=data.verify_identity(root);groups=read(root/'groups.json')
    all_ids=[x['id'] for v in groups.values() for x in v]
    assert len(all_ids)==len(set(all_ids))==252
    audits={}
    for bank in BANKS:
        ss=read(root/'proposals'/f'{bank}.slots.json');assert ss==data.slots(bank,groups)
        audits[bank]=data.audit(ss,bank,groups)
        pool=read(root/'proposals'/f'{bank}.pool.json')
        assert pool['proposals']==len(ss)*256 and pool['slots_sha256']==file_hash(root/'proposals'/f'{bank}.slots.json')
        for slot in ss[::max(1,len(ss)//48)]:
            for alt in (0,255,data.alternative_order(slot)[0]):
                r=data.proposal(slot,alt,groups);data.validate(r);assert r==data.proposal(slot,alt,groups)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='ap_resume_') as folder:resume=resume_check(Path(folder))
    value=dict(passed=True,identity=identity,banks=audits,examples=examples(),algebra=algebra(),
        schedule=schedule(groups),resume=resume,cuda_initialized=torch.cuda.is_initialized())
    if value['cuda_initialized']:raise AssertionError('CPU checks initialized CUDA')
    write(root/'cpu_verification.json',value);print('AP CPU verification passed',flush=True)
    return value
