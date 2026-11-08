"""CPU integration of portable hook/shape plumbing using a small synthetic model."""
from pathlib import Path
from types import SimpleNamespace
import re
import tempfile
import unittest
import torch
import review
from vendor import ap_engine, references
from vendor.tail import Tail


class Tokenizer:
    pad_token_id=1

    def __call__(self,text,**kwargs):
        spans=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
        return dict(input_ids=[0]+[sum(map(ord,text[a:b]))%2046+2 for a,b in spans],offset_mapping=[(0,0)]+spans)

    def decode(self,ids):
        return str(ids[0])


class Block(torch.nn.Module):
    def forward(self,h):
        return h+h.cumsum(1)*.001


class Inner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(2048,32)
        self.layers=torch.nn.ModuleList([Block() for _ in range(26)])
        self.norm=torch.nn.LayerNorm(32)

    def forward(self,input_ids,**kwargs):
        h=self.embedding(input_ids)
        for layer in self.layers:
            h=layer(h)
        return SimpleNamespace(last_hidden_state=self.norm(h))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Inner()
        self.lm_head=torch.nn.Linear(32,2048,bias=False)
        self.config=SimpleNamespace(final_logit_softcapping=50.)


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.q=torch.eye(32)[:,:4]
        ap_engine.ROOT=self.root
        ap_engine.AP=self.root/'ap.pt'
        torch.save(dict(basis=self.q),ap_engine.AP)
        self.model=Model().to(torch.bfloat16).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.budget=SimpleNamespace(tick=lambda:None)
        plan=review.read(review.ROOT/'data/ap_plan.json')
        self.engine=ap_engine.Engine(self.model,Tokenizer(),dict(contexts=plan['contexts'][:32]),self.budget)
        self.ref=references.selected_functions(review.ROOT/'vendor/source_references/mcqa_gradual_das.py',['answer_logits'])
        self.ref=references.selected_functions(review.ROOT/'vendor/source_references/mcqa_gradual_das_c.py',['run_frozen_mediator'],self.ref)

    def tearDown(self):
        self.assertFalse(any(layer._forward_hooks for layer in self.model.model.layers))
        self.temp.cleanup()

    def test_full_vector_identity_and_capture(self):
        ii=list(range(32))
        base=self.engine.forward(ii,capture=True)
        patched=self.engine.forward(ii,layer=0,donor=base['states'][:,0,0],slots=[0]*32)
        torch.testing.assert_close(patched['logits'],base['logits'],atol=2e-4,rtol=2e-4)

    def test_subspace_matches_pinned_reference(self):
        ii=list(range(32))
        natural=self.engine.forward(ii,capture=True)
        donor=natural['last'].roll(1,0)
        actual=self.engine.forward(ii,layer=17,donor=donor,subspace=True)['logits']
        ids,mask,_=self.engine.inputs(ii)
        bank=dict(base_input_ids=ids,base_attention_mask=mask,base_position_by_id={
            'last_token':torch.tensor([len(self.engine.encoded[i]['input_ids'])-1 for i in ii])})
        expected=self.ref['run_frozen_mediator'](self.model,bank,donor,self.q,17,None,batch_size=32,require_grad=False)
        torch.testing.assert_close(actual,expected,atol=2e-4,rtol=2e-4)

    def test_tail_partial_batch_padding(self):
        backend=SimpleNamespace(model=self.model,device=torch.device('cpu'),at_basis=self.q,at_layer=25,
            letters=tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),label_ids=list(range(26)),tie_tolerance=0.)
        tail=Tail(backend,self.root,self.budget)
        h=torch.randn(7,32)
        y=tail.logits(h,32)
        self.assertEqual(tuple(y.shape),(32,2048))
        padded=torch.cat((h,h[:1].expand(25,-1))).to(torch.bfloat16)
        expected=self.model.lm_head(self.model.model.norm(padded[:,None,:])[:,0]).float()
        expected=torch.tanh(expected/50)*50
        torch.testing.assert_close(y,expected,atol=2e-4,rtol=2e-4)


if __name__=='__main__':
    unittest.main()
