"""Explicit, annotation-free intervention supports."""
from dataclasses import dataclass, asdict
import re
import torch

@dataclass(frozen=True)
class Site:
    layer: int
    kind: str
    index: int = 0

    @property
    def count(self):
        return 4 if self.kind.startswith('joint_') else 1

    @property
    def id(self):
        return f'L{self.layer:02d}_{self.kind}_{self.index:03d}'

    def record(self):
        return asdict(self)

def catalog(width, joint=False):
    if joint:
        return [Site(l,k) for l in range(26) for k in ('joint_label','joint_color')]
    return [Site(l,k,i) for l in range(26) for k,indices in
            [('offset',range(width)),('label',range(4)),('color',range(4))] for i in indices]

def structural_positions(context, offsets):
    matches=list(re.finditer(r'(?m)^([A-Z])\. ([a-z]+)\r?$',context['prompt']))
    if len(matches)!=4:raise ValueError('Four complete option rows required')
    result={}
    for kind in ('label','color'):
        locations=[]
        for match in matches:
            start,end=match.span(1 if kind=='label' else 2)
            candidates=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start]
            if not candidates or (kind=='label' and len(candidates)!=1):
                raise ValueError('Structural token alignment is ambiguous')
            locations.append(candidates[0] if kind=='label' else candidates[-1])
        if len(set(locations))!=4:raise ValueError('Overlapping option supports')
        result[kind]=locations
    return result

def gather(hidden, positions):
    return hidden[torch.arange(len(hidden),device=hidden.device)[:,None],positions].flatten(1)

def scatter(hidden,positions,values,available):
    out=hidden.clone();values=values.reshape(len(hidden),positions.shape[1],hidden.shape[-1])
    rr=torch.arange(len(hidden),device=hidden.device)[:,None]
    out[rr,positions]=torch.where(available[:,None,None],values.to(hidden.dtype),hidden[rr,positions])
    return out
