"""Finite candidate bundles with metadata-only quotas and natural filtering."""
from dataclasses import asdict
from collections import Counter
import itertools
import json
from pathlib import Path
import random

from experiments.hai_c_at_evaluation import schema
from experiments.mcqa_c_composition_ablation import data as inventory_source
from experiments.mcqa_at_anchor_comparison.storage import read, write, digest, file_hash, status
from .protocol import ROOT, SALT, SIZES, DIRECTIONS, COLORS, LETTERS, PROTOCOL, FROZEN

def key(*parts):
    return digest([SALT, *parts])

def inventory():
    objects=[x for values in inventory_source.inventory().values() for x in values]
    objects.sort(key=lambda x:(key('objects',x['id']),x['id']))
    assert len({o['id'] for o in objects})==252
    return dict(train=objects[:96], calibration=objects[96:144], final=objects[144:])

def context(obj, color, labels, contents):
    prompt=obj['template'].format(color=color)+'\n'+'\n'.join(f'{a}. {c}' for a,c in zip(labels,contents))+'\nAnswer:'
    return asdict(schema.parse_context(dict(base_prompt=prompt, question=obj['id']),'base'))

def answer(c):
    return c['labels'][c['contents'].index(c['color'])]

def ap(c):
    return c['contents'].index(c['color'])

def slots(bank):
    n,p=SIZES[bank]; out=[]
    for b,s in DIRECTIONS:
        for i in range(n):
            out.append(dict(bank=bank, b=b, s=s, ordinal=i, kind='changing', cycle_slot=[v for v in range(4) if v not in (b,s)][i%2]))
    for b in range(4):
        for i in range(p):
            out.append(dict(bank=bank,b=b,s=b,ordinal=i,kind='preservation',cycle_slot=None))
    for i,x in enumerate(out): x['slot']=i
    return out

def alternative_order(slot):
    return sorted(range(256),key=lambda a:(key('alternative',slot,a),a))

def proposal(slot, alternative, groups):
    if not 0<=alternative<256: raise ValueError('Outside frozen pool')
    r=random.Random(int(key('proposal',slot,alternative),16)); bank=slot['bank']
    obj=r.choice(groups['final' if bank=='c_diagnostic' else bank])
    color=r.choice(COLORS); contents=r.sample([c for c in COLORS if c!=color],3)
    b,s=slot['b'],slot['s']; contents.insert(b,color); labels=r.sample(LETTERS,4)
    base=context(obj,color,labels,contents); sources=[]
    if bank=='c_diagnostic':
        slabels=labels.copy(); sc=contents.copy(); scolor=color
        if b!=s:
            scolor=contents[s]
            if slot['ordinal']%2==0: slabels[b],slabels[s]=slabels[s],slabels[b]
        elif slot['ordinal']%2==0:
            t=(b+1+slot['ordinal']//2%3)%4
            slabels[b],slabels[t]=slabels[t],slabels[b];sc[b],sc[t]=sc[t],sc[b]
        else:
            t=(b+1+slot['ordinal']//2%3)%4
            slabels[b],slabels[t]=slabels[t],slabels[b]
        sources=[('C_only' if b!=s and slot['ordinal']%2==0 else 'both' if b!=s else 'AP_only' if slot['ordinal']%2==0 else 'AT_only',context(obj,scolor,slabels,sc))]
    elif b!=s:
        order=list(range(4));order[b],order[s]=order[s],order[b]
        sources.append(('transposition',context(obj,color,[labels[j] for j in order],[contents[j] for j in order])))
        t=slot['cycle_slot'];order=list(range(4));order[s]=b;order[b]=t;order[t]=s
        sources.append(('cycle',context(obj,color,[labels[j] for j in order],[contents[j] for j in order])))
    else:
        others=[v for v in range(4) if v!=b]
        perms=[p for p in itertools.permutations(others) if list(p)!=others]
        order=list(range(4))
        for i,j in zip(others,r.choice(perms)):order[i]=j
        sources=[('preservation',context(obj,color,[labels[j] for j in order],[contents[j] for j in order]))]
    rows=[]
    for typ,source in sources:
        row=dict(id=key(base['id'],source['id']),base=base,source=source,
            target=labels[s],metadata=dict(slot,from_AP=b,to_AP=s,permutation=typ,
                group='final' if bank=='c_diagnostic' else bank,base_group=obj['id']),alternative=alternative)
        validate(row);rows.append(row)
    return rows

def validate(row):
    b,s,m=row['base'],row['source'],row['metadata']; origin,dest=m['from_AP'],m['to_AP']
    assert b['id']!=s['id'] and b['question']==s['question']
    assert ap(b)==origin and row['target']==b['labels'][dest]
    assert len(set(b['labels']))==len(set(b['contents']))==4
    if m['bank']=='c_diagnostic':
        assert b['contents'].index(s['color'])==dest
        assert (b['color']!=s['color'])==(origin!=dest)
    else:
        assert b['color']==s['color'] and answer(b)==answer(s) and ap(s)==dest
        assert sorted(zip(b['labels'],b['contents']))==sorted(zip(s['labels'],s['contents']))
        if m['permutation']=='cycle':
            assert s['labels'][origin]!=row['target']

def audit(rows,bank):
    n,p=SIZES[bank]; factor=1 if bank=='c_diagnostic' else 2
    assert len(rows)==12*n*factor+4*p
    for b,s in DIRECTIONS:
        rr=[r for r in rows if (r['metadata']['from_AP'],r['metadata']['to_AP'])==(b,s)]
        assert len(rr)==n*factor
        if bank=='final':assert len({r['base']['question'] for r in rr})>=32
    for b in range(4):assert sum(r['metadata']['from_AP']==r['metadata']['to_AP']==b for r in rows)==p
    assert len({r['id'] for r in rows})==len(rows)
    for r in rows:validate(r)
    return dict(count=len(rows),templates=len({r['base']['question'] for r in rows}),
        colors=dict(Counter(r['base']['color'] for r in rows)),
        target_letters=dict(Counter(r['target'] for r in rows)),
        natural_letters=dict(Counter(answer(r['base']) for r in rows)),
        label_orders=dict(Counter(''.join(r['base']['labels']) for r in rows)))

def prepare(root=ROOT):
    root=Path(root)
    if (root/'identity.json').exists():return verify(root)
    groups=inventory();write(root/'groups.json',groups);write(root/'protocol.json',json.loads(json.dumps(PROTOCOL)))
    excluded,records=inventory_source.historical_exclusions()
    for name in ('eval/hai_ap_composition_20261023_v2/plan.json',):
        value=read(name);excluded.extend(c['id'] for c in value['contexts'])
    for bank in ('final','c_diagnostic'):
        old=Path('eval/mcqa_ap_specialization_routing_20261023_v1/banks')/bank/'pairs.json'
        for r in read(old):
            excluded.extend(r[side]['id'] for side in ('base','source'))
    write(root/'historical_exclusions.json',sorted(set(excluded)))
    for bank in SIZES:write(root/'proposals'/f'{bank}.json',slots(bank))
    write(root/'proposals/controls.json',dict(alternatives=256,AP_donors=2,other_recipients=4,AT_donors=2,
        identity='Deterministic from parent pair ID, role, ordinal and alternative; generator source sealed'))
    refs=[FROZEN/f'results_joint_randomized_{method}'/'das_c.pt' for method in ('standard','gradual2.0')]
    refs.append(FROZEN/'results_joint_randomized_standard/das_at.pt')
    from diagnostics.hai_frozen_eval_20261020 import run as historical
    from .protocol import DIAGNOSIS
    from . import engine,training,banks,evaluation,verification,runner
    from experiments.mcqa_at_anchor_comparison import engine as base_engine,training as base_training,storage
    from experiments.mcqa_ap_specialization_routing import banks as old_banks
    from experiments.mcqa_c_at_localization import runtime
    from experiments.mcqa_c_at_it_gradual_pilot import engineering
    from diagnostics.ap_specialization_routing_20261023 import counter_transport,library_initialization
    from diagnostics.ap_thermal_20261021 import resume_ap
    modules=(inventory_source,schema,base_engine,base_training,storage,old_banks,runtime,engineering,
             counter_transport,library_initialization,resume_ap)
    import sys
    from experiments.toy_c_at_composition import core as toy_core
    sources=list(Path(__file__).parent.glob('*.py'))+[Path(m.__file__) for m in modules]+[Path(toy_core.__file__)]
    sources.extend(Path(m.__file__) for name,m in list(sys.modules.items())
        if name.startswith(('experiments.','diagnostics.')) and getattr(m,'__file__',None) and str(m.__file__).endswith('.py'))
    workspace=Path.cwd().resolve()
    sources=sorted({p.resolve().relative_to(workspace) for p in sources},key=str)
    files=sources+refs+[DIAGNOSIS,root/'groups.json',root/'protocol.json',root/'historical_exclusions.json',
                         *sorted((root/'proposals').glob('*.json'))]
    identity=dict(files={str(p):file_hash(p) for p in files},
        model=read(historical.DATA/'protocol.json')['model_files'],model_root=str(historical.MODEL),
        old_studies_unchanged=True,prior_budget_not_reset=True,new_gpu_allocation_seconds=86400)
    for p in sources:
        dest=root/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    write(root/'identity.json',identity);status(root,'prepared',GPU_seconds=0,final_accessed=False)
    return identity

def verify(root=ROOT,model=False):
    identity=read(Path(root)/'identity.json')
    for p,h in identity['files'].items():
        if file_hash(p)!=h:raise ValueError('Immutable input changed: '+p)
    if model:
        for p,h in identity['model'].items():
            if file_hash(Path(identity['model_root'])/p)!=h:raise ValueError('Model changed: '+p)
    return identity

