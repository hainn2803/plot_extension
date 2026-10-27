"""Finite, metadata-balanced AP proposals with pre-inference pool hashes."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import itertools
import platform
import subprocess

from experiments.hai_c_at_evaluation import schema
from experiments.mcqa_at_anchor_comparison import data as inventory_source
from experiments.mcqa_at_anchor_comparison.storage import read,write,digest,file_hash,status
from .protocol import ROOT,PARENT,SALT,MODEL,PROTOCOL,LETTERS,COLORS,DIRECTIONS,BANKS,in_domain

def keyed(*parts):
    return digest([SALT,*parts])

def environment():
    import torch,transformers
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],text=True).strip()
    return dict(python=platform.python_version(),platform=platform.platform(),torch=torch.__version__,
        transformers=transformers.__version__,cuda_runtime=torch.version.cuda,gpu_and_driver=gpu)

def inventory():
    all_objects=[x for v in inventory_source.inventory().values() for x in v]
    all_objects.sort(key=lambda x:(keyed('objects',x['id']),x['id']))
    assert len({x['id'] for x in all_objects})==252
    return {'train':all_objects[:96],'calibration':all_objects[96:144],'final':all_objects[144:]}

def slots(bank,groups):
    changing,preserve=BANKS[bank]; group='train' if bank=='train_master' else bank
    objects=groups[group]; out=[]
    for di,(b,s) in enumerate(DIRECTIONS):
        for i in range(changing):
            letter=(i+di)%4; block=i//4
            shift=1+(di*4+block+{'train':0,'calibration':7,'final':13}[group])%3
            out.append(dict(bank=bank,group=group,kind='changing',from_AP=b,to_AP=s,
                ordinal=i,block=block,base_group=(17*di+i)%len(objects),
                color=COLORS[(7*di+i)%23],target=LETTERS[letter],base_letter=LETTERS[(letter+shift)%4]))
    for p in range(4):
        for i in range(preserve):
            out.append(dict(bank=bank,group=group,kind='preservation',from_AP=p,to_AP=p,
                ordinal=i,block=i//4,base_group=(17*p+i)%len(objects),
                color=COLORS[(7*p+i)%23],target=LETTERS[(i+p)%4],base_letter=LETTERS[(i+p)%4]))
    for i,x in enumerate(out):x['slot']=i
    return out

def context(obj,color,labels,contents):
    prompt=obj['template'].format(color=color)+'\n'+'\n'.join(f'{a}. {c}' for a,c in zip(labels,contents))+'\nAnswer:'
    return asdict(schema.parse_context(dict(base_prompt=prompt,question=obj['id']),'base'))

def answer(c):return c['labels'][c['contents'].index(c['color'])]

def ap(c):return c['contents'].index(c['color'])

def target(base,source):return base['labels'][ap(source)]

def alternative_order(slot):
    return sorted(range(256),key=lambda a:(keyed('alternative',slot['bank'],slot['slot'],a),a))

def proposal(slot,alternative,groups):
    if not 0<=alternative<256:raise ValueError('Outside registered finite proposal pool')
    r=random.Random(int(keyed('proposal',slot['bank'],slot['slot'],alternative),16))
    obj=groups[slot['group']][slot['base_group']]; b,s=slot['from_AP'],slot['to_AP']
    color=COLORS[(COLORS.index(slot['color'])+alternative)%len(COLORS)]
    contents=r.sample([x for x in COLORS if x!=color],3)
    contents.insert(b,color)
    labels=[None]*4;labels[b]=slot['base_letter']
    if b!=s:labels[s]=slot['target']
    available=r.sample([x for x in LETTERS if x not in labels],labels.count(None))
    for i in range(4):
        if labels[i] is None:labels[i]=available.pop()
    base=context(obj,color,labels,contents)
    order=list(range(4))
    if b!=s:order[b],order[s]=order[s],order[b]
    else:
        other=[i for i in range(4) if i!=b]
        perms=[v for v in itertools.permutations(other) if list(v)!=other]
        for i,j in zip(other,r.choice(perms)):order[i]=j
    source=context(obj,color,[labels[i] for i in order],[contents[i] for i in order])
    value=dict(id=keyed(base['id'],source['id']),base=base,source=source,target=target(base,source),
        metadata=dict(slot,registered_slot_color=slot['color'],color=color),alternative=alternative)
    validate(value)
    return value

def validate(row):
    b,s,m=row['base'],row['source'],row['metadata']
    assert b['id']!=s['id'] and b['question']==s['question'] and b['color']==s['color']
    assert b['color']==m['color']
    assert answer(b)==answer(s)==m['base_letter']
    assert len(set(b['labels']))==len(set(b['contents']))==4
    assert set(b['labels'])==set(s['labels'])==set(LETTERS)
    assert sorted(zip(b['labels'],b['contents']))==sorted(zip(s['labels'],s['contents']))
    assert (ap(b),ap(s))==(m['from_AP'],m['to_AP'])
    assert row['target']==m['target']==target(b,s)
    assert (row['target']!=answer(b))==(m['kind']=='changing')

def view(rows,arm):
    return [r for r in rows if r['metadata']['kind']=='preservation' or
        (r['metadata']['block'] in PROTOCOL['shared_blocks'] if arm=='shared' else in_domain(arm,r['metadata']['to_AP']))]

def audit(rows,bank,groups,retained=False):
    meta=[r['metadata'] for r in rows] if retained else rows
    changing,preserve=BANKS[bank]
    assert len(meta)==12*changing+4*preserve
    for b,s in DIRECTIONS:
        ss=[r for r in meta if (r['from_AP'],r['to_AP'])==(b,s)]
        assert len(ss)==changing
        for field in ('target','base_letter'):
            counts=Counter(r[field] for r in ss)
            assert set(counts)==set(LETTERS) and max(counts.values())-min(counts.values())<=1
            assert set(counts.values())<={changing//4,(changing+3)//4}
        if bank=='final':assert len({r['base_group'] for r in ss})==104
    for p in range(4):
        ss=[r for r in meta if r['from_AP']==r['to_AP']==p]
        assert len(ss)==preserve
        counts=Counter(r['target'] for r in ss)
        assert set(counts)==set(LETTERS) and max(counts.values())-min(counts.values())<=1
        assert set(counts.values())<={preserve//4,(preserve+3)//4}
    for field in ('target','base_letter'):
        assert Counter(r[field] for r in meta)==Counter({x:len(meta)//4 for x in LETTERS})
    assert len({r['base_group'] for r in meta})==len(groups[meta[0]['group']])
    if retained:
        for r in rows:validate(r)
        assert len({r['id'] for r in rows})==len(rows)
    return dict(pairs=len(meta),templates=len({r['base_group'] for r in meta}),
        colors=dict(Counter(r['color'] for r in meta)),
        target_letters=dict(Counter(r['target'] for r in meta)),
        natural_letters=dict(Counter(r['base_letter'] for r in meta)),
        base_label_orders=dict(Counter(''.join(r['base']['labels']) for r in rows)) if retained else None,
        source_label_orders=dict(Counter(''.join(r['source']['labels']) for r in rows)) if retained else None,
        passed=True)

def historical_exclusions():
    excluded=set()
    for path in (Path('eval/hai_frozen_descriptive_20261020_v1/selected_pairs.json'),):
        cs,_=schema.import_pairs(read(path));excluded.update(c.id for c in cs)
    for name in ('contexts.json','readout_plan.json'):
        x=read(Path('eval/hai_c_at_full_chain_20261020_v3')/name)
        values=x.get('contexts',[]) if isinstance(x,dict) else x
        excluded.update(c['id'] for c in values)
    return sorted(excluded)

def register_amendment(root):
    from experiments.mcqa_ap_specialization_revised import data as parent_data
    parent_data.verify_identity(PARENT)
    ledger=read(PARENT/'engineering/gpu_budget.json')
    if ledger.get('active'):raise ValueError('Parent GPU session is still active')
    if (PARENT/'pause.request').exists():raise ValueError('Parent safety pause cannot be bypassed by a child')
    value=dict(parent=str(PARENT),parent_identity_sha256=file_hash(PARENT/'identity.json'),
        parent_budget_sha256=file_hash(PARENT/'engineering/gpu_budget.json'),
        parent_shortfall_sha256=file_hash(PARENT/'banks/train_master/shortfall_000000.json'),
        authorization='User explicitly approved shuffled A-D and resume; 26-example calibration cells use 6/7 per-letter counts; all other scientific settings and total compute cap unchanged',
        amendment=PROTOCOL['approved_amendment'],color_schedule=PROTOCOL['color_schedule'],
        carried_gpu_seconds=ledger['seconds'],carried_profile_seconds=ledger['profile_seconds'],
        parent_preserved=True,parent_scientific_trainings=0,
        preserved_diagnostics=['object_feasibility_v1','controlled_relabel_v1'])
    write(root/'amendment.json',value)
    path=root/'engineering/gpu_budget.json'
    if not path.exists():
        write(path,dict(ledger,active=False,stage='inherited_parent_accounting',
            provenance=str(root/'amendment.json')),immutable=False)
    current=read(path)
    if current['seconds']<ledger['seconds'] or current['profile_seconds']<ledger['profile_seconds']:
        raise ValueError('Child budget does not include all parent usage')

def prepare(root=ROOT):
    root=Path(root);groups=inventory()
    register_amendment(root)
    from .verification import reference
    reference(root)
    write(root/'protocol.json',json.loads(json.dumps(PROTOCOL)))
    write(root/'groups.json',groups);write(root/'historical_exclusions.json',historical_exclusions())
    pools={}
    for bank in BANKS:
        ss=slots(bank,groups);check=audit(ss,bank,groups)
        write(root/'proposals'/f'{bank}.slots.json',ss)
        receipt=root/'proposals'/f'{bank}.pool.json'
        if receipt.exists():
            existing=read(receipt)
            if existing['generator_sha256']!=file_hash(__file__):raise ValueError('Pool generator changed')
            pools[bank]=existing;continue
        h=hashlib.sha256()
        for i,slot in enumerate(ss):
            for alt in alternative_order(slot):
                value=proposal(slot,alt,groups)
                h.update(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n')
            if i%128==0:status(root,'preparing',bank=bank,slots_hashed=i,total=len(ss))
        value=dict(slots_sha256=file_hash(root/'proposals'/f'{bank}.slots.json'),
            generator_sha256=file_hash(__file__),complete_proposal_sha256=h.hexdigest(),
            proposals=len(ss)*256,alternatives=256,audit=check,
            preservation='All proposals regenerate exactly from frozen slots, groups, source and salted ordering; no unused proposals discarded')
        write(receipt,value);pools[bank]=value
    write(root/'proposals/manifest.json',pools)
    files=[*Path(__file__).parent.rglob('*.py'),
        Path(inventory_source.__file__),Path('experiments/mcqa_at_anchor_comparison/storage.py'),
        Path('experiments/mcqa_at_anchor_comparison/engine.py'),
        Path('experiments/mcqa_c_at_localization/runtime.py'),Path('experiments/mcqa_c_at_localization/profiling.py'),
        Path('experiments/hai_c_at_evaluation/core.py'),Path('experiments/hai_c_at_evaluation/schema.py'),
        Path('diagnostics/hai_full_chain_20261020/resume_shared_io.py'),
        Path('experiments/mcqa_c_at_it_gradual_pilot/engineering.py'),
        Path('experiments/mcqa_at_anchor_comparison/training.py'),
        Path('experiments/mcqa_c_at_it_das/training.py')]
    imports={str(p):file_hash(p) for p in sorted((inventory_source.parent.DATA/'generated').glob('*.jsonl'))}
    for p in (inventory_source.parent.DATA/'protocol.json',inventory_source.parent.DATA/'manifest.json',
              root/'reference/hai_ap.pt',root/'reference/receipt.json',root/'amendment.json',
              PARENT/'identity.json',PARENT/'engineering/gpu_budget.json'):
        imports[str(p)]=file_hash(p)
    identity=dict(source={str(p):file_hash(p) for p in files},imports=imports,environment=environment(),
        artifacts={p:file_hash(root/p) for p in ('protocol.json','groups.json','historical_exclusions.json','proposals/manifest.json')},
        model=read(inventory_source.parent.DATA/'protocol.json')['model_files'])
    write(root/'identity.json',identity);status(root,'prepared',frozen_proposals=sum(p['proposals'] for p in pools.values()))
    return identity

def verify_identity(root,model=False):
    identity=read(Path(root)/'identity.json')
    if identity['environment']!=environment():raise ValueError('Execution environment changed; migration validation required')
    for group in ('source','imports'):
        for p,h in identity[group].items():
            if file_hash(p)!=h:raise ValueError('Immutable source/input changed: '+p)
    for p,h in identity['artifacts'].items():
        if file_hash(Path(root)/p)!=h:raise ValueError('Immutable registered artifact changed: '+p)
    if model:
        for p,h in identity['model'].items():
            if file_hash(MODEL/p)!=h:raise ValueError('Pinned model/tokenizer changed: '+p)
    return digest(identity)
