"""Frozen, natural-only cross-question donor and recipient proposals."""
from pathlib import Path
import random
from experiments.mcqa_at_anchor_comparison.storage import read,write,digest,status
from . import data,banks
from .protocol import COLORS,LETTERS

ROLES=('AP_donor0','AP_donor1','recipient0','recipient1','recipient2','recipient3',
       'AT_donor0','AT_donor1','conflict0','conflict1')

def conflict_letter(row,index):
    other=[x for x in row['base']['labels'] if x!=row['target']]
    ordered=sorted(other,key=lambda x:data.key('conflict',row['id'],x))
    return ordered[index]

def proposal(row,role,alternative,objects):
    rng=random.Random(int(data.key('control',row['id'],role,alternative),16))
    pool=[o for o in objects if o['id'] not in (row['base']['question'],row['source']['question'])]
    obj=rng.choice(pool);colors=[c for c in COLORS if c not in (row['base']['color'],row['source']['color'])]
    color=rng.choice(colors);dest=row['metadata']['to_AP'];position=dest
    if role.startswith('recipient') and int(role[-1])>=2:position=(dest+int(role[-1])-1)%4
    if role.startswith(('AT','conflict')):position=rng.randrange(4)
    contents=rng.sample([c for c in COLORS if c!=color],3);contents.insert(position,color)
    labels=rng.sample(LETTERS,4)
    if role.startswith('AP'):
        natural=rng.choice([x for x in LETTERS if x!=row['target']]);labels=rng.sample([x for x in LETTERS if x!=natural],3);labels.insert(position,natural)
    elif role.startswith(('AT','conflict')):
        target=row['target'] if role.startswith('AT') else conflict_letter(row,int(role[-1]))
        labels=rng.sample([x for x in LETTERS if x!=target],3);labels.insert(position,target)
    context=data.context(obj,color,labels,contents)
    assert context['question']!=row['base']['question']
    if role.startswith('AP'):assert data.ap(context)==dest and data.answer(context)!=row['target']
    if role.startswith('AT'):assert data.answer(context)==row['target']
    return context

def filter_controls(root,rows,natural,label):
    root=Path(root);banks.locked(root);folder=root/'controls'/label
    objects=read(root/'groups.json')['final'];excluded=set(read(root/'historical_exclusions.json'));result={}
    for start in range(0,len(rows),32):
        page=folder/f'page_{start:06d}.json'
        if page.exists():result.update(read(page)['accepted']);continue
        piece=rows[start:start+32];pending={(r['id'],role):(r,role) for r in piece for role in ROLES}
        accepted={r['id']:{} for r in piece};used={r['id']:set() for r in piece};log=[]
        orders={k:sorted(range(256),key=lambda a:data.key('control-order',*k,a)) for k in pending}
        for rank in range(256):
            proposed={k:proposal(r,role,orders[k][rank],objects) for k,(r,role) in pending.items()}
            candidates={k:c for k,c in proposed.items() if c['id'] not in excluded and c['id'] not in used[k[0]]}
            natural.ensure(list(candidates.values()))
            for k,c in proposed.items():
                clean=k in candidates and bool(natural.field([c],'eligible')[0])
                previous=accepted[k[0]]
                if any(c['question']==v['question'] for v in previous.values()):clean=False
                partner=k[1][:-1]+str(1-int(k[1][-1])) if k[1].startswith(('AP_donor','AT_donor')) else None
                if partner in previous and (c['color']==previous[partner]['color'] or
                    (k[1].startswith('AP') and data.answer(c)==data.answer(previous[partner]))):clean=False
                log.append(dict(pair=k[0],role=k[1],alternative=orders[k][rank],rank=rank,eligible=clean,
                                historical_exclusion=c['id'] in excluded,context_id=c['id']))
                if clean:
                    accepted[k[0]][k[1]]=c;used[k[0]].add(c['id']);del pending[k]
            if not pending:break
        missing=[dict(pair=k[0],role=k[1]) for k in pending]
        write(page,dict(accepted=accepted,missing=missing,dispositions=log));result.update(accepted)
        status(root,'filtering_controls',bank=label,completed=start+len(piece),total=len(rows),missing_in_page=len(missing))
    missing=[dict(pair=r['id'],role=role) for r in rows for role in ROLES if role not in result[r['id']]]
    write(folder/'completion.json',dict(pairs=len(rows),missing=missing,complete=not missing,primary_pairs_removed=0))
    for role in ROLES:
        rr,_=rows_for(rows,result,role)
        if rr:natural.layout(rr,label+'_'+role)
    return result

def rows_for(rows,controls,role):
    output=[];indices=[]
    for i,r in enumerate(rows):
        c=controls[r['id']].get(role)
        if c is None:continue
        new=dict(r);m=dict(r['metadata']);new['metadata']=m
        if role.startswith('recipient'):
            new['base']=c;new['target']=c['labels'][m['to_AP']];m['from_AP']=data.ap(c)
        elif role.startswith('conflict'):
            new['source']=c;new['target']=data.answer(c)
        else:new['source']=c
        m['permutation']=role;m['cluster_question']=r['base']['question'];m['parent_id']=r['id']
        m['parent_transition']=f"{r['metadata']['from_AP']+1}->{r['metadata']['to_AP']+1}"
        new['id']=digest([r['id'],role,c['id']]);output.append(new);indices.append(i)
    return output,indices
