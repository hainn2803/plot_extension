"""Resumable natural-only bundle selection and immutable natural-state cache."""
from functools import lru_cache
from pathlib import Path
import time
import torch
from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,file_hash,status
from . import data

def score(engine,logits,targets):
    x=logits.detach().float();values,ids=x.topk(2,dim=1);ties=values[:,0]==values[:,1]
    decoded=[engine.decode(i).strip() for i in ids[:,0].cpu().tolist()]
    correct=torch.tensor([a==b for a,b in zip(decoded,targets)],device=x.device)&~ties
    label_ids=torch.tensor(engine.label_ids,device=x.device)
    probabilities=(x[:,label_ids]-x.logsumexp(-1,keepdim=True)).exp()
    return dict(correct=correct.cpu(),token_id=ids[:,0].cpu(),tied=ties.cpu(),margin=(values[:,0]-values[:,1]).cpu(),
        top_two_token_ids=ids.cpu(),top_two_logits=values.cpu(),
        label_probabilities=probabilities.cpu())

class NaturalCache:
    def __init__(self,root,engine):
        self.root=Path(root);self.engine=engine;self.index={};self.folder=self.root/'natural_cache'
        self.folder.mkdir(parents=True,exist_ok=True)
        for path in sorted(self.folder.glob('*.ids.json')):
            ids=read(path);tensor=path.with_name(path.name.replace('.ids.json','.pt'));load(tensor)
            for i,ident in enumerate(ids):
                if ident in self.index:raise ValueError('Duplicate cache prompt')
                self.index[ident]=(tensor,i)
        self.next=len(list(self.folder.glob('*.ids.json')))

    @lru_cache(maxsize=16)
    def chunk(self,path):return load(path)

    def ensure(self,contexts):
        rows=list({c['id']:c for c in contexts if c['id'] not in self.index}.values())
        for start in range(0,len(rows),128):
            piece=rows[start:start+128];values={}
            for i in range(0,len(piece),self.engine.batch):
                small=piece[i:i+self.engine.batch];n=len(small)
                with torch.no_grad():
                    logits,h=self.engine.forward(small);targets=[data.answer(c) for c in small]
                    predicted=score(self.engine,logits[:n],targets);native=score(self.engine,h['native'][:n],targets)
                v=dict(ap=h['ap'][:n].cpu().to(torch.bfloat16),c=h['c'][:n].cpu().to(torch.bfloat16),
                    native_correct=native['correct'],native_token_id=native['token_id'],**predicted)
                v['eligible']=v['correct']&v['native_correct']
                for k,x in v.items():values.setdefault(k,[]).append(x)
            tensor={k:torch.cat(x) for k,x in values.items()};path=self.folder/f'{self.next:07d}.pt'
            save(path,tensor);write(path.with_suffix('.ids.json'),[c['id'] for c in piece])
            for i,c in enumerate(piece):self.index[c['id']]=(path,i)
            self.next+=1

    def field(self,contexts,key):
        return torch.stack([self.chunk(self.index[c['id']][0])[key][self.index[c['id']][1]] for c in contexts])

    def layout(self,rows,label):
        for side in ('base','source'):
            for start in range(0,len(rows),self.engine.batch):
                cs=[r[side] for r in rows[start:start+self.engine.batch]];n=len(cs)
                with torch.no_grad():y,h=self.engine.forward(cs)
                for scores,field in ((score(self.engine,y[:n],[data.answer(c) for c in cs]),'token_id'),
                    (score(self.engine,h['native'][:n],[data.answer(c) for c in cs]),'native_token_id')):
                    if not scores['correct'].all() or not torch.equal(scores['token_id'],self.field(cs,field)):
                        write(self.root/'engineering'/f'layout_failure_{time.time_ns()}.json',dict(label=label,side=side,ids=[c['id'] for c in cs],tokens=scores['token_id'].tolist()))
                        raise ValueError('Matched natural layout failed: '+label)
        write(self.root/'engineering/layout'/f'{label}.json',dict(passed=True,count=len(rows)))

def locked(root):
    x=read(Path(root)/'locks/all_final.json')
    if len(x['slots'])!=6 or len(x['endpoints'])!=len(x['unique_configurations'])*3:
        raise ValueError('Final data sealed: confirmation endpoints incomplete')
    for v in x['endpoints'].values():
        if file_hash(v['checkpoint'])!=v['sha256']:raise ValueError('Locked endpoint changed')

def filter_bank(root,bank,cache):
    root=Path(root)
    if bank in ('final','c_diagnostic'):locked(root)
    folder=root/'banks'/bank
    if (folder/'completion.json').exists():return load_bank(root,bank)
    groups=read(root/'groups.json');slots=read(root/'proposals'/f'{bank}.json')
    excluded=set(read(root/'historical_exclusions.json'));used=set();selected=[];started=time.perf_counter()
    for start in range(0,len(slots),32):
        path=folder/f'page_{start:06d}.json'
        if path.exists():
            selected.extend(read(path)['selected']);used.update(r['id'] for r in selected);continue
        pending={s['slot']:s for s in slots[start:start+32]};accepted={};dispositions=[]
        orders={k:data.alternative_order(s) for k,s in pending.items()}
        for rank in range(256):
            proposals={k:data.proposal(s,orders[k][rank],groups) for k,s in pending.items()}
            candidates={};occupied={}
            for r in selected:
                m=r['metadata'];occupied.setdefault((m['from_AP'],m['to_AP']),set()).add(r['base']['question'])
            for rr in accepted.values():
                for r in rr:
                    m=r['metadata'];occupied.setdefault((m['from_AP'],m['to_AP']),set()).add(r['base']['question'])
            for k,rows in proposals.items():
                s=pending[k];duplicate=any(r['id'] in used for r in rows)
                history=any(c['id'] in excluded for r in rows for c in (r['base'],r['source']))
                coverage=bank=='final' and s['kind']=='changing' and s['ordinal']<32 and rows[0]['base']['question'] in occupied.get((s['b'],s['s']),set())
                if not (duplicate or history or coverage):candidates[k]=rows
                dispositions.append(dict(slot=k,alternative=orders[k][rank],rank=rank,duplicate=duplicate,historical_overlap=history,coverage_excluded=coverage))
            cache.ensure([c for rows in candidates.values() for r in rows for c in (r['base'],r['source'])])
            for k,rows in candidates.items():
                clean=bool(cache.field([c for r in rows for c in (r['base'],r['source'])],'eligible').all())
                s=pending[k];cell=(s['b'],s['s']);obj=rows[0]['base']['question']
                # Recheck coverage after earlier acceptances in this same inference batch.
                coverage=bank=='final' and s['kind']=='changing' and s['ordinal']<32 and obj in occupied.get(cell,set())
                dispositions.append(dict(slot=k,alternative=orders[k][rank],rank=rank,clean=clean,retained=clean and not coverage))
                if clean and not coverage and not any(r['id'] in used for r in rows):
                    accepted[k]=rows;used.update(r['id'] for r in rows);occupied.setdefault(cell,set()).add(obj);del pending[k]
            if not pending:break
        if pending:
            write(folder/f'shortfall_{start:06d}.json',dict(unfilled=list(pending),accepted=accepted,dispositions=dispositions))
            raise RuntimeError(f'Frozen 256-bundle pool exhausted: {bank} {list(pending)}')
        records=[r for s in slots[start:start+32] for r in accepted[s['slot']]]
        write(path,dict(selected=records,dispositions=dispositions));selected.extend(records)
        status(root,'filtering',bank=bank,accepted_slots=start+len(accepted),total_slots=len(slots),natural_prompts=len(cache.index))
    audit=data.audit(selected,bank);write(folder/'pairs.json',selected)
    write(folder/'completion.json',dict(audit=audit,pairs_sha256=file_hash(folder/'pairs.json'),seconds=time.perf_counter()-started,naturally_clean_only=True))
    cache.layout(selected,bank);return selected

def load_bank(root,bank):
    folder=Path(root)/'banks'/bank;receipt=read(folder/'completion.json')
    if file_hash(folder/'pairs.json')!=receipt['pairs_sha256']:raise ValueError('Bank hash changed')
    return read(folder/'pairs.json')


