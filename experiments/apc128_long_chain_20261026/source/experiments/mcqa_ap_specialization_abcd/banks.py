"""Resumable natural-only filtering; final bank stays sealed until all fits lock."""
from functools import lru_cache
from pathlib import Path
import time

import torch

from experiments.mcqa_at_anchor_comparison.storage import read,write,save,load,digest,file_hash,status
from . import data

class NaturalCache:
    def __init__(self,root,engine):
        self.root=Path(root);self.engine=engine;self.folder=self.root/'natural_cache';self.index={}
        self.folder.mkdir(parents=True,exist_ok=True)
        self.identity=digest(dict(run=data.verify_identity(root),execution=read(self.root/'execution.json')))
        write(self.folder/'identity.json',dict(identity=self.identity))
        for p in sorted(self.folder.glob('chunk_*.ids.json')):
            metadata=read(p);path=p.with_name(p.name.replace('.ids.json','.pt'))
            if metadata['identity']!=self.identity:raise ValueError('Natural cache execution changed')
            if file_hash(path)!=read(path.with_suffix('.receipt.json'))['sha256']:raise ValueError('Natural cache corruption')
            for i,key in enumerate(metadata['ids']):
                if key in self.index:raise ValueError('Duplicate natural cache ID')
                self.index[key]=(path,i)
        self.next=max([int(p.stem.split('_')[1]) for p in self.folder.glob('chunk_*.pt')],default=-1)+1

    @lru_cache(maxsize=16)
    def chunk(self,path):return load(path)

    def ensure(self,contexts):
        rows=list({c['id']:c for c in contexts if c['id'] not in self.index}.values())
        for start in range(0,len(rows),128):
            piece=rows[start:start+128];states=[];scores=[]
            for i in range(0,len(piece),self.engine.batch):
                small=piece[i:i+self.engine.batch];n=len(small)
                with torch.no_grad():
                    logits,ap=self.engine.forward(small)
                    score=self.engine.score(logits[:n],[data.answer(c) for c in small])
                states.append(ap[:n].cpu().to(torch.bfloat16));scores.append(score)
            value=dict(ap=torch.cat(states))
            for key in scores[0]:value[key]=torch.cat([torch.as_tensor(s[key]) for s in scores])
            path=self.folder/f'chunk_{self.next:07d}.pt';save(path,value)
            write(path.with_suffix('.ids.json'),dict(ids=[c['id'] for c in piece],identity=self.identity))
            for i,c in enumerate(piece):self.index[c['id']]=(path,i)
            self.next+=1

    def field(self,contexts,key):
        return torch.stack([self.chunk(self.index[c['id']][0])[key][self.index[c['id']][1]] for c in contexts])

    def audit_layout(self,rows,label):
        for side in ('base','source'):
            for start in range(0,len(rows),self.engine.batch):
                small=rows[start:start+self.engine.batch];cs=[r[side] for r in small]
                with torch.no_grad():
                    logits,_=self.engine.forward(cs);score=self.engine.score(logits[:len(cs)],[data.answer(c) for c in cs])
                if not bool(score['correct'].all()) or not torch.equal(torch.as_tensor(score['token_id']),self.field(cs,'token_id')):
                    write(self.root/'engineering'/f'clean_layout_failure_{time.time_ns()}.json',dict(label=label,side=side,start=start,
                        ids=[c['id'] for c in cs],correct=score['correct'].tolist(),tokens=score['token_id'].tolist()))
                    raise ValueError('Natural correctness/layout stability failed; no outcome-based replacement: '+label)
        write(self.root/'engineering/layout_audits'/f'{label}.json',dict(passed=True,pairs=len(rows),batch=self.engine.batch))

def seal_check(root):
    lock=read(Path(root)/'locks/all_final.json')
    if len(lock['runs'])!=9:raise ValueError('Final data remain sealed')
    for x in lock['runs'].values():
        if file_hash(x['checkpoint'])!=x['sha256']:raise ValueError('Locked final checkpoint changed')
        load(x['checkpoint'])

def filter_bank(root,bank,cache):
    root=Path(root)
    if bank=='final':seal_check(root)
    folder=root/'banks'/bank;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():return load_bank(root,bank)
    slots=read(root/'proposals'/f'{bank}.slots.json');groups=read(root/'groups.json')
    excluded=set(read(root/'historical_exclusions.json'));selected=[];used=set();start_time=time.perf_counter()
    for start in range(0,len(slots),64):
        page=folder/f'page_{start:06d}.json'
        if page.exists():
            records=read(page)['selected'];selected.extend(records);used.update(r['id'] for r in records);continue
        block=slots[start:start+64];pending={s['slot']:s for s in block};accepted={};dispositions=[]
        orders={s['slot']:data.alternative_order(s) for s in block}
        for rank in range(256):
            proposals=[(s,data.proposal(s,orders[s['slot']][rank],groups)) for s in pending.values()]
            eligible=[r for _,r in proposals if not any(r[side]['id'] in excluded for side in ('base','source')) and r['id'] not in used]
            cache.ensure([r[side] for r in eligible for side in ('base','source')])
            for s,r in proposals:
                history=any(r[side]['id'] in excluded for side in ('base','source'));duplicate=r['id'] in used
                clean=False if history or duplicate else bool(cache.field([r['base'],r['source']],'correct').all())
                keep=clean and not history and not duplicate
                dispositions.append(dict(slot=s['slot'],alternative=r['alternative'],hash_order_rank=rank,
                    historical_overlap=history,duplicate=duplicate,clean=clean,retained=keep))
                if keep:accepted[s['slot']]=r;used.add(r['id']);del pending[s['slot']]
            if not pending:break
        if pending:
            write(folder/f'shortfall_{start:06d}.json',dict(unfilled=list(pending),accepted=list(accepted.values()),
                dispositions=dispositions,alternatives=256,pool_unchanged=True))
            raise RuntimeError('Finite-pool shortfall: '+bank+' '+str(list(pending)))
        chunk=[accepted[s['slot']] for s in block]
        write(page,dict(selected=chunk,dispositions=dispositions,pool_sha256=file_hash(root/'proposals'/f'{bank}.pool.json')))
        selected.extend(chunk);status(root,'filtering',bank=bank,retained=len(selected),total=len(slots))
    check=data.audit(selected,bank,groups,retained=True)
    source=cache.field([r['source'] for r in selected],'ap').float()
    save(folder/'source_ap.pt',source);write(folder/'pairs.json',selected)
    write(folder/'completion.json',dict(bank=bank,count=len(selected),audit=check,strict_clean=True,
        seconds=time.perf_counter()-start_time,pairs_sha256=file_hash(folder/'pairs.json'),
        source_sha256=file_hash(folder/'source_ap.pt'),selection='Metadata/hash order and natural correctness only'))
    return selected,source

def load_bank(root,bank):
    folder=Path(root)/'banks'/bank;r=read(folder/'completion.json')
    if file_hash(folder/'pairs.json')!=r['pairs_sha256'] or file_hash(folder/'source_ap.pt')!=r['source_sha256']:
        raise ValueError('Retained bank changed')
    rows=read(folder/'pairs.json');assert len(rows)==r['count']
    return rows,load(folder/'source_ap.pt')

def training_view(master,arm):
    rows,source=master
    keep=[i for i,r in enumerate(rows) if r in data.view([r],arm)]
    result=[rows[i] for i in keep]
    if len(result)!=832:raise ValueError('Training view count mismatch')
    return result,source[keep]
