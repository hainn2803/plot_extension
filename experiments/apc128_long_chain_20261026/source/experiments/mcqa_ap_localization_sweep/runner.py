"""Signed, guarded orchestration of the five registered stages."""
import argparse
import gc
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from experiments.mcqa_at_anchor_comparison.storage import read,write,status,file_hash
from diagnostics.ap_specialization_routing_20261023.counter_transport import use_native_counter_environment,verify_live_counter
from diagnostics.ap_specialization_routing_20261023.library_initialization import initialize
from . import data,banks,training,evaluation,verification
from .engine import Engine,StateCache,profile
from .execution import Session
from .sites import Site
from .protocol import ROOT,DIAGNOSIS,SEEDS

def estimate(root,p,stage):
    root=Path(root);sites=read(root/'sites.json');times=p['times']
    def cost(site,rank,field):
        kind=site.kind if site.count==4 else 'offset'
        tt=sorted([t for t in times if t['kind']==kind and t['rank']==rank],key=lambda t:t['layer'])
        return float(np.interp(site.layer,[t['layer'] for t in tt],[t[field] for t in tt]))
    seconds=0.;operations={};map_seconds=0.;screen_seconds=0.
    for record in sites['single']:
        site=Site(**record)
        if not (root/'maps'/f'{site.id}.pt').exists():map_seconds+=512*cost(site,128,'inference')
    for record in sites['single']+sites['distributed']:
        site=Site(**record);folder=root/'screens'/(site.id+'_R0128_shared_S1')
        if (folder/'completion.json').exists():continue
        step=read(folder/'latest.json')['step'] if (folder/'latest.json').exists() else 0
        screen_seconds+=(50-step)*cost(site,128,'update')+1536*cost(site,128,'inference')
    if (root/'selection.json').exists():configs=list(read(root/'selection.json')['unique_configurations'].values())
    else:
        configs=[dict(site=Site(0,'offset',0).record(),rank=128,arm=a) for a in ('shared','edge','middle','shared')]
        configs += [dict(site=Site(0,'joint_color').record(),rank=128,arm='shared'),dict(site=Site(0,'offset',0).record(),rank=512,arm='shared')]
    fit_seconds=0.;final_seconds=0.
    for config in configs:
        site=Site(**config['site']);rank=config['rank'];arm=config['arm']
        for seed in SEEDS:
            name=site.id+f'_R{rank:04d}_{arm}_S{seed}';folder=root/'fits'/name
            if not (folder/'completion.json').exists():
                step=read(folder/'latest.json')['step'] if (folder/'latest.json').exists() else 0
                fit_seconds+=(1000-step)*cost(site,rank,'update')+6*(512+(1024 if arm in ('edge','middle') else 1792))*cost(site,rank,'inference')
            final_seconds+=(1792*12+256*12)*cost(site,rank,'inference')
    attempts=[]
    for bank in ('train','calibration'):
        for page in (root/'banks'/bank).glob('page_*.json'):
            value=read(page)
            for slot in {r['metadata']['slot'] for r in value['selected']}:
                attempts.append(1+max(d['rank'] for d in value['dispositions'] if d['slot']==slot))
    average=max(5.,2*sum(attempts)/max(1,len(attempts)))
    natural=max(t['natural'] for t in times)
    filtering=sum(len(data.slots(b))*3*average*natural for b in data.SIZES if not (root/'banks'/b/'completion.json').exists())
    filtering+=sum(len_expected*10*average*natural for name,len_expected in [('final',1792),('C',256)] if not (root/'controls'/name/'completion.json').exists())
    # Account for full-state cache transfers, disk receipts and reports, not just updates.
    cache_seconds=30000*natural+1800
    operations=dict(mapping=map_seconds,screening=screen_seconds,confirmation=fit_seconds,final_evaluation=final_seconds,
        filtering=filtering,natural_caches_and_IO=cache_seconds)
    seconds=sum(operations.values())
    result=dict(stage=stage,components=operations,total_seconds=seconds,filtering_attempts=average,
        timing='Piecewise interpolation over measured early/middle/late layers; unselected confirmation uses early-layer bound',
        completed_fits_and_maps_subtracted=True,remaining_evaluation_conservative=True)
    write(root/'engineering/estimates'/f'{time.time_ns()}.json',result)
    return result

def mapping(root,engine,master,natural,states):
    rows=training.subset(master);write(Path(root)/'map_pairs.json',rows)
    for item in read(Path(root)/'sites.json')['single']:
        site=Site(**item);path=Path(root)/'maps'/f'{site.id}.pt'
        if path.exists():load_check=training.saved_evaluate(path,None);continue
        donor=states.get([r['source'] for r in rows],site)
        training.saved_evaluate(path,lambda:training.evaluate(engine,rows,donor,None,site,natural))
        status(root,'mapping',site=site.id)

def screens(root,engine,master,cal,natural,states,joint=False):
    for item in read(Path(root)/'sites.json')['distributed' if joint else 'single']:
        training.fit(root,engine,Site(**item),128,'shared',1,50,master,cal,natural,states,screen=True)

def confirmation(root,engine,master,cal,natural,states):
    selected=read(Path(root)/'selection.json')
    for seed in SEEDS:
        for config in selected['unique_configurations'].values():
            training.fit(root,engine,Site(**config['site']),config['rank'],config['arm'],seed,1000,master,cal,natural,states)
    training.lock(root)

def gpu(root,command,bank=None):
    root=Path(root);data.verify(root,model=True);session=Session(root,'profile' if not (root/'profile.json').exists() else 'run')
    engine=None;attempt=root/'engineering/attempts'/str(time.time_ns())
    try:
        with use_native_counter_environment(attempt):
            initialize(attempt,DIAGNOSIS);engine=Engine(root,session);torch.cuda.synchronize()
            write(attempt/'live_counter_receipt.json',verify_live_counter())
            p=profile(root,engine);engine.batch=p['batch'];session.transition('run')
            natural=banks.NaturalCache(root,engine);states=StateCache(root,engine)
            if command=='profile':return
            if command=='filter':banks.filter_bank(root,bank,natural);return
            if command=='evaluate':
                gate=estimate(root,p,'evaluate');session.allow(gate['total_seconds'],'Complete remaining evaluation')
                evaluation.evaluate_all(root,engine,natural,states);return
            master=banks.filter_bank(root,'train',natural);cal=banks.filter_bank(root,'calibration',natural)
            for label,action in [('map',lambda:mapping(root,engine,master,natural,states)),
                ('screen',lambda:screens(root,engine,master,cal,natural,states)),
                ('distributed',lambda:screens(root,engine,master,cal,natural,states,True)),
                ('select',lambda:training.select(root)),
                ('train',lambda:confirmation(root,engine,master,cal,natural,states)),
                ('evaluate',lambda:evaluation.evaluate_all(root,engine,natural,states))]:
                if command!='run' and command!=label:continue
                if label!='select':
                    remaining=estimate(root,p,label);session.allow(remaining['total_seconds'],'Complete five-stage remaining workload before '+label)
                    status(root,'launch_gate_passed',stage_name=label,estimated_hours=remaining['total_seconds']*1.25/3600)
                action()
            if command=='run':write(root/'gpu_completion.json',dict(completed=True,seconds=session.ledger()['seconds']))
    finally:
        session.close();del engine;gc.collect()
        if torch.cuda.is_initialized():torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','verify','profile','filter','map','screen','distributed','select','train','evaluate','report','run'])
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--bank',choices=list(data.SIZES))
    args=parser.parse_args();root=args.root
    try:
        if args.command in ('prepare','run'):data.prepare(root)
        if args.command in ('verify','run'):verification.cpu(root)
        if args.command=='select':data.verify(root);training.select(root)
        elif args.command in ('profile','filter','map','screen','distributed','train','evaluate','run'):gpu(root,args.command,args.bank)
        if args.command in ('report','run'):
            from .report import report
            report(root)
    except BaseException as error:
        write(root/'terminal.json',dict(type=type(error).__name__,reason=str(error),traceback=traceback.format_exc(),time=time.time()),immutable=False)
        status(root,'stopped',reason=str(error),automatic_restart=False)
        from .report import partial
        partial(root)
        raise

if __name__=='__main__':main()
