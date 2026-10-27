"""User-authorized evaluation-only continuation after the exact final-phase AC stop."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import gc
import os
import shutil
import sys
import time
import traceback
from unittest.mock import patch

from .full_transfer_plan import ROOT,OUTPUT,TOTAL_LIMIT_SECONDS

STOP_TIME = 1793060266.819456
STOP_SECONDS = 31952.714111800044
STOP_CREATED = 1793048771.6270792


def require_stop(process,ledger,pause,terminal):
    if process['pid']!=31248 or abs(process['created']-STOP_CREATED)>=.01:
        raise ValueError('Only the explicitly authorized actual AC-stopped worker may resume')
    if process['cmdline'][-4:]!=['-B','-m','diagnostics.apc_128_20261026.resume_last_fit','run']:
        raise ValueError('Stopped worker command changed')
    if pause!=dict(reason='AC power disconnected',time=STOP_TIME):
        raise ValueError('Only the exact user-authorized AC pause may be archived')
    if terminal['pid']!=31248 or terminal['type']!='Paused' or terminal['reason']!='AC power disconnected; resume from the last committed checkpoint':
        raise ValueError('Not the authorized AC terminal condition')
    if ledger['active'] or ledger['pid']!=31248 or ledger['seconds']!=STOP_SECONDS or ledger['profile_seconds']!=853.1099996000121 or ledger['total_limit_seconds']!=43200:
        raise ValueError('Exact closed cumulative ledger required, without resetting charges')


def final_access_writer(original,root,access):
    def write(path,value,immutable=True):
        if Path(path)==Path(root)/'final_access.json':
            if {k:v for k,v in value.items() if k!='time'}!={k:v for k,v in access.items() if k!='time'}:
                raise ValueError('Final-access metadata differs; timestamp alone may be retained')
            if value['time']<access['time']:
                raise ValueError('Final-access clock moved backwards')
            return original(path,access,immutable=immutable)
        return original(path,value,immutable=immutable)
    return write


def frozen_bank(root,bank,cache):
    from experiments.mcqa_ap_gradual_chain_20261025 import banks
    from experiments.mcqa_at_anchor_comparison.storage import read
    if Path(root)!=ROOT or bank not in ('AT_final','AP_final','C_final'):
        raise ValueError('Only the already completed final-preparation phase may be reused')
    rows=banks.load_bank(root,bank)
    proof=read(ROOT/'engineering/layout'/(bank+'.json'))
    if proof!=dict(passed=True,count=len(rows)):
        raise ValueError('Completed final-bank matched-layout proof missing')
    return rows


def frozen_controls(root,rows,cache,bank):
    from .full_transfer_plan import controls_for
    from experiments.mcqa_ap_localization_sweep.controls import ROLES,rows_for
    from experiments.mcqa_at_anchor_comparison.storage import read
    if Path(root)!=ROOT or bank not in ('AP','C'):
        raise ValueError('Unexpected completed control-preparation phase')
    controls=controls_for(root,bank)
    if set(controls)!={r['id'] for r in rows}:
        raise ValueError('Control parent inventory changed')
    for role in ROLES:
        rr,ix=rows_for(rows,controls,role)
        if len(rr)!=len(rows) or ix!=list(range(len(rows))):
            raise ValueError('A missing control cannot shrink the denominator')
        if read(ROOT/'engineering/layout'/(bank+'_'+role+'.json'))!=dict(passed=True,count=len(rows)):
            raise ValueError('Completed control matched-layout proof missing')
    return controls


def remaining_estimate():
    from .full_transfer import estimate
    from experiments.mcqa_ap_gradual_chain_20261025.evaluation import inventory
    from experiments.mcqa_ap_gradual_chain_20261025.protocol import BANK_SIZES
    extra=estimate()
    saved={p.stem for p in (ROOT/'evaluation').glob('*.pt') if p.with_suffix('.json').exists()}
    missing={k:v for k,v in inventory().items() if k not in saved}
    full,tail=extra['observed_max_full_seconds_per_row'],extra['observed_max_tail_seconds_per_row']
    remaining=sum(BANK_SIZES[v]*(tail if k.startswith(('AT_','clean_')) or k.endswith(
        ('_AT_replay','_AT_chain','_AT_parallel','_AT_restore','_conflict0','_conflict1')) else full) for k,v in missing.items())
    validation=extra['components']['fresh_full_reference_validation']
    total=remaining+validation+extra['total_seconds']
    return dict(saved=len(saved),missing=list(missing),remaining_original_seconds=remaining,
        fresh_original_resume_validation_seconds=validation,supplement=extra,
        complete_remaining_seconds=total,complete_remaining_with25percent=total*1.25)


def prove():
    import torch
    from transformers import AutoTokenizer
    from experiments.mcqa_apc_128_20261026 import registration,auditing
    from experiments.mcqa_ap_gradual_chain_20261025 import banks,training,verification
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_at_anchor_comparison.storage import read,write,load,file_hash
    from experiments.mcqa_c_at_localization.runtime import power_status
    from experiments.mcqa_c_at_localization.profiling import thermals
    from .full_transfer_plan import verify_plan
    from .evaluation_resume_tests import tests
    output=ROOT/'engineering/authorized_evaluation_resumes'/str(time.time_ns())
    output.mkdir(parents=True)
    identity=registration.verify(ROOT,model=True)
    verify_plan()
    process,ledger,pause,terminal=(read(ROOT/n) for n in ('latest_process.json','engineering/gpu_budget.json','pause.request','terminal.json'))
    require_stop(process,ledger,pause,terminal)
    registration.require_exited(process)
    old_queue=read(OUTPUT/'queue.json')
    if old_queue['pid']!=41152 or abs(old_queue['created']-1793058205.576761)>=.01:
        raise ValueError('Unexpected old addendum controller')
    registration.require_exited(old_queue)
    queue_terminal=read(OUTPUT/'terminal.json')
    if queue_terminal['pid']!=41152 or queue_terminal['reason']!='Original worker stopped or paused; no automatic continuation allowed':
        raise ValueError('Addendum did not stop solely for the original AC pause')
    power,thermal=power_status(),thermals()
    if power['ACLineStatus']!=1 or thermal['active']:
        raise RuntimeError('Connected AC and clear thermals required')
    tests_value=tests(output)
    lock=banks.locked(ROOT)
    access=read(ROOT/'final_access.json')
    if access['time']<=lock['time']:
        raise ValueError('Final access was not after the frozen endpoint/readout lock')
    fits={}
    orders=0
    for path in sorted((ROOT/'fits').glob('*/completion.json')):
        done=read(path)
        wrapper=load(done['checkpoint'])
        p=wrapper['payload']
        if file_hash(done['checkpoint'])!=done['sha256'] or tensor_digest(p)!=wrapper['payload_digest'] or p['step']!=1000 or len(p['updates'])!=1000:
            raise ValueError('Frozen endpoint digest or update count failed')
        cfg=p['identity']
        initial=training.Subspace(cfg['seed'],width=9216 if cfg['variable']=='AP' else 2304)
        if tensor_digest(initial.state_dict())!=p['initialization']:
            raise ValueError('Original four-thread initialization changed')
        rows=banks.load_bank(ROOT,cfg['variable']+'_train')
        for i,u in enumerate(p['updates']):
            if u['indices']!=training.order(rows,cfg['variable'],cfg['seed'],i):
                raise ValueError('Saved original update orders changed')
        if any(int(v['step'])!=1000 for v in p['optimizer']['state'].values()) or not all(k in p for k in ('cpu_rng','cuda_rng','numpy_rng','python_rng')):
            raise ValueError('Frozen optimizer/RNG evidence changed')
        fits[path.parent.name]=done
        orders+=1000
    if len(fits)!=15 or orders!=15000:
        raise ValueError('All15 completed fits and15000 original orders required; no training may resume')
    for variable in ('AP','C'):
        for seed in (1,0,2):
            a,b=(fits[f'{variable}_{m}_S{seed}'] for m in ('standard','gradual'))
            if any(a[k]!=b[k] for k in ('initialization','order_digest','preservation_digest')):
                raise ValueError('Completed paired initialization/orders/preservation changed')
    parents={b:frozen_bank(ROOT,b+'_final',None) for b in ('AT','AP','C')}
    for bank in ('AP','C'):
        frozen_controls(ROOT,parents[bank],None,bank)
    tokenizer=AutoTokenizer.from_pretrained(identity['model_root'],local_files_only=True)
    from experiments.mcqa_at_anchor_comparison.engine import answer_label_ids
    label_ids=list(answer_label_ids(tokenizer))
    decoded={}
    predictions=0
    for path in sorted((ROOT/'evaluation').glob('*.pt')):
        arrays=load(path)
        rr=read(ROOT/'evaluation_rows'/(path.stem+'.json'))['rows']
        expected_ids=[label_ids[list('ABCDEFGHIJKLMNOPQRSTUVWXYZ').index(r['target'])] for r in rr]
        if list(arrays['label_ids'])!=label_ids or arrays['target_token_ids']!=expected_ids:
            raise ValueError('Independent pinned-tokenizer target identifiers changed')
        if arrays['rows_digest']!=verification.digest(rr) or arrays['pair_ids']!=[r['id'] for r in rr] or arrays['targets']!=[r['target'] for r in rr]:
            raise ValueError('Saved prediction alignment failed')
        for token in arrays['token_id'].tolist():
            if token not in decoded:
                decoded[token]=tokenizer.decode([token]).strip()
        ties=arrays['top_two_logits'][:,0]==arrays['top_two_logits'][:,1]
        correct=torch.tensor([decoded[int(t)]==r['target'] and not bool(tie) for r,t,tie in zip(rr,arrays['token_id'],ties)])
        if not torch.equal(ties,arrays['tied']) or not torch.equal(arrays['token_id'],arrays['top_two_token_ids'][:,0]) or not torch.equal(correct,arrays['correct']):
            raise ValueError('Saved independent token/top-two/tie audit failed')
        with patch.object(verification,'compare',auditing.compare):
            verification.audit_metrics(rr,arrays,read(path.with_suffix('.json')))
        predictions+=1
    if predictions!=175:
        raise ValueError('The exact AC-stopped175-output inventory is required')
    estimate=remaining_estimate()
    available=TOTAL_LIMIT_SECONDS-300-ledger['seconds']
    if estimate['complete_remaining_with25percent']>available:
        raise RuntimeError('Complete original continuation plus all144 extra outputs and both fresh validations cannot fit the authorized13h cap')
    immutable={str(ROOT/n):file_hash(ROOT/n) for n in ('identity.json','profile.json','execution.json','locks/all_final.json','final_access.json','AT_policy.json')}
    for folder in ('fits','readouts','banks','controls','proposals','natural_cache','evaluation','evaluation_rows'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file():
                immutable[str(p)]=file_hash(p)
    stop_files={}
    for base,names,tag in ((ROOT,('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json','engineering/gpu_budget.json'),'original'),
                          (OUTPUT,('terminal.json','queue.json','controller_launch.json','controller.stdout.log','controller.stderr.log'),'controller')):
        for name in names:
            p=base/name
            target=output/'prior_stop'/tag/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
            if file_hash(p)!=file_hash(target):
                raise ValueError('AC-stop evidence copy failed')
            stop_files[str(p)]=dict(sha256=file_hash(p),archived_copy=str(target))
    if torch.cuda.is_initialized():
        raise ValueError('CPU resume preflight initialized CUDA')
    value=dict(passed=True,authorization='Direct user: AC reconnected; please resume.',
        prior_process=process,prior_queue=old_queue,prior_pause=pause,prior_terminal=terminal,
        closed_ledger=ledger,power=power,thermals=thermal,verified_fits=15,verified_orders=orders,
        independently_scored_saved_outputs=predictions,remaining_original_outputs=68,
        final_preparation_phase_complete_and_reused=True,final_access=access,
        original_lock_sha256=file_hash(ROOT/'locks/all_final.json'),immutable_old_scientific_artifacts=immutable,
        original_supplement_plan_sha256=file_hash(OUTPUT/'plan.json'),stop_files=stop_files,
        complete_remaining_gate=dict(allowed=True,available_seconds=available,estimate=estimate),
        new_training_updates=0,all_prior_charges_retained=True,source_files={str(p):file_hash(p) for p in
        (Path(__file__),Path(__file__).with_name('evaluation_resume_tests.py'))},CPU_tests=tests_value,cuda_initialized=False,time=time.time())
    write(output/'authorization.json',value)
    print('EVALUATION_RESUME_PROOF '+str(output/'authorization.json'),flush=True)
    return value,output


def check_authorization(path):
    from experiments.mcqa_at_anchor_comparison.storage import read,file_hash
    value=read(path)
    if not value['passed'] or value['cuda_initialized'] or value['new_training_updates'] or value['verified_orders']!=15000:
        raise ValueError('Complete CPU evaluation-only resume proof required')
    for p,expected in {**value['immutable_old_scientific_artifacts'],**value['source_files']}.items():
        if file_hash(p)!=expected:
            raise ValueError('Authorized evaluation-resume input changed: '+p)
    if file_hash(OUTPUT/'plan.json')!=value['original_supplement_plan_sha256']:
        raise ValueError('Original full transfer plan changed')
    return value


def run(path):
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
    from experiments.mcqa_apc_128_20261026 import runner,registration
    from experiments.mcqa_ap_gradual_chain_20261025 import evaluation,profiling,banks
    from experiments.mcqa_ap_gradual_chain_20261025.engine import Engine,NaturalCache
    from experiments.mcqa_ap_gradual_chain_20261025.execution import Session
    from experiments.mcqa_ap_gradual_chain_20261025.protocol import DIAGNOSIS
    from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
    from diagnostics.ap_localization_sweep_20261024.scoped_counter import use_scoped_counter,verify_live_counter
    from diagnostics.ap_specialization_routing_20261023.library_initialization import initialize
    from experiments.mcqa_c_at_localization.runtime import power_status
    from experiments.mcqa_c_at_localization.profiling import thermals
    from .full_transfer import check_manifest
    auth=check_authorization(path)
    output=Path(path).parent
    registration.require_exited(auth['prior_process'])
    registration.require_exited(auth['prior_queue'])
    require_stop(*(read(ROOT/n) for n in ('latest_process.json','engineering/gpu_budget.json','pause.request','terminal.json')))
    if power_status()['ACLineStatus']!=1 or thermals()['active']:
        raise RuntimeError('Connected AC and clear thermals required immediately before resume')
    for name in ('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json'):
        source=ROOT/name
        if file_hash(source)!=auth['stop_files'][str(source)]['sha256']:
            raise ValueError('Stopped original worker evidence changed')
        target=output/'archived_root'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
    for name in ('terminal.json','queue.json','controller_launch.json','controller.stdout.log','controller.stderr.log'):
        source=OUTPUT/name
        if file_hash(source)!=auth['stop_files'][str(source)]['sha256']:
            raise ValueError('Stopped controller evidence changed')
        target=output/'archived_controller'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        source.rename(target)
    process=runner.stamp(ROOT)
    write(output/'resumed_process.json',process)
    session=Session(ROOT,'evaluation_resume_validation')
    engine=None
    cache=None
    completed=False
    try:
        with use_scoped_counter(ROOT/'engineering/attempts'/str(time.time_ns())):
            initialize(output,DIAGNOSIS)
            engine=Engine(ROOT,session)
            write(output/'live_counter.json',verify_live_counter())
            cache=NaturalCache(ROOT,engine)
            class NoNewCache(type(cache)):
                def ensure(self,contexts):
                    if any(c['id'] not in self.index for c in contexts):
                        raise ValueError('Evaluation resume may not generate new prompts or natural caches')
            cache.__class__=NoNewCache
            original_write=runner.write
            def teacher_writer(p,v,immutable=True):
                p=Path(p)
                if p.parent==ROOT/'engineering/actual_teacher_validation':
                    p=output/'fresh_actual_teachers'/p.name
                return original_write(p,v,immutable=immutable)
            with MemoryWatch() as watch:
                session.memory_watch=watch
                try:
                    profiling.profile(ROOT,engine,cache)
                    for variable in ('AT','AP'):
                        for split in ('train','calibration'):
                            bank='readout_'+variable+'_'+split
                            cache.layout(banks.load_bank(ROOT,bank),'evaluation_resume_'+bank)
                    with patch.object(runner,'write',teacher_writer):
                        for variable in ('AP','C'):
                            for seed in (1,0,2):
                                runner.validate_teachers(ROOT,engine,cache,variable,seed)
                    from experiments.mcqa_ap_gradual_chain_20261025 import training
                    from experiments.mcqa_at_anchor_comparison.engine import checked
                    from torch.nn import functional as F
                    next_rows=evaluation.originals(frozen_bank(ROOT,'AP_final',cache))[:32]
                    states=cache.states(next_rows)
                    q=training.basis(ROOT,'AP','standard',2,engine.device)
                    donor=dict(values=states['source_AP'],available=torch.ones(32,dtype=torch.bool))
                    with torch.no_grad():
                        y,h=engine.forward([r['base'] for r in next_rows],donor=donor,q=q,restore=True)
                        yr,hr=engine.forward([r['base'] for r in next_rows],donor=donor,q=q,restore=True,reference=True,full=True)
                        ids=engine.targets(next_rows)[0]
                        next_checks=[checked('actual_AP_standard_S2_restore_logits',y,yr),
                            checked('actual_AP_standard_S2_restored_state',h['site'],hr['site']),
                            checked('actual_AP_standard_S2_restore_full_vocab_CE',F.cross_entropy(y,ids),F.cross_entropy(yr,ids))]
                    write(output/'next_operation_validation.json',dict(passed=True,operation='AP_standard_S2_restore',
                        checks=next_checks,full_reference_batch=32,new_training_updates=0))
                    final_proofs=[]
                    from experiments.mcqa_ap_localization_sweep.controls import ROLES,rows_for
                    for bank in ('AT','AP','C'):
                        rr=frozen_bank(ROOT,bank+'_final',cache)
                        cache.layout(rr[:32],'evaluation_resume_spot_'+bank+'_final')
                        final_proofs.append(dict(bank=bank,old_full_layout_count=len(rr),new_process_spot_check_count=32))
                        if bank!='AT':
                            control=frozen_controls(ROOT,rr,cache,bank)
                            for role in ROLES:
                                small,_=rows_for(rr[:32],control,role)
                                cache.layout(small,'evaluation_resume_spot_'+bank+'_'+role)
                    estimate=remaining_estimate()
                    after=session.ledger()
                    remaining=estimate['remaining_original_seconds']+estimate['supplement']['total_seconds']
                    if remaining*1.25>TOTAL_LIMIT_SECONDS-300-after['seconds']:
                        raise RuntimeError('Full remaining original+supplement plus25% no longer fits cumulative13h after fresh validation')
                    session.allow(estimate['remaining_original_seconds'],'All68 remaining original outputs before original12h reserve')
                finally:
                    session.memory_watch=None
            memory=watch.result()
            profiling.require_memory(memory)
            write(output/'fresh_validation.json',dict(passed=True,memory=memory,counter=verify_live_counter(),
                original35_full_reference_checks_and15_saved_projectors_repeated=True,
                all6_train_calibration_and4_readout_layouts_repeated=True,
                all6_actual_teacher_proofs_repeated=True,all_final_and_control_preparation_reused_from_full_sealed_proofs=True,
                actual_next_restore_operation_full_reference_checks=next_checks,
                fresh_final_and_control_spot_checks_only=True,final_bank_proofs=final_proofs,
                complete_remaining_with25percent=remaining*1.25,available_cumulative13h_after_reserve=TOTAL_LIMIT_SECONDS-300-after['seconds'],
                no_training=True,time=time.time()))
            session.transition('evaluation')
            with MemoryWatch() as watch:
                session.memory_watch=watch
                try:
                    with (patch.object(evaluation,'write',final_access_writer(evaluation.write,ROOT,auth['final_access'])),
                          patch.object(evaluation.banks,'filter_bank',frozen_bank),
                          patch.object(evaluation.controls,'filter_controls',frozen_controls)):
                        evaluation.evaluate_all(ROOT,engine,cache)
                finally:
                    session.memory_watch=None
            memory=watch.result()
            profiling.require_memory(memory)
            write(output/'ongoing_memory.json',memory)
            completed=True
    finally:
        if cache is not None:
            cache.engine=None
        del engine
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        session.close()
    check_manifest(auth['immutable_old_scientific_artifacts'])
    if completed:
        write(ROOT/'gpu_completion.json',dict(completed=True,ledger=read(ROOT/'engineering/gpu_budget.json'),process=process,time=time.time()))
        runner.audit(ROOT)
        runner.reporting.complete(ROOT)
        runner.status(ROOT,'awaiting_worker_exit_verification',report=str(ROOT/'REPORT.md'),AT_qualification_disabled=True)
        write(output/'receipt.json',dict(passed=True,original175_outputs_unchanged=True,
            original15_fits_15000_orders_unchanged=True,final_access_and_lock_unchanged=True,
            all243_original_outputs_complete=True,ledger=read(ROOT/'engineering/gpu_budget.json'),time=time.time()))


def watch(path):
    from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
    from . import full_transfer
    auth=check_authorization(path)
    process=read(Path(path).parent/'resumed_process.json')
    if process['cmdline'][-5:-1]!=['-m','diagnostics.apc_128_20261026.resume_evaluation','run','--authorization']:
        raise ValueError('Unexpected evaluation-only resumed worker command')
    original_verify=full_transfer.verify_plan
    def amended_worker_plan(*args,**kwargs):
        value=original_verify(*args,**kwargs)
        if value['source_worker']!=auth['prior_process']:
            raise ValueError('Original plan historical worker identity changed')
        if read(ROOT/'latest_process.json')!=process:
            raise ValueError('Authorized new worker identity changed')
        return dict(value,source_worker=process)
    write(OUTPUT/'source_worker_resume_authorization.json',dict(user_authorized_evaluation_resume=str(path),
        authorization_sha256=file_hash(path),historical_plan_worker=auth['prior_process'],new_worker=process,
        original_plan_sha256=file_hash(OUTPUT/'plan.json'),only_expected_execution_identity_amended=True,time=time.time()))
    with patch.object(full_transfer,'verify_plan',amended_worker_plan):
        full_transfer.wait_then_run()


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles=setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    from experiments.mcqa_at_anchor_comparison.storage import write
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prove','run','watch'))
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(4)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils,'is_sklearn_available',return_value=False))
        stack.enter_context(patch.object(imports,'is_sklearn_available',return_value=False))
        try:
            if args.command=='prove':
                prove()
            elif args.command=='run':
                run(args.authorization)
            else:
                watch(args.authorization)
        except BaseException as error:
            folder=OUTPUT if args.command=='watch' else ROOT
            value=dict(type=type(error).__name__,reason=str(error),traceback=traceback.format_exc(),
                       time=time.time(),pid=os.getpid(),automatic_restart=False)
            write(folder/'engineering/evaluation_resume_failures'/str(time.time_ns())/'terminal.json',value)
            if args.command!='prove':
                write(folder/'terminal.json',value,immutable=False)
                write(folder/'status.json',dict(stage='stopped',reason=str(error),time=time.time(),pid=os.getpid()),immutable=False)
                (folder/'STOP_REPORT.md').write_text('# Evaluation Continuation Stopped\n\n'+str(error)+
                    '\n\nAll training endpoints remain frozen. Final data was already open before this continuation. No automatic restart.\n',encoding='utf-8')
            raise
    assert handles is not None


if __name__=='__main__':
    main()
