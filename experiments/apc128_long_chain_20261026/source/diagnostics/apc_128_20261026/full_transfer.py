"""Wait for the current worker, then evaluate its frozen handles without training."""
import argparse
from contextlib import ExitStack
from functools import lru_cache
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from .full_transfer_plan import ROOT, OUTPUT, SEEDS, TOTAL_LIMIT_SECONDS, RESERVE_SECONDS, jobs, prepare, verify_plan, controls_for, recipient_rows


def same_worker(process):
    import psutil
    try:
        live = psutil.Process(process['pid'])
        if abs(live.create_time()-process['created']) >= .01:
            return False
        if live.cmdline() != process['cmdline'] or Path(live.exe()).resolve() != Path(process['executable']).resolve():
            raise ValueError('Worker command/executable changed for the same creation identity')
        return True
    except psutil.NoSuchProcess:
        return False


def require_closed(ledger,total=43200):
    if ledger['active'] or ledger['total_limit_seconds'] != total or not 0 < ledger['seconds'] < total-RESERVE_SECONDS:
        raise ValueError('Verified closed cumulative ledger below the five-minute reserve required')
    if ledger['profile_seconds'] != 853.1099996000121:
        raise ValueError('Frozen profiling charge changed')


def source_manifest():
    from experiments.mcqa_at_anchor_comparison.storage import file_hash
    return {str(p):file_hash(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and not p.is_relative_to(OUTPUT)}


def check_manifest(manifest):
    from experiments.mcqa_at_anchor_comparison.storage import file_hash
    for path, expected in manifest.items():
        if file_hash(path) != expected:
            raise ValueError('Original scientific artifact changed: '+path)


def estimate():
    from experiments.mcqa_at_anchor_comparison.storage import read
    from experiments.mcqa_ap_gradual_chain_20261025 import banks
    timings = dict(full=[],tail=[])
    for path in (ROOT/'evaluation').glob('*.pt'):
        label = path.stem
        rows = ROOT/'evaluation_rows'/(label+'.json')
        if not rows.exists() or not path.with_suffix('.json').exists():
            continue
        n = len(read(rows)['rows'])
        seconds = path.stat().st_mtime-rows.stat().st_mtime
        if n <= 0 or seconds <= 0:
            continue
        if label.startswith(('AP_','C_')) and (label.endswith('_direct') or '_recipient' in label):
            timings['full'].append(seconds/n)
        if label.endswith(('_AT_replay','_AT_chain','_AT_parallel')):
            timings['tail'].append(seconds/n)
    if not all(timings.values()):
        raise ValueError('Observed full-forward and AT-tail timings required')
    full, tail = max(timings['full']), max(timings['tail'])
    collection = sum(1792*(tail if j['kind']=='AT_recipient' else full) for j in jobs())
    clean_layout_rows = 2*sum(len(banks.load_bank(ROOT, v+'_'+s))
        for v in ('AT','AP','C','readout_AT','readout_AP') for s in ('train','calibration'))
    validation_rows = clean_layout_rows + 15*32*8 + 5*32*12 + len(jobs())*32*4 + 6*32*8
    components = dict(collection=collection, fresh_full_reference_validation=validation_rows*full,
                      checkpoint_cache_IO=1200.)
    return dict(total_seconds=sum(components.values()), components=components,
        observed_max_full_seconds_per_row=full,observed_max_tail_seconds_per_row=tail,
        timing_samples={k:len(v) for k,v in timings.items()}, allowance=1.25,
        complete_inventory=144,new_training_updates=0)


def gate_without_gpu(ledger, estimate_value):
    require_closed(ledger)
    available = TOTAL_LIMIT_SECONDS-RESERVE_SECONDS-ledger['seconds']
    if estimate_value['total_seconds']*1.25 > available:
        raise RuntimeError('Full mediator/donor addendum plus 25% cannot fit the authorized cumulative13h budget')
    return dict(allowed=True,available_seconds=available,estimate=estimate_value)


def wait_then_run():
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
    from experiments.mcqa_apc_128_20261026 import registration
    plan = verify_plan()
    process = read(ROOT/'latest_process.json')
    expected = plan['source_worker']
    if process != expected:
        raise ValueError('Current worker differs from the explicitly queued identity')
    write(OUTPUT/'queue.json',dict(pid=os.getpid(),created=__import__('psutil').Process().create_time(),
        cmdline=__import__('psutil').Process().cmdline(),source_worker=process,
        original_worker_will_not_be_restarted=True,time=time.time()))
    while same_worker(process):
        if (ROOT/'terminal.json').exists() or (ROOT/'pause.request').exists():
            raise RuntimeError('Original worker stopped or paused; no automatic continuation allowed')
        if torch.cuda.is_initialized():
            raise RuntimeError('Waiting controller must not initialize CUDA')
        time.sleep(15)
    registration.require_exited(process)
    if (ROOT/'terminal.json').exists() or (ROOT/'pause.request').exists():
        raise RuntimeError('Original worker has a terminal stop or pause')
    complete = read(ROOT/'gpu_completion.json')
    if complete['process'] != process or not complete['completed']:
        raise ValueError('Successful completion from the exact original worker required')
    ledger = read(ROOT/'engineering/gpu_budget.json')
    require_closed(ledger)
    if ledger['pid'] != process['pid']:
        raise ValueError('Original ledger belongs to another worker')
    final_log = OUTPUT/'original_finalize.log'
    with final_log.open('w',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u','-B','-m','experiments.mcqa_apc_128_20261026','finalize'],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not read(ROOT/'completion.json')['completed']:
        raise ValueError('Original full verification/finalization failed')
    verify_plan()
    manifest = source_manifest()
    predicted = estimate()
    decision = gate_without_gpu(ledger,predicted)
    write(OUTPUT/'authorization.json',dict(allowed=True,source_process=process,source_ledger=ledger,
        source_completion_sha256=file_hash(ROOT/'completion.json'),original_manifest=manifest,
        pre_GPU_gate=decision,time=time.time(),additional_authorized_seconds=3600,
        cumulative_total_limit_seconds=TOTAL_LIMIT_SECONDS,final_already_open=True))
    print('ADDENDUM_ESTIMATE_HOURS_WITH_25_PERCENT '+str(predicted['total_seconds']*1.25/3600),flush=True)
    with (OUTPUT/'gpu.stdout.log').open('w',encoding='utf-8') as stdout, (OUTPUT/'gpu.stderr.log').open('w',encoding='utf-8') as stderr:
        subprocess.run([sys.executable,'-u','-B','-m','diagnostics.apc_128_20261026.full_transfer','gpu'],
                       stdout=stdout,stderr=stderr,check=True)
    gpu_done = read(OUTPUT/'gpu_completion.json')
    registration.require_exited(gpu_done['process'])
    require_closed(read(OUTPUT/'engineering/gpu_budget.json'),TOTAL_LIMIT_SECONDS)
    check_manifest(manifest)
    from .full_transfer_report import report
    result = report()
    check_manifest(read(OUTPUT/'report_manifest.json')['files'])
    check_manifest(manifest)
    write(OUTPUT/'completion.json',dict(completed=True,worker_exit_verified=True,source_unchanged=True,
        final_report=result,ledger=read(OUTPUT/'engineering/gpu_budget.json'),time=time.time()))
    write(OUTPUT/'status.json',dict(stage='complete',report=result['report'],time=time.time()),immutable=False)


def gpu():
    import psutil
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash,load
    from experiments.mcqa_apc_128_20261026 import registration,runner
    from experiments.mcqa_ap_gradual_chain_20261025 import banks,training,profiling,evaluation
    from experiments.mcqa_ap_gradual_chain_20261025.engine import Engine,NaturalCache,checked
    from experiments.mcqa_ap_gradual_chain_20261025.execution import Session
    from experiments.mcqa_ap_gradual_chain_20261025.protocol import DIAGNOSIS
    from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
    from diagnostics.ap_specialization_routing_20261023.library_initialization import initialize
    from diagnostics.ap_localization_sweep_20261024.scoped_counter import use_scoped_counter,verify_live_counter
    verify_plan()
    auth = read(OUTPUT/'authorization.json')
    registration.require_exited(auth['source_process'])
    check_manifest(auth['original_manifest'])
    if (OUTPUT/'engineering/gpu_budget.json').exists() or (OUTPUT/'gpu_completion.json').exists() or (OUTPUT/'terminal.json').exists():
        raise RuntimeError('No blind restart of a supplemental GPU attempt')
    registration.verify(ROOT,model=True)
    banks.locked(ROOT)
    ledger = read(ROOT/'engineering/gpu_budget.json')
    if ledger != auth['source_ledger']:
        raise ValueError('Original closed ledger changed before inheriting it')
    gate_without_gpu(ledger,estimate())
    write(OUTPUT/'engineering/gpu_budget.json',ledger)
    write(OUTPUT/'profile.json',read(ROOT/'profile.json'))
    proc = psutil.Process()
    process = dict(pid=os.getpid(),created=proc.create_time(),cmdline=proc.cmdline(),executable=proc.exe(),
                   executable_sha256=file_hash(proc.exe()))
    if process['executable_sha256'] != '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004':
        raise ValueError('Signed interpreter changed')
    write(OUTPUT/'latest_process.json',process)

    class ReadOnlyCache(NaturalCache):
        def ensure(self,contexts):
            if any(c['id'] not in self.index for c in contexts):
                raise ValueError('New natural context/cache generation is forbidden in this addendum')

    class GuardedSession(Session):
        def ledger(self):
            value = super().ledger()
            value.update(total_limit_seconds=TOTAL_LIMIT_SECONDS,
                         additional_evaluation_only_seconds_authorized=3600,
                         inherited_original_total_limit_seconds=43200)
            return value

        def check(self):
            from experiments.mcqa_c_at_localization import runtime
            from experiments.mcqa_at_anchor_comparison.storage import BudgetStop
            if (ROOT/'pause.request').exists() or (ROOT/'terminal.json').exists():
                raise RuntimeError('Original study received a pause or terminal condition')
            while True:
                runtime.guard()
                watch = getattr(self,'memory_watch',None)
                if watch is not None:
                    if watch.error:
                        raise RuntimeError('Original memory audit unavailable: '+watch.error)
                    memory = watch.result()
                    if not memory['no_added_shared_memory'] or not memory['all_process_device_usage_within_capacity']:
                        raise RuntimeError('Original ongoing memory thresholds failed')
                value = self.ledger()
                write(self.root/'engineering/gpu_budget.json',value,immutable=False)
                if value['seconds'] >= TOTAL_LIMIT_SECONDS-RESERVE_SECONDS:
                    raise BudgetStop('Authorized13h cumulative cap: five-minute reserve reached')
                if self.stage=='profile' and value['profile_seconds'] >= 1770:
                    raise BudgetStop('Initial profiling30-minute ceiling')
                thermal = self.root/'engineering/thermal_state.json'
                if not thermal.exists() or not read(thermal)['active']:
                    break
                time.sleep(1)

        def allow(self,seconds,label):
            import math
            from experiments.mcqa_at_anchor_comparison.storage import BudgetStop
            self.check()
            if not isinstance(seconds,(int,float)) or seconds<0 or not math.isfinite(seconds):
                raise ValueError('Invalid complete-work estimate')
            available = TOTAL_LIMIT_SECONDS-RESERVE_SECONDS-self.ledger()['seconds']
            value = dict(label=label,estimated_remaining_seconds=seconds,allowance=1.25,
                         available_seconds=available,allowed=seconds*1.25<=available,
                         cumulative_total_limit_seconds=TOTAL_LIMIT_SECONDS)
            write(self.root/'engineering/launch_gates'/f'{time.time_ns()}.json',value)
            if not value['allowed']:
                raise BudgetStop('Full supplemental workload plus25% cannot fit the authorized cumulative13h cap')
            return value

    session = GuardedSession(OUTPUT,'fresh_evaluation_validation')
    engine = None
    cache = None
    attempt = OUTPUT/'engineering/attempts'/str(time.time_ns())
    completed = False
    try:
        with use_scoped_counter(attempt):
            initialize(attempt,DIAGNOSIS)
            engine = Engine(ROOT,session)
            engine.batch = 32
            write(attempt/'live_counter.json',verify_live_counter())
            cache = ReadOnlyCache(ROOT,engine)
            cache.root = OUTPUT
            def redirected(path,value,immutable=True):
                path = Path(path)
                if path.is_relative_to(ROOT/'engineering'):
                    path = OUTPUT/path.relative_to(ROOT)
                return write(path,value,immutable=immutable)
            with MemoryWatch() as watch:
                session.memory_watch = watch
                try:
                    with patch.object(profiling,'write',redirected):
                        profile = profiling.profile(ROOT,engine,cache)
                    for variable in ('AT','AP'):
                        for split in ('train','calibration'):
                            name = 'readout_'+variable+'_'+split
                            cache.layout(banks.load_bank(ROOT,name),'fresh_'+name)
                    with patch.object(runner,'write',redirected):
                        for variable in ('AP','C'):
                            for seed in SEEDS:
                                runner.validate_teachers(ROOT,engine,cache,variable,seed)
                    plan_jobs = jobs()
                    parents = {b:banks.load_bank(ROOT,b+'_final') for b in ('AP','C')}
                    control = {b:controls_for(ROOT,b) for b in ('AP','C')}

                    @lru_cache(maxsize=4)
                    def trace(label):
                        return load(ROOT/'evaluation'/(label+'.pt'))

                    def inputs(job,small=False):
                        rr,ix = recipient_rows(parents[job['bank']],control[job['bank']],job)
                        if small:
                            rr,ix = rr[:32],ix[:32]
                        variable = 'AT' if job['kind']=='AT_recipient' else 'AP'
                        method = 'standard' if variable=='AT' else job['ap_evaluator']
                        q = training.basis(ROOT,variable,method,job['seed'],engine.device)
                        original = trace(job['source_trace'])
                        if original['pair_ids'] != [r['id'] for r in parents[job['bank']]]:
                            raise ValueError('Induced-state trace alignment changed')
                        return rr,ix,q,original[variable+'_state'][ix].float()

                    checks = []
                    for job in plan_jobs:
                        rr,ix,q,induced = inputs(job,small=True)
                        contexts = [r['base'] for r in rr]
                        with torch.no_grad():
                            clean,live = engine.forward(contexts,full=True,reference=True)
                            checks.append(checked(job['label']+'_cached_clean_AT',cache.field(contexts,'AT').to(engine.device),live['at'][:len(rr)]))
                            if job['kind']=='AT_recipient':
                                baseline = cache.field(contexts,'AT').float().to(engine.device)
                                from experiments.mcqa_at_anchor_comparison.engine import projection
                                actual = engine.tail(projection(baseline,induced.to(engine.device),q).to(torch.bfloat16))[:len(rr)]
                                reference,_ = engine.forward(contexts,at_donor=induced,at_q=q,reference=True,full=True)
                            else:
                                donor=dict(values=induced,available=torch.ones(len(rr),dtype=torch.bool))
                                actual,_ = engine.forward(contexts,donor=donor,q=q)
                                reference,_ = engine.forward(contexts,donor=donor,q=q,reference=True,full=True)
                            checks.append(checked(job['label']+'_full_reference_logits',actual,reference[:len(rr)]))
                    write(attempt/'actual_recipient_validation.json',dict(passed=True,checks=checks,
                        paths=144,unchanged_atol=2e-4,unchanged_rtol=2e-4,scientific_updates=0))
                finally:
                    session.memory_watch = None
            memory = watch.result()
            profiling.require_memory(memory)
            write(attempt/'fresh_validation.json',dict(passed=True,memory=memory,counter=verify_live_counter(),
                profile_sha256=file_hash(ROOT/'profile.json'),no_new_fitting=True))
            remaining = estimate()
            remaining['components']['fresh_full_reference_validation'] = 0.
            remaining['total_seconds'] = sum(remaining['components'].values())
            session.allow(remaining['total_seconds'],'All144 supplemental outputs plus I/O; authorized cumulative13h budget')
            session.transition('supplemental_evaluation')
            with MemoryWatch() as watch:
                session.memory_watch = watch
                try:
                    for job in plan_jobs:
                        rr,ix,q,induced = inputs(job)
                        if job['kind']=='AT_recipient':
                            fn = lambda: evaluation.tail_evaluate(engine,cache,rr,induced,q)
                        else:
                            states = cache.states(rr)
                            fn = lambda: training.evaluate(engine,rr,states,q,'AP',keep=True,override=induced)
                        metric,arrays = evaluation.put(OUTPUT,job['label'],rr,fn,ix)
                        if job['kind']=='AP_recipient_trace':
                            old = load(ROOT/'evaluation'/(job['existing']+'.pt'))
                            for key in ('correct','token_id','tied','top_two_token_ids','top_two_logits','target_log_probability'):
                                if not torch.equal(arrays[key],old[key]):
                                    raise ValueError('Fresh saved-state capture changed an original prediction: '+job['existing']+'/'+key)
                finally:
                    session.memory_watch = None
            memory = watch.result()
            profiling.require_memory(memory)
            write(attempt/'ongoing_memory.json',memory)
            if {p.stem for p in (OUTPUT/'evaluation').glob('*.pt')} != {j['label'] for j in jobs()}:
                raise ValueError('Supplemental output coverage incomplete')
            completed = True
    finally:
        if cache is not None:
            cache.engine = None
        del engine
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        session.close()
    check_manifest(auth['original_manifest'])
    if completed:
        write(OUTPUT/'gpu_completion.json',dict(completed=True,process=process,
              ledger=read(OUTPUT/'engineering/gpu_budget.json'),source_unchanged=True,time=time.time()))


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    from experiments.mcqa_at_anchor_comparison.storage import write,read
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','wait','gpu','report'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils,'is_sklearn_available',return_value=False))
        stack.enter_context(patch.object(imports,'is_sklearn_available',return_value=False))
        try:
            if args.command=='prepare':
                from .full_transfer_tests import tests
                proof = tests()
                write(OUTPUT/'CPU_tests.json',proof)
                value = prepare()
                print('PREPARED '+str(OUTPUT/'plan.json'),flush=True)
                print('INVENTORY '+str(value['new_artifacts'])+' prediction/state artifacts; no GPU initialized',flush=True)
            elif args.command=='wait':
                wait_then_run()
            elif args.command=='gpu':
                gpu()
            else:
                from .full_transfer_report import report
                print(report(),flush=True)
        except BaseException as error:
            write(OUTPUT/'failures'/str(time.time_ns())/'terminal.json',dict(type=type(error).__name__,reason=str(error),
                traceback=traceback.format_exc(),time=time.time(),pid=os.getpid(),command=args.command,
                automatic_restart=False,cuda_initialized=torch.cuda.is_initialized()))
            write(OUTPUT/'terminal.json',dict(type=type(error).__name__,reason=str(error),time=time.time(),pid=os.getpid(),
                  command=args.command,automatic_restart=False),immutable=False)
            raise
    assert handles is not None


if __name__=='__main__':
    main()
