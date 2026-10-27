"""Explicitly authorized AC-stop continuation; original science and guards unchanged."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

PAUSE = dict(reason='AC power disconnected',time=1793048255.5425231)
SECONDS = 20669.35758750004
PROFILE_SECONDS = 853.1099996000121
CHECKPOINT_SHA = '3d6f2b6e17d110cbe0a2aa967c0ff63d920695cba1b7fad39e176a434e7cc83d'


def validate_stop(process,ledger,terminal,pause):
    if pause != PAUSE or terminal['type']!='Paused' or terminal['reason']!='AC power disconnected; resume from the last committed checkpoint':
        raise ValueError('Only the exact explicitly authorized AC stop may be archived')
    if process['pid']!=42908 or terminal['pid']!=42908 or ledger['pid']!=42908:
        raise ValueError('Prior worker identity changed')
    if abs(process['created']-1793044671.5596907)>=.01 or process['cmdline'][-4:]!=['-B','-m','diagnostics.apc_128_20261026.resume_after_ac','run']:
        raise ValueError('Prior command or creation time changed')
    if ledger['active'] or ledger['seconds']!=SECONDS or ledger['profile_seconds']!=PROFILE_SECONDS or ledger['total_limit_seconds']!=43200:
        raise ValueError('Exact cumulative closed ledger required')


def preflight(root,output):
    import torch
    from transformers import AutoTokenizer
    from experiments.mcqa_at_anchor_comparison.storage import read,write,load,file_hash
    from experiments.mcqa_at_anchor_comparison.engine import Subspace
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_c_at_localization.runtime import power_status
    from experiments.mcqa_c_at_localization.profiling import thermals
    from experiments.mcqa_ap_gradual_chain_20261025 import banks,training,verification
    from experiments.mcqa_apc_128_20261026 import registration,auditing
    from .last_fit_resume_tests import cpu
    identity = registration.verify(root,model=True)
    process = read(root/'latest_process.json')
    registration.require_exited(process)
    ledger,terminal,pause = (read(root/n) for n in ('engineering/gpu_budget.json','terminal.json','pause.request'))
    validate_stop(process,ledger,terminal,pause)
    power,thermal = power_status(),thermals()
    if power['ACLineStatus']!=1 or thermal['active']:
        raise RuntimeError('Connected AC and clear thermals required')
    tests = read(root/'engineering/cpu_tests.json')
    if not tests['passed'] or tests['original_test_count']!=14 or tests['additional_test_count']!=7:
        raise ValueError('Original 21 CPU checks required')
    cpu(output,validate_stop,process,ledger,terminal,pause)
    expected = {f'AT_S{s}' for s in (1,0,2)} | {f'AP_{m}_S{s}' for m in ('standard','gradual') for s in (1,0,2)} | {'C_standard_S1','C_gradual_S1','C_standard_S0','C_gradual_S0','C_standard_S2'}
    if {p.parent.name for p in (root/'fits').glob('*/completion.json')}!=expected:
        raise ValueError('Completed inventory differs from the verified AC stop')
    train_rows = {v:banks.load_bank(root,v+'_train') for v in ('AT','AP','C')}
    tokenizer = AutoTokenizer.from_pretrained(identity['model_root'],local_files_only=True)
    decoded = {}
    def decode(t):
        t = int(t)
        if t not in decoded:
            decoded[t] = tokenizer.decode([t]).strip()
        return decoded[t]
    verified,evaluations,orders = [],0,0
    with patch.object(verification,'compare',auditing.compare):
        for latest_path in sorted((root/'fits').glob('*/latest.json')):
            latest = read(latest_path)
            wrapper = load(latest['path'])
            p = wrapper['payload']
            if file_hash(latest['path'])!=latest['sha256'] or tensor_digest(p)!=wrapper['payload_digest'] or p['step']!=len(p['updates']):
                raise ValueError('Checkpoint receipt/internal digest/update count failed')
            cfg = p['identity']
            variable,method,seed = cfg['variable'],cfg['method'],cfg['seed']
            if variable not in train_rows or cfg['maximum']!=1000:
                raise ValueError('Unexpected fitting stage')
            if tensor_digest(Subspace(seed,width=9216 if variable=='AP' else 2304).state_dict())!=p['initialization']:
                raise ValueError('Original four-thread initialization changed')
            rows = train_rows[variable]
            for i,u in enumerate(p['updates']):
                if u['step']!=i+1 or u['indices']!=training.order(rows,variable,seed,i) or sum(rows[j]['metadata']['kind']=='changing' for j in u['indices'])!=24:
                    raise ValueError('Saved update order or 24/8 quota changed')
            if any(int(s['step'])!=p['step'] for s in p['optimizer']['state'].values()) or not all(k in p for k in ('cpu_rng','cuda_rng','numpy_rng','python_rng')) or not p['cuda_rng']:
                raise ValueError('Saved optimizer/RNG state failed')
            if variable!='AT':
                if cfg['run']!=file_hash(root/'identity.json') or cfg['execution']!=file_hash(root/'execution.json'):
                    raise ValueError('Scientific/execution identity changed')
                teacher = training.teachers(root,SimpleNamespace(device=torch.device('cpu')),seed,'gradual' if variable=='C' else None) if method=='gradual' else None
                if cfg['teachers']!=tensor_digest(teacher):
                    raise ValueError('Frozen teacher changed')
            if latest_path.parent.name=='C_gradual_S2':
                if p['step']!=525 or latest['sha256']!=CHECKPOINT_SHA:
                    raise ValueError('Exact saved C step525 required')
                partial_path,partial_prefix,partial_init = latest['path'],tensor_digest(p['updates']),p['initialization']
            elif p['step']!=1000:
                raise ValueError('Unexpected additional partial fit')
            if variable!='AT':
                for arrays_path in sorted((latest_path.parent/'evaluations').glob('*.pt')):
                    _,split = arrays_path.stem.split('_',1)
                    rr = banks.load_bank(root,variable+'_'+split)
                    arrays = load(arrays_path)
                    if arrays['pair_ids']!=[r['id'] for r in rr] or arrays['targets']!=[r['target'] for r in rr]:
                        raise ValueError('Saved evaluation alignment changed')
                    correct = [decode(t)==r['target'] and not bool(tie) for r,t,tie in zip(rr,arrays['token_id'],arrays['tied'])]
                    if correct!=arrays['correct'].tolist():
                        raise ValueError('Independent saved-token scoring failed')
                    verification.audit_metrics(rr,arrays,read(arrays_path.with_suffix('.json')))
                    evaluations += 1
            orders += p['step']
            verified.append(dict(fit=latest_path.parent.name,step=p['step'],checkpoint=latest['path'],sha256=latest['sha256']))
    for variable,seeds in (('AP',(1,0,2)),('C',(1,0))):
        for seed in seeds:
            pair = [read(root/'fits'/f'{variable}_{m}_S{seed}'/'completion.json') for m in ('standard','gradual')]
            if any(pair[0][k]!=pair[1][k] for k in ('initialization','order_digest','preservation_digest')):
                raise ValueError('Completed paired initialization/preservation/order mismatch')
    standard = load(read(root/'fits/C_standard_S2/completion.json')['checkpoint'])['payload']
    partial = load(partial_path)['payload']
    if partial_init!=standard['initialization'] or [u['indices'] for u in partial['updates']]!=[u['indices'] for u in standard['updates'][:525]]:
        raise ValueError('Partial paired C initialization/preservation prefix changed')
    if orders!=14525 or len(verified)!=15 or evaluations!=140 or (root/'final_access.json').exists():
        raise ValueError('Unexpected stage, coverage, update count or final access')
    readouts = []
    for path in sorted((root/'readouts').glob('*.pt')):
        head,meta = load(path),read(path.with_suffix('.json'))
        label = path.stem
        done = read(root/'fits'/label/'completion.json')
        q = load(done['checkpoint'])['payload']['basis']
        classes = 26 if label.startswith('AT_') else 4
        if head['weight'].shape!=(classes,128) or head['bias'].shape!=(classes,) or not torch.isfinite(head['weight']).all() or not torch.isfinite(head['bias']).all() or meta['q_digest']!=tensor_digest(q):
            raise ValueError('Saved factual readout failed')
        readouts.append(dict(path=str(path),sha256=file_hash(path)))
    if len(readouts)!=9:
        raise ValueError('All nine factual readouts required')
    manifest = {str(p):file_hash(p) for folder in ('fits','readouts') for p in sorted((root/folder).rglob('*')) if p.is_file() and p.name!='latest.json'}
    for name in ('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json','engineering/gpu_budget.json'):
        source,dest = root/name,output/'prior_pause'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
        if file_hash(source)!=file_hash(dest):
            raise ValueError('Stop archive hash mismatch')
    for source in (Path(__file__),Path(__file__).with_name('last_fit_resume_tests.py'),Path(__file__).with_name('ac_resume_tests.py'),Path(__file__).with_name('resume_tests.py')):
        shutil.copy2(source,output/source.name)
    receipt = dict(passed=True,authorization='Direct user command: resume after verified AC stop',pause=pause,
          prior_terminal=terminal,prior_process=process,prior_closed_ledger=ledger,power=power,thermal=thermal,
          verified_fits=verified,verified_orders=orders,independently_scored_saved_evaluations=evaluations,
          readouts=readouts,partial_checkpoint=partial_path,partial_prefix_digest=partial_prefix,
          immutable_old_artifacts=manifest,profile_sha256=file_hash(root/'profile.json'),execution_sha256=file_hash(root/'execution.json'),
          identity_sha256=file_hash(root/'identity.json'),helper_sha256=file_hash(__file__),
          test_helper_sha256=file_hash(Path(__file__).with_name('last_fit_resume_tests.py')),
          prior_C50_test_helper_sha256=file_hash(Path(__file__).with_name('ac_resume_tests.py')),
          prior_test_helper_sha256=file_hash(Path(__file__).with_name('resume_tests.py')),
          cuda_initialized=torch.cuda.is_initialized(),final_unopened=True,time=time.time())
    if receipt['cuda_initialized']:
        raise RuntimeError('CPU preflight initialized CUDA')
    write(output/'authorization.json',receipt)
    return receipt


def next_batch_validation(root,engine,cache,output,profile_fn):
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read,write,load
    from experiments.mcqa_at_anchor_comparison.engine import Subspace,checked
    from experiments.mcqa_ap_gradual_chain_20261025 import training,banks,profiling
    from experiments.mcqa_ap_gradual_chain_20261025.engine import total_loss
    from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
    profile = profile_fn(root,engine,cache)
    with MemoryWatch() as watch:
        engine.session.memory_watch = watch
        try:
            payload = load(read(root/'fits/C_gradual_S2/latest.json')['path'])['payload']
            bank = banks.load_bank(root,'C_train')
            rows = [bank[i] for i in training.order(bank,'C',2,525)]
            cached,live = cache.states(rows),profiling.live_states(engine,rows)
            sub = Subspace(2).to(engine.device)
            sub.load_state_dict(payload['subspace'])
            teacher = training.teachers(root,engine,2,'gradual')
            checks = [checked('saved_basis_next_batch',sub.basis(),payload['basis'].to(engine.device))]
            checks.extend(checked('saved_cache_'+k,cached[k].to(engine.device),live[k].to(engine.device)) for k in cached)
            terms,y = engine.losses('C','gradual',rows,cached,sub,teacher)
            loss = total_loss(terms,'gradual')
            coefficients = torch.stack(torch.autograd.grad(loss,tuple(terms.values()),retain_graph=True))
            checks.append(checked('saved_loss_coefficients',coefficients,torch.tensor([1. if k=='direct' else 2. for k in terms],device=engine.device)))
            gradient = torch.autograd.grad(loss,sub.raw)[0]
            ref,yr = engine.losses('C','gradual',rows,live,sub,teacher,reference=True)
            ref_loss = total_loss(ref,'gradual')
            ref_gradient = torch.autograd.grad(ref_loss,sub.raw)[0]
            checks.extend([checked('saved_next_logits',y,yr),checked('saved_next_combined_CE',loss,ref_loss),checked('saved_next_combined_gradient',gradient,ref_gradient)])
            checks.extend(checked('saved_next_'+k,terms[k],ref[k]) for k in terms)
        finally:
            engine.session.memory_watch = None
    memory = watch.result()
    profiling.require_memory(memory)
    write(output/'next_batch_validation.json',dict(passed=True,next_update=526,checks=checks,memory=memory,scientific_updates_before_validation=0))
    return profile


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    from experiments.mcqa_at_anchor_comparison.storage import read,write,file_hash
    from experiments.mcqa_c_at_localization.runtime import power_status
    from experiments.mcqa_c_at_localization.profiling import thermals
    from experiments.mcqa_apc_128_20261026 import runner,registration
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=('prove','run'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = ROOT/'engineering/authorized_ac_resumes'/str(time.time_ns())
    output.mkdir(parents=True,exist_ok=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils,'is_sklearn_available',return_value=False))
        stack.enter_context(patch.object(imports,'is_sklearn_available',return_value=False))
        try:
            authorization = preflight(ROOT,output)
        except BaseException as error:
            import traceback
            write(output/'CPU_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),cuda_initialized=torch.cuda.is_initialized(),ledger=read(ROOT/'engineering/gpu_budget.json'),time=time.time()))
            raise
        print('AUTHORIZED_AC_RESUME '+str(output),flush=True)
        if args.command=='prove':
            return
        if power_status()['ACLineStatus']!=1 or thermals()['active']:
            raise RuntimeError('Power/thermal conditions changed before archiving the authorized stop')
        for name in ('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json'):
            source = ROOT/name
            if file_hash(source)!=file_hash(output/'prior_pause'/name):
                raise ValueError('Stop evidence changed during authorization')
            dest = output/'archived_root'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            source.rename(dest)
        original_write,profile_fn = runner.write,runner.profiling.profile
        def writer(path,value,immutable=True):
            path = Path(path)
            if path.parent==ROOT/'engineering/actual_teacher_validation' and path.exists():
                path = output/'fresh_actual_teachers'/path.name
            return original_write(path,value,immutable=immutable)
        try:
            stack.enter_context(patch.object(runner,'write',writer))
            stack.enter_context(patch.object(runner.profiling,'profile',lambda r,e,c: next_batch_validation(r,e,c,output,profile_fn)))
            runner.main('run')
        finally:
            registration.verify(ROOT)
            for path,expected in authorization['immutable_old_artifacts'].items():
                if file_hash(path)!=expected:
                    raise ValueError('Preserved scientific artifact changed: '+path)
            if file_hash(ROOT/'profile.json')!=authorization['profile_sha256'] or file_hash(ROOT/'execution.json')!=authorization['execution_sha256']:
                raise ValueError('Immutable profile/execution changed')
            write(output/'receipt.json',dict(authorization_sha256=file_hash(output/'authorization.json'),after=read(ROOT/'engineering/gpu_budget.json'),
                  all_old_artifacts_unchanged=True,profile_and_execution_unchanged=True,helper_sha256=file_hash(__file__),time=time.time()))
    assert handles is not None


if __name__=='__main__':
    main()
