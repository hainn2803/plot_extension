"""Authorized resume of the exact saved AP/C user pause, with fresh validation."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from unittest.mock import patch

PAUSE = dict(reason='User requested a safe pause; no resume until explicit user authorization', time=1793036076.5805082)
SECONDS = 12988.62537730002
PROFILE_SECONDS = 853.1099996000121
CHECKPOINT_SHA = '3de88d7393a76b746f9119d93762406fb2682a38f96ccc66e896836e4f85bb7b'


def preflight(root, output):
    import torch
    from transformers import AutoTokenizer
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash, digest
    from experiments.mcqa_at_anchor_comparison.engine import Subspace
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_c_at_localization.runtime import power_status
    from experiments.mcqa_c_at_localization.profiling import thermals
    from experiments.mcqa_ap_gradual_chain_20261025 import banks, training, verification
    from experiments.mcqa_apc_128_20261026 import registration, auditing
    from .resume_tests import cpu
    identity = registration.verify(root, model=True)
    proc = read(root/'latest_process.json')
    registration.require_exited(proc)
    ledger, terminal, pause = (read(root/name) for name in ('engineering/gpu_budget.json', 'terminal.json', 'pause.request'))
    if pause != PAUSE or terminal['type'] != 'Paused' or terminal['pid'] != proc['pid'] or not terminal['reason'].startswith(PAUSE['reason']):
        raise ValueError('Only the exact explicitly authorized user pause may be archived')
    if ledger['active'] or ledger['pid'] != proc['pid'] or ledger['seconds'] != SECONDS or ledger['profile_seconds'] != PROFILE_SECONDS:
        raise ValueError('Verified exact closed pause ledger required')
    power, thermal = power_status(), thermals()
    if power['ACLineStatus'] != 1 or thermal['active']:
        raise RuntimeError('Verified AC and clear thermals required before resume')
    cpu(output)
    expected_complete = {'AT_S1','AT_S0','AT_S2','AP_standard_S1','AP_gradual_S1','AP_standard_S0','AP_gradual_S0','AP_standard_S2'}
    if {p.parent.name for p in (root/'fits').glob('*/completion.json')} != expected_complete:
        raise ValueError('Completed fit inventory differs from the verified user pause')
    train_rows = {v: banks.load_bank(root, v+'_train') for v in ('AT', 'AP')}
    decoded = {}
    tokenizer = AutoTokenizer.from_pretrained(identity['model_root'], local_files_only=True)
    def decode(token):
        token = int(token)
        if token not in decoded:
            decoded[token] = tokenizer.decode([token]).strip()
        return decoded[token]
    verified, evaluations, orders = [], 0, 0
    with patch.object(verification, 'compare', auditing.compare):
        for latest_path in sorted((root/'fits').glob('*/latest.json')):
            latest = read(latest_path)
            w = load(latest['path'])
            p = w['payload']
            if tensor_digest(p) != w['payload_digest'] or p['step'] != len(p['updates']):
                raise ValueError('Saved internal checkpoint digest/update count failed')
            cfg = p['identity']
            variable, method, seed = cfg['variable'], cfg['method'], cfg['seed']
            if variable not in ('AT','AP') or cfg['maximum'] != 1000:
                raise ValueError('Unexpected fitting stage at pause')
            initial = tensor_digest(Subspace(seed, width=9216 if variable == 'AP' else 2304).state_dict())
            if initial != p['initialization']:
                raise ValueError('Original four-thread initialization changed')
            rows = train_rows[variable]
            for i, update in enumerate(p['updates']):
                if update['step'] != i+1 or update['indices'] != training.order(rows, variable, seed, i):
                    raise ValueError('Saved update order changed')
                if sum(rows[j]['metadata']['kind'] == 'changing' for j in update['indices']) != 24:
                    raise ValueError('24/8 preservation quota changed')
            if any(int(s['step']) != p['step'] for s in p['optimizer']['state'].values()):
                raise ValueError('Optimizer step mismatch')
            if not all(k in p for k in ('cpu_rng','cuda_rng','numpy_rng','python_rng')) or not p['cuda_rng']:
                raise ValueError('Saved RNG state missing')
            if variable == 'AP':
                if cfg['run'] != file_hash(root/'identity.json') or cfg['execution'] != file_hash(root/'execution.json'):
                    raise ValueError('Scientific/execution identity changed')
                teacher = training.teachers(root, SimpleNamespace(device=torch.device('cpu')), seed) if method == 'gradual' else None
                if cfg['teachers'] != tensor_digest(teacher):
                    raise ValueError('Frozen AT teacher identity changed')
            if latest_path.parent.name == 'AP_gradual_S2':
                if p['step'] != 100 or latest['sha256'] != CHECKPOINT_SHA or file_hash(latest['path']) != CHECKPOINT_SHA:
                    raise ValueError('Exact step100 partial checkpoint required')
                partial_prefix = tensor_digest(p['updates'])
                partial_path = latest['path']
            elif p['step'] != 1000:
                raise ValueError('Unexpected additional partial fit')
            if variable == 'AP':
                for arrays_path in sorted((latest_path.parent/'evaluations').glob('*.pt')):
                    step, split = arrays_path.stem.split('_', 1)
                    rr = banks.load_bank(root, 'AP_'+split)
                    arrays = load(arrays_path)
                    if arrays['pair_ids'] != [r['id'] for r in rr] or arrays['targets'] != [r['target'] for r in rr]:
                        raise ValueError('Saved evaluation alignment changed')
                    correct = [decode(t) == r['target'] and not bool(tie) for r,t,tie in zip(rr, arrays['token_id'], arrays['tied'])]
                    if correct != arrays['correct'].tolist():
                        raise ValueError('Independent saved-token scoring failed')
                    verification.audit_metrics(rr, arrays, read(arrays_path.with_suffix('.json')))
                    evaluations += 1
            orders += p['step']
            verified.append(dict(fit=latest_path.parent.name, step=p['step'], checkpoint=latest['path'], sha256=file_hash(latest['path'])))
    for seed in (1,0):
        pair = [read(root/'fits'/f'AP_{m}_S{seed}'/'completion.json') for m in ('standard','gradual')]
        if any(pair[0][k] != pair[1][k] for k in ('initialization','order_digest','preservation_digest')):
            raise ValueError('Completed pair initialization/preservation/order mismatch')
    if orders != 8100 or (root/'final_access.json').exists() or list((root/'fits').glob('C_*/latest.json')):
        raise ValueError('Unexpected stage, update total or final access')
    manifest = {}
    for folder in ('fits','readouts'):
        for p in sorted((root/folder).rglob('*')):
            if p.is_file() and p.name != 'latest.json':
                manifest[str(p)] = file_hash(p)
    for name in ('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json','engineering/gpu_budget.json'):
        source, dest = root/name, output/'prior_pause'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if file_hash(source) != file_hash(dest):
            raise ValueError('Pause evidence archive mismatch')
    for source in (Path(__file__), Path(__file__).with_name('resume_tests.py')):
        shutil.copy2(source, output/source.name)
    receipt = dict(passed=True, authorization='Direct user command: resume', pause=pause, prior_terminal=terminal,
          prior_process=proc, prior_closed_ledger=ledger, power=power, thermal=thermal, verified_fits=verified,
          verified_orders=orders, independently_scored_saved_evaluations=evaluations,
          partial_checkpoint=partial_path, partial_prefix_digest=partial_prefix,
          immutable_old_artifacts=manifest, profile_sha256=file_hash(root/'profile.json'),
          execution_sha256=file_hash(root/'execution.json'), identity_sha256=file_hash(root/'identity.json'),
          helper_sha256=file_hash(__file__), test_helper_sha256=file_hash(Path(__file__).with_name('resume_tests.py')),
          cuda_initialized=torch.cuda.is_initialized(), final_unopened=True, time=time.time())
    if receipt['cuda_initialized']:
        raise RuntimeError('CPU preflight initialized CUDA')
    write(output/'authorization.json', receipt)
    return receipt


def next_batch_validation(root, engine, cache, output, profile_fn):
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load
    from experiments.mcqa_at_anchor_comparison.engine import Subspace, checked
    from experiments.mcqa_ap_gradual_chain_20261025 import training, banks, profiling
    from experiments.mcqa_ap_gradual_chain_20261025.engine import total_loss
    from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
    profile = profile_fn(root, engine, cache)
    with MemoryWatch() as watch:
        engine.session.memory_watch = watch
        try:
            latest = read(root/'fits/AP_gradual_S2/latest.json')
            payload = load(latest['path'])['payload']
            bank = banks.load_bank(root,'AP_train')
            rows = [bank[i] for i in training.order(bank,'AP',2,100)]
            cached, live = cache.states(rows), profiling.live_states(engine, rows)
            sub = Subspace(2,width=9216).to(engine.device)
            sub.load_state_dict(payload['subspace'])
            teacher = training.teachers(root, engine, 2)
            checks = [checked('saved_basis_next_batch',sub.basis(),payload['basis'].to(engine.device))]
            checks.extend(checked('saved_cache_'+k,cached[k].to(engine.device),live[k].to(engine.device)) for k in cached)
            terms,y = engine.losses('AP','gradual',rows,cached,sub,teacher)
            loss = total_loss(terms,'gradual')
            gradient = torch.autograd.grad(loss,sub.raw)[0]
            ref,yr = engine.losses('AP','gradual',rows,live,sub,teacher,reference=True)
            ref_loss = total_loss(ref,'gradual')
            ref_gradient = torch.autograd.grad(ref_loss,sub.raw)[0]
            checks.extend([checked('saved_next_logits',y,yr),checked('saved_next_combined_CE',loss,ref_loss),
                           checked('saved_next_combined_gradient',gradient,ref_gradient)])
            checks.extend(checked('saved_next_'+k,terms[k],ref[k]) for k in terms)
        finally:
            engine.session.memory_watch = None
    memory = watch.result()
    profiling.require_memory(memory)
    write(output/'next_batch_validation.json',dict(passed=True,next_update=101,checks=checks,memory=memory,scientific_updates_before_validation=0))
    return profile


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    from experiments.mcqa_apc_128_20261026 import runner, registration
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prove','run'))
    args=parser.parse_args()
    torch.set_num_threads(4)
    output=ROOT/'engineering/authorized_resumes'/str(time.time_ns())
    output.mkdir(parents=True,exist_ok=False)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils,'is_sklearn_available',return_value=False))
        stack.enter_context(patch.object(imports,'is_sklearn_available',return_value=False))
        try:
            authorization=preflight(ROOT,output)
        except BaseException as error:
            import traceback
            write(output/'CPU_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
                  cuda_initialized=torch.cuda.is_initialized(),ledger=read(ROOT/'engineering/gpu_budget.json'),time=time.time()))
            raise
        print('AUTHORIZED_RESUME '+str(output),flush=True)
        if args.command=='prove':
            return
        for name in ('pause.request','terminal.json','status.json','STOP_REPORT.md','latest_process.json'):
            source=ROOT/name
            if file_hash(source)!=file_hash(output/'prior_pause'/name):
                raise ValueError('Pause evidence changed during authorization')
            dest=output/'archived_root'/name
            dest.parent.mkdir(parents=True,exist_ok=True)
            source.rename(dest)
        original_write=runner.write
        profile_fn=runner.profiling.profile
        def writer(path,value,immutable=True):
            path=Path(path)
            if path.parent==ROOT/'engineering/actual_teacher_validation' and path.exists():
                path=output/'fresh_actual_teachers'/path.name
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
            write(output/'receipt.json',dict(authorization_sha256=file_hash(output/'authorization.json'),
                  after=read(ROOT/'engineering/gpu_budget.json'),all_old_artifacts_unchanged=True,
                  profile_and_execution_unchanged=True,helper_sha256=file_hash(__file__),time=time.time()))
    assert handles is not None


if __name__=='__main__':
    main()
