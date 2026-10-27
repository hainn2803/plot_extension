"""Read-only CPU checks of the first committed continuation after the AC stop."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read,write,load,file_hash
    from experiments.mcqa_at_anchor_comparison.engine import Subspace
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_ap_gradual_chain_20261025 import training,banks
    from experiments.mcqa_apc_128_20261026 import registration
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('authorization',type=Path)
    parser.add_argument('launch',type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    output = args.authorization
    a,launch = read(output/'authorization.json'),read(args.launch)
    process,ledger = read(ROOT/'latest_process.json'),read(ROOT/'engineering/gpu_budget.json')
    assert process['pid']==launch['pid']==ledger['pid']
    assert abs(process['created']-datetime.fromisoformat(launch['created_utc']).timestamp())<.01
    assert process['cmdline'][-4:]==['-B','-m','diagnostics.apc_128_20261026.resume_last_fit','run']
    assert ledger['active'] and 20669.35758750004<ledger['seconds']<42900
    assert ledger['profile_seconds']==853.1099996000121 and ledger['total_limit_seconds']==43200
    registration.verify(ROOT)
    assert file_hash(ROOT/'identity.json')==a['identity_sha256']
    assert file_hash(ROOT/'profile.json')==a['profile_sha256'] and file_hash(ROOT/'execution.json')==a['execution_sha256']
    assert not any((ROOT/n).exists() for n in ('pause.request','terminal.json','final_access.json'))
    for path,expected in a['immutable_old_artifacts'].items():
        assert file_hash(path)==expected,path
    latest = read(ROOT/'fits/C_gradual_S2/latest.json')
    old = load(a['partial_checkpoint'])['payload']
    wrapper = load(latest['path'])
    p = wrapper['payload']
    assert file_hash(latest['path'])==latest['sha256'] and tensor_digest(p)==wrapper['payload_digest']
    assert 550<=p['step']==len(p['updates'])<=1000
    assert tensor_digest(p['updates'][:525])==a['partial_prefix_digest']==tensor_digest(old['updates'])
    assert p['identity']==old['identity']
    assert p['initialization']==old['initialization']==tensor_digest(Subspace(2).state_dict())
    assert all(int(s['step'])==p['step'] for s in p['optimizer']['state'].values())
    assert all(k in p for k in ('cpu_rng','cuda_rng','numpy_rng','python_rng')) and p['cuda_rng']
    rows = banks.load_bank(ROOT,'C_train')
    for i,u in enumerate(p['updates']):
        assert u['step']==i+1 and u['indices']==training.order(rows,'C',2,i)
        assert sum(rows[j]['metadata']['kind']=='changing' for j in u['indices'])==24
    next_batch = read(output/'next_batch_validation.json')
    assert next_batch['passed'] and next_batch['next_update']==526 and next_batch['scientific_updates_before_validation']==0
    assert len(next_batch['checks'])==14 and all(c['passed'] and c['atol']==.0002 and c['rtol']==.0002 for c in next_batch['checks'])
    assert next_batch['memory']['no_added_shared_memory'] and next_batch['memory']['all_process_device_usage_within_capacity']
    numerical,fresh,gates = [],[],[]
    for path in sorted((ROOT/'engineering/resume_validations').glob('*.json')):
        if path.stat().st_mtime>=process['created']:
            v = read(path)
            assert v['passed'] and len(v['checks'])==35 and len(v['saved_projectors'])==15
            checks = v['checks']+[c for saved in v['saved_projectors'] for c in saved['checks']]
            assert all(c['passed'] and c['atol']==.0002 and c['rtol']==.0002 for c in checks)
            assert v['memory']['no_added_shared_memory'] and v['memory']['all_process_device_usage_within_capacity']
            assert v['live_counter']['passed'] and v['live_counter']['pid']==process['pid']
            numerical.append(dict(path=str(path),full_reference_checks=35,saved_projector_checks=len(checks)-35,
                                  maximum_absolute_error=max(c['max_absolute_error'] for c in checks)))
    for path in sorted((ROOT/'engineering/attempts').glob('*/fresh_validation.json')):
        if path.stat().st_mtime>=process['created']:
            v = read(path)
            assert v['passed'] and v['AT_qualification_disabled']
            assert v['counter']['passed'] and v['counter']['all_worker_instances_valid'] and v['counter']['pid']==process['pid']
            assert v['original_profile_sha256']==a['profile_sha256']
            assert v['memory']['no_added_shared_memory'] and v['memory']['all_process_device_usage_within_capacity']
            fresh.append(str(path))
    for path in sorted((ROOT/'engineering/launch_gates').glob('*.json')):
        if path.stat().st_mtime>=process['created']:
            v = read(path)
            assert v['allowed'] and v['allowance']==1.25 and v['estimated_remaining_seconds']*1.25<=v['available_seconds']
            gates.append(dict(path=str(path),**v))
    assert len(numerical)==len(fresh)==1 and gates
    result = dict(passed=True,process=process,ledger=ledger,fit='C_gradual_S2',step=p['step'],
                  checkpoint=latest['path'],sha256=latest['sha256'],original_525_update_prefix_exact=True,
                  identity_initialization_and_teacher_unchanged=True,optimizer_step=p['step'],
                  old_artifacts_unchanged=len(a['immutable_old_artifacts']),profile_and_execution_unchanged=True,
                  numerical=numerical,next_batch_validation=str(output/'next_batch_validation.json'),
                  fresh_validation=fresh,launch_gates=gates,final_unopened=True,cuda_initialized=torch.cuda.is_initialized(),time=time.time())
    assert not result['cuda_initialized']
    write(output/'resume_commit_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    assert handles is not None


if __name__=='__main__':
    main()
