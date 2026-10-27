"""CPU-only verification of a committed continuation from the authorized pause."""
import json
from pathlib import Path
import time


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash
    from experiments.mcqa_c_at_it_das.training import tensor_digest
    from experiments.mcqa_at_anchor_comparison.engine import Subspace
    from experiments.mcqa_ap_gradual_chain_20261025 import training, banks
    from experiments.mcqa_apc_128_20261026 import registration
    from experiments.mcqa_apc_128_20261026.protocol import ROOT
    torch.set_num_threads(4)
    output = ROOT/'engineering/authorized_resumes/1793038236236201000'
    auth = read(output/'authorization.json')
    launch = read(ROOT/'launch_resume_20261026_181033.json')
    process = read(ROOT/'latest_process.json')
    ledger = read(ROOT/'engineering/gpu_budget.json')
    assert process['pid'] == launch['pid'] == ledger['pid'] == 48648
    assert process['cmdline'][-4:] == ['-B','-m','diagnostics.apc_128_20261026.resume','run']
    assert abs(process['created'] - 1793038233.999943) < .01
    assert ledger['active'] and 12988.62537730002 < ledger['seconds'] < 42900
    assert ledger['profile_seconds'] == 853.1099996000121 and ledger['total_limit_seconds'] == 43200
    registration.verify(ROOT)
    assert file_hash(ROOT/'profile.json') == auth['profile_sha256']
    assert file_hash(ROOT/'execution.json') == auth['execution_sha256']
    assert not (ROOT/'pause.request').exists() and not (ROOT/'terminal.json').exists()
    assert not (ROOT/'final_access.json').exists()
    for path, expected in auth['immutable_old_artifacts'].items():
        assert file_hash(path) == expected, path
    latest = read(ROOT/'fits/AP_gradual_S2/latest.json')
    old = load(auth['partial_checkpoint'])['payload']
    wrapper = load(latest['path'])
    payload = wrapper['payload']
    assert file_hash(latest['path']) == latest['sha256']
    assert tensor_digest(payload) == wrapper['payload_digest']
    assert 125 <= payload['step'] == len(payload['updates']) <= 1000
    assert tensor_digest(payload['updates'][:100]) == auth['partial_prefix_digest'] == tensor_digest(old['updates'])
    assert payload['identity'] == old['identity']
    assert payload['initialization'] == old['initialization'] == tensor_digest(Subspace(2,width=9216).state_dict())
    for state in payload['optimizer']['state'].values():
        assert int(state['step']) == payload['step']
    assert all(k in payload for k in ('cpu_rng','cuda_rng','numpy_rng','python_rng')) and payload['cuda_rng']
    rows = banks.load_bank(ROOT,'AP_train')
    for step, update in enumerate(payload['updates']):
        assert update['step'] == step+1
        assert update['indices'] == training.order(rows,'AP',2,step)
        assert sum(rows[i]['metadata']['kind']=='changing' for i in update['indices']) == 24
    next_batch = read(output/'next_batch_validation.json')
    assert next_batch['passed'] and next_batch['next_update'] == 101
    assert next_batch['scientific_updates_before_validation'] == 0
    assert all(c['passed'] and c['atol']==.0002 and c['rtol']==.0002 for c in next_batch['checks'])
    numerical = []
    for path in sorted((ROOT/'engineering/resume_validations').glob('*.json')):
        if path.stat().st_mtime >= process['created']:
            item = read(path)
            assert item['passed'] and len(item['checks']) == 35
            assert len(item['saved_projectors']) == 9
            checks = item['checks'] + [c for p in item['saved_projectors'] for c in p['checks']]
            assert all(c['passed'] and c['atol']==.0002 and c['rtol']==.0002 for c in checks)
            assert item['memory']['no_added_shared_memory'] and item['memory']['all_process_device_usage_within_capacity']
            assert item['live_counter']['passed'] and item['live_counter']['pid']==process['pid']
            numerical.append(dict(path=str(path),full_reference_checks=len(item['checks']),
                                  saved_projector_checks=len(checks)-len(item['checks']),
                                  maximum_absolute_error=max(c['max_absolute_error'] for c in checks)))
    assert len(numerical) == 1
    fresh = []
    for path in sorted((ROOT/'engineering/attempts').glob('*/fresh_validation.json')):
        if path.stat().st_mtime >= process['created']:
            item = read(path)
            assert item['passed'] and item['AT_qualification_disabled']
            assert item['counter']['passed'] and item['counter']['all_worker_instances_valid']
            assert item['counter']['pid'] == process['pid']
            assert item['original_profile_sha256'] == auth['profile_sha256']
            assert item['memory']['no_added_shared_memory'] and item['memory']['all_process_device_usage_within_capacity']
            fresh.append(str(path))
    assert len(fresh) == 1
    gates = []
    for path in sorted((ROOT/'engineering/launch_gates').glob('*.json')):
        if path.stat().st_mtime >= process['created']:
            item = read(path)
            assert item['allowed'] and item['allowance']==1.25
            assert item['estimated_remaining_seconds']*1.25 <= item['available_seconds']
            gates.append(dict(path=str(path),**item))
    assert gates
    result = dict(passed=True,process=process,ledger=ledger,fit='AP_gradual_S2',step=payload['step'],
                  checkpoint=latest['path'],sha256=latest['sha256'],original_100_update_prefix_exact=True,
                  identity_and_initialization_unchanged=True,optimizer_step=payload['step'],
                  old_artifacts_unchanged=len(auth['immutable_old_artifacts']),profile_and_execution_unchanged=True,
                  numerical=numerical,next_batch_validation=str(output/'next_batch_validation.json'),fresh_validation=fresh,
                  launch_gates=gates,final_unopened=True,cuda_initialized=torch.cuda.is_initialized(),time=time.time())
    assert not result['cuda_initialized']
    write(output/'resume_commit_verification.json',result)
    print(json.dumps(result,indent=2),flush=True)
    assert handles is not None


if __name__=='__main__':
    main()
