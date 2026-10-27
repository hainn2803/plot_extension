"""Explicit post-lock evaluation addendum; never changes the registered study."""
from pathlib import Path

ROOT = Path('eval/mcqa_apc_128_20261026_v1')
OUTPUT = ROOT / 'supplemental/full_mediator_donor_v1'
PDF = Path('C:/Users/zgzg1/Downloads/gradual_causal_abstraction.pdf')
TOTAL_LIMIT_SECONDS = 46800
RESERVE_SECONDS = 300
SEEDS = (1, 0, 2)
METHODS = ('standard', 'gradual')
AT_ROLES = ('AT_donor0', 'AT_donor1', 'conflict0', 'conflict1')
AP_ROLES = ('recipient0', 'recipient1', 'recipient2', 'recipient3')


def jobs():
    result = []
    for seed in SEEDS:
        for method in METHODS:
            for variable in ('AP', 'C'):
                label = f'{variable}_{method}_S{seed}'
                routes = [(label + '_direct', 'direct', None)]
                if variable == 'C':
                    routes += [(label + '_via_' + ap + '_AP_replay', 'via_AP', ap)
                               for ap in METHODS]
                for trace, route, ap in routes:
                    for role in AT_ROLES:
                        result.append(dict(label=trace + '_AT_recipient_' + role,
                            kind='AT_recipient', bank=variable, seed=seed, method=method,
                            ap_evaluator=ap, route=route, source_trace=trace, role=role,
                            condition='same' if role.startswith('AT_') else 'different'))
            for ap in METHODS:
                label = f'C_{method}_S{seed}'
                for role in AP_ROLES:
                    result.append(dict(label=label + '_via_' + ap + '_' + role + '_trace',
                        kind='AP_recipient_trace', bank='C', seed=seed, method=method,
                        ap_evaluator=ap, route='direct', source_trace=label + '_direct',
                        existing=label + '_via_' + ap + '_' + role, role=role,
                        condition='same' if int(role[-1]) < 2 else 'different'))
    if len(result) != 144 or len({j['label'] for j in result}) != 144:
        raise ValueError('Full transfer inventory changed')
    return result


def controls_for(root, bank):
    from experiments.mcqa_at_anchor_comparison.storage import read
    folder = Path(root) / 'controls' / bank
    complete = read(folder / 'completion.json')
    if not complete['complete'] or complete['missing'] or complete['primary_pairs_removed']:
        raise ValueError('Full frozen controls required; missing cases cannot be dropped')
    control = {}
    for path in sorted(folder.glob('page_*.json')):
        for key, value in read(path)['accepted'].items():
            if key in control:
                raise ValueError('Duplicate parent in control pages')
            control[key] = value
    return control


def recipient_rows(parents, controls, job):
    from experiments.mcqa_ap_gradual_chain_20261025 import data
    from experiments.mcqa_at_anchor_comparison.storage import digest
    from experiments.mcqa_ap_localization_sweep.controls import rows_for
    if job['kind'] == 'AP_recipient_trace':
        rows, indices = rows_for(parents, controls, job['role'])
    else:
        rows, indices = [], []
        for i, parent in enumerate(parents):
            context = controls[parent['id']].get(job['role'])
            if context is None:
                raise ValueError('Missing frozen recipient: no denominator exclusion allowed')
            row = dict(parent, base=context, target=parent['target'])
            row['metadata'] = dict(parent['metadata'], permutation=job['role'],
                cluster_question=parent['base']['question'], parent_id=parent['id'],
                parent_transition=f"{parent['metadata']['from_AP']+1}->{parent['metadata']['to_AP']+1}")
            row['id'] = digest([parent['id'], 'AT_recipient', job['role'], context['id']])
            rows.append(row)
            indices.append(i)
    if len(rows) != len(parents) or indices != list(range(len(parents))):
        raise ValueError('All parent cases must remain in every recipient test')
    for parent, row in zip(parents, rows):
        recipient = row['base']
        if recipient['question'] in (parent['base']['question'], parent['source']['question']):
            raise ValueError('Recipient is not a different question')
        actual = data.ap(recipient) if job['kind'] == 'AP_recipient_trace' else data.answer(recipient)
        intended = parent['metadata']['to_AP'] if job['kind'] == 'AP_recipient_trace' else parent['target']
        if (actual == intended) != (job['condition'] == 'same'):
            raise ValueError('Same/different condition is not the intended counterfactual value')
        target = recipient['labels'][intended] if job['kind'] == 'AP_recipient_trace' else intended
        if row['target'] != target:
            raise ValueError('Recipient counterfactual target is wrong')
    return rows, indices


def verify_plan(output=OUTPUT):
    from experiments.mcqa_at_anchor_comparison.storage import read, file_hash
    plan = read(Path(output) / 'plan.json')
    if plan['inventory'] != jobs() or plan['new_training_updates'] != 0:
        raise ValueError('Supplement scope changed')
    if (plan['allocation_seconds_added'],plan['shared_total_limit_seconds'],plan['reserve_seconds']) != (3600,TOTAL_LIMIT_SECONDS,RESERVE_SECONDS):
        raise ValueError('Explicit13h evaluation-only budget authorization changed')
    if any(plan[k] for k in ('new_handles','new_readouts','new_contexts')):
        raise ValueError('The addendum cannot fit handles/readouts or select new contexts')
    for path, expected in plan['frozen_inputs'].items():
        if file_hash(path) != expected:
            raise ValueError('Frozen addendum input changed: ' + path)
    for path, expected in plan['source_files'].items():
        if file_hash(path) != expected:
            raise ValueError('Sealed addendum implementation changed: ' + path)
    return plan


def prepare(output=OUTPUT):
    import time
    import torch
    from experiments.mcqa_ap_gradual_chain_20261025 import banks
    from experiments.mcqa_apc_128_20261026 import registration
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    output = Path(output)
    if (output / 'plan.json').exists():
        return verify_plan(output)
    registration.verify(ROOT)
    lock = banks.locked(ROOT)
    access = read(ROOT / 'final_access.json')
    if access['time'] <= lock['time']:
        raise ValueError('Existing lock-before-final ordering is invalid')
    inputs = {str(ROOT / name): file_hash(ROOT / name) for name in
        ('identity.json', 'protocol.json', 'execution.json', 'profile.json',
         'AT_policy.json', 'locks/all_final.json', 'final_access.json')}
    for folder in ('banks', 'controls', 'readouts'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file():
                inputs[str(path)] = file_hash(path)
    for item in [*lock['endpoints'].values(), *lock['readouts'].values()]:
        inputs[item['path']] = item['sha256']
    import shutil
    reference = output/'reference.pdf'
    if reference.exists() and file_hash(reference) != file_hash(PDF):
        raise ValueError('Preserved PDF reference changed')
    if not reference.exists():
        shutil.copy2(PDF,reference)
    inputs[str(reference)] = file_hash(reference)
    inputs[str(output/'CPU_tests.json')] = file_hash(output/'CPU_tests.json')
    checks = []
    for bank in ('AP', 'C'):
        rows, control = banks.load_bank(ROOT, bank + '_final'), controls_for(ROOT, bank)
        for job in [j for j in jobs() if j['bank'] == bank and j['seed'] == 1 and j['method'] == 'standard']:
            rr, ix = recipient_rows(rows, control, job)
            checks.append(dict(bank=bank, label=job['label'], parents=len(rr), missing=0,
                               same_or_different=job['condition'], all_parent_indices_retained=True))
    if torch.cuda.is_initialized():
        raise ValueError('Preparation must be CPU-only')
    sources = {str(p): file_hash(p) for p in sorted(Path(__file__).parent.glob('full_transfer*.py'))}
    value = dict(authorization='User: report IIA as well as all mediator and all donor performance for the current run and learned handles.',
        reference_pdf=str(PDF), reference_sections=['2.1.4', '2.1.5', '2.4', '2.5'],
        created=time.time(), source_root=str(ROOT), output=str(output), inventory=jobs(),
        source_worker=read(ROOT/'latest_process.json'),
        new_artifacts=144, new_prediction_rows=144*1792, new_training_updates=0,
        new_handles=0, new_readouts=0, new_contexts=0, original_registered_artifacts=243,
        allocation_seconds_added=3600, shared_total_limit_seconds=TOTAL_LIMIT_SECONDS,
        reserve_seconds=RESERVE_SECONDS,
        additional_budget_authorization='User explicitly authorized one extra hour of evaluation-only time; 13-hour total cap.',
        preserved_reference_pdf=str(reference), parents_per_bank=1792,
        original_finals_already_open=True, original_lock_and_final_access_unchanged=True,
        recipient_policy=dict(AP_same=['recipient0','recipient1'], AP_different=['recipient2','recipient3'],
            AT_same=['AT_donor0','AT_donor1'], AT_different=['conflict0','conflict1'],
            condition_uses='Intended high-level counterfactual value, never an observed model prediction',
            independent_recipients_per_parent_per_condition=2, no_new_proposals=True),
        frozen_inputs=inputs, source_files=sources, CPU_recipient_checks=checks,
        cuda_initialized=False, report_requires_all_seeds_and_all_artifacts=True)
    write(output / 'plan.json', value)
    return value
