"""Independent CPU scoring, baseline checks, paired intervals and artifact receipts."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from diagnostics.hai_followup_checks_20261022 import run as follow
from diagnostics.hai_full_chain_20261020.verify_results import intervals
from diagnostics.hai_new_checkpoints_20261021 import recheck
from diagnostics.hai_frozen_eval_20261020 import run as old, descriptive as desc
from experiments.hai_c_at_evaluation import core, schema


def main():
    torch.set_num_threads(2)
    root = follow.ROOT
    plan, identity = old.read(root / 'plan.json'), old.read(root / 'identity.json')
    for path, expected in identity['sources'].items():
        if schema.file_hash(path) != expected:
            raise ValueError('Registered source changed')
    if not old.read(root / 'gpu_completion.json')['complete']:
        raise ValueError('GPU execution incomplete')
    budget = old.read(root / 'engineering/gpu_budget.json')
    if budget['active'] or budget['consumed_seconds'] > 900:
        raise ValueError('Budget accounting is not closed within the cap')
    numeric = old.read(root / 'numerical.json')
    if not numeric['passed'] or any(not x['passed'] or x['atol'] != 2e-4 or x['rtol'] != 2e-4 for x in numeric['checks']):
        raise ValueError('Matched numerical checks failed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(old.MODEL, local_files_only=True)
    contexts = [schema.Context(**c) for c in plan['contexts']]
    rows = plan['rows']
    natural = follow.arrays(root / 'natural_predictions.npz')
    shared = {}
    for r in rows:
        b, s = contexts[r['base']], contexts[r['source']]
        if b.question != s.question or b.color != s.color or b.answer != s.answer:
            raise ValueError('AP control changes a fixed variable')
        if b.labels[follow.position(s)] != r['target'] or s.labels[follow.position(b)] != r['copy_target']:
            raise ValueError('Independent AP/copy target check failed')
        if (r['target'] == r['copy_target']) != (r['variant'] == 'transposition'):
            raise ValueError('Competing predictions not separated')
        shared.setdefault(r['pair_id'], True)
        shared[r['pair_id']] &= bool(natural['correct'][r['base']] and natural['correct'][r['source']])
    mask = np.array([shared[r['pair_id']] for r in rows])
    actual_rows = 0
    examples = []
    per_transition = {}
    for name in ['full_L'+str(i).zfill(2) for i in range(26)] + ['learned_AP_L17']:
        path = root / 'operations' / (name+'.npz')
        if schema.file_hash(path) != old.read(path.with_suffix('.json'))['sha256']:
            raise ValueError('Operation checksum mismatch')
        pred = follow.arrays(path)
        output = np.array([tokenizer.decode([int(i)]).strip() for i in pred['token_id']])
        correct = (output == np.array([r['target'] for r in rows])) & ~pred['tied']
        if not np.array_equal(correct, pred['correct']) or len(correct) != 288:
            raise ValueError('Independent strict scoring differs')
        actual_rows += len(correct)
        per_transition[name] = {}
        for start in range(4):
            for dest in range(4):
                if start == dest:
                    continue
                cell = mask & np.array([r['start'] == start and r['dest'] == dest for r in rows])
                per_transition[name][str(start)+'->'+str(dest)] = {
                    variant: core.rate(correct[cell & np.array([(r['variant']=='transposition') == (variant=='transposition') for r in rows])])
                    for variant in ('transposition', 'cycles')}
        if name == 'full_L08':
            for j, r in enumerate(rows):
                if mask[j] and r['variant'] != 'transposition' and output[j] == r['copy_target'] and not pred['tied'][j]:
                    examples.append(dict(base=contexts[r['base']].prompt, source=contexts[r['source']].prompt,
                        expected_AP=r['target'], predicted_copy=r['copy_target'], observed_output=output[j], layer=8,
                        base_position=r['start']+1, source_position=r['dest']+1))
                if len(examples) == 3:
                    break
    if actual_rows != 27*288:
        raise ValueError('Operation rows incomplete')
    contexts_old, pairs, historical, _ = recheck.load()
    indices = np.asarray(historical['original_pair_indices'])
    group = np.array([p.cluster for p in pairs])
    changing = np.array([contexts_old[p.base].color != contexts_old[p.source].color for p in pairs])
    controls = __import__('experiments.hai_c_at_full_chain.gpu', fromlist=['read_operation']).read_operation(recheck.ROOT, 'pair_controls')['correct'].reshape(len(pairs), 2)
    both = controls.all(1)
    predictions = {m: follow.arrays(recheck.ROOT / m / 'predictions.npz') for m in ('standard','gradual')}
    paired = {}
    for label, selected in (('all_changing', changing), ('changing_with_both_clean_donors', changing & both)):
        paired[label] = {}
        for metric in ('direct', 'replay', 'joint'):
            def values(method):
                a = predictions[method]
                d, r = a['direct_correct'][indices], a['replay_correct'][indices]
                return d if metric == 'direct' else r if metric == 'replay' else d & r
            a, b = values('standard'), values('gradual')
            paired[label][metric] = dict(standard=core.rate(a[selected]), gradual=core.rate(b[selected]),
                difference=float(b[selected].mean()-a[selected].mean()),
                percentile_95=intervals(a,b,group,selected).tolist(), bootstrap_resamples=2000,
                conditioning='Descriptive diagnostic' if label != 'all_changing' else 'Unchanged primary changing population')
    # Readouts are compared with the clean state from the exact base execution layout.
    base = desc.load_observed(recheck.ROOT / 'operations/matched_base')
    head_path = recheck.SNAPSHOT / recheck.FOLDERS['standard'] / 'at_readout_LogisticRegression.pt'
    head = torch.load(head_path, map_location='cpu', weights_only=True)
    q = recheck.checkpoint('standard','AT')['basis'].float()
    def decoded(h):
        return ((h.float() @ q) @ head['W_AT'].T + head['b_AT']).argmax(-1).numpy()
    targets = np.array(['ABCDEFGHIJKLMNOPQRSTUVWXYZ'.index(contexts_old[p.base].answer) for p in pairs])
    baseline_class = decoded(base.at[indices])
    readout = {}
    for method in ('standard','gradual'):
        direct = desc.load_observed(recheck.ROOT / 'operations' / (method+'_direct'))
        induced_class = decoded(direct.at[indices])
        readout[method] = {}
        for family in ('AP_only', 'AT_only'):
            cell = np.array([p.family == family for p in pairs])
            readout[method][family] = dict(clean_AT_readout=core.rate((baseline_class==targets)[cell]),
                after_C_AT_readout=core.rate((induced_class==targets)[cell]),
                clean_induced_readout_agreement=core.rate((baseline_class==induced_class)[cell]),
                matched_clean_induced_AT_max_error=float((base.at[indices][cell]-direct.at[indices][cell]).abs().max()))
    samples = [json.loads(line) for line in (root/'engineering/thermal_samples.jsonl').read_text().splitlines()]
    if any(x['power']['ACLineStatus'] != 1 or x['active'] for x in samples):
        raise ValueError('Unexpected power/thermal event requires inspection')
    result = dict(passed=True, operations=27, intervention_rows=actual_rows, natural_contexts=384,
        shared_clean_pairs=sum(shared.values()), shared_clean_transition_counts=dict(Counter(
            str(r['start'])+'->'+str(r['dest']) for r in rows if mask[rows.index(r)] and r['variant']=='transposition')),
        matched_numerical_checks=numeric, per_transition=per_transition, paired=paired,
        preservation_readout_baseline=readout, concrete_AP_counterexamples=examples,
        thermal_samples=len(samples), max_temperature_C=max(s['temperature_C'] for s in samples),
        GPU_seconds=budget['consumed_seconds'], no_training=True, no_fitting=True,
        historical_data_limit='Reconstructed common bank, not claimed exact reproduction of Hai retained pairs',
        source=schema.file_hash(__file__))
    follow.save('independent_verification.json', result)
    # Preserve the manifest itself separately so its contents do not hash themselves.
    artifacts = {str(p.relative_to(root)): schema.file_hash(p) for p in root.rglob('*')
        if p.is_file() and p.name not in ('completion.json','worker.log')}
    follow.save('completion.json', dict(complete=True, artifacts=artifacts, no_training=True,
        preserved_historical_result_sha256=schema.file_hash(recheck.ROOT/'results.json')))
    print(json.dumps(dict(verified=True, intervention_rows=actual_rows, paired=paired,
        preservation_readout=readout, concrete_AP_counterexamples=examples,
        GPU_seconds=budget['consumed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
