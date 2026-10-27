"""CPU-only IIA, readout, same-base mediator and cross-context donor tables."""
import csv
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .full_transfer_plan import ROOT, OUTPUT, SEEDS, METHODS, jobs, verify_plan, controls_for


def report(output=OUTPUT):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from experiments.mcqa_ap_gradual_chain_20261025 import banks, training, verification, evaluation
    from experiments.mcqa_apc_128_20261026 import auditing
    from experiments.mcqa_at_anchor_comparison.storage import read, write, load, digest, file_hash
    from unittest.mock import patch
    output = Path(output)
    plan = verify_plan(output)
    expected = {j['label'] for j in jobs()}
    if {p.stem for p in (output/'evaluation').glob('*.pt')} != expected:
        raise ValueError('All 144 supplemental artifacts required before full reporting')
    tokenizer = AutoTokenizer.from_pretrained(read(ROOT/'identity.json')['model_root'], local_files_only=True)
    decoded = {}
    audited = {}
    supplemental_jobs = {j['label']:j for j in jobs()}
    parent_banks = {b:banks.load_bank(ROOT,b+'_final') for b in ('AP','C','AT')}
    frozen_controls = {b:controls_for(ROOT,b) for b in ('AP','C')}
    torch.set_num_threads(4)

    @lru_cache(maxsize=4)
    def artifact(folder, label):
        folder = Path(folder)
        path = folder/'evaluation'/(label+'.pt')
        arrays = load(path)
        rr = read(folder/'evaluation_rows'/(label+'.json'))['rows']
        if arrays['rows_digest'] != digest(rr) or arrays['pair_ids'] != [r['id'] for r in rr] or arrays['targets'] != [r['target'] for r in rr]:
            raise ValueError('Prediction rows/targets changed: '+label)
        if folder == output:
            job = supplemental_jobs[label]
            original_rows = parent_banks[job['bank']]
            if len(rr) != plan['parents_per_bank'] or len(rr) != len(original_rows):
                raise ValueError('Full original parent denominator required: '+label)
            from experiments.mcqa_ap_gradual_chain_20261025 import data
            for parent, row in zip(original_rows,rr):
                recipient = frozen_controls[job['bank']][parent['id']][job['role']]
                if row['base'] != recipient or row['metadata']['parent_id'] != parent['id']:
                    raise ValueError('Saved recipient differs from the frozen control: '+label)
                target = parent['target'] if job['kind']=='AT_recipient' else recipient['labels'][parent['metadata']['to_AP']]
                natural = data.answer(recipient) if job['kind']=='AT_recipient' else data.ap(recipient)
                intended = parent['target'] if job['kind']=='AT_recipient' else parent['metadata']['to_AP']
                if row['target'] != target or ((natural==intended) != (job['condition']=='same')):
                    raise ValueError('Independent recipient counterfactual/condition check failed: '+label)
        for token in arrays['token_id'].tolist():
            if token not in decoded:
                decoded[token] = tokenizer.decode([token]).strip()
        ties = arrays['top_two_logits'][:, 0] == arrays['top_two_logits'][:, 1]
        if not torch.equal(ties, arrays['tied']) or not torch.equal(arrays['token_id'], arrays['top_two_token_ids'][:, 0]):
            raise ValueError('Saved top-two/tie evidence disagrees: '+label)
        correct = torch.tensor([decoded[int(t)] == r['target'] and not bool(tie)
                               for r, t, tie in zip(rr, arrays['token_id'], ties)])
        if not torch.equal(correct, arrays['correct']):
            raise ValueError('Independent decoded-token score differs: '+label)
        ids = [arrays['label_ids'][list('ABCDEFGHIJKLMNOPQRSTUVWXYZ').index(r['target'])] for r in rr]
        if arrays['target_token_ids'] != ids:
            raise ValueError('Target token identifiers changed: '+label)
        with patch.object(verification, 'compare', auditing.compare):
            verification.audit_metrics(rr, arrays, read(path.with_suffix('.json')))
        audited[str(path)] = file_hash(path)
        return rr, arrays

    @lru_cache(maxsize=9)
    def head(variable, method, seed):
        q = training.basis(ROOT, variable, method, seed, 'cpu')
        label = f'AT_S{seed}' if variable == 'AT' else f'AP_{method}_S{seed}'
        return q, load(ROOT/'readouts'/(label+'.pt'))

    def readout(arrays, rows, variable, method, seed):
        q, h = head(variable, method, seed)
        z = (arrays[variable+'_state'].float()@q)@h['weight'].T+h['bias']
        target = torch.tensor([list('ABCDEFGHIJKLMNOPQRSTUVWXYZ').index(r['target']) if variable == 'AT'
                               else r['metadata']['to_AP'] for r in rows])
        top = z.topk(2, dim=-1)
        return ((top.indices[:, 0] == target) & (top.values[:, 0] > top.values[:, 1])).numpy()

    parents = {b: {r['id']: r for r in parent_banks[b]} for b in ('AP', 'C', 'AT')}
    observations = []

    def emit(seed, method, bank, path, metric, folder, label, ap='', setting='base', condition='original'):
        rr, arrays = artifact(str(folder), label)
        if metric == 'Y':
            correct = arrays['correct'].numpy()
        elif metric == 'AT_readout':
            correct = readout(arrays, rr, 'AT', 'standard', seed)
        elif metric == 'AP_readout':
            correct = readout(arrays, rr, 'AP', ap, seed)
        elif metric == 'AT_readout_and_Y':
            correct = arrays['correct'].numpy() & readout(arrays, rr, 'AT', 'standard', seed)
        elif metric == 'AP_AT_readouts_and_Y':
            correct = arrays['correct'].numpy() & readout(arrays, rr, 'AT', 'standard', seed) & readout(arrays, rr, 'AP', ap, seed)
        else:
            raise ValueError('Unregistered report metric: '+metric)
        observations.append(dict(seed=seed, method=method, bank=bank, path=path, metric=metric,
            ap_evaluator=ap, setting=setting, condition=condition, rows=rr, correct=correct,
            artifact=str(Path(folder)/'evaluation'/(label+'.pt'))))

    for seed in SEEDS:
        emit(seed, 'common', 'AT', 'AT_direct', 'Y', ROOT, f'AT_S{seed}_direct', setting='direct')
        for method in METHODS:
            for bank in ('AP', 'C'):
                p = f'{bank}_{method}_S{seed}'
                emit(seed, method, bank, bank+'_direct', 'Y', ROOT, p+'_direct', setting='direct')
                emit(seed, method, bank, bank+'_induced_AT', 'AT_readout', ROOT, p+'_direct', setting='induced')
                emit(seed, method, bank, bank+'_induced_AT', 'AT_readout_and_Y', ROOT, p+'_direct', setting='induced')
                label = p+('_AT_replay' if bank == 'AP' else '_AT_parallel')
                for metric in ('Y','AT_readout','AT_readout_and_Y'):
                    emit(seed, method, bank, bank+'_to_AT', metric, ROOT, label)
                if bank == 'C':
                    for ap in METHODS:
                        emit(seed, method, bank, 'C_induced_AP', 'AP_readout', ROOT, p+'_direct', ap, setting='induced')
                        emit(seed, method, bank, 'C_full_joint', 'AP_AT_readouts_and_Y', ROOT, p+'_direct', ap, setting='induced')
                        label = p+'_via_'+ap
                        for metric in ('Y', 'AP_readout', 'AT_readout', 'AT_readout_and_Y','AP_AT_readouts_and_Y'):
                            emit(seed, method, bank, 'C_to_AP', metric, ROOT, label+'_AP_replay', ap)
                        for metric in ('Y','AT_readout','AT_readout_and_Y'):
                            emit(seed, method, bank, 'C_to_AP_to_AT', metric, ROOT, label+'_AT_chain', ap)
    for job in jobs():
        path = ('C_to_AP' if job['kind'] == 'AP_recipient_trace' else
                job['bank']+('_to_AP_to_AT' if job['route'] == 'via_AP' else '_to_AT'))
        metrics = ('Y','AP_readout','AT_readout','AT_readout_and_Y','AP_AT_readouts_and_Y') if job['kind'] == 'AP_recipient_trace' else ('Y','AT_readout','AT_readout_and_Y')
        for metric in metrics:
            emit(job['seed'], job['method'], job['bank'], path, metric, output, job['label'],
                 job['ap_evaluator'] or '', 'recipient', job['condition'])

    # Two frozen recipient replicates stay together in each parent-question bootstrap.
    pooled = {}
    for obs in observations:
        key = tuple(obs[k] for k in ('seed','method','bank','path','metric','ap_evaluator','setting','condition'))
        if key not in pooled:
            pooled[key] = dict(obs, rows=list(obs['rows']), correct=obs['correct'].copy(), artifacts=[obs['artifact']])
        else:
            pooled[key]['rows'].extend(obs['rows'])
            pooled[key]['correct'] = np.concatenate([pooled[key]['correct'], obs['correct']])
            pooled[key]['artifacts'].append(obs['artifact'])

    summaries, comparison_data = [], {}
    for key, obs in pooled.items():
        rr = obs['rows']
        original = [parents[obs['bank']][r['metadata'].get('parent_id', r['id'])] for r in rr]
        groups = dict(all=np.ones(len(rr), dtype=bool),
                      changing=np.array([r['metadata']['kind']=='changing' for r in original]))
        groups['preservation'] = ~groups['changing']
        groups['target_in_recipient_options'] = np.array([r['target'] in r['base'].get('labels',[]) for r in rr])
        groups['target_outside_recipient_options'] = ~groups['target_in_recipient_options']
        for family in sorted({r['metadata']['permutation'] for r in original}):
            groups['family:'+family] = np.array([r['metadata']['permutation']==family for r in original])
        for group, select in groups.items():
            correct, count = int(obs['correct'][select].sum()), int(select.sum())
            row = {k:obs[k] for k in ('seed','method','bank','path','metric','ap_evaluator','setting','condition')}
            row.update(group=group, correct=correct, count=count, accuracy=correct/count if count else None,
                unique_parent_pairs=len({r['metadata'].get('parent_id',r['id']) for r, use in zip(rr,select) if use}),
                recipients_per_parent=len(obs['artifacts']) if obs['setting']=='recipient' else 0,
                artifacts=';'.join(obs['artifacts']))
            summaries.append(row)
            if obs['method'] in METHODS and count:
                paired_key = tuple(row[k] for k in ('seed','bank','path','metric','ap_evaluator','setting','condition','group'))
                comparison_data.setdefault(paired_key, {})[obs['method']] = (obs, select)

    effects = []
    for key, methods in comparison_data.items():
        if set(methods) != set(METHODS):
            raise ValueError('Missing paired comparison cell')
        a, select = methods['standard']
        b, other_select = methods['gradual']
        if [r['id'] for r in a['rows']] != [r['id'] for r in b['rows']] or not np.array_equal(select,other_select):
            raise ValueError('Paired donor/mediator row alignment changed')
        boot = evaluation.bootstrap(a['rows'], a['correct'], b['correct'], select)
        # Reconstruct cluster sums independently rather than accepting the report calculation.
        names = sorted({r['metadata'].get('cluster_question',r['base']['question']) for r in a['rows']})
        lookup = {n:i for i,n in enumerate(names)}
        ix = np.array([lookup[r['metadata'].get('cluster_question',r['base']['question'])] for r in a['rows']])
        sums = np.bincount(ix[select], weights=(b['correct'].astype(int)-a['correct'].astype(int))[select], minlength=len(names))
        counts = np.bincount(ix[select], minlength=len(names))
        draws = np.random.default_rng(20261025).integers(0,len(names),(2000,len(names)))
        den = counts[draws].sum(1)
        valid = den > 0
        values = sums[draws].sum(1)[valid]/den[valid]
        independent = dict(count=int(select.sum()), difference=float((b['correct'].astype(int)-a['correct'].astype(int))[select].mean()),
            lower=float(np.quantile(values,.025)), upper=float(np.quantile(values,.975)), resamples=2000,
            valid_resamples=int(valid.sum()), seed=20261025, cluster_count=len(names))
        verification.compare(boot, independent, '/'.join(map(str,key)))
        row = dict(zip(('seed','bank','path','metric','ap_evaluator','setting','condition','group'),key))
        row.update({('bootstrap_seed' if name == 'seed' else name): value for name, value in boot.items()})
        effects.append(row)

    def table(path, rows):
        with path.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    table(output/'ALL_METRICS.csv', summaries)
    table(output/'PAIRED_EFFECTS.csv', effects)
    write(output/'results.json', dict(metrics=summaries, paired_effects=effects,
        scope='All IIA, readouts, same-base mediator, same/different new-recipient donor paths',
        original_registered_joint_is_distinct_from_PDF_readout_joint=True, native_mechanism_proven=False))
    lines = ['# Complete IIA, Mediator and Donor Evaluation', '',
        'Evaluation-only addendum on the exact frozen rank-128 handles from the matched AP/C run.',
        'No fitting, model selection, readout fitting, new contexts or new pools. The user authorized the evaluation time needed to finish; the current recorded cumulative cap is14h, retaining all prior charges and the five-minute reserve. The original13h preparation plan remains preserved as historical evidence.',
        'The original 243-artifact study remains unchanged. This addendum supplies 144 additional prediction/state artifacts.', '',
        '## Definitions', '',
        '- IIA: direct full-vocabulary, tie-failing output correctness.',
        '- Mediator: replay the induced downstream AP or AT handle into a fresh clean run of the original base.',
        '- Donor: replay that induced handle into a clean different-question recipient.',
        '- Same/different compares the recipient natural AP/AT against the intended counterfactual value, never the observed predicted value.',
        '- AP transplant targets use the new recipient label at the counterfactual position. AT transplant targets retain the counterfactual answer letter.',
        '- Frozen factual readouts and readout/output joints are reported separately from replay/output joints.',
        '- All rows remain in the denominators, including upstream failures; no both-direct-correct filtering.',
        '- With A-Z labels, a transferred AT letter can be absent from a new recipient option list. Present/absent strata are reported without changing the targets, recipient selection or primary denominators.',
        '- There are two frozen recipient replicates per parent per condition, clustered together by original question for 2,000 paired bootstrap draws.', '',
        '## Main Tables', '',
        'All cells, counts, strata and evidence paths: [ALL_METRICS.csv](ALL_METRICS.csv).',
        'Every-seed Gradual-minus-Standard intervals: [PAIRED_EFFECTS.csv](PAIRED_EFFECTS.csv).',
        'Both Standard and Gradual AP evaluators are retained; the common Standard AP comparison remains the original primary.', '',
        '| Seed | Path | Setting | Condition | AP evaluator | Standard Y (changing) | Gradual Y (changing) | Difference and 95% interval |',
        '|---|---|---|---|---|---:|---:|---:|']
    index = {(r['seed'],r['method'],r['bank'],r['path'],r['metric'],r['ap_evaluator'],r['setting'],r['condition'],r['group']):r for r in summaries}
    for e in effects:
        if e['metric']!='Y' or e['group']!='changing':
            continue
        k = tuple(e[n] for n in ('bank','path','metric','ap_evaluator','setting','condition','group'))
        a,b = (index[(e['seed'],m,*k)] for m in METHODS)
        lines.append(f"| {e['seed']} | {e['path']} | {e['setting']} | {e['condition']} | {e['ap_evaluator'] or '-'} | {a['correct']}/{a['count']} ({a['accuracy']:.2%}) | {b['correct']}/{b['count']} ({b['accuracy']:.2%}) | {100*e['difference']:+.2f} pp [{100*e['lower']:+.2f}, {100*e['upper']:+.2f}] |")
    lines += ['', '## Limitations', '',
        'The imported AT teachers and factual readouts remain imperfect and explicitly ungated. A common teacher controls paired comparisons but is neither a reliability guarantee nor a fixed ceiling.',
        'These are post-lock diagnostic evaluations, not new confirmatory endpoints or proof of a unique native computation.',
        'The reference PDF uses other datasets/handle settings and in some sections restricts AP targets to {0,3}; this addendum does not make those restrictions.',
        'Source: '+str(plan['reference_pdf'])+' (sections 2.1.4, 2.1.5, 2.4 and 2.5).']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if torch.cuda.is_initialized():
        raise ValueError('Independent report/audit must be CPU-only')
    write(output/'independent_verification.json',dict(passed=True,
        predictions=144, prediction_rows=144*plan['parents_per_bank'], independent_decoded_tokens=True,
        top_two_ties_and_full_vocabulary_CE_and_strata=True, audited_artifacts=audited,
        independent_paired_bootstraps=len(effects), resamples=2000, all_frozen_inputs_unchanged=True,
        frozen_plan_sha256=file_hash(output/'plan.json'), cuda_initialized=False))
    write(output/'report_manifest.json',dict(files={str(output/name):file_hash(output/name) for name in
          ('ALL_METRICS.csv','PAIRED_EFFECTS.csv','REPORT.md','results.json','independent_verification.json')}))
    return dict(metrics=len(summaries),comparisons=len(effects),report=str(output/'REPORT.md'))
