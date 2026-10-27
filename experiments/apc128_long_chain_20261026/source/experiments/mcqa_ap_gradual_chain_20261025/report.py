"""Saved-evidence tables, figures, and explicit partial/completed outcomes."""
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash
from .protocol import SEEDS, METHODS


def partial(root):
    root = Path(root)
    terminal = read(root/'terminal.json') if (root/'terminal.json').exists() else {}
    ledger = read(root/'engineering/gpu_budget.json') if (root/'engineering/gpu_budget.json').exists() else dict(seconds=0., active=False)
    fits = {p.parent.name: read(p) for p in (root/'fits').glob('*/completion.json')}
    latest = {p.parent.name: read(p) for p in (root/'fits').glob('*/latest.json')}
    gate = {p.name: read(p) for p in root.glob('AT_gate*.json')}
    value = dict(completed=False, reason=terminal.get('reason', 'Incomplete'), ledger=ledger, completed_fits=len(fits),
                 latest=latest, AT_gates=gate, final_opened=(root/'final_access.json').exists(), automatic_restart=False)
    write(root/'incomplete_study.json', value, immutable=False)
    lines = ['# Incomplete Matched Standard/Gradual Study', '', value['reason'], '',
             f"Completed fixed fits: {len(fits)}/15. Final opened: {value['final_opened']}.",
             f"GPU-session time: {ledger['seconds']/3600:.4f}/12 hours; active ledger: {ledger['active']}.",
             '', 'No coverage, updates, seeds, thresholds, or safety guards were reduced. No automatic restart.']
    for name, result in gate.items():
        lines.extend(['', '## '+name, '', '```json', json.dumps(result, indent=2), '```'])
    (root/'STOP_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return value


def complete(root):
    root = Path(root)
    audit = read(root/'independent_verification.json')
    if not audit['passed']:
        raise ValueError('Cannot report completion without independent verification')
    result = read(root/'results.json')
    lines = ['# Matched Standard vs Gradual AT-AP-C', '',
             'Fresh AT teachers; paired Standard/Gradual AP and C fits. All auxiliary coefficients are 2.',
             'AP is one rank-128 subspace of the four L14 option-color vectors concatenated into 9,216 dimensions.', '',
             f"Closed GPU-session ledger: {audit['closed_ledger']['seconds']/3600:.4f}/12 hours. Previous study time remains separately preserved.",
             f"Independent audit: 15 fits, 15,000 saved update orders, {audit['evaluation_artifacts']} evaluation artifacts, "
             f"{audit['prediction_rows']} prediction rows, and {audit['bootstrap_checks']} paired bootstrap checks with 2,000 draws each.", '',
             '## Direct Output and AP-to-AT Consistency', '',
             '| Seed | Fit | Changing | Preservation | Direct + AT replay (changing) |',
             '|---|---|---:|---:|---:|']
    for seed in SEEDS:
        for variable in ('AP', 'C'):
            for method in METHODS:
                entry = result['seeds'][str(seed)][variable+'_'+method]
                joint = entry.get('direct_and_AT_replay', {}).get('changing', {}).get('accuracy')
                lines.append(f"| {seed} | {variable} {method} | {entry['direct']['changing']['accuracy']:.2%} | "
                             f"{entry['direct']['preservation']['accuracy']:.2%} | {joint:.2%} |" if joint is not None else
                             f"| {seed} | {variable} {method} | {entry['direct']['changing']['accuracy']:.2%} | {entry['direct']['preservation']['accuracy']:.2%} | n/a |")
    lines += ['', '## Complete Chain Cross-Evaluation', '',
              'Each entry requires correct direct C output, correct AP replay, and correct downstream AT replay. All changing pairs remain in the denominator.', '',
              '| Seed | C fit | Standard AP evaluator | Gradual AP evaluator |', '|---|---|---:|---:|']
    for seed in SEEDS:
        for method in METHODS:
            e = result['seeds'][str(seed)]['C_'+method]['via_AP']
            lines.append(f"| {seed} | {method} | {e['standard']['changing']['accuracy']:.2%} | {e['gradual']['changing']['accuracy']:.2%} |")
    lines += ['', '## Paired Effects', '', '| Seed | Comparison | Gradual minus Standard | Paired 95% interval |', '|---|---|---:|---:|']
    comparisons = []
    for seed in SEEDS:
        for key, v in result['comparisons'][str(seed)].items():
            comparisons.append(dict(seed=seed, comparison=key, **v))
            if v['difference'] is not None:
                lines.append(f"| {seed} | {key} | {100*v['difference']:+.2f} pp | [{100*v['lower']:+.2f}, {100*v['upper']:+.2f}] pp |")
            else:
                lines.append(f'| {seed} | {key} | undefined | no eligible pairs |')
    with (root/'PAIRED_EFFECTS.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    lines += ['', '## Registered Decision and Limitations', '',
              f"Strong practical superiority across both primary comparisons: **{result['strong_practical_superiority']}**.",
              f"All controls available: {result['controls_complete']}. Missing controls are not removed from primary examples.",
              'The common Standard AP evaluator is the primary C comparison; both AP evaluators and both complete pipelines are reported.',
              'Readouts are auxiliary measurements, not proof that the network uniquely uses these abstractions. The comparison does not establish native switching.',
              'Training-time C auxiliaries use directly C-induced AP and AT states. Complete three-hop chaining was reserved for evaluation.',
              'The existing object split and vocabulary were reused. Final prompts are fresh and excluded from inspected finals and the new fits/readout banks.',
              'Three paired C fits are genuine new training replicates here, unlike the prior frozen-C diagnostic.',
              '', '## Evidence', '',
              '- Exact intervention pairs: `banks/*/pairs.json`; control pairs: `evaluation_rows/*.json`.',
              '- Saved token IDs, top-two logits, probabilities, induced states and continuous restoration errors: `evaluation/*.pt`.',
              '- Detailed transition/preservation summaries: `evaluation/*.json` and `STRATIFIED_METRICS.csv`.',
              '- Frozen-readout diagnostics and independent hashes: `independent_verification.json`.',
              '- Actual worker exit is certified separately in `completion.json`; this report alone is not an exit certificate.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    strata, failures = [], []
    for path in sorted((root/'evaluation').glob('*.json')):
        if path.name.endswith('.receipt.json'):
            continue
        metric = read(path)
        for field in ('transition', 'parent_transition', 'target_letter', 'family', 'preservation_position'):
            for cell, v in metric.get(field, {}).items():
                strata.append(dict(operation=path.stem, stratum=field, cell=cell, **v))
        arr = load(path.with_suffix('.pt'))
        if arr['pair_ids']:
            rr = read(root/'evaluation_rows'/path.name)['rows']
            for i, row in enumerate(rr):
                if not arr['correct'][i]:
                    failures.append(dict(operation=path.stem, pair=row['id'], target=row['target'], predicted_token_id=int(arr['token_id'][i]),
                                         tied=bool(arr['tied'][i]), metadata=row['metadata'], base=row['base']['prompt'], source=row['source']['prompt']))
    with (root/'STRATIFIED_METRICS.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(strata[0]))
        writer.writeheader()
        writer.writerows(strata)
    write(root/'FAILURES.json', failures)
    figures(root, result)
    write(root/'report_manifest.json', dict(time=time.time(), files={str(p): file_hash(p) for p in
          [root/'REPORT.md', root/'PAIRED_EFFECTS.csv', root/'STRATIFIED_METRICS.csv', root/'FAILURES.json', *sorted(root.glob('*.png'))]}))


def figures(root, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(standard='#287c74', gradual='#cf6726')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, variable in zip(axes, ('AT', 'AP', 'C')):
        for seed in SEEDS:
            for method in (('standard',) if variable == 'AT' else METHODS):
                label = f'AT_S{seed}' if variable == 'AT' else f'{variable}_{method}_S{seed}'
                done = read(root/'fits'/label/'completion.json')
                points = sorted((int(k), v['calibration']['changing']['accuracy']) for k, v in
                                load(done['checkpoint'])['payload']['evaluations'].items())
                ax.plot([x[0] for x in points], [x[1] for x in points], color=colors[method], alpha=.7, label=f'{method} S{seed}')
        ax.set(title=variable+' calibration', xlabel='Update', ylim=(0, 1.02))
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Changing accuracy')
    fig.tight_layout()
    fig.savefig(root/'LEARNING_CURVES.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, seed in zip(axes, SEEDS):
        matrix = np.array([[result['seeds'][str(seed)]['C_'+method]['via_AP'][ap_method]['changing']['accuracy'] for ap_method in METHODS] for method in METHODS])
        ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i,j]:.1%}', ha='center', va='center', color='white' if matrix[i,j] < .5 else 'black')
        ax.set(title=f'Seed {seed}', xticks=[0, 1], xticklabels=['Standard AP', 'Gradual AP'], yticks=[0, 1], yticklabels=['Standard C', 'Gradual C'])
    fig.suptitle('Joint C -> AP -> AT success, changing pairs')
    fig.tight_layout()
    fig.savefig(root/'C_AP_MATRIX.png', dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, method in enumerate(METHODS):
        for j, seed in enumerate(SEEDS):
            metric = result['seeds'][str(seed)]['AP_'+method]['direct']['transition']
            matrix = np.array([[metric[f'{b+1}->{s+1}']['accuracy'] for s in range(4)] for b in range(4)])
            axes[i, j].imshow(matrix, vmin=0, vmax=1, cmap='viridis')
            axes[i, j].set(title=f'{method}, seed {seed}', xlabel='Target position', ylabel='Original position', xticks=range(4), yticks=range(4))
            for b in range(4):
                for s in range(4):
                    axes[i, j].text(s, b, f'{matrix[b,s]:.0%}', ha='center', va='center', color='white' if matrix[b,s] < .5 else 'black', fontsize=8)
    fig.suptitle('AP direct transitions; diagonal is preservation')
    fig.tight_layout()
    fig.savefig(root/'AP_TRANSITIONS.png', dpi=140)
    plt.close(fig)
    operations = []
    for p in sorted((root/'evaluation').glob('*restore.json')):
        arr = load(p.with_suffix('.pt'))
        if arr['pair_ids']:
            operations.append((p.stem, float(arr['correct'].float().mean()), float(arr['restoration_l2'].mean())))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].bar(range(len(operations)), [v[1] for v in operations], color='#287c74')
    axes[1].bar(range(len(operations)), [v[2] for v in operations], color='#cf6726')
    axes[0].set(ylabel='Original-answer recovery', ylim=(0, 1.02))
    axes[1].set(ylabel='Mean full-state L2 error', xticks=range(len(operations)), xticklabels=[v[0] for v in operations])
    axes[1].tick_params(axis='x', labelrotation=90, labelsize=6)
    fig.tight_layout()
    fig.savefig(root/'RESTORATION.png', dpi=140)
    plt.close(fig)
