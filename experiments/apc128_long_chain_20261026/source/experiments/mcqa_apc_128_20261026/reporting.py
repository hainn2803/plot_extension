"""Original full reports, with explicit imported/ungated teacher provenance."""
from pathlib import Path
from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
from experiments.mcqa_ap_gradual_chain_20261025 import report as original


def complete(root):
    root = Path(root)
    original.complete(root)
    path = root/'REPORT.md'
    text = path.read_text(encoding='utf-8')
    text = text.replace('Fresh AT teachers; paired Standard/Gradual AP and C fits. All auxiliary coefficients are 2.',
        'Existing rank-128 AT teachers are frozen and shared within each paired seed. AT accuracy qualification was explicitly disabled; no AT refit/replacement occurred. All AP/C auxiliary coefficients remain 2.')
    text = text.replace('Previous study time remains separately preserved.',
        'This cumulative12h allocation includes all prior AT training, rank-sweep, validation, cooling and failed-attempt charges; no budget reset occurred.')
    text = text.replace('Independent audit: 15 fits, 15,000 saved update orders,',
        'Independent audit: 12 new AP/C fits plus 3 preserved AT fits, 15,000 saved update orders,')
    policy = read(root/'AT_policy.json')
    lines = ['', '## Ungated AT Teacher Limitation', '',
        'The AT calibration and natural-readout weaknesses below remain in the evidence; removing the prerequisite does not make the teachers reliable or establish internal faithfulness.', '',
        '| Seed | Changing calibration | Preservation calibration | Factual readout calibration |', '|---|---:|---:|---:|']
    for seed, entry in policy['teachers'].items():
        c, r = entry['calibration'], entry['factual_readout']
        lines.append(f"| {seed} | {c['changing']['correct']/c['changing']['count']:.2%} | {c['preservation']['correct']/c['preservation']['count']:.2%} | {r['accuracy']['correct']/r['accuracy']['count']:.2%} |")
    lines += ['', 'These imperfect teachers can bias auxiliary losses and downstream measurements. The common frozen teacher controls the paired comparison; it is not an accuracy ceiling or proof of a unique native mechanism.',
        'Localization here means fitting subspaces at the already registered L14 joint option-color AP and L0 question-color C sites. No new site, rank, or objective search was performed.']
    path.write_text(text+'\n'.join(lines)+'\n', encoding='utf-8')
    manifest = read(root/'report_manifest.json')
    manifest['files'][str(path)] = file_hash(path)
    write(root/'report_manifest.json', manifest, immutable=False)
