"""Plot 18 MCQA intervention metrics across the seeds listed in SEED_FOLDERS.

Edit only SEED_FOLDERS to choose 1, 2, 5, or any number of available seeds.
Each path points to a folder containing config_summary.csv.
Run with no arguments:
    python3 plot_mcqa_interventions.py

Plot layout:
    AT: line plot across AT layers
    AP: heatmaps (AT x AP)
    C: vertically stacked heatmaps (one per C layer)

Each metric shows mean +/- sample std (%) across the listed seeds.
With one seed, std is NA (undefined). Values are TEST, not CAL.
"""

import re
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Metric:
    name: str
    intervention: str
    source: str
    family: str


METRICS = [
    Metric('IIA_AT', 'AT', 'AT_Y_after_AT', 'Direct IIA'),
    Metric('Readout_AT', 'AT', 'AT_AT_after_AT', 'Readout'),
    Metric('Joint_AT_Y_AT_intervene', 'AT', 'AT_Joint_AT_Y_after_AT', 'Joint'),

    Metric('IIA_AP', 'AP', 'AP_Y_after_AP', 'Direct IIA'),
    Metric('Readout_AP', 'AP', 'AP_AP_after_AP', 'Readout'),
    Metric('Joint_AP_AT_Y_AP_intervene', 'AP', 'AP_Joint_AP_AT_Y_after_AP', 'Joint'),
    Metric('Y_via_AT_replay_AP_intervene', 'AP', 'AP_Y_via_AT_replay', 'Replay'),
    Metric('Y_via_AT_restored_AP_intervene', 'AP', 'AP_Y_restored_via_AT', 'Restoration'),
    Metric('Y_via_AT_replay_AP_intervene_recipient', 'AP',
           'AP_only_Y_after_AT_replay_recipient_different', 'Recipient'),

    Metric('IIA_C', 'C', 'C_Y_after_C', 'Direct IIA'),
    Metric('Readout_C', 'C', 'C_C_after_C', 'Readout'),
    Metric('Joint_C_AP_AT_Y_C_intervene', 'C', 'C_Joint_C_AP_AT_Y_after_C', 'Joint'),
    Metric('Y_via_AP_replay_C_intervene', 'C', 'C_Y_via_AP_replay', 'Replay'),
    Metric('Y_via_AT_replay_C_intervene', 'C', 'C_Y_via_AT_replay', 'Replay'),
    Metric('Y_via_AP_restored_C_intervene', 'C', 'C_Y_restored_via_AP', 'Restoration'),
    Metric('Y_via_AT_restored_C_intervene', 'C', 'C_Y_restored_via_AT', 'Restoration'),
    Metric('Y_via_AP_replay_C_intervene_recipient', 'C',
           'C_only_Y_after_AP_replay_recipient_different', 'Recipient'),
    Metric('Y_via_AT_replay_C_intervene_recipient', 'C',
           'C_only_Y_after_AT_replay_recipient_different', 'Recipient'),
]
assert len(METRICS) == 18

# ============================================================================
# EDIT ONLY THIS SECTION. Include exactly the seeds you want to aggregate.
# Use a LIST of (seed_id, exact_eval_folder) tuples, not a dictionary of methods.
# Every folder must contain config_summary.csv.
# Any number of seeds is supported: 1, 2, 3, 5, ...
# ============================================================================
# SEED_FOLDERS = [
#     (0, 'results_gradual_222200_selected_layers_controlled_seed0_no_distractor/eval_configs_gradual_controlled'),
#     (1, 'results_gradual_222200_selected_layers_controlled_seed1_no_distractor/eval_configs_gradual_controlled'),
#     (2, 'results_gradual_222200_selected_layers_controlled_seed2_no_distractor/eval_configs_gradual_controlled'),
#     (3, 'results_gradual_222200_selected_layers_controlled_seed3_no_distractor/eval_configs_gradual_controlled'),
#     (4, 'results_gradual_222200_selected_layers_controlled_seed4_no_distractor/eval_configs_gradual_controlled'),
# ]

SEED_FOLDERS = [
    # (0, 'results_gradual_222222_selected_layers_controlled_seed0_no_distractor/eval_configs_gradual_controlled'),
    (1, 'results_gradual_222222_selected_layers_controlled_seed1_no_distractor/eval_configs_gradual_controlled'),
]

# SEED_FOLDERS = [
#         (0, 'results_standard_all_layers_controlled_seed0_no_distractor/eval_configs_controlled'),
#         (1, 'results_standard_all_layers_controlled_seed1_no_distractor/eval_configs_controlled'),
#         (2, 'results_standard_all_layers_controlled_seed2_no_distractor/eval_configs_controlled'),
#         (3, 'results_standard_all_layers_controlled_seed3_no_distractor/eval_configs_controlled'),
#         (4, 'results_standard_all_layers_controlled_seed4_no_distractor/eval_configs_controlled'),
# ]

def find_summaries(root):
    """Read the exact eval folders selected above; never guess an eval profile."""
    found = {}
    for seed, folder in SEED_FOLDERS:
        if seed in found:
            raise ValueError(f'Duplicate seed {seed} in SEED_FOLDERS')
        if not folder or not str(folder).strip():
            raise ValueError(f'Seed {seed}: folder path is empty in SEED_FOLDERS')
        path = Path(folder).expanduser()
        if not path.is_absolute():
            path = root / path
        summary = path if path.name == 'config_summary.csv' else path / 'config_summary.csv'
        if not summary.is_file():
            raise FileNotFoundError(f'Seed {seed}: cannot find {summary}')
        found[seed] = summary
    if not found:
        raise ValueError('SEED_FOLDERS is empty: enter at least one seed')
    return found


def infer_plot_label(paths):
    """Automatically choose a descriptive output name from the selected paths."""
    names = [str(p).lower() for p in paths.values()]
    if all('gradual' in name for name in names):
        label = 'GDAS'
        configs = [re.search(r'results_gradual_(\d+)_', name) for name in names]
        if all(match is not None for match in configs):
            ids = {match.group(1) for match in configs}
            if len(ids) == 1:
                label += '_' + ids.pop()
        return label
    if all('standard' in name for name in names):
        return 'SDAS'
    return 'MCQA'


def config_key(metric, row):
    at = int(row['at_layer'])
    if metric.intervention == 'AT':
        return (at,)
    ap = int(row['ap_layer'])
    if metric.intervention == 'AP':
        return (at, ap)
    return (at, ap, int(row['c_layer']))


def load_scores(paths):
    # (metric_name, layer_key) -> {seed: score in [0,1]}
    scores = defaultdict(dict)
    seen_sources = set()
    for seed, path in sorted(paths.items()):
        with path.open(newline='') as fh:
            reader = csv.DictReader(fh)
            columns = set(reader.fieldnames or ())
            mandatory = {'at_layer', 'ap_layer', 'c_layer'}
            if not mandatory <= columns:
                raise ValueError(f'Missing layer columns in {path}: {mandatory-columns}')
            present = [m for m in METRICS if m.source in columns]
            missing = [m.name for m in METRICS if m.source not in columns]
            if missing:
                print(f'[warning] seed {seed}: {len(missing)} missing metrics: {", ".join(missing)}')
            seen_sources.update(m.name for m in present)
            rows = 0
            for row in reader:
                rows += 1
                for metric in present:
                    raw = row.get(metric.source, '').strip()
                    if not raw:
                        continue
                    value = float(raw)
                    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                        raise ValueError(f'{path}: {metric.source}={raw!r} not in [0,1]')
                    key = (metric.name, config_key(metric, row))
                    previous = scores[key].get(seed)
                    if previous is not None and abs(previous - value) > 1e-6:
                        raise ValueError(f'Inconsistent repeated metrics for {key} in seed {seed}: '
                                         f'{previous} vs {value}')
                    scores[key][seed] = value
            print(f'[read] seed {seed}: {rows} configurations, {len(present)} metrics, {path}')
    if not scores:
        raise ValueError('No valid metrics found in selected summaries')
    return scores


def aggregate(scores, expected_seeds, strict):
    result = []
    expected = set(expected_seeds)
    for metric in METRICS:
        items = [(key, scores[(name, key)]) for name, key in scores if name == metric.name]
        for key, by_seed in sorted(items):
            missing = sorted(expected - set(by_seed))
            if missing:
                msg = f'{metric.name} {key}: missing seeds {missing}'
                if strict:
                    raise ValueError(msg)
                print(f'[warning] {msg}')
            values = np.array(list(by_seed.values()), dtype=float) * 100
            at = key[0]
            ap = key[1] if len(key) >= 2 else None
            c = key[2] if len(key) >= 3 else None
            result.append({
                'metric': metric.name, 'family': metric.family,
                'intervention': metric.intervention,
                'at_layer': at, 'ap_layer': ap, 'c_layer': c,
                'mean_pct': float(values.mean()),
                'std_pct': float(values.std(ddof=1)) if len(values) > 1 else float('nan'),
                'n_seeds': len(values),
                'seeds': ','.join(str(x) for x in sorted(by_seed)),
            })
    return result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def annotation(rec, total_seeds):
    if rec is None:
        return ''
    std = rec['std_pct']
    mark = '*' if rec['n_seeds'] < total_seeds else ''
    std_s = f'{std:.1f}' if np.isfinite(std) else 'NA'
    return f'{rec["mean_pct"]:.1f}\n±{std_s}{mark}'


def heatmap(ax, rows, at_layers, ap_layers, total_seeds, *, c_layer=None):
    lookup = {(int(r['at_layer']), int(r['ap_layer'])): r for r in rows}
    data = np.full((len(at_layers), len(ap_layers)), np.nan)
    for iy, at in enumerate(at_layers):
        for ix, ap in enumerate(ap_layers):
            rec = lookup.get((at, ap))
            if rec is not None:
                data[iy, ix] = rec['mean_pct']
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('#e8e8e8')
    im = ax.imshow(data, vmin=0, vmax=100, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(ap_layers)), [f'L{x}' for x in ap_layers])
    ax.set_yticks(range(len(at_layers)), [f'L{x}' for x in at_layers])
    ax.set_xlabel('AP layer')
    ax.set_ylabel('AT layer')
    if c_layer is not None:
        ax.set_title(f'C layer = L{c_layer}', weight='bold', loc='left', fontsize=12)
    ax.set_xticks(np.arange(len(ap_layers) + 1) - .5, minor=True)
    ax.set_yticks(np.arange(len(at_layers) + 1) - .5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    for iy, at in enumerate(at_layers):
        for ix, ap in enumerate(ap_layers):
            rec = lookup.get((at, ap))
            if rec is None:
                continue
            score = rec['mean_pct']
            ax.text(ix, iy, annotation(rec, total_seeds),
                    ha='center', va='center', fontsize=8.5,
                    color='white' if score < 60 else 'black')
    return im


def plot_at(rows, output, method, dpi, total_seeds):
    metric_rows = {m.name: [r for r in rows if r['metric'] == m.name]
                   for m in METRICS if m.intervention == 'AT'}
    if not any(metric_rows.values()):
        print('[skip] No AT data')
        return
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name, entries in metric_rows.items():
        if not entries:
            continue
        entries = sorted(entries, key=lambda r: r['at_layer'])
        x = [r['at_layer'] for r in entries]
        y = [r['mean_pct'] for r in entries]
        errs = [r['std_pct'] if np.isfinite(r['std_pct']) else 0 for r in entries]
        ax.errorbar(x, y, yerr=errs, fmt='o-', capsize=4, linewidth=2,
                    markersize=6, label=name)
    ax.set_title(f'{method} · AT intervention (mean ± std across seeds)')
    ax.set_xlabel('AT layer')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 103)
    ax.set_xticks(sorted({r['at_layer'] for entries in metric_rows.values() for r in entries}))
    ax.grid(alpha=.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', output)


def plot_ap(metric, rows, output, method, dpi, total_seeds):
    if not rows:
        print('[skip] no data:', metric.name)
        return
    at_layers = sorted({r['at_layer'] for r in rows})
    ap_layers = sorted({r['ap_layer'] for r in rows})
    width = max(8, len(ap_layers) * 1.04 + 2.3)
    fig, ax = plt.subplots(figsize=(width, max(4.6, len(at_layers) * 1.1 + 2)))
    im = heatmap(ax, rows, at_layers, ap_layers, total_seeds)
    ax.set_title(f'{method} · AP intervention · {metric.name}', fontsize=13, pad=14)
    fig.colorbar(im, ax=ax, pad=.012, fraction=.025, label='Mean accuracy (%)')
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', output)


def plot_c(metric, rows, output, method, dpi, total_seeds):
    if not rows:
        print('[skip] no data:', metric.name)
        return
    at_layers = sorted({r['at_layer'] for r in rows})
    ap_layers = sorted({r['ap_layer'] for r in rows})
    c_layers = sorted({r['c_layer'] for r in rows})
    width = max(9, len(ap_layers) * 1.04 + 2.6)
    height = max(4.4 * len(c_layers), (len(at_layers) * .95 + 1.6) * len(c_layers))
    fig, axes = plt.subplots(len(c_layers), 1, figsize=(width, height), squeeze=False)
    for i, c in enumerate(c_layers):
        ax = axes[i, 0]
        subset = [r for r in rows if r['c_layer'] == c]
        im = heatmap(ax, subset, at_layers, ap_layers, total_seeds, c_layer=c)
        # One common color scale across C layers and across SDAS/GDAS.
    fig.suptitle(f'{method} · C intervention · {metric.name}', y=.995, fontsize=14)
    fig.subplots_adjust(hspace=.44, right=.9, top=.96, bottom=.06)
    cbar = fig.colorbar(im, ax=[ax[0] for ax in axes], fraction=.018, pad=.015)
    cbar.set_label('Mean accuracy (%)')
    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', output)


def make_diagnostic(rows, output):
    """Exploratory per-config summary, with all upstream IIAs (no config collapse)."""
    by_name = defaultdict(dict)
    for r in rows:
        name = r['metric']
        at, ap, c = r['at_layer'], r['ap_layer'], r['c_layer']
        if name in ('IIA_AT', 'Joint_AT_Y_AT_intervene'):
            key = (at,)
        elif name in ('IIA_AP', 'Joint_AP_AT_Y_AP_intervene'):
            key = (at, ap)
        elif name in ('IIA_C', 'Joint_C_AP_AT_Y_C_intervene'):
            key = (at, ap, c)
        else:
            continue
        by_name[name][key] = r

    diagnostics = []
    for at, ap, c in sorted(by_name['IIA_C']):
        v_at = by_name['IIA_AT'].get((at,))
        v_ap = by_name['IIA_AP'].get((at, ap))
        v_c = by_name['IIA_C'].get((at, ap, c))
        joint = by_name['Joint_C_AP_AT_Y_C_intervene'].get((at, ap, c))
        if not all((v_at, v_ap, v_c, joint)):
            continue
        diagnostics.append({
            'at_layer': at, 'ap_layer': ap, 'c_layer': c,
            'IIA_AT_mean_pct': round(v_at['mean_pct'], 4),
            'IIA_AP_mean_pct': round(v_ap['mean_pct'], 4),
            'IIA_C_mean_pct': round(v_c['mean_pct'], 4),
            'Joint_C_mean_pct': round(joint['mean_pct'], 4),
            'gap_IIA_C_minus_Joint_C_pct': round(v_c['mean_pct']-joint['mean_pct'], 4),
            'minimum_upstream_IIA_pct': round(min(v_at['mean_pct'], v_ap['mean_pct'], v_c['mean_pct']), 4),
            'n_seeds_min': min(v_at['n_seeds'], v_ap['n_seeds'], v_c['n_seeds'], joint['n_seeds']),
        })
    diagnostics.sort(key=lambda r: (-r['gap_IIA_C_minus_Joint_C_pct'],
                                     -r['minimum_upstream_IIA_pct']))
    if diagnostics:
        write_csv(output, diagnostics, list(diagnostics[0]))
        print('[saved]', output)


def main():
    # Paths resolve relative to this script, not the shell's working directory.
    root = Path(__file__).resolve().parent
    paths = find_summaries(root)
    label = infer_plot_label(paths)
    output = root / 'plots_by_intervention' / label

    print(f'[info] Selected seeds: {sorted(paths)} (n={len(paths)})')
    scores = load_scores(paths)
    rows = aggregate(scores, paths, strict=False)

    output.mkdir(parents=True, exist_ok=True)
    for variable in ('AT', 'AP', 'C'):
        (output / variable).mkdir(exist_ok=True)
    fields = ['metric', 'family', 'intervention', 'at_layer', 'ap_layer', 'c_layer',
              'mean_pct', 'std_pct', 'n_seeds', 'seeds']
    write_csv(output / 'aggregated_metrics.csv', rows, fields)
    make_diagnostic(rows, output / 'C_config_diagnostics.csv')

    dpi = 180
    plot_at(rows, output / 'AT' / 'AT_intervention.png', label, dpi, len(paths))
    for metric in METRICS:
        if metric.intervention == 'AT':
            continue
        subset = [r for r in rows if r['metric'] == metric.name]
        location = output / metric.intervention / f'{metric.name}.png'
        if metric.intervention == 'AP':
            plot_ap(metric, subset, location, label, dpi, len(paths))
        else:
            plot_c(metric, subset, location, label, dpi, len(paths))
    print(f'Finished {label}: {output} (seed count = {len(paths)})')


if __name__ == '__main__':
    main()
