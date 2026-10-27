"""Immutable development inputs, method-neutral selection, and finite teachers."""
from collections import Counter
from pathlib import Path
import tarfile

import numpy as np

from experiments.mcqa_c_at_localization.storage import read, write, digest, file_hash
from experiments.mcqa_c_at_localization.protocol import COLORS, LABELS
from experiments.mcqa_c_at_localization.data import from_record, targets
from experiments.mcqa_c_at_it_das import protocol as parent

ROOT = Path('eval/mcqa_c_at_it_gradual_20261018_v2')
ENGINEERING_PARENT = Path('eval/mcqa_c_at_it_gradual_20261018_v1')
PARENT = parent.ROOT
PLAN = Path('diagnostics/c_at_postquery_20261018/NEXT_EXPERIMENT.md')
POOL = Path('eval/mcqa_c_at_localization_20261017_local_v8/proposals/train.json')
NATURAL = Path('eval/mcqa_c_at_bank_feasibility_20261017_it_v1')
PROTOCOL = {
    'model': 'google/gemma-2-2b-it', 'revision': '299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8',
    'layer': 19, 'positions': [57, 58, 81], 'rank': 128, 'seed': 1,
    'methods': {'standard': [1, 0, 0], 'gradual': [1, 2, 2]},
    'optimizer': 'Adam', 'lr': .01, 'effective_batch': 32,
    'small_pairs': 384, 'steps': 500, 'evaluations': [0, 50, 100, 250, 500],
    'checkpoint_every': 25, 'AT_layer': 24, 'AT_position': 81, 'AT_rank': 8,
    'arithmetic': 'Frozen BF16 SDPA, FP32 QR centered projections, full-sequence head, FP32 softcap, TF32 disabled',
    'target': 'Source requested color mapped through BASE contents and labels for every loss',
    'atol': 2e-4, 'rtol': 2e-4,
    'teacher': {'fit': 512, 'validation': 256, 'groups': [8, 8], 'penalties': [1e-4, 1e-3, 1e-2], 'epochs': 200},
    'selection': 'Half mean of seven changing cells plus half mean of nine preserving cells; common site maximizes mean across methods at step500, earliest position ties',
    'fitting_gate': 'At least one fixed step500 fit >=.95 training changing AND >=.95 preservation',
    'conditional_stage2': {'seeds': [1, 0, 2], 'methods': ['standard', 'gradual'], 'pairs': 1536, 'epochs': 20, 'steps': 960},
    'safety': 'Unchanged AC/battery/60-second thermal guards; no automatic restart after stop; 12h session cap',
    'authorization': 'User approved local six fitting tests, profile first, autonomous overnight continuation of the proposed experiment',
    'final_data': 'Current heldout bank is historical only; no access in fitting worker; fresh confirmation requires locks',
}


def configs():
    return [dict(layer=19, position=p, rank=128, seed=1, method=m, steps=500)
            for p in (57, 58, 81) for m in ('standard', 'gradual')]


def score(rows, correct):
    assert len(rows) == len(correct) and len(rows)
    cells = {}
    for row, value in zip(rows, correct):
        key = row['family'] + '/' + row['condition']
        changed = bool(row['C_changed'])
        cell = cells.setdefault(key, {'changed': changed, 'values': []})
        assert cell['changed'] == changed
        cell['values'].append(bool(value))
    assert len(cells) == 16
    groups = [[np.mean(c['values']) for c in cells.values() if c['changed'] == flag] for flag in (True, False)]
    assert [len(g) for g in groups] == [7, 9]
    mask = np.array([r['C_changed'] for r in rows], dtype=bool)
    values = np.asarray(correct, dtype=float)
    return {'score': float(.5 * np.mean(groups[0]) + .5 * np.mean(groups[1])),
            'changing': float(values[mask].mean()), 'preservation': float(values[~mask].mean()),
            'changing_n': int(mask.sum()), 'preservation_n': int((~mask).sum()),
            'cells': {k: {'n': len(c['values']), 'accuracy': float(np.mean(c['values']))} for k,c in sorted(cells.items())}}


def fitting_gate(results):
    assert len(results) == 6
    return any(r['train']['changing'] >= .95 and r['train']['preservation'] >= .95 for r in results)


def choose_common(results):
    assert len(results) == 6
    values = {}
    for p in (57, 58, 81):
        subset = [r for r in results if r['position'] == p]
        assert len(subset) == 2 and {r['method'] for r in subset} == {'standard', 'gradual'}
        values[p] = sum(r['calibration']['score'] for r in subset) / 2
    return min(values, key=lambda p: (-values[p], p))


def sources():
    out = parent.sources()
    out.update({p.as_posix(): file_hash(p) for p in Path(__file__).parent.rglob('*.py')})
    return out


def prepare(root):
    reg = parent.verify(PARENT, assets=True)
    completion = read(PARENT / 'training_completion.json')
    lock = read(PARENT / 'selections/all_locked.json')
    for p, sha in lock['files'].items():
        assert file_hash(PARENT / p) == sha
    assert completion['independent_readouts_complete']
    imports = {}
    for name in ('train', 'calibration', 'readout_fit', 'readout_validation'):
        for suffix in ('', '_inputs'):
            src = PARENT / 'banks' / (name + suffix + '.json')
            dest = root / 'banks' / src.name
            write(dest, read(src), immutable=True)
            assert file_hash(dest) == file_hash(src)
            imports[dest.relative_to(root).as_posix()] = file_hash(dest)
    small = read(parent.parent.ROOT / 'banks/localization.json')
    parent.parent.audit_rows(small)
    assert {r['id'] for r in small} <= {r['id'] for r in read(root / 'banks/train.json')}
    write(root / 'banks/small.json', small, immutable=True)
    write(root / 'token_map.json', read(PARENT / 'token_map.json'), immutable=True)
    imports['banks/small.json'] = file_hash(root / 'banks/small.json')
    imports['token_map.json'] = file_hash(root / 'token_map.json')
    for r in small:
        assert targets(from_record(r['base']), from_record(r['source'])) == r['targets']
    anchors = {s: read(PARENT / 'selections/chain_seeds.json')[s]['AT'] for s in ('1', '0', '2')}
    assert all((a['layer'],a['position'],a['rank']) == (24,81,8) for a in anchors.values())
    write(root / 'anchors.json', anchors, immutable=True)
    imports['anchors.json'] = file_hash(root / 'anchors.json')
    groups = sorted({tuple(r['group']) for r in read(root / 'banks/train.json')},
                    key=lambda g: digest(['postquery_teacher_groups_20261018_v1', list(g)]))
    assert len(groups) == 16
    # Freeze the complete finite training-only teacher pool before selecting by natural correctness.
    excluded = {r[side]['id'] for n in ('train','calibration') for r in read(root / f'banks/{n}.json') for side in ('base','source')}
    pool, seen = {'fit': [], 'validation': []}, set()
    for slot in read(POOL):
        for pair in slot['alternatives']:
            for side in ('base','source'):
                p = pair[side]
                if p['id'] in excluded or p['id'] in seen:
                    continue
                seen.add(p['id'])
                group = tuple(p['objects'])
                assert group in groups
                pool['fit' if group in groups[:8] else 'validation'].append(p)
    for name in pool:
        pool[name].sort(key=lambda p: digest(['postquery_teacher_prompts_20261018_v1',p['id']]))
    write(root / 'teacher/proposals.json', pool, immutable=True)
    imports['teacher/proposals.json'] = file_hash(root / 'teacher/proposals.json')
    write(root / 'teacher/groups.json', {'fit': groups[:8], 'validation': groups[8:], 'overlap_DAS_training_objects': True}, immutable=True)
    imports['teacher/groups.json'] = file_hash(root / 'teacher/groups.json')
    # The engineering parent froze this identical pool before any teacher filtering.
    # Import its complete scored/rejected evidence rather than scoring it again.
    if (ENGINEERING_PARENT/'teacher/selection.json').exists():
        assert file_hash(root/'teacher/proposals.json')==file_hash(ENGINEERING_PARENT/'teacher/proposals.json')
        for name in ('selection.json','seed0.pt','seed0.json','seed1.pt','seed1.json','seed2.pt','seed2.json'):
            src=ENGINEERING_PARENT/'teacher'/name
            dest=root/'teacher'/name
            if name.endswith('.pt'):
                import shutil
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists(): shutil.copyfile(src,dest)
            else: write(dest,read(src),immutable=True)
            assert file_hash(src)==file_hash(dest)
            imports[dest.relative_to(root).as_posix()]=file_hash(dest)
    result = {'protocol': PROTOCOL, 'plan': file_hash(PLAN), 'source': sources(), 'imported': imports,
              'parent_registration': file_hash(PARENT / 'registration.json'),
              'parent_completion': file_hash(PARENT / 'training_completion.json'),
              'parent_lock': file_hash(PARENT / 'selections/all_locked.json'), 'original_pool': file_hash(POOL),
              'model_files': reg['model_files'], 'created_before_teacher_filtering': True,
              'engineering_parent': {'path': ENGINEERING_PARENT.as_posix(),
                 'registration': file_hash(ENGINEERING_PARENT/'registration.json'),
                 'status': file_hash(ENGINEERING_PARENT/'status.json'),
                 'reason': 'Separate BF16 backward passes and a combined backward have different adjoint rounding; use same-arithmetic trusted combined gradient at unchanged tolerances, preserve separate-gradient discrepancy as diagnostic'}}
    write(root / 'registration.json', result, immutable=True)
    if not (root / 'source.tar.gz').exists():
        with tarfile.open(root / 'source.tar.gz', 'w:gz') as archive:
            for path in result['source']:
                archive.add(path, arcname=path)
    return result


def verify(root, assets=False):
    reg = read(root / 'registration.json')
    assert digest(reg['protocol']) == digest(PROTOCOL) and reg['source'] == sources(), 'Source/configuration changed; refuse resume'
    assert reg['plan'] == file_hash(PLAN)
    for p, sha in reg['imported'].items():
        assert file_hash(root / p) == sha, 'Imported input changed: ' + p
    for filename, key in (('registration.json','parent_registration'),('training_completion.json','parent_completion'),('selections/all_locked.json','parent_lock')):
        assert file_hash(PARENT / filename) == reg[key]
    if assets:
        assert parent.parent.it.assets() == reg['model_files']
    return reg
