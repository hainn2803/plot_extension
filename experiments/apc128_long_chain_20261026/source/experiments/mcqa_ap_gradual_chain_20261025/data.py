"""Frozen structural proposals; selection uses natural correctness only."""
from collections import Counter
from pathlib import Path
import itertools
import json
import random
import shutil
import sys
import time

from experiments.mcqa_ap_localization_sweep import data as old
from experiments.mcqa_at_anchor_comparison.storage import read, write, digest, file_hash
from .protocol import ROOT, PRIOR, SALT, LETTERS, COLORS, BANK_SIZES, PROTOCOL

context, answer, ap = old.context, old.answer, old.ap


def key(*parts):
    return digest([SALT, *parts])


def slots(bank):
    if bank.startswith('readout_'):
        _, variable, split = bank.split('_')
        classes = list(LETTERS) if variable == 'AT' else list(range(4))
        return [dict(bank=bank, split=split, variable=variable, cls=c, ordinal=i, slot=j)
                for j, (c, i) in enumerate(itertools.product(classes, range(BANK_SIZES[bank] // len(classes))))]
    variable, split = bank.split('_')
    out = []
    if variable == 'AT':
        changing, preservation = (16, 4) if split == 'calibration' else (48, 16)
        for letter in LETTERS:
            for kind, n in (('changing', changing), ('preservation', preservation)):
                out.extend(dict(bank=bank, split=split, variable=variable, letter=letter, kind=kind, ordinal=i) for i in range(n))
    else:
        n, p = (16, 32) if split == 'calibration' else (64, 64)
        for b in range(4):
            for s in range(4):
                count = (n if variable == 'AP' else 2*n) if b != s else p
                out.extend(dict(bank=bank, split=split, variable=variable, b=b, s=s,
                                kind='changing' if b != s else 'preservation', ordinal=i) for i in range(count))
    for i, slot in enumerate(out):
        slot['slot'] = i
    return out


def alternatives(slot):
    return sorted(range(256), key=lambda a: (key('alternative', slot, a), a))


def natural(rng, objects, position=None, letter=None):
    obj = rng.choice(objects)
    color = rng.choice(COLORS)
    position = rng.randrange(4) if position is None else position
    contents = rng.sample([c for c in COLORS if c != color], 3)
    contents.insert(position, color)
    labels = rng.sample(LETTERS, 4) if letter is None else rng.sample([x for x in LETTERS if x != letter], 3)
    if letter is not None:
        labels.insert(position, letter)
    return context(obj, color, labels, contents)


def proposal(slot, alternative, groups):
    if alternative not in range(256):
        raise ValueError('Outside frozen pool')
    rng = random.Random(int(key('proposal', slot, alternative), 16))
    objects = groups[slot['split']]
    variable = slot['variable']
    if slot['bank'].startswith('readout_'):
        c = natural(rng, objects, position=slot['cls'] if variable == 'AP' else None,
                    letter=slot['cls'] if variable == 'AT' else None)
        return [dict(id=key('readout', slot['bank'], c['id']), base=c, source=c, target=answer(c),
                     alternative=alternative, metadata=dict(slot, kind='natural', from_AP=ap(c), to_AP=ap(c),
                                                            permutation='natural', base_group=c['question']))]
    if variable == 'AT':
        target = slot['letter']
        base_letter = target if slot['kind'] == 'preservation' else rng.choice([x for x in LETTERS if x != target])
        base = natural(rng, objects, letter=base_letter)
        source = natural(rng, [o for o in objects if o['id'] != base['question']], letter=target)
        sources = [('cross_question_AT', source)]
        b, s = ap(base), ap(source)
    else:
        b, s = slot['b'], slot['s']
        obj = rng.choice(objects)
        color = rng.choice(COLORS)
        contents = rng.sample([c for c in COLORS if c != color], 3)
        contents.insert(b, color)
        labels = rng.sample(LETTERS, 4)
        base = context(obj, color, labels, contents)
        target = labels[s]
        if variable == 'C':
            slabels, sc, scolor = labels.copy(), contents.copy(), color
            if b != s:
                scolor = contents[s]
                if slot['ordinal'] % 2 == 0:
                    slabels[b], slabels[s] = slabels[s], slabels[b]
                typ = 'C_only' if slot['ordinal'] % 2 == 0 else 'both'
            else:
                t = (b + 1 + slot['ordinal'] // 2 % 3) % 4
                slabels[b], slabels[t] = slabels[t], slabels[b]
                if slot['ordinal'] % 2 == 0:
                    sc[b], sc[t] = sc[t], sc[b]
                typ = 'AP_only' if slot['ordinal'] % 2 == 0 else 'AT_only'
            sources = [(typ, context(obj, scolor, slabels, sc))]
        elif b != s:
            order = list(range(4))
            order[b], order[s] = order[s], order[b]
            sources = [('transposition', context(obj, color, [labels[j] for j in order], [contents[j] for j in order]))]
            t = [v for v in range(4) if v not in (b, s)][slot['ordinal'] % 2]
            order = list(range(4))
            order[s], order[b], order[t] = b, t, s
            sources.append(('cycle', context(obj, color, [labels[j] for j in order], [contents[j] for j in order])))
        else:
            others = [v for v in range(4) if v != b]
            perms = [p for p in itertools.permutations(others) if list(p) != others]
            order = list(range(4))
            for i, j in zip(others, rng.choice(perms)):
                order[i] = j
            sources = [('preservation', context(obj, color, [labels[j] for j in order], [contents[j] for j in order]))]
    rows = [dict(id=key(base['id'], source['id']), base=base, source=source, target=target, alternative=alternative,
                 metadata=dict(slot, from_AP=b, to_AP=s, permutation=typ, base_group=base['question'])) for typ, source in sources]
    for row in rows:
        validate(row)
    return rows


def validate(row):
    b, s, m = row['base'], row['source'], row['metadata']
    if m['kind'] == 'natural':
        assert row['target'] == answer(b)
        return
    assert b['id'] != s['id']
    assert (row['target'] != answer(b)) == (m['kind'] == 'changing')
    if m['variable'] == 'AT':
        assert row['target'] == answer(s) and b['question'] != s['question']
    else:
        assert row['target'] == b['labels'][m['to_AP']] and ap(b) == m['from_AP']
        assert b['question'] == s['question']
        if m['variable'] == 'AP':
            assert answer(b) == answer(s) and b['color'] == s['color'] and ap(s) == m['to_AP']
            assert sorted(zip(b['labels'], b['contents'])) == sorted(zip(s['labels'], s['contents']))
        else:
            assert b['contents'].index(s['color']) == m['to_AP']


def audit(rows, bank):
    assert len(rows) == BANK_SIZES[bank] and len({r['id'] for r in rows}) == len(rows)
    variable = bank.split('_')[1] if bank.startswith('readout_') else bank.split('_')[0]
    for row in rows:
        validate(row)
    if variable in ('AP', 'C') and not bank.startswith('readout_'):
        cells = Counter((r['metadata']['from_AP'], r['metadata']['to_AP']) for r in rows)
        for b in range(4):
            for s in range(4):
                expected = (32 if bank.endswith('calibration') else 64) if b == s else (32 if bank.endswith('calibration') else 128)
                assert cells[b, s] == expected
                if bank.endswith('final') and b != s:
                    assert len({r['base']['question'] for r in rows if (r['metadata']['from_AP'], r['metadata']['to_AP']) == (b, s)}) >= 32
    if bank.startswith('readout_'):
        counts = Counter(answer(r['base']) if variable == 'AT' else ap(r['base']) for r in rows)
        classes = LETTERS if variable == 'AT' else tuple(range(4))
        assert set(counts) == set(classes) and set(counts.values()) == {len(rows)//len(classes)}
    elif variable == 'AT':
        for letter in LETTERS:
            counts = Counter(r['metadata']['kind'] for r in rows if r['target'] == letter)
            assert counts == (dict(changing=16, preservation=4) if bank.endswith('calibration') else dict(changing=48, preservation=16))
    elif variable == 'C':
        for b in range(4):
            for s in range(4):
                counts = Counter(r['metadata']['permutation'] for r in rows if (r['metadata']['from_AP'], r['metadata']['to_AP']) == (b, s))
                per_family = (16 if bank.endswith('calibration') else 32) if b == s else (16 if bank.endswith('calibration') else 64)
                assert counts == {k: per_family for k in (('AP_only', 'AT_only') if b == s else ('C_only', 'both'))}
    return dict(count=len(rows), templates=len({r['base']['question'] for r in rows}),
                kinds=dict(Counter(r['metadata']['kind'] for r in rows)),
                target_letters=dict(Counter(r['target'] for r in rows)))


def prepare(root=ROOT):
    root = Path(root)
    if (root / 'identity.json').exists():
        return verify(root)
    groups = read(PRIOR / 'groups.json')
    write(root / 'groups.json', groups)
    write(root / 'protocol.json', json.loads(json.dumps(PROTOCOL)))
    excluded = set(read(PRIOR / 'historical_exclusions.json'))
    for bank in ('final', 'c_diagnostic'):
        for row in read(PRIOR / 'banks' / bank / 'pairs.json'):
            excluded.update(row[side]['id'] for side in ('base', 'source'))
    for path in (PRIOR / 'controls').glob('*/page_*.json'):
        for roles in read(path)['accepted'].values():
            excluded.update(c['id'] for c in roles.values())
    write(root / 'historical_exclusions.json', sorted(excluded))
    imports = {}
    for split in ('train', 'calibration'):
        path = PRIOR / 'banks' / split / 'pairs.json'
        rows = read(path)
        for row in rows:
            row['metadata'] = dict(row['metadata'], variable='AP', split=split, bank='AP_' + split)
        folder = root / 'banks' / ('AP_' + split)
        write(folder / 'pairs.json', rows)
        write(folder / 'completion.json', dict(audit=audit(rows, 'AP_' + split), pairs_sha256=file_hash(folder / 'pairs.json'),
                                              imported_from=str(path), imported_sha256=file_hash(path)))
        imports[str(path)] = file_hash(path)
    for bank in BANK_SIZES:
        write(root / 'proposals' / (bank + '.json'), slots(bank))
    write(root / 'proposals/controls.json', dict(roles=['AP_donor0', 'AP_donor1', 'recipient0', 'recipient1', 'recipient2',
                                                      'recipient3', 'AT_donor0', 'AT_donor1', 'conflict0', 'conflict1'], alternatives=256))
    prior_identity = read(PRIOR / 'identity.json')
    sources = set(Path(__file__).parent.glob('*.py'))
    sources.update(Path('.external/hai_review_publication_20261022')/name for name in
                   ('mcqa_joint_das_ap_at_c_balanced.py', 'mcqa_gradual_das_ap.py', 'mcqa_gradual_das_c.py'))
    sources.update(Path(p) for p in prior_identity['files'] if p.endswith('.py'))
    sources.update(Path(m.__file__) for name, m in list(sys.modules.items()) if name.startswith(('experiments.', 'diagnostics.'))
                   and getattr(m, '__file__', None) and str(m.__file__).endswith('.py'))
    workspace = Path.cwd().resolve()
    sources = sorted({p.resolve().relative_to(workspace) for p in sources}, key=str)
    fixed = [root / 'groups.json', root / 'protocol.json', root / 'historical_exclusions.json', *sorted((root / 'proposals').glob('*.json'))]
    fixed.extend(p for bank in ('AP_train', 'AP_calibration') for p in (root / 'banks' / bank).glob('*.json'))
    fixed.append(root/'engineering/cpu_tests.json')
    import torch
    import transformers
    identity = dict(files={str(p): file_hash(p) for p in sources + fixed}, imports=imports,
                    model=prior_identity['model'], model_root=prior_identity['model_root'],
                    prior_ledger=read(PRIOR / 'engineering/gpu_budget.json'), prior_ledger_sha256=file_hash(PRIOR / 'engineering/gpu_budget.json'),
                    new_allocation_seconds=43200, prior_study_unchanged=True, created=time.time(),
                    runtime=dict(python=sys.version, executable=sys.executable, executable_sha256=file_hash(sys.executable),
                                 torch=torch.__version__, transformers=transformers.__version__))
    assert identity['prior_ledger']['active'] is False
    for path in sources:
        destination = root / 'source_snapshot' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write(root / 'identity.json', identity)
    return identity


def verify(root=ROOT, model=False):
    identity = read(Path(root) / 'identity.json')
    for path, expected in {**identity['files'], **identity['imports']}.items():
        if file_hash(path) != expected:
            raise ValueError('Immutable input changed: ' + path)
    if file_hash(PRIOR / 'engineering/gpu_budget.json') != identity['prior_ledger_sha256']:
        raise ValueError('Prior study ledger changed')
    if model:
        for path, expected in identity['model'].items():
            if file_hash(Path(identity['model_root']) / path) != expected:
                raise ValueError('Pinned model changed: ' + path)
    return identity
