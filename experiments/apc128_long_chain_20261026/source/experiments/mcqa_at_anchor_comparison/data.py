"""Finite proposals with fixed target, family and template quota slots."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

from experiments.hai_c_at_evaluation import schema
from diagnostics.hai_frozen_eval_20261020 import run as parent
from .storage import read, write, digest, file_hash, status

ROOT = Path('eval/mcqa_at_anchor_comparison_20261020_v1')
SALT = 'mcqa_at_anchor_comparison_20261020_v1'
LETTERS = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
COLORS = tuple('beige black blue brown burgundy crimson cyan gray green indigo magenta maroon navy olive orange pink purple red tan teal turquoise white yellow'.split())
DIRECTIONS = tuple(itertools.permutations(COLORS, 2))
FAMILIES = ('C_only', 'both', 'AT_only', 'preserve')
ALLOCATIONS = [('train',96), ('calibration',32), ('qualification',32), ('teacher_fit',16),
               ('teacher_validation',12), ('measurement_fit',16), ('measurement_validation',16), ('final',32)]
SIZES = {'at_train':6656, 'at_calibration':3328, 'at_qualification':6656,
         'c_train':3328, 'c_calibration':1664, 'c_final':26312,
         'teacher_fit':2392, 'teacher_validation':1196, 'measurement_fit':2392, 'measurement_validation':1196}
PROTOCOL = dict(salt=SALT, model='google/gemma-2-2b', revision='c5ebcd40d208330abc697524c919956e692655cf',
    colors=COLORS, letters=LETTERS, alternatives=32, rank=128, at_layer=25, c_layer=0,
    at_position='last_token', c_position='first_literal_color_subtoken', seeds=[1,0,2],
    optimizer='Adam', lr=.01, effective_batch=32, at_updates=2000, c_updates=1000,
    at_selection_steps=[250,500,1000,1500,2000], c_evaluation_steps=[0,100,250,500,750,1000],
    objective={'standard':[1,0,0], 'gradual':[1,1,1]}, checkpoint_every=25,
    atol=2e-4, rtol=2e-4, at_budget_seconds=5400, total_budget_seconds=21600,
    bootstrap=2000, bootstrap_seed=20261020, batch_candidates=[1,2,4,8,16,32],
    arithmetic='frozen BF16 SDPA; FP32 QR/projections and softcap; TF32 disabled',
    selection='natural full-vocabulary correctness with exact ties failed; no intervention selection',
    final_seal='C checkpoints, teachers and AT selections locked before final natural filtering')


def rng(*parts):
    return random.Random(int(digest([SALT, *parts]),16))


def inventory():
    templates = {}
    for path in sorted((parent.DATA / 'generated').glob('*.jsonl')):
        for line in path.open(encoding='utf-8'):
            row = json.loads(line)
            c = schema.parse_context(dict(base_prompt=row['prompt'], question=row['question_key']), 'base')
            first = c.prompt.splitlines()[0]
            template = first[:c.color_span[0]] + '{color}' + first[c.color_span[1]:]
            if c.question in templates and templates[c.question] != template:
                raise ValueError('Template normalization merged distinct wording')
            templates[c.question] = template
    if len(templates) != 252:
        raise ValueError('Expected the frozen 252 object templates')
    keys = sorted(templates, key=lambda x:(digest([SALT,x]),x))
    groups, start = {}, 0
    for name,count in ALLOCATIONS:
        groups[name] = [dict(id=k, template=templates[k]) for k in keys[start:start+count]]
        start += count
    assert start == 252
    return groups


def slots(bank, groups):
    n = SIZES[bank]
    group = bank.split('_',1)[1] if bank.startswith(('at_','c_')) else bank
    objects = groups[group]
    rows = []
    if bank.startswith('at_'):
        for category,fraction in [('preserve',2), ('in_options',1), ('outside_options',1)]:
            for same in (True,False):
                count = n*fraction//8
                for i in range(count):
                    li, rep = i%26, i//26
                    target = LETTERS[li]
                    base_letter = target if category == 'preserve' else LETTERS[(li+1+rep%25)%26]
                    rows.append(dict(kind='AT', category=category, same_color=same,
                        base_group=(rep+17*li)%len(objects), target=target, base_letter=base_letter,
                        base_color=COLORS[(rep+li)%23], ordinal=i))
    elif bank.startswith('c_'):
        for family in FAMILIES:
            for i in range(n//4):
                direction, repeat = divmod(i,13) if bank == 'c_final' else (i%506,i//506)
                cb,cs = DIRECTIONS[direction%506]
                if family in ('AT_only','preserve'):
                    cb = cs = COLORS[i%23]
                if bank == 'c_final' and family in ('C_only','both'):
                    bgroup = (direction + repeat + (13 if family == 'both' else 0))%32
                else:
                    bgroup = i%len(objects)
                rows.append(dict(kind='C', family=family, same_question=i%2==0,
                    base_group=bgroup, base_color=cb, source_color=cs, target=LETTERS[i%26],
                    subtype='fixed' if (i//2+i%2)%2==0 else 'disagree', ordinal=i))
    else:
        for i in range(n):
            color,letter = COLORS[(i//26)%23],LETTERS[i%26]
            rows.append(dict(kind='readout', base_group=((i//26)+i%26)%len(objects),
                             base_color=color,target=letter,ordinal=i))
    assert len(rows) == n
    for i,row in enumerate(rows):
        row.update(bank=bank,slot=i,group=group)
    return rows


def context(group, color, labels, contents):
    prompt = group['template'].format(color=color) + '\n' + '\n'.join(f'{a}. {c}' for a,c in zip(labels,contents)) + '\nAnswer:'
    parsed = schema.parse_context(dict(base_prompt=prompt,question=group['id']), 'base')
    return asdict(parsed)


def answer(c):
    return c['labels'][c['contents'].index(c['color'])]


def mapped(c,color):
    return c['labels'][c['contents'].index(color)]


def make_state(randomizer, group, color, letter, required=(), forbidden=(), palette=()):
    colors = list(dict.fromkeys([color,*palette]))
    colors += randomizer.sample([x for x in COLORS if x not in colors],4-len(colors))
    labels = list(dict.fromkeys([letter,*required]))
    labels += randomizer.sample([x for x in LETTERS if x not in labels and x not in forbidden],4-len(labels))
    randomizer.shuffle(colors); randomizer.shuffle(labels)
    p,q = colors.index(color),labels.index(letter)
    labels[p],labels[q] = labels[q],labels[p]
    return context(group,color,labels,colors)


def proposal(slot, alternative, groups):
    if not 0 <= alternative < 32:
        raise ValueError('Outside frozen finite proposal pool')
    r = rng(slot['bank'],slot['slot'],alternative)
    objects = groups[slot['group']]
    bg = objects[slot['base_group']]
    other = r.choice([x for x in objects if x['id'] != bg['id']])
    color,target = slot['base_color'],slot['target']
    if slot['kind'] == 'readout':
        base = make_state(r,bg,color,target)
        return dict(slot=slot['slot'],alternative=alternative,base=base,target=target)
    if slot['kind'] == 'AT':
        category = slot['category']
        base = make_state(r,bg,color,slot['base_letter'],
                          required=[target] if category == 'in_options' else [],
                          forbidden=[target] if category == 'outside_options' else [])
        sc = color if slot['same_color'] else r.choice([x for x in COLORS if x != color])
        source = make_state(r,other,sc,target)
    else:
        family,sc = slot['family'],slot['source_color']
        sg = bg if slot['same_question'] else other
        if family in ('C_only','both'):
            bl = r.choice([x for x in LETTERS if x != target])
            base = make_state(r,bg,color,bl,required=[target],palette=[sc])
            labels,contents = list(base['labels']),list(base['contents'])
            p,q = contents.index(sc),labels.index(target)
            labels[p],labels[q] = labels[q],labels[p]
            base = context(bg,color,labels,contents)
            if family == 'C_only':
                bp,sp = contents.index(color),contents.index(sc)
                contents[bp],contents[sp] = contents[sp],contents[bp]
                source = context(sg,sc,labels,contents)
            elif slot['subtype'] == 'fixed':
                source = context(sg,sc,labels,contents)
            else:
                # Four distinct outcomes: base, source letter, source position, color.
                others = [x for x in labels if x not in (bl,target)]
                source_letter, position_letter = others
                sp = labels.index(position_letter)
                scontents = list(contents); q = scontents.index(sc)
                scontents[sp],scontents[q] = scontents[q],scontents[sp]
                slabels = list(labels); q = slabels.index(source_letter)
                slabels[sp],slabels[q] = slabels[q],slabels[sp]
                source = context(sg,sc,slabels,scontents)
        else:
            base = make_state(r,bg,color,target)
            labels,contents = list(base['labels']),list(base['contents'])
            if family == 'AT_only':
                labels[contents.index(color)] = r.choice([x for x in LETTERS if x not in labels])
            else:
                p = r.randrange(4); q = (p+r.randrange(1,4))%4
                labels[p],labels[q] = labels[q],labels[p]
                contents[p],contents[q] = contents[q],contents[p]
            source = context(sg,color,labels,contents)
    result = dict(slot=slot['slot'],alternative=alternative,base=base,source=source,target=target)
    validate(slot,result)
    return result


def validate(slot,row):
    b = row['base']
    assert len(set(b['contents'])) == len(set(b['labels'])) == 4
    if slot['kind'] == 'readout':
        assert answer(b) == row['target']
        return
    s = row['source']
    assert b['id'] != s['id']
    if slot['kind'] == 'AT':
        assert b['question'] != s['question']
        assert answer(s) == row['target']
        assert (b['color'] == s['color']) == slot['same_color']
        assert (answer(b) == answer(s)) == (slot['category'] == 'preserve')
        assert (answer(s) in b['labels']) == (slot['category'] != 'outside_options')
    else:
        cc,ac = b['color'] != s['color'], answer(b) != answer(s)
        assert (cc,ac) == {'C_only':(True,False),'both':(True,True),'AT_only':(False,True),'preserve':(False,False)}[slot['family']]
        assert (b['question'] == s['question']) == slot['same_question']
        assert mapped(b,s['color']) == row['target']
        if slot['family'] == 'both' and slot['subtype'] == 'disagree':
            assert len({answer(b),answer(s),mapped(b,s['color']),b['labels'][s['contents'].index(s['color'])]}) == 4


def audit_slots(bank, rows, groups):
    assert len(rows) == SIZES[bank]
    if bank.startswith('at_'):
        cells = Counter((x['category'],x['same_color']) for x in rows)
        assert sorted(cells.values()) == [len(rows)//8]*4+[len(rows)//4]*2
        for cell in cells:
            subset = [x for x in rows if (x['category'],x['same_color'])==cell]
            assert len(set(Counter(x['target'] for x in subset).values())) == 1
        if bank == 'at_qualification':
            for category in ('preserve','in_options','outside_options'):
                for letter in LETTERS:
                    assert len({x['base_group'] for x in rows if x['category']==category and x['target']==letter}) >= 25
    elif bank.startswith('c_'):
        assert set(Counter(x['family'] for x in rows).values()) == {len(rows)//4}
        for family in FAMILIES:
            subset = [x for x in rows if x['family']==family]
            assert sum(x['same_question'] for x in subset) == len(subset)//2
        if bank == 'c_final':
            for cb,cs in DIRECTIONS:
                subset = [x for x in rows if x['family'] in ('C_only','both') and (x['base_color'],x['source_color'])==(cb,cs)]
                assert len(subset) == len({x['base_group'] for x in subset}) == 26
    else:
        assert set(Counter((x['base_color'],x['target']) for x in rows).values()) == {len(rows)//598}
    return dict(count=len(rows),templates=len({x['base_group'] for x in rows}),passed=True)


def prepare(root=ROOT):
    root = Path(root)
    groups = inventory()
    write(root/'protocol.json',json.loads(json.dumps(PROTOCOL)))
    write(root/'groups.json',groups)
    excluded = set()
    for bankpath in (Path('eval/hai_frozen_descriptive_20261020_v1/selected_pairs.json'),):
        contexts,_ = schema.import_pairs(read(bankpath))
        excluded.update(c.id for c in contexts)
    # Includes all additional natural prompts used by the full-chain diagnostic.
    for record in read(Path('eval/hai_c_at_full_chain_20261020_v3/readout_plan.json'))['contexts']:
        excluded.add(record['id'])
    write(root/'historical_prompt_exclusions.json',sorted(excluded))
    imports = {str(parent.DATA/'manifest.json'):file_hash(parent.DATA/'manifest.json')}
    for path in sorted((parent.DATA/'generated').glob('*.jsonl')):
        imports[str(path)] = file_hash(path)
    manifest = {}
    for bank in SIZES:
        rows = slots(bank,groups)
        audit = audit_slots(bank,rows,groups)
        write(root/'proposals'/f'{bank}.slots.json',rows)
        receipt = root/'proposals'/f'{bank}.pool.json'
        if receipt.exists():
            existing = read(receipt)
            if existing['generator_sha256'] != file_hash(__file__) or existing['slots_sha256'] != file_hash(root/'proposals'/f'{bank}.slots.json'):
                raise ValueError('Frozen slots changed')
            manifest[bank] = existing
            continue
        h = hashlib.sha256()
        for slot in rows:
            for alternative in range(32):
                value = proposal(slot,alternative,groups)
                h.update(json.dumps(value,sort_keys=True,separators=(',',':')).encode()+b'\n')
        result = dict(bank=bank,slots_sha256=file_hash(root/'proposals'/f'{bank}.slots.json'),
            generator_sha256=file_hash(__file__),
            complete_proposal_sha256=h.hexdigest(),proposals=len(rows)*32,alternatives=32,
            encoding='Canonical compact JSON plus newline, slots then alternatives ascending; deterministically regenerated',audit=audit)
        write(receipt,result)
        manifest[bank] = result
        status(root,'preparing',bank=bank,proposals=result['proposals'])
    write(root/'proposals/manifest.json',manifest)
    sources = [*Path(__file__).parent.glob('*.py'),Path(parent.__file__),
               *Path('experiments/hai_c_at_evaluation').glob('*.py'),
               Path('experiments/mcqa_c_at_localization/runtime.py'),
               Path('experiments/mcqa_c_at_localization/profiling.py'),
               *Path('experiments/hai_c_at_full_chain').glob('*.py'),
               *Path('experiments/toy_c_at_composition').glob('*.py'),
               Path('experiments/mcqa_c_at_it_das/training.py'),
               Path('experiments/mcqa_c_at_it_gradual_pilot/engineering.py'),
               Path('diagnostics/hai_full_chain_20261020/resume_shared_io.py')]
    identity = dict(source={str(p):file_hash(p) for p in sources},imports=imports,
        protocol=file_hash(root/'protocol.json'),groups=file_hash(root/'groups.json'),
        pools=file_hash(root/'proposals/manifest.json'),exclusions=file_hash(root/'historical_prompt_exclusions.json'),
        model=read(parent.DATA/'protocol.json')['model_files'])
    write(root/'identity.json',identity)
    status(root,'prepared',banks=len(SIZES),frozen_proposals=sum(SIZES.values())*32,scientific_training_started=False)
    return identity


def verify_identity(root, model=False):
    root=Path(root); ident=read(root/'identity.json')
    for group in ('source','imports'):
        for p,h in ident[group].items():
            if file_hash(p)!=h:
                raise ValueError('Immutable source/input changed: '+p)
    for key,path in [('protocol','protocol.json'),('groups','groups.json'),('pools','proposals/manifest.json'),('exclusions','historical_prompt_exclusions.json')]:
        if file_hash(root/path)!=ident[key]:
            raise ValueError('Registered artifact changed: '+path)
    if model:
        for p,h in ident['model'].items():
            if file_hash(parent.MODEL/p)!=h:
                raise ValueError('Model/tokenizer hash mismatch: '+p)
    return digest(ident)
