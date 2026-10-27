"""Frozen C-pair reservoirs; no model calls or outcome-based selection.

prepare(root) registers slots, groups, exclusions and all 256 pair hashes per
slot. proposal(slot, alternative, groups) regenerates a complete candidate pair.
The runner selects the first eligible naturally clean pair in ascending
alternative order and writes banks/<bank>/pairs.json. load_rows(root, bank)
reads and checks that selected bank; it does not certify natural correctness.
"""
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random

from experiments.hai_c_at_evaluation import schema
from experiments.mcqa_at_anchor_comparison import data as at_anchor


WORKSPACE = Path(__file__).resolve().parents[2]
SALT = '20261023'
ALGORITHM = 'c_composition_pair_reservoir_v1'
ALGORITHM_SEED = 20261023
ALTERNATIVES = 256
LETTERS = at_anchor.LETTERS
COLORS = at_anchor.COLORS
FAMILIES = at_anchor.FAMILIES
ALLOCATIONS = (('train', 96), ('calibration', 48), ('final', 108))
SIZES = {'train': 2048, 'calibration': 512, 'final': 2048}
SOURCE = WORKSPACE / at_anchor.parent.DATA
HISTORICAL_PAIRS = (
    (WORKSPACE / 'eval/hai_frozen_descriptive_20261020_v1/selected_pairs.json', 1600),
    (WORKSPACE / 'eval/hai_frozen_clean_only_20261020_v1/selected_pairs.json', 1476),
)
AP_PLAN = WORKSPACE / 'eval/hai_followup_checks_20261022_v1/plan.json'


def _bank(bank):
    bank = 'calibration' if bank == 'cal' else bank
    if bank not in SIZES:
        raise ValueError('Unknown bank: ' + str(bank))
    return bank


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def inventory():
    """Normalize the existing generated inventory and hash-resplit its objects."""
    templates = {}
    for path in sorted((SOURCE / 'generated').glob('*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                c = schema.parse_context(
                    dict(base_prompt=row['prompt'], question=row['question_key']), 'base')
                first = c.prompt.splitlines()[0]
                template = first[:c.color_span[0]] + '{color}' + first[c.color_span[1]:]
                if c.question in templates and templates[c.question] != template:
                    raise ValueError('Template normalization merged distinct wording')
                templates[c.question] = template
    if len(templates) != 252:
        raise ValueError('Expected the frozen 252 object templates')
    keys = sorted(templates, key=lambda key: (schema.digest([SALT, key]), key))
    groups, start = {}, 0
    for name, count in ALLOCATIONS:
        groups[name] = [dict(id=key, template=templates[key])
                        for key in keys[start:start + count]]
        start += count
    return groups


def slots(bank, groups):
    """Balance only family, query relation and the two both-family subtypes."""
    bank = _bank(bank)
    result = []
    for family in FAMILIES:
        for ordinal in range(SIZES[bank] // 4):
            index = len(result)
            seed = schema.digest([ALGORITHM, ALGORITHM_SEED, bank, index])
            r = random.Random(int(seed, 16))
            result.append(dict(bank=bank, group=bank, slot=index, kind='C',
                family=family, same_question=ordinal % 2 == 0,
                subtype=('fixed' if ordinal // 2 % 2 == 0 else 'disagree')
                        if family == 'both' else None,
                ordinal=ordinal, seed=seed,
                base_group=r.randrange(len(groups[bank])),
                base_color=r.choice(COLORS), source_color=r.choice(COLORS),
                target=r.choice(LETTERS)))
    return result


def proposal(slot, alternative, groups):
    """Regenerate one of exactly 256 PAIRS, not a product of donor marginals."""
    if type(alternative) is not int or not 0 <= alternative < ALTERNATIVES:
        raise ValueError('Outside the registered 256-pair reservoir')
    bank = _bank(slot['bank'])
    block, inner = divmod(alternative, 32)
    color_index = (COLORS.index(slot['base_color']) + alternative) % len(COLORS)
    color = COLORS[color_index]
    source_color = color
    if slot['family'] in ('C_only', 'both'):
        offset = (COLORS.index(slot['source_color']) + alternative // len(COLORS))
        source_color = COLORS[(color_index + 1 + offset % (len(COLORS) - 1)) % len(COLORS)]
    virtual = dict(slot,
        bank=f'{ALGORITHM}:{SALT}:{bank}:block_{block}',
        base_group=(slot['base_group'] + alternative) % len(groups[bank]),
        base_color=color, source_color=source_color,
        target=LETTERS[(LETTERS.index(slot['target']) + alternative) % len(LETTERS)])
    row = at_anchor.proposal(virtual, inner, groups)
    metadata = dict(virtual, bank=bank)
    metadata['proposal_seed'] = schema.digest(
        [at_anchor.SALT, virtual['bank'], slot['slot'], inner])
    row.update(id=schema.digest([row['base']['id'], row['source']['id']]),
               alternative=alternative, metadata=dict(slot), proposal_metadata=metadata)
    validate(slot, row)
    return row


def answer(context):
    """Return the factual letter from metadata, without evaluating the prompt."""
    return at_anchor.answer(context)


def validate(slot, row):
    """Check the fixed family/query/subtype, not the slot's initial coverage."""
    at_anchor.validate(slot, row)
    for key in ('bank', 'group', 'slot', 'kind', 'family', 'same_question', 'subtype'):
        if row['metadata'][key] != slot[key]:
            raise ValueError('Candidate changed fixed slot metadata: ' + key)
    if slot['family'] == 'both' and slot['subtype'] == 'fixed':
        if (row['base']['labels'] != row['source']['labels'] or
                row['base']['contents'] != row['source']['contents']):
            raise ValueError('Fixed both-family candidate changed its options')
    key = [row['base']['id'], row['source']['id']]
    if row['id'] != schema.digest(key):
        raise ValueError('Candidate pair ID differs from its exact prompt IDs')


def historical_exclusions():
    """Exact prompt-ID union, including failed natural cases and AP cycles."""
    excluded, records, pair_sets = set(), [], []
    for path, count in HISTORICAL_PAIRS:
        contexts, pairs = schema.import_pairs(_read(path))
        if len(pairs) != count:
            raise ValueError('Historical selected-bank size changed: ' + str(path))
        ids = {c.id for c in contexts}
        excluded.update(ids)
        pair_sets.append({(contexts[p.base].id, contexts[p.source].id) for p in pairs})
        records.append(dict(path=str(path), sha256=schema.file_hash(path),
                            pairs=count, prompts=len(ids)))
    if not pair_sets[1] <= pair_sets[0]:
        raise ValueError('The 1476-pair bank is not a subset of the 1600-pair bank')
    contexts = _read(AP_PLAN)['contexts']
    for c in contexts:
        parsed = asdict(schema.parse_context(
            dict(base_prompt=c['prompt'], question=c['question']), 'base'))
        if _plain(parsed) != c:
            raise ValueError('AP-plan prompt ID or metadata differs from its exact text')
        excluded.add(c['id'])
    records.append(dict(path=str(AP_PLAN), sha256=schema.file_hash(AP_PLAN),
                        prompts=len(contexts)))
    return sorted(excluded), records


def _inputs():
    manifest_path = SOURCE / 'manifest.json'
    manifest = _read(manifest_path)
    generated = sorted((SOURCE / 'generated').glob('*.jsonl'))
    if {p.name for p in generated} != {'train.jsonl', 'validation.jsonl', 'test.jsonl'}:
        raise ValueError('Historical generated-data inventory changed')
    for path in generated:
        if schema.file_hash(path) != manifest['artifacts']['generated/' + path.name]:
            raise ValueError('Historical generated-data hash differs: ' + str(path))
    sources = [Path(__file__).resolve(), Path(at_anchor.__file__).resolve(),
               Path(schema.__file__).resolve()]
    paths = [manifest_path, *generated, *(p for p, _ in HISTORICAL_PAIRS), AP_PLAN]
    return dict(source={str(p): schema.file_hash(p) for p in sources},
                imports={str(p): schema.file_hash(p) for p in paths})


def _profile():
    return {key: Counter() for key in
            ('families', 'query_relation', 'both_subtypes', 'base_templates',
             'source_templates', 'target_letters', 'base_letters', 'source_letters',
             'base_colors', 'source_colors', 'color_directions')}


def _count(profile, row):
    b, s, m = row['base'], row['source'], row['metadata']
    values = dict(families=m['family'],
        query_relation='same' if m['same_question'] else 'different',
        base_templates=b['question'], source_templates=s['question'],
        target_letters=row['target'], base_letters=at_anchor.answer(b),
        source_letters=at_anchor.answer(s), base_colors=b['color'],
        source_colors=s['color'], color_directions=b['color'] + '->' + s['color'])
    if m['family'] == 'both':
        values['both_subtypes'] = m['subtype']
    for key, value in values.items():
        profile[key][value] += 1


def _audit_balance(bank, metadata):
    if len(metadata) != SIZES[bank]:
        raise ValueError('Incomplete bank: ' + bank)
    cells = Counter((m['family'], m['same_question']) for m in metadata)
    expected = Counter({(family, same): SIZES[bank] // 8
                        for family in FAMILIES for same in (True, False)})
    both = Counter((m['same_question'], m['subtype']) for m in metadata
                   if m['family'] == 'both')
    expected_both = Counter({(same, subtype): SIZES[bank] // 16
                            for same in (True, False) for subtype in ('fixed', 'disagree')})
    if cells != expected or both != expected_both:
        raise ValueError('Family/query/subtype balance differs: ' + bank)


def _settings():
    return dict(algorithm=ALGORITHM, algorithm_seed=ALGORITHM_SEED, salt=SALT,
        alternatives=ALTERNATIVES, sizes=SIZES, allocations=dict(ALLOCATIONS),
        colors=COLORS, letters=LETTERS, families=FAMILIES,
        python=platform.python_version(),
        split_order='SHA256 canonical JSON [salt, normalized template ID], then template ID',
        alternative_order='0..255 ascending; exactly 256 paired alternatives, no Cartesian expansion',
        variable_metadata='Target letter, factual colors and base template vary within each slot',
        fixed_metadata='Family, same/different question, and both fixed/disagree subtype',
        coverage_policy='Report intended and reservoir coverage; no per-template, letter or color quotas',
        selection='Runner only: first eligible pair by metadata, IDs and strict natural baseline; no interventions',
        historical_scope='Both exact selected banks (1600/1476 pairs) plus AP-control-plan contexts')


def verify_registration(root):
    """Verify frozen data and its source inputs without evaluating any prompt."""
    root = Path(root)
    registration = _read(root / 'data_registration.json')
    signature = registration['signature']
    if schema.digest({k: v for k, v in registration.items() if k != 'signature'}) != signature:
        raise ValueError('Registration signature differs')
    if registration['settings'] != _plain(_settings()):
        raise ValueError('Registered data algorithm/settings differ')
    for group in ('source', 'imports'):
        for name, expected in registration['inputs'][group].items():
            if schema.file_hash(name) != expected:
                raise ValueError('Registered source/input changed: ' + name)
    for name, expected in registration['artifacts'].items():
        if schema.file_hash(root / name) != expected:
            raise ValueError('Registered artifact changed: ' + name)
    return registration


def prepare(root):
    """Freeze all finite pair pools and return their signed registration dict."""
    root = Path(root).resolve()
    if (root / 'data_registration.json').exists():
        return verify_registration(root)
    inputs = _inputs()
    for name in inputs['source'] | inputs['imports']:
        path = Path(name)
        if path.is_relative_to(root) or root.is_relative_to(path.parent):
            raise ValueError('Output root overlaps historical inputs or scientific source: ' + str(root))
    groups = inventory()
    excluded, history = historical_exclusions()
    artifacts = {}

    def freeze(name, value):
        at_anchor.write(root / name, _plain(value))
        artifacts[name] = schema.file_hash(root / name)

    freeze('groups.json', groups)
    freeze('historical_prompt_exclusions.json', excluded)
    freeze('historical_exclusion_sources.json', history)
    manifest, coverage = {}, {}
    for bank in SIZES:
        ss = slots(bank, groups)
        _audit_balance(bank, ss)
        freeze(f'proposals/{bank}.slots.json', ss)
        pools, intended, reservoir = [], _profile(), _profile()
        for slot in ss:
            hashes = []
            for alternative in range(ALTERNATIVES):
                row = proposal(slot, alternative, groups)
                hashes.append(schema.digest(row))
                _count(reservoir, row)
                if alternative == 0:
                    _count(intended, row)
            if len(set(hashes)) != ALTERNATIVES:
                raise ValueError('Repeated complete candidate within slot: ' + str(slot['slot']))
            pools.append(dict(slot=slot['slot'], seed=slot['seed'], sha256=hashes))
        name = f'proposals/{bank}.pool.json'
        freeze(name, dict(algorithm=ALGORITHM, algorithm_seed=ALGORITHM_SEED,
            bank=bank, alternatives=ALTERNATIVES, slots=len(ss),
            proposals=len(ss) * ALTERNATIVES, encoding='SHA256 of canonical compact sorted JSON per complete pair',
            pools=pools))
        manifest[bank] = dict(slots_path=f'proposals/{bank}.slots.json', pool_path=name,
            slots=len(ss), alternatives=ALTERNATIVES, proposals=len(ss) * ALTERNATIVES,
            selected_path=f'banks/{bank}/pairs.json')
        coverage[bank] = dict(intended=intended, reservoir=reservoir,
            interpretation='Intended is alternative 0 before any natural filtering; actual selected coverage must be reported separately',
            rigid_template_letter_color_quotas=False)
    freeze('proposals/manifest.json', manifest)
    freeze('intended_coverage.json', coverage)
    if inputs != _inputs():
        raise ValueError('Source or historical input changed during preparation')
    registration = _plain(dict(settings=_settings(), inputs=inputs, artifacts=artifacts,
        banks=manifest, historical_prompt_count=len(excluded),
        proposal_count=sum(SIZES.values()) * ALTERNATIVES,
        prompts_evaluated=False, naturally_clean_banks_created=False))
    registration['signature'] = schema.digest(registration)
    at_anchor.write(root / 'data_registration.json', registration)
    return verify_registration(root)


def load_rows(root, bank):
    """Read a runner-selected bank; check frozen membership, IDs and metadata.

    Natural correctness and first-eligible ordering require the runner's baseline
    evidence and are deliberately not certified by this metadata-only loader.
    """
    root, bank = Path(root), _bank(bank)
    registration = verify_registration(root)
    entry = registration['banks'][bank]
    rows = _read(root / entry['selected_path'])
    if not isinstance(rows, list) or len(rows) != SIZES[bank]:
        raise ValueError('Expected a complete selected pair list: ' + bank)
    ss = _read(root / entry['slots_path'])
    pools = _read(root / entry['pool_path'])['pools']
    excluded = set(_read(root / 'historical_prompt_exclusions.json'))
    seen = set()
    for index, row in enumerate(rows):
        alternative = row['alternative']
        if type(alternative) is not int or not 0 <= alternative < ALTERNATIVES:
            raise ValueError('Selected alternative outside frozen reservoir')
        if row['slot'] != index or row['metadata']['slot'] != index:
            raise ValueError('Selected rows must be in frozen slot order')
        if schema.digest(row) != pools[index]['sha256'][alternative]:
            raise ValueError('Selected pair differs from its frozen complete-pair hash')
        for key in ('bank', 'group', 'kind', 'family', 'same_question', 'subtype', 'ordinal', 'seed'):
            if row['metadata'][key] != ss[index][key]:
                raise ValueError('Selected slot metadata differs: ' + key)
        at_anchor.validate(row['metadata'], row)
        key = (row['base']['id'], row['source']['id'])
        if key in seen or any(value in excluded for value in key):
            raise ValueError('Repeated pair or historical evaluation prompt overlap')
        if row['id'] != schema.digest(list(key)):
            raise ValueError('Selected pair ID differs from its prompt IDs')
        seen.add(key)
    _audit_balance(bank, [row['metadata'] for row in rows])
    return rows
