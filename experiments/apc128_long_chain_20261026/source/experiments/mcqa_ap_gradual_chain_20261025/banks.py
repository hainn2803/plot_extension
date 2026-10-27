from pathlib import Path
import time

from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash, status
from . import data


def locked(root):
    from .protocol import fit_names
    value = read(Path(root) / 'locks/all_final.json')
    if set(value['endpoints']) != set(fit_names()) or len(value['readouts']) != 9:
        raise ValueError('All 15 endpoints and nine readouts must precede final access')
    for item in [*value['endpoints'].values(), *value['readouts'].values()]:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Locked endpoint/readout changed')
    return value


def load_bank(root, bank):
    folder = Path(root) / 'banks' / bank
    done = read(folder / 'completion.json')
    if file_hash(folder / 'pairs.json') != done['pairs_sha256']:
        raise ValueError('Bank hash changed: ' + bank)
    return read(folder / 'pairs.json')


def exclusions(root, bank):
    excluded = set(read(Path(root) / 'historical_exclusions.json'))
    if bank.startswith('readout_') or bank.endswith('_final'):
        for path in (Path(root) / 'banks').glob('*/pairs.json'):
            name = path.parent.name
            if name == bank or name.endswith('_final'):
                continue
            for row in read(path):
                excluded.update(row[side]['id'] for side in ('base', 'source'))
    return excluded


def filter_bank(root, bank, cache):
    root = Path(root)
    final = bank.endswith('_final')
    if final:
        locked(root)
    folder = root / 'banks' / bank
    if (folder / 'completion.json').exists():
        rows = load_bank(root, bank)
        cache.layout(rows, bank)
        return rows
    groups = read(root / 'groups.json')
    slots = read(root / 'proposals' / (bank + '.json'))
    excluded = exclusions(root, bank)
    write(folder / 'exclusions.json', sorted(excluded))
    selected, used, occupied = [], set(), {}
    started = time.perf_counter()
    for start in range(0, len(slots), 32):
        path = folder / f'page_{start:06d}.json'
        if path.exists():
            records = read(path)['selected']
        else:
            pending = {s['slot']: s for s in slots[start:start+32]}
            orders = {k: data.alternatives(s) for k, s in pending.items()}
            accepted, dispositions = {}, []
            for rank in range(256):
                proposed = {k: data.proposal(s, orders[k][rank], groups) for k, s in pending.items()}
                candidates = {}
                for k, rows in proposed.items():
                    history = any(r[side]['id'] in excluded for r in rows for side in ('base', 'source'))
                    if not history and not any(r['id'] in used for r in rows):
                        candidates[k] = rows
                    dispositions.append(dict(slot=k, alternative=orders[k][rank], rank=rank, historical_overlap=history))
                cache.ensure([r[side] for rr in candidates.values() for r in rr for side in ('base', 'source')])
                for k, rows in candidates.items():
                    slot = pending[k]
                    cell = (slot.get('b'), slot.get('s'))
                    needs_coverage = final and slot['variable'] in ('AP', 'C') and slot['kind'] == 'changing' and slot['ordinal'] < 32
                    repeated = needs_coverage and rows[0]['base']['question'] in occupied.get(cell, set())
                    clean = bool(cache.field([r[side] for r in rows for side in ('base', 'source')], 'eligible').all())
                    retained = clean and not repeated and not any(r['id'] in used for r in rows)
                    dispositions.append(dict(slot=k, rank=rank, clean=clean, coverage_excluded=repeated, retained=retained))
                    if retained:
                        accepted[k] = rows
                        used.update(r['id'] for r in rows)
                        occupied.setdefault(cell, set()).add(rows[0]['base']['question'])
                        del pending[k]
                if not pending:
                    break
            if pending:
                write(folder / f'shortfall_{start:06d}.json', dict(unfilled=list(pending), accepted=accepted, dispositions=dispositions))
                raise RuntimeError('Frozen 256-alternative bank pool exhausted: ' + bank)
            records = [r for s in slots[start:start+32] for r in accepted[s['slot']]]
            write(path, dict(selected=records, dispositions=dispositions))
        selected.extend(records)
        used.update(r['id'] for r in records)
        for r in records:
            m = r['metadata']
            occupied.setdefault((m.get('b'), m.get('s')), set()).add(r['base']['question'])
        status(root, 'filtering', bank=bank, accepted=len(selected), total=data.BANK_SIZES[bank])
    audit = data.audit(selected, bank)
    write(folder / 'pairs.json', selected)
    cache.layout(selected, bank)
    write(folder / 'completion.json', dict(audit=audit, pairs_sha256=file_hash(folder / 'pairs.json'),
                                          seconds=time.perf_counter()-started, naturally_clean_only=True))
    return selected
