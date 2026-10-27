"""The prior study's exact structural control roles, with new parent pairs."""
from pathlib import Path
from experiments.mcqa_ap_localization_sweep.controls import ROLES, proposal, rows_for
from experiments.mcqa_at_anchor_comparison.storage import read, write, status
from . import banks, data


def filter_controls(root, rows, cache, label):
    root = Path(root)
    banks.locked(root)
    folder = root/'controls'/label
    objects = read(root/'groups.json')['final']
    excluded = banks.exclusions(root, label+'_final')
    result = {}
    for start in range(0, len(rows), 32):
        path = folder/f'page_{start:06d}.json'
        if path.exists():
            result.update(read(path)['accepted'])
            continue
        piece = rows[start:start+32]
        pending = {(r['id'], role): (r, role) for r in piece for role in ROLES}
        accepted = {r['id']: {} for r in piece}
        used = {r['id']: set() for r in piece}
        orders = {k: sorted(range(256), key=lambda a: data.key('control-order', *k, a)) for k in pending}
        log = []
        for rank in range(256):
            proposed = {k: proposal(r, role, orders[k][rank], objects) for k, (r, role) in pending.items()}
            candidates = {k: c for k, c in proposed.items() if c['id'] not in excluded and c['id'] not in used[k[0]]}
            cache.ensure(list(candidates.values()))
            for k, c in proposed.items():
                clean = k in candidates and bool(cache.field([c], 'eligible')[0])
                previous = accepted[k[0]]
                if any(c['question'] == v['question'] for v in previous.values()):
                    clean = False
                partner = k[1][:-1]+str(1-int(k[1][-1])) if k[1].startswith(('AP_donor', 'AT_donor')) else None
                if partner in previous and (c['color'] == previous[partner]['color'] or
                    (k[1].startswith('AP') and data.answer(c) == data.answer(previous[partner]))):
                    clean = False
                log.append(dict(pair=k[0], role=k[1], alternative=orders[k][rank], rank=rank,
                                eligible=clean, historical_overlap=c['id'] in excluded, context_id=c['id']))
                if clean:
                    accepted[k[0]][k[1]] = c
                    used[k[0]].add(c['id'])
                    del pending[k]
            if not pending:
                break
        missing = [dict(pair=k[0], role=k[1]) for k in pending]
        write(path, dict(accepted=accepted, missing=missing, dispositions=log))
        result.update(accepted)
        status(root, 'filtering_controls', bank=label, completed=start+len(piece), total=len(rows), missing=len(missing))
    missing = [dict(pair=r['id'], role=role) for r in rows for role in ROLES if role not in result[r['id']]]
    write(folder/'completion.json', dict(pairs=len(rows), missing=missing, complete=not missing, primary_pairs_removed=0))
    for role in ROLES:
        rr, _ = rows_for(rows, result, role)
        if rr:
            cache.layout(rr, label+'_'+role)
    return result
