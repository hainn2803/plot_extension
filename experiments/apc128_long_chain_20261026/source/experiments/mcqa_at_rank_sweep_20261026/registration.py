"""Seal exact nonfinal inputs and read-only cache provenance before GPU work."""
from functools import lru_cache
from pathlib import Path
import shutil
import sys
import time

import torch
from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash
from experiments.mcqa_ap_gradual_chain_20261025 import data, banks
from .protocol import ROOT, PARENT, BANKS, PROTOCOL


def verify_parent_exit():
    import psutil
    process = read(PARENT/'latest_process.json')
    try:
        current = psutil.Process(process['pid'])
        if abs(current.create_time()-process['created']) < .01:
            raise RuntimeError('Actual parent worker is still alive')
    except psutil.NoSuchProcess:
        pass
    ledger = read(PARENT/'engineering/gpu_budget.json')
    if ledger['active'] or ledger['seconds'] != PROTOCOL['inherited_seconds']:
        raise RuntimeError('Parent closed accounting changed')
    if (PARENT/'pause.request').exists():
        raise RuntimeError('Parent safety pause remains enforced')
    if (PARENT/'final_access.json').exists():
        raise RuntimeError('Unexpected parent final access')
    return process, ledger


def prepare(cpu_receipt, root=ROOT):
    root = Path(root)
    if (root/'identity.json').exists():
        return verify(root)
    parent = data.verify(PARENT)
    process, ledger = verify_parent_exit()
    write(root/'protocol.json', PROTOCOL)
    shutil.copy2(PARENT/'groups.json', root/'groups.json')
    inputs = {str(PARENT/'identity.json'): file_hash(PARENT/'identity.json'),
              str(PARENT/'engineering/gpu_budget.json'): file_hash(PARENT/'engineering/gpu_budget.json')}
    wanted = set()
    for bank in BANKS:
        rows = banks.load_bank(PARENT, bank)
        data.audit(rows, bank)
        for name in ('pairs.json', 'completion.json'):
            source, destination = PARENT/'banks'/bank/name, root/'banks'/bank/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            inputs[str(source)] = file_hash(source)
            inputs[str(destination)] = file_hash(destination)
        wanted.update(row[side]['id'] for row in rows for side in ('base', 'source'))
    index = {}
    for path in sorted((PARENT/'natural_cache').glob('*.ids.json')):
        ids = read(path)
        matches = wanted.intersection(ids)
        if not matches:
            continue
        tensor = path.with_name(path.name.replace('.ids.json', '.pt'))
        chunk = load(tensor)
        for i, ident in enumerate(ids):
            if ident in matches:
                if ident in index or not bool(chunk['eligible'][i]):
                    raise ValueError('Duplicate or ineligible imported natural context')
                index[ident] = dict(path=str(tensor), index=i)
        for fixed in (path, tensor, tensor.with_suffix('.receipt.json')):
            inputs[str(fixed)] = file_hash(fixed)
    if set(index) != wanted:
        raise ValueError('Missing sealed nonfinal natural states')
    write(root/'cache_index.json', index)
    done_path = PARENT/'fits/AT_S1/completion.json'
    done = read(done_path)
    imported = dict(rank=128, seed=1, folder=str(done_path.parent), completion=done,
                    readout=str(PARENT/'readouts/AT_S1.pt'), original_stop_preserved=True)
    write(root/'imported_rank128_seed1.json', imported)
    for path in [done_path, Path(done['checkpoint']), Path(done['checkpoint']).with_suffix('.receipt.json'),
                 PARENT/'fits/AT_S1/latest.json', PARENT/'readouts/AT_S1.pt',
                 PARENT/'readouts/AT_S1.receipt.json', PARENT/'readouts/AT_S1.json',
                 *sorted((PARENT/'fits/AT_S1/evaluations').glob('*'))]:
        if path.is_file():
            inputs[str(path)] = file_hash(path)
    sources = {p.resolve() for p in Path(__file__).parent.glob('*.py')}
    sources.update(Path(m.__file__).resolve() for name, m in list(sys.modules.items())
                   if name.startswith(('experiments.', 'diagnostics.')) and (getattr(m, '__file__', None) or '').endswith('.py'))
    workspace = Path.cwd().resolve()
    for source in sorted(sources):
        relative = source.relative_to(workspace)
        inputs[str(relative)] = file_hash(source)
        dest = root/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    for path in (root/'protocol.json', root/'groups.json', root/'cache_index.json',
                 root/'imported_rank128_seed1.json', cpu_receipt):
        inputs[str(path)] = file_hash(path)
    for path in sorted((root/'engineering/registration_failure_1793021387').glob('*')):
        if path.is_file():
            inputs[str(path)] = file_hash(path)
    write(root/'identity.json', dict(files=inputs, parent_identity=file_hash(PARENT/'identity.json'),
          model=parent['model'], model_root=parent['model_root'], inherited_ledger=ledger,
          inherited_process=process, created=time.time(), final_access=False))
    write(root/'engineering/gpu_budget.json', dict(ledger, stage='amendment_registered',
          accounting='Inherited closed parent ledger; all new GPU sessions add to these charges'))
    return verify(root)


def verify(root=ROOT, model=False):
    root = Path(root)
    data.verify(PARENT, model=model)
    identity = read(root/'identity.json')
    for path, expected in identity['files'].items():
        if file_hash(path) != expected:
            raise ValueError('Sweep sealed input changed: '+path)
    for folder in (PARENT, root):
        if (folder/'final_access.json').exists() or list((folder/'banks').glob('*final*/pairs.json')):
            raise ValueError('Final data unexpectedly opened')
    return identity


class Cache:
    """Read only the previously validated natural states; never add contexts."""

    def __init__(self, root=ROOT):
        self.index = read(Path(root)/'cache_index.json')

    @lru_cache(maxsize=32)
    def chunk(self, path):
        return load(path)

    def field(self, contexts, key):
        if any(c['id'] not in self.index for c in contexts):
            raise ValueError('Unregistered context requested from read-only cache')
        return torch.stack([self.chunk(self.index[c['id']]['path'])[key][self.index[c['id']]['index']] for c in contexts])

    def states(self, rows):
        return dict(base_AT=self.field([r['base'] for r in rows], 'AT').float(),
                    source_AT=self.field([r['source'] for r in rows], 'AT').float())
