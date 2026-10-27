"""Preserve parent studies and seal a separate explicit protocol amendment."""
from pathlib import Path
import shutil
import sys
import time

from experiments.mcqa_at_anchor_comparison.storage import read, write, load, file_hash
from experiments.mcqa_ap_gradual_chain_20261025 import data, banks
from experiments.mcqa_at_rank_sweep_20261026.registration import verify as verify_sweep
from experiments.mcqa_at_rank_sweep_20261026.training import locations
from .protocol import ROOT, PARENT, SWEEP, PROTOCOL, SEEDS


def require_exited(process):
    import psutil
    try:
        actual = psutil.Process(process['pid'])
        if abs(actual.create_time()-process['created']) < .01:
            raise RuntimeError('Actual prior worker is still running')
    except psutil.NoSuchProcess:
        pass


def copy_file(source, destination, imports, fixed):
    source, destination = Path(source), Path(destination)
    expected = file_hash(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination) != expected:
            raise ValueError('Partial import differs: '+str(destination))
    else:
        shutil.copy2(source, destination)
    if file_hash(destination) != expected:
        raise ValueError('Import copy mismatch')
    imports[str(source)] = expected
    fixed[str(destination)] = expected


def prepare(root=ROOT):
    root = Path(root)
    if (root/'identity.json').exists():
        return verify(root)
    parent = data.verify(PARENT)
    verify_sweep(SWEEP)
    completion = read(SWEEP/'completion.json')
    ledger = read(SWEEP/'engineering/gpu_budget.json')
    if not completion['passed'] or ledger['active'] or ledger['seconds'] != PROTOCOL['inherited_seconds']:
        raise ValueError('Verified completed sweep and exact closed accounting required')
    require_exited(read(SWEEP/'latest_process.json'))
    if (SWEEP/'pause.request').exists() or (SWEEP/'final_access.json').exists() or (PARENT/'final_access.json').exists():
        raise ValueError('Unexpected pause or final access in imported studies')
    imports, fixed = {}, {}
    write(root/'protocol.json', PROTOCOL)
    for name in ('groups.json', 'historical_exclusions.json', 'execution.json', 'profile.json'):
        copy_file(PARENT/name, root/name, imports, fixed)
    for folder in ('proposals', 'banks', 'natural_cache'):
        for source in sorted((PARENT/folder).rglob('*')):
            if not source.is_file():
                continue
            if folder == 'banks' and any(part.endswith('_final') for part in source.relative_to(PARENT/'banks').parts):
                raise ValueError('No final bank may be imported')
            copy_file(source, root/source.relative_to(PARENT), imports, fixed)
    teacher_policy = dict(AT_qualification_enabled=False, explicitly_user_authorized=True, teachers={})
    for seed in SEEDS:
        folder, head = locations(128, seed, SWEEP)
        done = read(folder/'completion.json')
        load(done['checkpoint'])
        for name in ('completion.json', 'latest.json'):
            copy_file(folder/name, root/'fits'/f'AT_S{seed}'/name, imports, fixed)
        load(head)
        for suffix in ('.pt', '.json'):
            copy_file(head.with_suffix(suffix), root/'readouts'/f'AT_S{seed}{suffix}', imports, fixed)
        source_receipt = head.with_suffix('.receipt.json')
        imports[str(source_receipt)] = file_hash(source_receipt)
        destination_receipt = root/'readouts'/f'AT_S{seed}.receipt.json'
        receipt = dict(file=f'AT_S{seed}.pt', sha256=file_hash(head))
        if destination_receipt.exists() and read(destination_receipt) != receipt:
            if read(destination_receipt) != read(source_receipt):
                raise ValueError('Unexpected imported readout receipt mismatch')
            archive = root/'engineering/import_receipt_provenance'/f'AT_S{seed}.source_filename.receipt.json'
            archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination_receipt, archive)
            fixed[str(archive)] = file_hash(archive)
        write(destination_receipt, receipt, immutable=False)
        fixed[str(destination_receipt)] = file_hash(destination_receipt)
        write(root/'engineering/import_receipt_provenance'/f'AT_S{seed}.json', dict(
            source=str(head), source_receipt=str(source_receipt), source_receipt_sha256=file_hash(source_receipt),
            source_tensor_sha256=file_hash(head), destination_tensor_sha256=file_hash(root/'readouts'/f'AT_S{seed}.pt'),
            only_receipt_basename_rebased=True))
        # The checkpoint stays at its preserved source path. Seal both wrapper
        # and receipt; the new completion record points to those exact bytes.
        for p in (Path(done['checkpoint']), Path(done['checkpoint']).with_suffix('.receipt.json')):
            imports[str(p)] = file_hash(p)
        teacher_policy['teachers'][str(seed)] = dict(checkpoint=done['checkpoint'], checkpoint_sha256=done['sha256'],
            calibration=done['endpoint']['calibration'], factual_readout=read(head.with_suffix('.json')),
            trained_updates=1000, imported=True, qualified=False, use_authorized_despite_accuracy=True)
    for name in ('identity.json', 'completion.json', 'engineering/gpu_budget.json', 'independent_verification.json'):
        imports[str(SWEEP/name)] = file_hash(SWEEP/name)
    write(root/'AT_policy.json', teacher_policy)
    from . import tests
    tests.cpu(root)
    sources = {p.resolve() for p in Path(__file__).parent.glob('*.py')}
    sources.update(Path(p).resolve() for p in parent['files'] if p.endswith('.py'))
    sources.update(Path(m.__file__).resolve() for name, m in list(sys.modules.items())
                   if name.startswith(('experiments.', 'diagnostics.')) and (getattr(m, '__file__', None) or '').endswith('.py'))
    workspace = Path.cwd().resolve()
    for source in sorted(sources):
        path = source.relative_to(workspace)
        fixed[str(path)] = file_hash(path)
        destination = root/'source_snapshot'/path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for path in (root/'protocol.json', root/'AT_policy.json', root/'engineering/cpu_tests.json'):
        fixed[str(path)] = file_hash(path)
    identity = dict(files=fixed, imports=imports, model=parent['model'], model_root=parent['model_root'],
          prior_ledger=parent['prior_ledger'], prior_ledger_sha256=parent['prior_ledger_sha256'],
          new_allocation_seconds=0, shared_total_limit_seconds=43200, inherited_ledger=ledger,
          parent_identity_sha256=file_hash(PARENT/'identity.json'), sweep_identity_sha256=file_hash(SWEEP/'identity.json'),
          original_studies_unchanged=True, created=time.time())
    write(root/'identity.json', identity)
    write(root/'engineering/gpu_budget.json', dict(ledger, stage='amendment_registered',
          accounting='Inherited exact cumulative closed ledger; same12h cap, no new allocation'))
    return verify(root)


def verify(root=ROOT, model=False):
    value = data.verify(root, model=model)
    if value['new_allocation_seconds'] != 0 or value['shared_total_limit_seconds'] != 43200:
        raise ValueError('Unexpected budget reset/allocation')
    return value
