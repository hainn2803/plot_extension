"""Recorded signed-runtime migration; no scientific source or guard changes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time


IDENTITY = '0a25afe9da3db9ce72ca59cf7165f3a9231a2e6f5813d4a1c3f7fe252b55d487'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def setup():
    workspace = Path(__file__).resolve().parents[2]
    os.chdir(workspace)
    candidate = workspace / 'diagnostics/ap_thermal_20261021/evidence/python_candidate_20261022'
    if sys.version_info[:3] != (3, 11, 17) or Path(sys.executable).resolve() != candidate / 'payload/python.exe':
        raise RuntimeError('Unregistered Python runtime')
    if sha(sys.executable) != '0c81a500d76bd4a9af739d528acd57bde001dc9e666effb3494bbaec30a71004':
        raise RuntimeError('Signed interpreter changed')
    manifest = candidate / 'dependency_receipts.json'
    if sha(manifest) != 'de9540bdc79540c0fab2f498c8940bfe6fee468f72ae3d0bcf23f5f224c297a8':
        raise RuntimeError('Signed dependency manifest changed')
    handles = []
    for receipt in json.loads(manifest.read_text(encoding='utf-8-sig')):
        for dll in receipt['dlls']:
            if sha(dll['file']) != dll['sha256'].lower() or dll['signature'] != 'Valid':
                raise RuntimeError('Signed dependency changed')
        handles.append(os.add_dll_directory(str(Path(receipt['payload']) / 'Library/bin')))
    sys.path.insert(0, str(workspace / '.venv_mcqa_c_at/Lib/site-packages'))
    sys.path.insert(0, str(workspace))
    return handles


def reconstruct(root, cache):
    from experiments.mcqa_ap_specialization_abcd import data
    from experiments.mcqa_at_anchor_comparison.storage import read
    missing = set(cache.index)
    found = {}
    groups = read(root / 'groups.json')
    # No page has committed yet; all preserved cache entries are from page zero.
    slots = read(root / 'proposals/train_master.slots.json')[:64]
    orders = {slot['slot']: data.alternative_order(slot) for slot in slots}
    for rank in range(256):
        for slot in slots:
            row = data.proposal(slot, orders[slot['slot']][rank], groups)
            for side in ('base', 'source'):
                context = row[side]
                if context['id'] in missing:
                    found[context['id']] = context
                    missing.remove(context['id'])
        if not missing:
            return found
    raise RuntimeError('Preserved cache cannot be reconstructed from its frozen page: ' + str(len(missing)))


def check_cache(root, output):
    import torch
    from experiments.mcqa_ap_specialization_abcd import data
    from experiments.mcqa_ap_specialization_abcd.banks import NaturalCache
    from experiments.mcqa_ap_specialization_abcd.engine import Engine, donors, numerical
    from experiments.mcqa_ap_specialization_abcd.execution import Session
    from experiments.mcqa_c_at_localization import runtime
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    ledger_path = root / 'engineering/gpu_budget.json'
    ledger = read(ledger_path)
    if ledger['active'] or runtime.power_status()['ACLineStatus'] != 1:
        raise RuntimeError('Inactive prior session and AC power required')
    protected = [root / x for x in ('pause.request', 'identity.json', 'protocol.json',
        'execution.json', 'profile.json', 'status.json', 'groups.json')]
    protected += [Path('experiments/mcqa_c_at_localization/runtime.py'),
        Path('experiments/mcqa_c_at_localization/profiling.py')]
    before = {str(p): file_hash(p) for p in protected}
    write(output / 'engineering/gpu_budget.json', dict(ledger, active=False))
    result = dict(started=time.time(), passed=False, protected_before=before,
        atol=2e-4, rtol=2e-4, selection_changed=False, final_data_accessed=False)
    session = engine = cache = None
    try:
        session = Session(output, 'signed_runtime_cache_migration')
        session.allow(180, 'bounded cache migration audit')
        engine = Engine(root, session)
        versions = [p._version for p in engine.model.parameters()]
        cache = NaturalCache(root, engine)
        if len(cache.index) != 2460:
            raise RuntimeError('Unexpected preserved cache size')
        contexts = reconstruct(root, cache)
        reports = []
        for ids_path in sorted(cache.folder.glob('chunk_*.ids.json')):
            ids = read(ids_path)['ids']
            states = []
            scores = []
            for start in range(0, len(ids), engine.batch):
                cs = [contexts[key] for key in ids[start:start+engine.batch]]
                with torch.no_grad():
                    logits, ap = engine.forward(cs)
                    scores.append(engine.score(logits[:len(cs)], [data.answer(c) for c in cs]))
                states.append(ap[:len(cs)].cpu())
            current = torch.cat(states)
            original = cache.field([contexts[key] for key in ids], 'ap').float()
            if not torch.allclose(current, original, atol=2e-4, rtol=2e-4):
                raise RuntimeError('Cached AP states changed')
            errors = {'ap': float((current-original).abs().max())}
            for key in scores[0]:
                actual = torch.cat([torch.as_tensor(score[key]) for score in scores])
                expected = cache.field([contexts[k] for k in ids], key)
                if actual.is_floating_point():
                    if not torch.allclose(actual.double(), expected.double(), atol=2e-4, rtol=2e-4):
                        raise RuntimeError('Cached natural score changed: ' + key)
                    errors[key] = float((actual.double()-expected.double()).abs().max())
                elif not torch.equal(actual, expected):
                    raise RuntimeError('Cached strict natural prediction changed: ' + key)
            reports.append(dict(ids_file=str(ids_path), count=len(ids), max_errors=errors))
            print(json.dumps(dict(stage='cache_migration', chunks=len(reports), prompts=sum(x['count'] for x in reports))), flush=True)
        slots = read(root / 'proposals/train_master.slots.json')
        groups = read(root / 'groups.json')
        rows = [data.proposal(slots[i*31], data.alternative_order(slots[i*31])[0], groups) for i in range(32)]
        result['numerical_checks'] = numerical(engine, rows, donors(engine, rows))
        result['backbone_unchanged'] = versions == [p._version for p in engine.model.parameters()]
        if not result['backbone_unchanged']:
            raise RuntimeError('Backbone mutated')
        result.update(passed=True, chunks=reports, prompts=len(cache.index))
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        cache = engine = None
        gc.collect()
        torch.cuda.empty_cache()
        if session is not None:
            session.close()
            closed = read(output / 'engineering/gpu_budget.json')
            if read(ledger_path) != ledger:
                raise RuntimeError('Concurrent budget change')
            write(ledger_path, dict(ledger, active=False, seconds=closed['seconds'],
                diagnostic=str(output), diagnostic_seconds=closed['seconds']-ledger['seconds'],
                time=time.time(), stage='signed_runtime_cache_migration_complete'), immutable=False)
            result['charged_seconds'] = closed['seconds']-ledger['seconds']
        result['protected_after'] = {str(p): file_hash(p) for p in protected}
        result['scientific_and_safety_unchanged'] = before == result['protected_after']
        result['finished'] = time.time()
        if not result['scientific_and_safety_unchanged']:
            result['passed'] = False
        write(output / 'receipt.json', result)
    if not result['passed']:
        raise RuntimeError('Runtime cache migration failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['verify', 'cache', 'run'])
    parser.add_argument('--migration', type=Path)
    args = parser.parse_args()
    handles = setup()
    import torch
    from experiments.mcqa_ap_specialization_abcd import data, protocol, verification
    from experiments.mcqa_at_anchor_comparison.storage import read, write, digest
    root = protocol.ROOT
    if data.verify_identity(root, model=args.mode != 'verify') != IDENTITY:
        raise RuntimeError('Scientific identity changed')
    output = root / 'engineering/signed_runtime_migration' / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'registration.json', dict(mode=args.mode, started=time.time(),
        executable=sys.executable, python=sys.version, executable_sha256=sha(sys.executable),
        bootstrap_sha256=sha(__file__), original_packages='.venv_mcqa_c_at/Lib/site-packages',
        scientific_identity=IDENTITY, runtime_vendor_change='Anaconda build, exact Python 3.11.17; original runtime preserved',
        security_policy_changed=False, scientific_source_changed=False, safety_source_changed=False))
    print(json.dumps(dict(stage='signed_ap_runtime', mode=args.mode, output=str(output))), flush=True)
    try:
        if args.mode == 'verify':
            result = verification.verify(root)
            import pytest
            test_tmp = tempfile.mkdtemp(prefix='ap_cpu_')
            code = pytest.main(['experiments/mcqa_ap_specialization/tests',
                'experiments/mcqa_ap_specialization_revised/tests',
                'experiments/mcqa_ap_specialization_abcd/tests',
                'diagnostics/ap_thermal_20261021/test_telemetry.py',
                '-q', '--import-mode=importlib', '-p', 'no:cacheprovider', '--basetemp=' + test_tmp])
            if code or torch.cuda.is_initialized():
                raise RuntimeError('CPU suite failed or initialized CUDA')
            write(output / 'receipt.json', dict(passed=True, verification=result, pytest_exit=code,
                pytest_temp=test_tmp, GPU_work=False))
        elif args.mode == 'cache':
            check_cache(root, output)
        else:
            if args.migration is None:
                raise RuntimeError('Verified migration receipt required')
            registration = read(args.migration / 'registration.json')
            receipt = read(args.migration / 'receipt.json')
            if (registration['bootstrap_sha256'] != sha(__file__) or not receipt['passed']
                    or receipt['prompts'] != 2460 or not receipt['scientific_and_safety_unchanged']):
                raise RuntimeError('Invalid runtime migration receipt')
            if (root / 'pause.request').exists():
                raise RuntimeError('Authorized pause archival required; launcher never clears guards')
            sys.argv = ['experiments.mcqa_ap_specialization_abcd', 'run', '--root', str(root)]
            runpy.run_module('experiments.mcqa_ap_specialization_abcd', run_name='__main__')
            write(output / 'receipt.json', dict(passed=read(root/'status.json')['stage']=='complete',
                status=read(root/'status.json'), migration=str(args.migration), automatic_resume=False))
    except BaseException as error:
        if not (output / 'receipt.json').exists():
            write(output / 'receipt.json', dict(passed=False, error=repr(error), mode=args.mode))
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
