"""Repair report-only fit/bootstrap seed names without changing saved science."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from .full_transfer_plan import ROOT, OUTPUT


ERROR = "dict() got multiple values for keyword argument 'seed'"
AUTH = ROOT/'engineering/authorized_evaluation_resumes/1793067534071768100/authorization.json'


def comparison_view(value):
    result = deepcopy(value)
    for fit_seed, comparisons in result['comparisons'].items():
        if int(fit_seed) not in (1, 0, 2):
            raise ValueError('Unexpected fit seed')
        for cell in comparisons.values():
            if cell.get('seed') != 20261025 or 'bootstrap_seed' in cell:
                raise ValueError('Unexpected bootstrap seed representation')
            cell['bootstrap_seed'] = cell.pop('seed')
    return result


def tests():
    value = {'comparisons': {str(s): {'primary': dict(seed=20261025, difference=.125,
             lower=-.1, upper=.2, count=1536, resamples=2000)} for s in (1, 0, 2)},
             'science': {'targets': ['A', 'Z'], 'counts': [5, 9]}}
    before = deepcopy(value)
    view = comparison_view(value)
    assert value == before
    rows = [dict(seed=int(s), comparison=k, **cell)
            for s, cells in view['comparisons'].items() for k, cell in cells.items()]
    assert [r['seed'] for r in rows] == [1, 0, 2]
    assert all(r['bootstrap_seed'] == 20261025 for r in rows)
    restored = deepcopy(view)
    for cells in restored['comparisons'].values():
        for cell in cells.values():
            cell['seed'] = cell.pop('bootstrap_seed')
    assert restored == value
    for kind in ('wrong_bootstrap', 'already_renamed', 'wrong_fit'):
        wrong = deepcopy(value)
        if kind == 'wrong_bootstrap':
            wrong['comparisons']['1']['primary']['seed'] = 1
        elif kind == 'already_renamed':
            wrong['comparisons']['1']['primary']['bootstrap_seed'] = 20261025
        else:
            wrong['comparisons']['3'] = wrong['comparisons'].pop('1')
        try:
            comparison_view(wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid representation accepted: '+kind)
    return dict(passed=True, test_count=6, exact_scientific_fields_retained=True,
                fit_seed_preserved=True, bootstrap_seed_preserved=True,
                wrong_representations_rejected=True)


def repair():
    import csv
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    from experiments.mcqa_apc_128_20261026 import registration, reporting, runner
    from .finish_transfer import check_manifest, verify_amendment, same_worker

    verify_amendment()
    done = read(ROOT/'gpu_completion.json')
    registration.require_exited(done['process'])
    queue = read(OUTPUT/'queue.json')
    if same_worker(dict(pid=queue['pid'], created=queue['created'],
                        cmdline=queue['cmdline'], executable=queue['cmdline'][0])):
        raise ValueError('Existing controller must have exited')
    ledger = read(ROOT/'engineering/gpu_budget.json')
    terminal = read(ROOT/'terminal.json')
    supplemental_terminal = read(OUTPUT/'terminal.json')
    audit = read(ROOT/'independent_verification.json')
    if (not done['completed'] or ledger['active'] or ledger != done['ledger'] or
            terminal['pid'] != done['process']['pid'] or terminal['reason'] != ERROR or
            'comparisons.append(dict(seed=seed, comparison=key, **v))' not in terminal['traceback'] or
            supplemental_terminal['reason'] != 'Original worker has a terminal stop or pause' or
            not audit['passed'] or audit['evaluation_artifacts'] != 243 or
            audit['prediction_rows'] != 434304 or (ROOT/'pause.request').exists()):
        raise ValueError('Exact completed-evaluation/report-only failure required')
    registration.verify(ROOT)
    auth = read(AUTH)
    preserved = dict(auth['immutable_old_scientific_artifacts'])
    for folder in ('evaluation', 'evaluation_rows'):
        for path in (ROOT/folder).glob('*'):
            if path.is_file():
                preserved[str(path)] = file_hash(path)
    for name in ('results.json', 'independent_verification.json', 'gpu_completion.json',
                 'engineering/gpu_budget.json', 'locks/all_final.json', 'final_access.json'):
        preserved[str(ROOT/name)] = file_hash(ROOT/name)
    check_manifest(preserved)
    original_result = read(ROOT/'results.json')
    expected = comparison_view(original_result)
    original_read = reporting.original.read

    def report_read(path):
        value = original_read(path)
        return comparison_view(value) if Path(path).resolve() == (ROOT/'results.json').resolve() else value

    with patch.object(reporting.original, 'read', report_read):
        reporting.complete(ROOT)
    check_manifest(preserved)
    with (ROOT/'PAIRED_EFFECTS.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != sum(len(v) for v in expected['comparisons'].values()):
        raise ValueError('Comparison coverage changed')
    for row in rows:
        fit_seed = row['seed']
        cell = expected['comparisons'][fit_seed][row['comparison']]
        if int(row['bootstrap_seed']) != cell['bootstrap_seed']:
            raise ValueError('Bootstrap seed changed in the CSV')
        for field, value in cell.items():
            if field == 'bootstrap_seed':
                continue
            if value is None:
                if row[field] != '':
                    raise ValueError('Undefined comparison changed')
            elif float(row[field]) != value:
                raise ValueError('Comparison value changed: '+field)
    text = (ROOT/'REPORT.md').read_text(encoding='utf-8')
    if 'no AT refit/replacement occurred' not in text or 'no budget reset occurred' not in text:
        raise ValueError('Imported ungated AT/shared-budget limitation missing')
    if torch.cuda.is_initialized():
        raise ValueError('CPU report repair initialized CUDA')

    folder = ROOT/'engineering/report_repairs'/str(time.time_ns())
    paths = [ROOT/n for n in ('terminal.json', 'status.json', 'STOP_REPORT.md')]
    paths += [OUTPUT/n for n in ('terminal.json', 'status.json', 'STOP_REPORT.md', 'queue.json',
                                'controller.stdout.log', 'controller.stderr.log', 'controller_launch.json')]
    archive = {}
    for path in paths:
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        target = folder/'prior_report_failure'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        old_hash = file_hash(path)
        shutil.copy2(path, target)
        if file_hash(target) != old_hash or file_hash(path) != old_hash:
            raise ValueError('Failure archive verification failed')
        archive[str(path)] = dict(copy=str(target), sha256=old_hash)
    write(folder/'failure_archive.json', archive)
    for path in paths:
        if path.exists():
            if file_hash(path) != archive[str(path)]['sha256']:
                raise ValueError('Failure evidence changed before archive')
            path.unlink()
    runner.status(ROOT, 'awaiting_worker_exit_verification', report=str(ROOT/'REPORT.md'),
                  AT_qualification_disabled=True, report_seed_repair=True)
    receipt = dict(passed=True, tests=tests(), source_process=done['process'],
        actual_source_and_old_controller_exited=True, source_ledger=ledger,
        all243_outputs_and_saved_science_unchanged=preserved,
        source_results_sha256=file_hash(ROOT/'results.json'),
        original_reporting_source_sha256=file_hash(Path(reporting.original.__file__)),
        repair_source_sha256=file_hash(Path(__file__)),
        reporting_only_view='bootstrap seed renamed seed -> bootstrap_seed; fit seed remains outer seed',
        original_results_unchanged=True, no_training=True, cuda_initialized=False,
        original_authorization=str(AUTH), original_authorization_sha256=file_hash(AUTH),
        archived_failure_evidence=archive, time=time.time())
    write(folder/'receipt.json', receipt)
    print('REPORT_REPAIR_RECEIPT '+str(folder/'receipt.json'), flush=True)
    return folder/'receipt.json'


def watch(receipt_path):
    from experiments.mcqa_at_anchor_comparison.storage import read, file_hash
    from . import finish_transfer
    receipt = read(receipt_path)
    if not receipt['passed'] or receipt['repair_source_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Report repair source or receipt changed')
    finish_transfer.check_manifest(receipt['all243_outputs_and_saved_science_unchanged'])
    if file_hash(AUTH) != receipt['original_authorization_sha256']:
        raise ValueError('Original finish authorization changed')
    original_verify = finish_transfer.verify_plan
    auth = read(AUTH)

    def amended_worker_plan(*args, **kwargs):
        plan = original_verify(*args, **kwargs)
        if plan['source_worker'] != auth['historical_plan_worker']:
            raise ValueError('Historical prepared plan identity changed')
        if read(ROOT/'latest_process.json') != receipt['source_process']:
            raise ValueError('Completed current source identity changed')
        return dict(plan, source_worker=receipt['source_process'])

    with patch.object(finish_transfer, 'verify_plan', amended_worker_plan):
        finish_transfer.wait_then_run()


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('test', 'repair', 'watch'))
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils, 'is_sklearn_available', return_value=False))
        stack.enter_context(patch.object(imports, 'is_sklearn_available', return_value=False))
        if args.command == 'test':
            print(tests(), flush=True)
        elif args.command == 'repair':
            repair()
        else:
            watch(args.receipt)
    assert handles is not None


if __name__ == '__main__':
    main()
