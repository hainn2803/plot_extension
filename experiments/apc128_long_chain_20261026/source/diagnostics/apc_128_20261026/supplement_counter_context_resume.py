"""Resume the empty supplement after proving a sandbox-only counter failure."""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

from .full_transfer_plan import ROOT, OUTPUT


def charged_previous(original, failed):
    if (original['active'] or failed['active'] or
            original['total_limit_seconds'] != 43200 or failed['total_limit_seconds'] != 50400 or
            original['profile_seconds'] != 853.1099996000121 or
            failed['profile_seconds'] != original['profile_seconds'] or
            failed['seconds'] <= original['seconds'] or failed['seconds'] >= 50100):
        raise ValueError('Closed, fully charged source and failed-attempt ledgers required')
    return dict(failed)


def tests():
    a = dict(active=False, seconds=36446.22345140009,
             profile_seconds=853.1099996000121, total_limit_seconds=43200)
    b = dict(a, seconds=36447.54109190006, total_limit_seconds=50400)
    assert charged_previous(a, b) == b
    for name, value in (('active', True), ('total_limit_seconds', 43200),
                        ('profile_seconds', 0.), ('seconds', a['seconds']), ('seconds', 50100)):
        wrong = dict(b, **{name: value})
        try:
            charged_previous(a, wrong)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid ledger accepted')
    assert a['seconds'] == 36446.22345140009
    return dict(passed=True, test_count=7, prior_failure_charge_retained=True,
                original_source_ledger_unchanged=True, reserve_and_profile_unchanged=True)


def prove():
    import torch
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    from experiments.mcqa_apc_128_20261026 import registration
    from . import finish_transfer
    finish_transfer.verify_amendment()
    source = read(ROOT/'completion.json')
    previous = read(OUTPUT/'latest_process.json')
    registration.require_exited(previous)
    queue = read(OUTPUT/'queue.json')
    controller = dict(pid=queue['pid'], created=queue['created'],
                      cmdline=queue['cmdline'], executable=queue['cmdline'][0])
    if finish_transfer.same_worker(controller):
        raise ValueError('Previous controller is still alive')
    current_terminal = read(OUTPUT/'terminal.json')
    original_failures = [read(p) for p in (OUTPUT/'failures').glob('*/terminal.json')]
    matches = [v for v in original_failures if v.get('pid') == previous['pid'] and
               v.get('command') == 'gpu' and v.get('type') == 'CalledProcessError']
    if len(matches) != 1:
        raise ValueError('Exact original GPU-counter failure receipt required')
    terminal = matches[0]
    if current_terminal.get('pid') != previous['pid']:
        if (current_terminal.get('command') != 'all' or
                current_terminal.get('cuda_initialized') is not False or
                current_terminal.get('type') != 'CalledProcessError' or
                'GPU Process Memory(*)' not in current_terminal['reason']):
            raise ValueError('A later genuine failure cannot be cleared by this repair')
    if (not source['completed'] or not source['worker_exit_verified'] or previous['pid'] != 53608 or
            terminal['pid'] != previous['pid'] or terminal['type'] != 'CalledProcessError' or
            'GPU Process Memory(pid_53608_*)' not in terminal['reason'] or
            list((OUTPUT/'evaluation').glob('*.pt')) or (ROOT/'pause.request').exists()):
        raise ValueError('Only the verified empty counter-startup attempt may be resumed')
    counter_files = list((OUTPUT/'engineering/attempts').glob('*/scoped_memory_counter.json'))
    if len(counter_files) != 1:
        raise ValueError('Exact failed counter attempt required')
    counter = read(counter_files[0])
    if (len(counter['samples']) != 1 or counter['samples'][0]['passed'] or
            counter['retries'] != 0 or counter['missing_as_zero'] or
            not counter['global_diagnostics'] or
            counter['global_diagnostics'][0]['errors'] != ['The specified object was not found on the computer.']):
        raise ValueError('The recorded counter-visibility diagnosis differs')
    command = "$ErrorActionPreference='Stop'; $s=@((Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples); if($s.Count -eq 0){throw 'Missing GPU counter objects'}; @($s | Select-Object InstanceName,CookedValue,Status) | ConvertTo-Json -Compress"
    outside = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                             capture_output=True, text=True, timeout=15, check=True)
    import json
    samples = json.loads(outside.stdout)
    samples = samples if isinstance(samples, list) else [samples]
    if not samples or not any(s['Status'] == 0 for s in samples):
        raise ValueError('Normal Windows context has no valid GPU counter objects')
    original = read(ROOT/'engineering/gpu_budget.json')
    failed = read(OUTPUT/'engineering/gpu_budget.json')
    charged_previous(original, failed)
    authorization = read(OUTPUT/'authorization.json')
    finish_transfer.check_manifest(authorization['original_manifest'])
    estimate = finish_transfer.estimate()
    available = 50100-failed['seconds']
    if estimate['total_seconds']*1.25 > available:
        raise ValueError('Full144 evaluation plus25percent cannot fit the recorded cap')
    if torch.cuda.is_initialized():
        raise ValueError('Counter-context proof must remain CPU-only')
    folder = OUTPUT/'engineering/counter_context_resumes'/str(time.time_ns())
    names = ('latest_process.json', 'terminal.json', 'STOP_REPORT.md', 'status.json',
             'engineering/gpu_budget.json', 'gpu.stdout.log', 'gpu.stderr.log')
    archive = {}
    for name in names:
        path = OUTPUT/name
        if not path.exists():
            continue
        dest = folder/'prior_counter_stop'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        digest = file_hash(path)
        shutil.copy2(path, dest)
        if file_hash(dest) != digest or file_hash(path) != digest:
            raise ValueError('Counter-stop archive differs')
        archive[str(path)] = dict(path=str(dest), sha256=digest)
    proof = dict(passed=True, tests=tests(), no_training=True, cuda_initialized=False,
        previous_process=previous, previous_controller=controller,
        previous_source_ledger=original, previous_failed_ledger=failed,
        retained_failure_seconds=failed['seconds']-original['seconds'],
        original_manifest=authorization['original_manifest'], archived_stop=archive,
        counter_failure_receipt_sha256=file_hash(counter_files[0]),
        normal_Windows_counter_probe=dict(command=command, samples=samples, stderr=outside.stderr),
        native_Windows_PowerShell_module_path=os.environ.get('PSModulePath'),
        prior_CPU_environment_probe_failure=current_terminal,
        environment_change_scope='Child process only: inherit native Windows PowerShell module path instead of PowerShell7 modules. No system or execution-policy change.',
        counter_guard_unchanged=True, no_missing_as_zero=True, retries=0,
        original_MemoryWatch_and_full_reference_validation_unchanged=True,
        full144_remaining_gate=dict(estimate=estimate, available_seconds=available, allowed=True),
        source_sha256=file_hash(Path(__file__)), source_completion_sha256=file_hash(ROOT/'completion.json'),
        budget_authorization='Direct user: add as much as needed, but finish; current finite cap14h.',
        cumulative_total_limit_seconds=50400, time=time.time())
    write(folder/'authorization.json', proof)
    for name in names:
        path = OUTPUT/name
        if path.exists():
            if file_hash(path) != archive[str(path)]['sha256']:
                raise ValueError('Counter-stop evidence changed before archival')
            path.unlink()
    print('COUNTER_CONTEXT_AUTHORIZATION '+str(folder/'authorization.json'), flush=True)
    return folder/'authorization.json'


def gpu(path):
    from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash
    from experiments.mcqa_apc_128_20261026 import registration
    from experiments.mcqa_ap_gradual_chain_20261025 import execution
    from . import finish_transfer
    auth = read(path)
    if not auth['passed'] or auth['source_sha256'] != file_hash(Path(__file__)):
        raise ValueError('Counter-context authorization/source changed')
    registration.require_exited(auth['previous_process'])
    finish_transfer.check_manifest(auth['original_manifest'])
    original_session = execution.Session
    initialized = []

    class FullyChargedSession(original_session):
        def __init__(self, root, stage):
            if Path(root) != OUTPUT or initialized:
                raise ValueError('Exactly one supplemental session is permitted')
            if read(OUTPUT/'engineering/gpu_budget.json') != auth['previous_source_ledger']:
                raise ValueError('Original source ledger copy differs')
            value = charged_previous(auth['previous_source_ledger'], auth['previous_failed_ledger'])
            write(OUTPUT/'engineering/gpu_budget.json', value, immutable=False)
            initialized.append(True)
            super().__init__(root, stage)
            if self.previous != value['seconds']:
                raise ValueError('Previous failure charge was not inherited')

    with patch.object(execution, 'Session', FullyChargedSession):
        finish_transfer.gpu()


def all_work():
    import torch
    import psutil
    from experiments.mcqa_at_anchor_comparison.storage import read, write
    from experiments.mcqa_apc_128_20261026 import registration
    from . import finish_transfer
    authorization = prove()
    proc = psutil.Process()
    write(authorization.parent/'controller.json', dict(pid=proc.pid, created=proc.create_time(),
          cmdline=proc.cmdline(), executable=proc.exe(), time=time.time()))
    with (OUTPUT/'counter_resume.gpu.stdout.log').open('w', encoding='utf-8') as stdout, \
         (OUTPUT/'counter_resume.gpu.stderr.log').open('w', encoding='utf-8') as stderr:
        subprocess.run([sys.executable, '-u', '-B', '-m',
            'diagnostics.apc_128_20261026.supplement_counter_context_resume', 'gpu',
            '--authorization', str(authorization)], stdout=stdout, stderr=stderr, check=True)
    done = read(OUTPUT/'gpu_completion.json')
    registration.require_exited(done['process'])
    finish_transfer.require_closed(read(OUTPUT/'engineering/gpu_budget.json'), 50400)
    finish_transfer.check_manifest(read(authorization)['original_manifest'])
    if torch.cuda.is_initialized():
        raise ValueError('Waiting/report controller must remain CPU-only')
    from .finish_transfer_report import report
    result = report()
    finish_transfer.check_manifest(read(OUTPUT/'report_manifest.json')['files'])
    finish_transfer.check_manifest(read(authorization)['original_manifest'])
    write(OUTPUT/'completion.json', dict(completed=True, worker_exit_verified=True, source_unchanged=True,
          final_report=result, ledger=read(OUTPUT/'engineering/gpu_budget.json'),
          counter_context_resume_authorization=str(authorization), time=time.time()))
    write(OUTPUT/'status.json', dict(stage='complete', report=result['report'], time=time.time()), immutable=False)


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    from experiments.mcqa_at_anchor_comparison.storage import write
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('test', 'all', 'gpu'))
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils, 'is_sklearn_available', return_value=False))
        stack.enter_context(patch.object(imports, 'is_sklearn_available', return_value=False))
        try:
            if args.command == 'test':
                print(tests(), flush=True)
            elif args.command == 'all':
                all_work()
            else:
                gpu(args.authorization)
        except BaseException as error:
            value = dict(type=type(error).__name__, reason=str(error), traceback=traceback.format_exc(),
                         command=args.command, pid=os.getpid(), time=time.time(), automatic_restart=False,
                         cuda_initialized=torch.cuda.is_initialized())
            write(OUTPUT/'failures'/str(time.time_ns())/'terminal.json', value)
            write(OUTPUT/'terminal.json', value, immutable=False)
            raise
    assert handles is not None


if __name__ == '__main__':
    main()
