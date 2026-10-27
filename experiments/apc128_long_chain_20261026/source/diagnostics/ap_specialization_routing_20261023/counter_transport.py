"""Isolate legacy PowerShell module discovery for the original memory query."""
from contextlib import contextmanager
from pathlib import Path
import os
import json
import subprocess
import time


SHELL = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe'


def child_environment():
    # Windows environment keys are case-insensitive; dict keys need not be.
    return {k: v for k, v in os.environ.items() if k.upper() != 'PSMODULEPATH'}


def query():
    return ("$x=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples; "
            f"$s=($x | Where-Object {{$_.InstanceName -like 'pid_{os.getpid()}_*'}} | Measure-Object -Property CookedValue -Sum).Sum; "
            "[long]$s")


class CounterTransport:
    def __init__(self, original, samples):
        self.original, self.samples = original, samples

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, args, **kwargs):
        if args[0] != 'powershell.exe':
            return self.original.run(args, **kwargs)
        if args != ['powershell.exe', '-NoProfile', '-Command', query()]:
            raise ValueError('Unexpected command in scoped memory-counter transport')
        if kwargs != dict(capture_output=True, text=True, check=True, timeout=15):
            raise ValueError('Memory-counter timeout or error handling changed')
        selected = list(args)
        sample = dict(time=time.time(), original=args, executed=selected, timeout_seconds=15,
                      child_module_path='PowerShell default discovery')
        started = time.perf_counter()
        try:
            result = self.original.run(selected, env=child_environment(), **kwargs)
            if result.stderr.strip():
                raise RuntimeError('Memory counter reported an error: ' + result.stderr.strip())
            shared = int(result.stdout.strip())
            if shared < 0:
                raise ValueError('Invalid negative shared-memory counter')
            sample.update(passed=True, shared_bytes=shared, stderr=result.stderr)
            return result
        except BaseException as error:
            sample.update(passed=False, error=repr(error))
            raise
        finally:
            sample['seconds'] = time.perf_counter() - started
            self.samples.append(sample)


@contextmanager
def use_native_counter_environment(output):
    from experiments.mcqa_c_at_it_gradual_pilot import engineering
    from experiments.mcqa_at_anchor_comparison.storage import file_hash, write
    original = engineering.subprocess
    samples = []
    engineering.subprocess = CounterTransport(original, samples)
    try:
        yield
    finally:
        engineering.subprocess = original
        write(Path(output) / 'memory_counter_transport.json', dict(
            shell='powershell.exe', samples=samples, helper_sha256=file_hash(__file__),
            only_removed_child_environment_key='PSMODULEPATH',
            same_counter_and_pid_filter=True, same_timeout_seconds=15,
            original_memory_watch_and_thresholds=True))


def verify_cpu():
    """Check both shells without a CUDA context or scientific execution."""
    from experiments.mcqa_at_anchor_comparison.storage import file_hash
    import torch
    if torch.cuda.is_initialized():
        raise ValueError('CPU transport verification must precede CUDA')
    samples = []
    for shell in ('powershell.exe', str(SHELL)):
        started = time.perf_counter()
        result = subprocess.run([shell, '-NoProfile', '-Command', query()], env=child_environment(),
                                capture_output=True, text=True, check=True, timeout=15)
        if result.stderr.strip():
            raise RuntimeError('Counter verification reported an error: ' + result.stderr.strip())
        samples.append(dict(shell=shell, shared_bytes=int(result.stdout.strip()),
                            seconds=time.perf_counter() - started, stderr=result.stderr))
    if samples[0]['shared_bytes'] != samples[1]['shared_bytes'] or samples[0]['shared_bytes'] != 0:
        raise ValueError('CPU process counters disagree')
    return dict(passed=True, samples=samples, shell_sha256=file_hash(SHELL),
                cuda_initialized=False, helper_sha256=file_hash(__file__))


def verify_live_counter():
    """Require actual valid Windows counter instances for the loaded worker."""
    command = ("$ErrorActionPreference='Stop'; "
               "$s=(Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples; "
               f"@($s | Where-Object {{$_.InstanceName -like 'pid_{os.getpid()}_*'}} | "
               "Select-Object InstanceName,CookedValue,Status) | ConvertTo-Json -Compress")
    result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                            env=child_environment(), capture_output=True, text=True, check=True, timeout=15)
    if result.stderr.strip():
        raise RuntimeError('Live counter validation failed: ' + result.stderr.strip())
    samples = json.loads(result.stdout)
    if isinstance(samples, dict):
        samples = [samples]
    if not samples or any(s['Status'] != 0 or s['CookedValue'] < 0 for s in samples):
        raise RuntimeError('Missing or invalid live worker counter instances')
    return dict(passed=True, pid=os.getpid(), samples=samples, command=command)
