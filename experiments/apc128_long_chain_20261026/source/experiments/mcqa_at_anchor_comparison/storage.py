"""Root-scoped atomic metadata, immutable receipts and cumulative GPU time."""
import json
import os
from pathlib import Path
import threading
import time

from experiments.hai_c_at_evaluation.schema import digest, file_hash
from experiments.mcqa_c_at_localization import runtime
from diagnostics.hai_full_chain_20261020 import resume_shared_io as shared

_replace = os.replace
_roots = set()


def read(path):
    return shared.read_shared(path)


def replace(src, dst):
    if not any(Path(dst).resolve().is_relative_to(p) for p in _roots):
        return _replace(src, dst)
    for attempt in range(7):
        try:
            with shared.metadata_lock(dst):
                return _replace(src, dst)
        except PermissionError:
            if attempt == 6:
                raise
            runtime.guard()
            time.sleep(min(.02 * 2**attempt, .4))


def install(root):
    _roots.add(Path(root).resolve())
    os.replace = replace
    # The policy, samples and stop timer remain those of the original watchdog.
    runtime.read = read


def write(path, value, immutable=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with shared.metadata_lock(path):
        if path.exists() and immutable:
            if read(path) != value:
                raise ValueError('Immutable artifact differs: ' + str(path))
            return
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        replace(tmp, path)


def save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.receipt.json').exists():
        raise ValueError('Checkpoint already exists: ' + str(path))
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    replace(tmp, path)
    receipt = dict(file=path.name, sha256=file_hash(path))
    write(path.with_suffix('.receipt.json'), receipt)
    return receipt


def load(path):
    import torch
    path = Path(path)
    receipt = read(path.with_suffix('.receipt.json'))
    if receipt['file'] != path.name or file_hash(path) != receipt['sha256']:
        raise ValueError('Checkpoint hash failed: ' + str(path))
    return torch.load(path, map_location='cpu', weights_only=True)


class BudgetStop(RuntimeError):
    pass


class Session:
    def __init__(self, root, stage='AT'):
        self.root, self.stage = Path(root), stage
        install(root)
        path = self.root / 'engineering/gpu_budget.json'
        previous = read(path) if path.exists() else {}
        if previous.get('active'):
            raise RuntimeError('Unclosed session needs verified accounting before resume')
        if (self.root / 'pause.request').exists():
            raise runtime.Paused('Existing pause.request requires authorized resume')
        self.previous = previous.get('seconds', 0.)
        self.at_previous = previous.get('at_seconds', 0.)
        self.started = time.perf_counter()
        self.active = True
        try:
            runtime.configure(root)
            self.tick()
        except BaseException:
            self.close()
            raise

    def elapsed(self):
        return time.perf_counter() - self.started

    def ledger(self):
        dt = self.elapsed()
        return dict(active=self.active, seconds=self.previous + dt,
            at_seconds=self.at_previous + (dt if self.stage == 'AT' else 0),
            total_limit_seconds=21600, at_limit_seconds=5400, pid=os.getpid(),
            stage=self.stage, time=time.time(), accounting='GPU-session wall time including loading, I/O, cooling and failures')

    def tick(self):
        runtime.guard()
        value = self.ledger()
        write(self.root / 'engineering/gpu_budget.json', value, immutable=False)
        if value['seconds'] >= 21590 or (self.stage == 'AT' and value['at_seconds'] >= 5390):
            raise BudgetStop('Registered cumulative GPU time ceiling reached')

    def check(self):
        self.tick()
        start = None
        path = self.root / 'engineering/thermal_state.json'
        while path.exists() and read(path)['active']:
            if start is None:
                start = time.time()
            time.sleep(1)
            self.tick()
        if start is not None:
            write(self.root / 'engineering/cooling' / f'{time.time_ns()}.json',
                  dict(start=start, end=time.time(), guard_reset=False))

    def allow(self, seconds, label):
        self.tick()
        now = self.ledger()
        remaining = min(21600-now['seconds'], 5400-now['at_seconds']) if self.stage == 'AT' else 21600-now['seconds']
        result = dict(label=label, estimated_remaining_seconds=seconds,
                      allowance=1.25, available_seconds=remaining, allowed=1.25*seconds <= remaining)
        write(self.root / 'engineering/launch_gates' / f'{time.time_ns()}.json', result)
        if not result['allowed']:
            raise BudgetStop('Measured workload exceeds remaining budget: ' + label)
        return result

    def close(self):
        self.active = False
        write(self.root / 'engineering/gpu_budget.json', self.ledger(), immutable=False)
        runtime.close()


def status(root, stage, **fields):
    value = dict(stage=stage, time=time.time(), pid=os.getpid(), **fields)
    write(Path(root) / 'status.json', value, immutable=False)
    print(json.dumps(value, sort_keys=True), flush=True)
