import os
import time
from pathlib import Path

from experiments.mcqa_ap_localization_sweep.execution import Session as OriginalSession
from experiments.mcqa_at_anchor_comparison.storage import read, write, BudgetStop
from experiments.mcqa_c_at_localization import runtime
from .protocol import PROTOCOL


class Session(OriginalSession):
    """The original AC/thermal runtime, with a separate 12-hour allocation."""

    def ledger(self):
        now = time.perf_counter()
        return dict(active=self.active, seconds=self.previous + now - self.started,
                    profile_seconds=self.profile_previous + (now - self.stage_started if self.stage == 'profile' else 0),
                    stage=self.stage, pid=os.getpid(), time=time.time(), total_limit_seconds=43200,
                    accounting='GPU-session wall time including load, profiling, validation, I/O, cooling and failures')

    def check(self):
        while True:
            runtime.guard()
            watch = getattr(self, 'memory_watch', None)
            if watch is not None:
                if watch.error:
                    raise RuntimeError('Original memory audit unavailable: ' + watch.error)
                result = watch.result()
                if not result['no_added_shared_memory'] or not result['all_process_device_usage_within_capacity']:
                    raise RuntimeError('Original ongoing memory thresholds failed')
            v = self.ledger()
            write(self.root / 'engineering/gpu_budget.json', v, immutable=False)
            if v['seconds'] >= 42900:
                raise BudgetStop('12-hour budget: five-minute reserve reached')
            if self.stage == 'profile' and v['profile_seconds'] >= 1770:
                raise BudgetStop('Initial profiling 30-minute ceiling')
            thermal = self.root / 'engineering/thermal_state.json'
            if not thermal.exists() or not read(thermal)['active']:
                break
            time.sleep(1)

    def allow(self, seconds, label):
        self.check()
        if not isinstance(seconds, (float, int)) or seconds < 0 or not seconds < float('inf'):
            raise ValueError('Invalid complete-work estimate')
        available = 42900 - self.ledger()['seconds']
        result = dict(label=label, estimated_remaining_seconds=seconds, allowance=1.25,
                      available_seconds=available, allowed=seconds * 1.25 <= available)
        write(self.root / 'engineering/launch_gates' / f'{time.time_ns()}.json', result)
        if not result['allowed']:
            raise BudgetStop('Complete registered workload cannot fit: ' + label)
        return result
