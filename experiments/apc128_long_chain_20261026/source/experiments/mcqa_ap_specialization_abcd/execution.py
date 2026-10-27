"""Cumulative six-hour accounting with the existing unmodified safety policy."""
import os
from pathlib import Path
import time

from experiments.mcqa_c_at_localization import runtime
from experiments.mcqa_at_anchor_comparison.storage import read,write,install,BudgetStop

class Session:
    def __init__(self,root,stage='profile'):
        self.root=Path(root);self.stage=stage;install(root)
        path=self.root/'engineering/gpu_budget.json'
        old=read(path) if path.exists() else {}
        if old.get('active'):raise RuntimeError('Unclosed prior GPU session requires verified accounting')
        if (self.root/'pause.request').exists():raise runtime.Paused('Existing safety stop requires authorized resume')
        self.previous=old.get('seconds',0.);self.profile_previous=old.get('profile_seconds',0.)
        self.started=time.perf_counter();self.stage_started=self.started;self.active=True
        try:runtime.configure(root);self.check()
        except BaseException:self.close();raise

    def ledger(self):
        now=time.perf_counter()
        return dict(active=self.active,seconds=self.previous+now-self.started,
            profile_seconds=self.profile_previous+(now-self.stage_started if self.stage=='profile' else 0),
            stage=self.stage,pid=os.getpid(),time=time.time(),total_limit_seconds=21600,
            initial_profile_limit_seconds=1800,
            accounting='GPU-session wall time including loading, I/O, cooling and interrupted attempts')

    def check(self):
        runtime.guard();v=self.ledger()
        write(self.root/'engineering/gpu_budget.json',v,immutable=False)
        if v['seconds']>=21300:raise BudgetStop('Six-hour budget: five-minute checkpoint reserve reached')
        if self.stage=='profile' and v['profile_seconds']>=1770:raise BudgetStop('Initial 30-minute profiling budget reached')
        p=self.root/'engineering/thermal_state.json'
        while p.exists() and read(p)['active']:
            time.sleep(1);runtime.guard();v=self.ledger()
            write(self.root/'engineering/gpu_budget.json',v,immutable=False)
            if v['seconds']>=21300 or (self.stage=='profile' and v['profile_seconds']>=1770):
                raise BudgetStop('Budget exhausted during guarded thermal wait')

    def transition(self,stage):
        if self.stage=='profile':self.profile_previous=self.ledger()['profile_seconds']
        self.stage=stage;self.stage_started=time.perf_counter();self.check()

    def allow(self,seconds,label):
        self.check();available=21300-self.ledger()['seconds']
        row=dict(label=label,estimated_seconds=seconds,allowance=1.25,available_seconds=available,
            allowed=1.25*seconds<=available)
        write(self.root/'engineering/launch_gates'/f'{time.time_ns()}.json',row)
        if not row['allowed']:raise BudgetStop('Measured complete remaining workload does not fit: '+label)
        return row

    def close(self):
        self.active=False
        write(self.root/'engineering/gpu_budget.json',self.ledger(),immutable=False)
        runtime.close()
