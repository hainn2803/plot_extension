"""New 24-hour ledger using the unchanged AC/thermal watchdog."""
import os
import time
from pathlib import Path
from experiments.mcqa_c_at_localization import runtime
from experiments.mcqa_at_anchor_comparison.storage import read,write,install,BudgetStop
from .protocol import PROTOCOL

class Session:
    def __init__(self,root,stage='profile'):
        self.root=Path(root);self.stage=stage;install(root)
        path=self.root/'engineering/gpu_budget.json';old=read(path) if path.exists() else {}
        if old.get('active'):raise RuntimeError('Unclosed prior session: verify accounting before resume')
        if (self.root/'pause.request').exists():raise runtime.Paused('Explicit authorized resume required')
        self.previous=old.get('seconds',0.);self.profile_previous=old.get('profile_seconds',0.)
        self.started=self.stage_started=time.perf_counter();self.active=True
        try:runtime.configure(root);self.check()
        except BaseException:self.close();raise

    def ledger(self):
        now=time.perf_counter()
        return dict(active=self.active,seconds=self.previous+now-self.started,
            profile_seconds=self.profile_previous+(now-self.stage_started if self.stage=='profile' else 0),
            stage=self.stage,pid=os.getpid(),time=time.time(),total_limit_seconds=86400,
            accounting='GPU-session wall time including load, profiling, I/O, cooling and failures')

    def check(self):
        while True:
            runtime.guard();v=self.ledger();write(self.root/'engineering/gpu_budget.json',v,immutable=False)
            if v['seconds']>=86100:raise BudgetStop('24-hour budget: five-minute reserve reached')
            if self.stage=='profile' and v['profile_seconds']>=1770:raise BudgetStop('Initial profiling 30-minute ceiling')
            thermal=self.root/'engineering/thermal_state.json'
            if not thermal.exists() or not read(thermal)['active']:break
            time.sleep(1)

    def transition(self,stage):
        if self.stage=='profile':self.profile_previous=self.ledger()['profile_seconds']
        self.stage=stage;self.stage_started=time.perf_counter();self.check()

    def allow(self,seconds,label):
        self.check();available=86100-self.ledger()['seconds']
        result=dict(label=label,estimated_remaining_seconds=seconds,allowance=1.25,
                    available_seconds=available,allowed=seconds*1.25<=available)
        write(self.root/'engineering/launch_gates'/f'{time.time_ns()}.json',result)
        if not result['allowed']:raise BudgetStop('Complete registered workload cannot fit: '+label)
        return result

    def close(self):
        if not self.active:return
        self.active=False;write(self.root/'engineering/gpu_budget.json',self.ledger(),immutable=False);runtime.close()
