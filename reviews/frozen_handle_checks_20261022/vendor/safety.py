"""Bounded wall-time accounting around the unchanged AC/thermal monitor."""
import time
from pathlib import Path
from . import runtime
from .storage import read, write


class Budget:
    def __init__(self, root, seconds):
        self.root, self.limit = Path(root), float(seconds)
        self.started = time.perf_counter()
        self.active = True
        self.tick()

    def elapsed(self):
        return time.perf_counter()-self.started

    def tick(self):
        write(self.root/'engineering/gpu_budget.json',dict(seconds=self.elapsed(),limit=self.limit,active=self.active))
        runtime.guard()
        if self.elapsed() >= self.limit-10:
            raise runtime.Paused('Registered GPU-session budget reached; committed results retained')

    def close(self):
        self.active=False
        write(self.root/'engineering/gpu_budget.json',dict(seconds=self.elapsed(),limit=self.limit,active=False))


def cooling(root,budget):
    while True:
        budget.tick()
        p=Path(root)/'engineering/thermal_state.json'
        if not p.exists() or not read(p)['active']:
            return
        time.sleep(1)
