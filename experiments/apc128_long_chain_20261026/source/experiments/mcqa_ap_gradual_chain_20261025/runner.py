"""Signed, guarded stage dispatch; no selection or automatic safety restarts."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback
import psutil
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash, status
from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from diagnostics.ap_localization_sweep_20261024.scoped_counter import use_scoped_counter, verify_live_counter
from diagnostics.ap_specialization_routing_20261023.library_initialization import initialize
from .protocol import ROOT, DIAGNOSIS, SEEDS, METHODS
from .execution import Session
from .engine import Engine, NaturalCache
from . import data, banks, training, profiling, evaluation, tests, verification, report


def stamp(root):
    proc = psutil.Process()
    value = dict(pid=os.getpid(), created=proc.create_time(), executable=proc.exe(), cmdline=proc.cmdline(),
                 executable_sha256=file_hash(proc.exe()), time=time.time())
    write(Path(root)/'engineering/processes'/f'{time.time_ns()}.json', value)
    write(Path(root)/'latest_process.json', value, immutable=False)
    return value


def gate(root, engine, profile, stage):
    estimate = profiling.estimate(root, profile, stage)
    value = engine.session.allow(estimate['total_seconds'], 'Complete remaining workload before '+stage)
    status(root, 'launch_gate_passed', next_stage=stage, estimated_hours_with_allowance=estimate['total_seconds']*1.25/3600,
           available_hours=value['available_seconds']/3600)


def gpu(root, command):
    data.verify(root, model=True)
    process = stamp(root)
    session = Session(root, 'profile' if not (root/'profile.json').exists() else 'resume_validation')
    attempt = root/'engineering/attempts'/str(time.time_ns())
    engine = None
    completed = False
    try:
        with use_scoped_counter(attempt):
            initialize(attempt, DIAGNOSIS)
            engine = Engine(root, session)
            torch.cuda.synchronize()
            write(attempt/'live_counter.json', verify_live_counter())
            cache = NaturalCache(root, engine)
            profile = profiling.profile(root, engine, cache)
            session.transition('run')
            if command == 'profile':
                return
            gate(root, engine, profile, 'AT')
            with MemoryWatch() as watch:
                session.memory_watch = watch
                try:
                    if command == 'run' or command == 'train_at':
                        session.transition('AT')
                        for bank in ('AP_train', 'AP_calibration', 'AT_train', 'AT_calibration', 'C_train', 'C_calibration', 'readout_AT_train', 'readout_AT_calibration',
                                     'readout_AP_train', 'readout_AP_calibration'):
                            banks.filter_bank(root, bank, cache)
                        for seed in SEEDS:
                            gate(root, engine, profile, f'AT seed {seed}')
                            training.fit(root, engine, cache, 'AT', 'standard', seed)
                            training.readout(root, engine, cache, 'AT', 'standard', seed)
                            training.gate_at(root, (seed,))
                        training.gate_at(root)
                        if command == 'train_at':
                            return
                    if command in ('run', 'train_ap'):
                        training.gate_at(root)
                        session.transition('AP')
                        for seed in SEEDS:
                            gate(root, engine, profile, f'AP seed {seed}')
                            teacher = training.teachers(root, engine, seed)
                            for method in METHODS:
                                training.fit(root, engine, cache, 'AP', method, seed, teacher if method == 'gradual' else None)
                                training.readout(root, engine, cache, 'AP', method, seed)
                        if command == 'train_ap':
                            return
                    if command in ('run', 'train_c'):
                        training.gate_at(root)
                        session.transition('C')
                        for seed in SEEDS:
                            gate(root, engine, profile, f'C seed {seed}')
                            teacher = training.teachers(root, engine, seed, 'gradual')
                            for method in METHODS:
                                training.fit(root, engine, cache, 'C', method, seed, teacher if method == 'gradual' else None)
                        training.lock(root)
                        if command == 'train_c':
                            return
                    if command in ('run', 'evaluate'):
                        banks.locked(root)
                        session.transition('evaluation')
                        gate(root, engine, profile, 'complete locked final evaluation')
                        evaluation.evaluate_all(root, engine, cache)
                        completed = True
                finally:
                    session.memory_watch = None
            memory = watch.result()
            write(attempt/'ongoing_memory.json', memory)
            profiling.require_memory(memory)
    finally:
        session.close()
        del engine
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    if completed:
        ledger = read(root/'engineering/gpu_budget.json')
        if ledger['active'] or ledger['pid'] != process['pid']:
            raise RuntimeError('Worker ledger did not close')
        write(root/'gpu_completion.json', dict(completed=True, ledger=ledger, process=process, time=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'verify_cpu', 'profile', 'train_at', 'train_ap', 'train_c', 'evaluate', 'report', 'finalize', 'run'])
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    try:
        if args.command in ('prepare', 'run'):
            if not (root/'identity.json').exists():
                tests.cpu(root)
                data.prepare(root)
            else:
                data.verify(root)
                tests.cpu(root/'engineering/preflights'/str(time.time_ns()))
        if args.command == 'verify_cpu':
            tests.cpu(root/'engineering/preflights'/str(time.time_ns()))
        if args.command in ('profile', 'train_at', 'train_ap', 'train_c', 'evaluate', 'run'):
            gpu(root, args.command)
        if args.command in ('report', 'finalize', 'run') and (root/'results.json').exists():
            verification.final(root)
            report.complete(root)
            status(root, 'awaiting_worker_exit_verification', report=str(root/'REPORT.md'))
        if args.command == 'finalize':
            verification.exited(root)
            status(root, 'complete', report=str(root/'REPORT.md'), independent_audit_passed=True)
    except BaseException as error:
        attempt = root/'engineering/failures'/str(time.time_ns())
        value = dict(type=type(error).__name__, reason=str(error), traceback=traceback.format_exc(), time=time.time(),
                     pid=os.getpid(), automatic_restart=False)
        write(attempt/'terminal.json', value)
        write(root/'terminal.json', value, immutable=False)
        status(root, 'stopped', reason=str(error), automatic_restart=False)
        report.partial(root)
        raise


if __name__ == '__main__':
    main()
