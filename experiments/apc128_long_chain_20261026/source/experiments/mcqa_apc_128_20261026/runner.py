"""AP then C only; no AT fitting or accuracy gate, all engineering guards intact."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import time
import traceback
from unittest.mock import patch
import psutil
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, file_hash, status
from experiments.mcqa_at_anchor_comparison.engine import Subspace, checked
from experiments.mcqa_ap_gradual_chain_20261025.execution import Session
from experiments.mcqa_ap_gradual_chain_20261025.engine import Engine, NaturalCache, total_loss
from experiments.mcqa_ap_gradual_chain_20261025 import data, banks, training, profiling, evaluation, verification
from experiments.mcqa_ap_gradual_chain_20261025.profiling import require_memory
from experiments.mcqa_ap_gradual_chain_20261025.protocol import DIAGNOSIS
from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
from diagnostics.ap_localization_sweep_20261024.scoped_counter import use_scoped_counter, verify_live_counter
from diagnostics.ap_specialization_routing_20261023.library_initialization import initialize
from .protocol import ROOT, SEEDS, METHODS
from . import registration, auditing, reporting


def stamp(root):
    proc = psutil.Process()
    value = dict(pid=os.getpid(), created=proc.create_time(), executable=proc.exe(), cmdline=proc.cmdline(),
                 executable_sha256=file_hash(proc.exe()), time=time.time())
    write(root/'latest_process.json', value)
    return value


def gate(root, engine, profile, stage):
    estimate = profiling.estimate(root, profile, stage)
    decision = engine.session.allow(estimate['total_seconds'], 'Complete remaining AP/C and final workload before '+stage)
    status(root, 'launch_gate_passed', next_stage=stage,
           estimated_hours_with_allowance=estimate['total_seconds']*1.25/3600,
           available_hours=decision['available_seconds']/3600)
    return decision


def validate_teachers(root, engine, cache, variable, seed):
    rows = banks.load_bank(root, variable+'_train')
    rows = [rows[i] for i in training.order(rows, variable, seed, 0)]
    states = cache.states(rows)
    teacher = training.teachers(root, engine, seed, 'gradual' if variable == 'C' else None)
    sub = Subspace(seed, width=9216 if variable == 'AP' else 2304).to(engine.device)
    terms, y = engine.losses(variable, 'gradual', rows, states, sub, teacher)
    loss = total_loss(terms, 'gradual')
    coefficients = torch.stack(torch.autograd.grad(loss, tuple(terms.values()), retain_graph=True))
    checks = [checked('actual_teacher_loss_coefficients', coefficients,
              torch.tensor([1. if k == 'direct' else 2. for k in terms], device=engine.device))]
    gradient = torch.autograd.grad(loss, sub.raw)[0]
    ref, yr = engine.losses(variable, 'gradual', rows, states, sub, teacher, reference=True)
    ref_loss = total_loss(ref, 'gradual')
    ref_gradient = torch.autograd.grad(ref_loss, sub.raw)[0]
    checks.extend([checked('actual_teacher_logits', y, yr), checked('actual_teacher_total_CE', loss, ref_loss),
                   checked('actual_teacher_combined_gradient', gradient, ref_gradient)])
    checks.extend(checked('actual_teacher_'+key, terms[key], ref[key]) for key in terms)
    if any(v.requires_grad for key, v in teacher.items() if isinstance(v, torch.Tensor)):
        raise ValueError('Frozen teacher basis unexpectedly trainable')
    for key in ('AT_head', 'AP_head'):
        if key in teacher and any(v.requires_grad for v in teacher[key].values() if isinstance(v, torch.Tensor)):
            raise ValueError('Frozen teacher head unexpectedly trainable')
    write(root/'engineering/actual_teacher_validation'/f'{variable}_S{seed}.json', dict(passed=True,
          variable=variable, seed=seed, loss_terms=list(terms), checks=checks, AT_accuracy_gate_enabled=False))
    return teacher


def train_apc(root, engine, cache, profile):
    for variable in ('AP', 'C'):
        engine.session.transition(variable)
        for seed in SEEDS:
            gate(root, engine, profile, variable+' seed '+str(seed))
            teacher = validate_teachers(root, engine, cache, variable, seed)
            for method in METHODS:
                training.fit(root, engine, cache, variable, method, seed, teacher if method == 'gradual' else None)
                if variable == 'AP':
                    training.readout(root, engine, cache, 'AP', method, seed)
    training.lock(root)


def gpu(root):
    registration.verify(root, model=True)
    if (root/'terminal.json').exists() or (root/'gpu_completion.json').exists():
        raise RuntimeError('No blind restart of a terminal worker')
    process = stamp(root)
    session = Session(root, 'resume_validation')
    engine = None
    output = root/'engineering/attempts'/str(time.time_ns())
    finished = False
    try:
        with use_scoped_counter(output):
            initialize(output, DIAGNOSIS)
            engine = Engine(root, session)
            engine.batch = 32
            write(output/'live_counter.json', verify_live_counter())
            cache = NaturalCache(root, engine)
            profile = profiling.profile(root, engine, cache)
            with MemoryWatch() as watch:
                session.memory_watch = watch
                try:
                    for variable in ('AT', 'AP'):
                        for split in ('train', 'calibration'):
                            bank = f'readout_{variable}_{split}'
                            cache.layout(banks.load_bank(root, bank), 'fresh_'+bank)
                finally:
                    session.memory_watch = None
            memory = watch.result()
            require_memory(memory)
            write(output/'fresh_validation.json', dict(passed=True, memory=memory, counter=verify_live_counter(),
                  AT_qualification_disabled=True, original_profile_sha256=file_hash(root/'profile.json')))
            gate(root, engine, profile, 'all12newfits and full final/control evaluation')
            with MemoryWatch() as watch:
                session.memory_watch = watch
                try:
                    train_apc(root, engine, cache, profile)
                    gate(root, engine, profile, 'locked final evaluation')
                    session.transition('evaluation')
                    evaluation.evaluate_all(root, engine, cache)
                    finished = True
                finally:
                    session.memory_watch = None
            memory = watch.result()
            require_memory(memory)
            write(output/'ongoing_memory.json', memory)
    finally:
        del engine
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        session.close()
    if finished:
        ledger = read(root/'engineering/gpu_budget.json')
        if ledger['active'] or ledger['pid'] != process['pid']:
            raise ValueError('Actual GPU ledger did not close')
        write(root/'gpu_completion.json', dict(completed=True, ledger=ledger, process=process, time=time.time()))


def audit(root):
    with patch.object(verification, 'compare', auditing.compare), \
         patch.object(verification, 'write', auditing.stable_writer(write, read, root)):
        return verification.final(root)


def finalize(root):
    registration.verify(root)
    process = read(root/'gpu_completion.json')['process']
    registration.require_exited(process)
    ledger = read(root/'engineering/gpu_budget.json')
    if ledger['active'] or ledger['pid'] != process['pid']:
        raise ValueError('Actual worker exit and closed ledger required')
    for path, expected in read(root/'report_manifest.json')['files'].items():
        if file_hash(path) != expected:
            raise ValueError('Report artifact changed: '+path)
    audit(root)
    verification.exited(root)
    if torch.cuda.is_initialized():
        raise ValueError('Finalization must remain CPU-only')
    status(root, 'complete', report=str(root/'REPORT.md'), AT_qualification_disabled=True, worker_exit_verified=True)


def main(command):
    root = ROOT
    try:
        if command == 'finalize':
            return finalize(root)
        registration.prepare(root)
        if command == 'prepare':
            p = read(root/'profile.json')
            estimate = profiling.estimate(root, p, 'CPU complete-work preflight')
            available = 42900-read(root/'engineering/gpu_budget.json')['seconds']
            if estimate['total_seconds']*1.25 > available:
                raise RuntimeError('Complete registered workload plus25% cannot fit inherited budget')
            print('REGISTERED '+str(root/'identity.json'), flush=True)
            print('ESTIMATED_HOURS_WITH_ALLOWANCE '+str(estimate['total_seconds']*1.25/3600), flush=True)
            return
        gpu(root)
        audit(root)
        reporting.complete(root)
        status(root, 'awaiting_worker_exit_verification', report=str(root/'REPORT.md'), AT_qualification_disabled=True)
    except BaseException as error:
        value = dict(type=type(error).__name__, reason=str(error), traceback=traceback.format_exc(),
                     time=time.time(), pid=os.getpid(), automatic_restart=False)
        write(root/'engineering/failures'/str(time.time_ns())/'terminal.json', value)
        write(root/'terminal.json', value, immutable=False)
        status(root, 'stopped', reason=str(error), automatic_restart=False)
        (root/'STOP_REPORT.md').write_text('# AP/C Study Stopped\n\n'+str(error)+'\n\nAT accuracy qualification alone was removed by explicit amendment. All numerical, safety, memory, budget and scientific coverage guards remain in force. No automatic restart.\n', encoding='utf-8')
        raise
