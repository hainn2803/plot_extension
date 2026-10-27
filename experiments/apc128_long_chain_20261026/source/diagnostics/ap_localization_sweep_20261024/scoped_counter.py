"""Collect the same Windows GPU-memory counter for the worker PID only.

Invalid, missing or negative worker samples are fatal. There are no retries,
cached readings, zero fallbacks, timeout changes or memory-threshold changes.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
import unittest

from diagnostics.ap_specialization_routing_20261023 import counter_transport as legacy


def query(pid):
    pid = int(pid)
    if pid <= 0:
        raise ValueError('Invalid PID')
    return ("$ErrorActionPreference='Stop'; "
            f"$s=@((Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage').CounterSamples); "
            "if($s.Count -eq 0){throw 'Missing worker counter instances'}; "
            "@($s | Select-Object InstanceName,CookedValue,Status) | ConvertTo-Json -Compress")


def validate(samples, pid):
    if isinstance(samples, dict):
        samples = [samples]
    if not isinstance(samples, list) or not samples:
        raise ValueError('Missing worker counter instances')
    names = set()
    total = 0
    for sample in samples:
        name = sample['InstanceName']
        value = sample['CookedValue']
        if not isinstance(name, str) or not name.startswith(f'pid_{pid}_') or name in names:
            raise ValueError('Unexpected or duplicate worker counter instance')
        if sample['Status'] != 0:
            raise ValueError('Invalid worker counter status')
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError('Invalid worker counter value type')
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError('Invalid worker counter value')
        names.add(name)
        total += int(value)
    if total > 2**63 - 1:
        raise ValueError('Worker memory counter overflow')
    return total


def collect(pid=None):
    pid = os.getpid() if pid is None else pid
    command = query(pid)
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],
                            env=legacy.child_environment(),capture_output=True,text=True,
                            check=True,timeout=15)
    if result.stderr.strip():
        raise RuntimeError('Scoped memory counter reported an error: ' + result.stderr.strip())
    samples = json.loads(result.stdout)
    if isinstance(samples, dict):
        samples = [samples]
    total = validate(samples, pid)
    return dict(passed=True,pid=pid,samples=samples,shared_bytes=total,command=command,time=time.time())


def global_diagnostic(pid):
    # Diagnostic only: retain every error and status, without using a failed
    # global query as a guard reading. Operational reads use collect() above.
    command = ("$e=@(); $s=@((Get-Counter '\\GPU Process Memory(*)\\Shared Usage' "
               "-ErrorAction SilentlyContinue -ErrorVariable +e).CounterSamples); "
               "[pscustomobject]@{errors=@($e|ForEach-Object {$_.ToString()}); "
               "samples=@($s|Select-Object InstanceName,CookedValue,Status)} | ConvertTo-Json -Depth 5 -Compress")
    result = subprocess.run(['powershell.exe','-NoProfile','-Command',command],
                            env=legacy.child_environment(),capture_output=True,text=True,
                            check=True,timeout=15)
    if result.stderr.strip():
        raise RuntimeError('Global diagnostic transport error: ' + result.stderr.strip())
    value = json.loads(result.stdout)
    value.update(time=time.time(),pid=pid,command=command)
    value['worker'] = [s for s in value['samples'] if s['InstanceName'].startswith(f'pid_{pid}_')]
    value['unrelated_invalid'] = [s for s in value['samples']
        if not s['InstanceName'].startswith(f'pid_{pid}_') and s['Status'] != 0]
    return value


def verify_live_counter():
    checks = []
    for _ in range(3):
        scoped = collect()
        global_read = global_diagnostic(os.getpid())
        total = validate(global_read['worker'], os.getpid())
        if total != scoped['shared_bytes']:
            raise RuntimeError('Scoped/global worker memory counters disagree at idle')
        a = sorted(s['InstanceName'] for s in scoped['samples'])
        b = sorted(s['InstanceName'] for s in global_read['worker'])
        if a != b:
            raise RuntimeError('Scoped/global worker instance sets disagree')
        checks.append(dict(scoped=scoped,global_reference=global_read))
    return dict(passed=True,pid=os.getpid(),comparisons=checks,timeout_seconds=15,
                counter='GPU Process Memory / Shared Usage',all_worker_instances_valid=True)


class ScopedTransport:
    def __init__(self, original, samples):
        self.original,self.samples = original,samples
        self.diagnostic_worker_error = None

    def __getattr__(self, name):
        return getattr(self.original, name)

    def run(self, args, **kwargs):
        if args[0] != 'powershell.exe':
            return self.original.run(args, **kwargs)
        if args != ['powershell.exe','-NoProfile','-Command',legacy.query()]:
            raise ValueError('Unexpected memory-counter caller')
        if kwargs != dict(capture_output=True,text=True,check=True,timeout=15):
            raise ValueError('Counter timeout or error policy changed')
        sample = dict(time=time.time(),original=args,timeout_seconds=15)
        started = time.perf_counter()
        try:
            if self.diagnostic_worker_error is not None:
                raise RuntimeError('Invalid worker instance in independent diagnostic: '+self.diagnostic_worker_error)
            value = collect()
            sample.update(value)
            # MemoryWatch and device_memory retain their original API/policy.
            return subprocess.CompletedProcess(args,0,str(value['shared_bytes'])+'\n','')
        except BaseException as error:
            sample.update(passed=False,error=repr(error))
            raise
        finally:
            sample['seconds'] = time.perf_counter()-started
            self.samples.append(sample)


@contextmanager
def use_scoped_counter(output):
    from experiments.mcqa_c_at_it_gradual_pilot import engineering
    from experiments.mcqa_at_anchor_comparison.storage import write,file_hash
    output = Path(output)
    original = engineering.subprocess
    samples,diagnostics = [],[]
    proxy = ScopedTransport(original,samples)
    stop = threading.Event()

    def observe():
        while not stop.is_set():
            try:
                value = global_diagnostic(os.getpid())
                if value['worker']:
                    try:
                        validate(value['worker'],os.getpid())
                    except BaseException as error:
                        proxy.diagnostic_worker_error = repr(error)
                diagnostics.append(value)
            except BaseException as error:
                diagnostics.append(dict(time=time.time(),diagnostic_error=repr(error)))
            # The observer is limited to real profiling; it is not a replacement
            # for the unchanged operational MemoryWatch.
            if (output.parents[2]/'profile.json').exists():
                break
            stop.wait(2)

    engineering.subprocess = proxy
    thread = threading.Thread(target=observe,daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set();thread.join(timeout=20)
        engineering.subprocess = original
        write(output/'scoped_memory_counter.json',dict(samples=samples,global_diagnostics=diagnostics,
            helper_sha256=file_hash(__file__),same_counter_and_worker_pid=True,
            collection_filter_moved_before_sampling=True,same_timeout_seconds=15,
            original_memory_watch_and_thresholds=True,retries=0,missing_as_zero=False,
            diagnostic_worker_error=proxy.diagnostic_worker_error))


def cpu_tests():
    class Tests(unittest.TestCase):
        def good(self):
            return dict(InstanceName='pid_123_luid_0x0_0x1_phys_0',CookedValue=4096.,Status=0)

        def test_sum(self):
            a=self.good();b=dict(a,InstanceName='pid_123_luid_0x0_0x2_phys_0',CookedValue=8192.)
            self.assertEqual(validate([a,b],123),12288)

        def test_missing(self):
            for value in ([],None,{}):
                with self.assertRaises((ValueError,KeyError)):validate(value,123)

        def test_invalid(self):
            for key,value in [('Status',1),('CookedValue',-1),('CookedValue',float('nan')),
                              ('CookedValue',float('inf')),('CookedValue',.5),('CookedValue',True)]:
                with self.assertRaises(ValueError):validate([dict(self.good(),**{key:value})],123)

        def test_wrong_pid(self):
            with self.assertRaises(ValueError):validate([self.good()],12)

        def test_duplicates(self):
            with self.assertRaises(ValueError):validate([self.good(),self.good()],123)

        def test_request_scope(self):
            self.assertIn('GPU Process Memory(pid_123_*)',query(123))
            self.assertIn("$ErrorActionPreference='Stop'",query(123))
            self.assertNotIn('SilentlyContinue',query(123))

        def test_bad_caller(self):
            proxy=ScopedTransport(subprocess,[])
            with self.assertRaises(ValueError):proxy.run(['powershell.exe','-Command','0'])
            with self.assertRaises(ValueError):proxy.run(['powershell.exe','-NoProfile','-Command',legacy.query()],timeout=30)

        def test_single_object_json(self):
            from unittest.mock import patch
            response=subprocess.CompletedProcess([],0,json.dumps(self.good()),'')
            with patch.object(subprocess,'run',return_value=response) as call:
                value=collect(123)
                self.assertEqual(value['samples'],[self.good()])
                self.assertEqual(value['shared_bytes'],4096)
                self.assertEqual(call.call_count,1)
                self.assertEqual(call.call_args.kwargs['timeout'],15)

        def test_transport_errors_fatal(self):
            from unittest.mock import patch
            response=subprocess.CompletedProcess([],0,json.dumps(self.good()),'counter error')
            with patch.object(subprocess,'run',return_value=response) as call:
                with self.assertRaises(RuntimeError):collect(123)
                self.assertEqual(call.call_count,1)
            with patch.object(subprocess,'run',side_effect=subprocess.TimeoutExpired('powershell.exe',15)) as call:
                with self.assertRaises(subprocess.TimeoutExpired):collect(123)
                self.assertEqual(call.call_count,1)

        def test_original_limits(self):
            from experiments.mcqa_c_at_it_gradual_pilot.engineering import MemoryWatch
            obj=MemoryWatch();obj.samples=[dict(shared_bytes_this_process=100,used_device_bytes_all_processes=500,physical_device_bytes=1000),
                                         dict(shared_bytes_this_process=101,used_device_bytes_all_processes=500,physical_device_bytes=1000)]
            self.assertFalse(obj.result()['no_added_shared_memory'])
            obj.samples[1]['shared_bytes_this_process']=100
            obj.samples[1]['used_device_bytes_all_processes']=1001
            self.assertFalse(obj.result()['all_process_device_usage_within_capacity'])

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful():
        raise RuntimeError('Scoped-counter CPU tests failed')
    return dict(passed=True,tests=result.testsRun,retries=0,missing_as_zero=False)
