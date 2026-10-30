"""Execution-only safeguards; inactive outside an explicitly configured worker."""
import ctypes
import os
from pathlib import Path
import threading
import time

from .storage import read, write

_root = None
_stop = threading.Event()
_thread = None
_failure = None


class Paused(RuntimeError):
    pass


def power_status():
    if os.name != "nt":
        return {"ACLineStatus": None, "platform": os.name}
    class Power(ctypes.Structure):
        _fields_ = [("ac", ctypes.c_ubyte), ("flag", ctypes.c_ubyte),
                    ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                    ("seconds", ctypes.c_uint32), ("full_seconds", ctypes.c_uint32)]
    value = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)):
        raise ctypes.WinError()
    return {"ACLineStatus": value.ac, "battery_percent": value.percent,
            "battery_flag": value.flag}


class ThermalPolicy:
    def __init__(self):
        self.active_since = None

    def observe(self, sample):
        if not sample["active"]:
            self.active_since = None
        elif self.active_since is None:
            self.active_since = sample["time"]
        return self.active_since is not None and sample["time"] - self.active_since >= 60


def request_pause(root, reason):
    path = Path(root) / "pause.request"
    if not path.exists():
        write(path, {"reason": reason, "time": time.time()})


def _watch():
    global _failure
    from .profiling import thermals
    policy = ThermalPolicy()
    try:
        while not _stop.is_set():
            sample = thermals()
            sample["power"] = power_status()
            write(_root / "engineering/thermal_state.json", sample)
            with (_root / "engineering/thermal_samples.jsonl").open("a", encoding="utf-8") as stream:
                import json
                stream.write(json.dumps(sample, sort_keys=True) + "\n")
            if sample["power"]["ACLineStatus"] == 0:
                request_pause(_root, "AC power disconnected")
            if policy.observe(sample):
                request_pause(_root, "Thermal throttling persisted across at least 60 seconds of monitoring")
            _stop.wait(15)
    except BaseException as error:
        _failure = str(error)
        request_pause(_root, "Safety monitor failed: " + str(error))


def configure(root):
    global _root, _thread, _failure
    if _root is not None:
        raise RuntimeError("A local runtime is already configured")
    root = Path(root)
    (root / "engineering").mkdir(parents=True, exist_ok=True)
    if os.name == "nt" and power_status()["ACLineStatus"] != 1:
        raise RuntimeError("Verified AC power is required before sustained laptop GPU work")
    _root, _failure = root, None
    _stop.clear()
    if os.name == "nt":
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            _root = None
            raise ctypes.WinError()
    _thread = threading.Thread(target=_watch, name="mcqa-local-safety", daemon=True)
    _thread.start()
    guard()


def guard():
    if _root is None:
        return
    if _failure:
        raise Paused("Safety monitor failed: " + _failure)
    path = _root / "pause.request"
    if path.exists():
        try:
            reason = read(path).get("reason", "User pause requested")
        except (ValueError, OSError):
            reason = "User pause requested"
        raise Paused(reason + "; resume from the last committed checkpoint")


def active(root):
    return _root is not None and _root.resolve() == Path(root).resolve()


def close():
    global _root, _thread
    _stop.set()
    if _thread is not None:
        _thread.join(timeout=20)
    if os.name == "nt" and _root is not None:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    _root, _thread = None, None
