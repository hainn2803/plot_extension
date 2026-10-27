"""Strict integer-count audit with explicit FP32-rate representation handling."""
from pathlib import Path
import math
import numpy as np


def compare(a, b, label):
    if a.keys() != b.keys():
        raise ValueError('Independent audit key mismatch: '+label)
    rate = set(a) == {'correct', 'count', 'accuracy'}
    if rate:
        if a['correct'] != b['correct'] or a['count'] != b['count']:
            raise ValueError('Independent integer count mismatch: '+label)
        expected = a['correct']/a['count'] if a['count'] else None
        representations = (None,) if expected is None else (expected, float(np.float32(expected)))
        if a['accuracy'] not in representations or b['accuracy'] not in representations:
            raise ValueError('Rate is neither exact ratio nor original FP32 mean: '+label)
        return
    for key in a:
        if isinstance(a[key], (int, float)) and not isinstance(a[key], bool) and b[key] is not None:
            if not math.isclose(a[key], b[key], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Independent numerical metric mismatch: '+label+'/'+key)
        elif a[key] != b[key]:
            raise ValueError('Independent metric mismatch: '+label+'/'+key)


def stable_writer(original, read, root):
    target = (Path(root)/'independent_verification.json').resolve()
    def write(path, value, immutable=True):
        if immutable and Path(path).resolve() == target and Path(path).exists():
            old = read(path)
            candidate = dict(value, time=old['time'])
            if candidate != old:
                raise ValueError('Repeated independent audit genuinely differs')
            return original(path, old, immutable=True)
        return original(path, value, immutable=immutable)
    return write
