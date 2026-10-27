"""Synthetic recipient, scoring, reporting and accounting tests; never run a model."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from verify import verify


def main():
    verify()
    source = Path(__file__).resolve().parent/'source'
    sys.path.insert(0, str(source))
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU checks cannot start in an initialized CUDA process')
    torch.set_num_threads(4)
    original_cwd = Path.cwd()
    temporary_root = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix='apc128_source_tests_') as scratch, ExitStack() as stack:
        if Path(scratch).resolve().parent != temporary_root:
            raise ValueError('Temporary fixture escaped the intended temporary directory')
        os.chdir(scratch)
        try:
            stack.enter_context(patch.object(utils, 'is_sklearn_available', return_value=False))
            stack.enter_context(patch.object(imports, 'is_sklearn_available', return_value=False))
            from diagnostics.apc_128_20261026 import full_transfer_tests, finish_transfer_tests
            from diagnostics.apc_128_20261026 import report_seed_repair, supplement_counter_context_resume
            values = [full_transfer_tests.tests(), finish_transfer_tests.tests(),
                      report_seed_repair.tests(), supplement_counter_context_resume.tests()]
            for name, module in list(sys.modules.items()):
                if not name.startswith(('experiments.', 'diagnostics.')):
                    continue
                path = getattr(module, '__file__', None)
                if path and not Path(path).resolve().is_relative_to(source):
                    raise ValueError('Local-source fallback outside this snapshot: '+name)
        finally:
            os.chdir(original_cwd)
    if torch.cuda.is_initialized() or not all(v['passed'] for v in values):
        raise RuntimeError('Synthetic CPU checks failed or initialized CUDA')
    print(json.dumps(dict(passed=True, synthetic_checks=sum(v['test_count'] for v in values),
          suites=values, cuda_initialized=False, no_model_weights_or_run_artifacts_required=True), indent=2))


if __name__ == '__main__':
    main()
