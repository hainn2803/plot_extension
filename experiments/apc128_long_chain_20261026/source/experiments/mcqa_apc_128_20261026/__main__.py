import argparse
from contextlib import ExitStack
import sys
from unittest.mock import patch


def main():
    from diagnostics.ap_thermal_20261021.resume_ap import setup
    handles = setup()
    import torch
    import transformers.utils as utils
    import transformers.utils.import_utils as imports
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'run', 'finalize'))
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError('Runtime preparation must precede CUDA')
    with ExitStack() as stack:
        stack.enter_context(patch.object(utils, 'is_sklearn_available', return_value=False))
        stack.enter_context(patch.object(imports, 'is_sklearn_available', return_value=False))
        from transformers import AutoModelForCausalLM
        from transformers.models.gemma2.modeling_gemma2 import Gemma2ForCausalLM
        if any(name == 'sklearn' or name.startswith('sklearn.') for name in sys.modules):
            raise RuntimeError('Unused blocked optional dependency unexpectedly imported')
        from .runner import main as run
        run(args.command)
    assert handles is not None and AutoModelForCausalLM is not None and Gemma2ForCausalLM is not None


if __name__ == '__main__':
    main()
