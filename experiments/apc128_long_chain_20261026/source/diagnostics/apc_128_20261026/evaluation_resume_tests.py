"""No-CUDA checks for exact-stop scope and phase-preserving final evaluation."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch


def tests(output):
    import torch
    from . import resume_evaluation as resume
    from .full_transfer_tests import tests as transfer_tests
    from experiments.mcqa_at_anchor_comparison.storage import save,write,file_hash
    from experiments.mcqa_ap_gradual_chain_20261025 import training,banks

    class Tests(unittest.TestCase):
        def stopped(self):
            return (dict(pid=31248,created=resume.STOP_CREATED,
                         cmdline=['python','-u','-B','-m','diagnostics.apc_128_20261026.resume_last_fit','run']),
                    dict(active=False,pid=31248,seconds=resume.STOP_SECONDS,profile_seconds=853.1099996000121,total_limit_seconds=43200),
                    dict(reason='AC power disconnected',time=resume.STOP_TIME),
                    dict(pid=31248,type='Paused',reason='AC power disconnected; resume from the last committed checkpoint'))

        def test_exact_stop_accepted(self):
            resume.require_stop(*self.stopped())

        def test_other_pause_worker_command_or_budget_rejected(self):
            base=self.stopped()
            changes=[(0,'pid',123),(0,'created',1.),(0,'cmdline',['python','unrelated.py']),
                     (1,'active',True),(1,'seconds',0.),(1,'profile_seconds',0.),(1,'total_limit_seconds',86400),
                     (2,'time',1.),(2,'reason','User pause'),(3,'type','ValueError'),(3,'reason','Numerical failure')]
            for index,key,value in changes:
                args=copy.deepcopy(base)
                args[index][key]=value
                with self.assertRaises(ValueError):
                    resume.require_stop(*args)

        def test_final_access_keeps_original_timestamp_only(self):
            before=dict(time=100.,after_all_endpoints_and_readouts=True)
            writes=[]
            writer=resume.final_access_writer(lambda p,v,immutable=True:writes.append((p,v,immutable)),Path('root'),before)
            writer(Path('root/final_access.json'),dict(time=200.,after_all_endpoints_and_readouts=True))
            self.assertEqual(writes[0][1],before)
            with self.assertRaises(ValueError):
                writer(Path('root/final_access.json'),dict(time=200.,after_all_endpoints_and_readouts=False))
            with self.assertRaises(ValueError):
                writer(Path('root/final_access.json'),dict(time=50.,after_all_endpoints_and_readouts=True))
            writer(Path('root/other.json'),dict(anything=True))
            self.assertEqual(writes[-1][1],dict(anything=True))

        def test_existing_saved_output_never_evaluated_or_rewritten(self):
            path=Path(output)/'CPU_regression_existing.pt'
            save(path,dict(correct=torch.tensor([True,False])))
            write(path.with_suffix('.json'),dict(example='old immutable evaluation'))
            expected=file_hash(path)
            stamp=path.stat().st_mtime_ns
            def forbidden():
                raise AssertionError('A completed output must not be recomputed')
            metric,arrays=training.saved(path,forbidden)
            self.assertEqual(metric,dict(example='old immutable evaluation'))
            self.assertEqual(arrays['correct'].tolist(),[True,False])
            self.assertEqual(file_hash(path),expected)
            self.assertEqual(path.stat().st_mtime_ns,stamp)

        def test_completed_bank_proof_required_and_no_new_bank(self):
            with patch.object(banks,'load_bank',return_value=[dict(id='one')]):
                with patch('experiments.mcqa_at_anchor_comparison.storage.read',return_value=dict(passed=True,count=1)):
                    self.assertEqual(resume.frozen_bank(resume.ROOT,'AP_final',None),[dict(id='one')])
                    with self.assertRaises(ValueError):
                        resume.frozen_bank(resume.ROOT,'AP_train',None)
                with patch('experiments.mcqa_at_anchor_comparison.storage.read',return_value=dict(passed=True,count=0)):
                    with self.assertRaises(ValueError):
                        resume.frozen_bank(resume.ROOT,'AP_final',None)

        def test_evaluation_only_source_has_no_training_entrypoint(self):
            import ast
            tree=ast.parse(Path(resume.__file__).read_text())
            run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
            for node in ast.walk(run):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('fit','train_apc','update','readout','lock','filter_bank','filter_controls'))

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not result.wasSuccessful() or torch.cuda.is_initialized():
        raise RuntimeError('Evaluation-only CPU regression failed or initialized CUDA')
    extra=transfer_tests()
    proof=dict(passed=True,evaluation_resume_tests=result.testsRun,full_transfer_tests=extra,
               final_timestamp_preserved=True,completed_outputs_not_recomputed=True,
               wrong_stop_or_budget_rejected=True,no_training_entrypoint=True,cuda_initialized=False)
    write(Path(output)/'CPU_tests.json',proof)
    return proof
