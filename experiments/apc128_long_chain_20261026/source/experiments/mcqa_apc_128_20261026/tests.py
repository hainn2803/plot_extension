"""CPU proof that only AT accuracy gating changes, not training or safeguards."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import uuid
import numpy as np
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, load
from experiments.mcqa_ap_gradual_chain_20261025.protocol import PROTOCOL as ORIGINAL
from experiments.mcqa_ap_gradual_chain_20261025.tests import cpu as original_cpu
from .protocol import PROTOCOL, SEEDS, METHODS
from . import runner, auditing


def cpu(root):
    torch.set_num_threads(4)
    base = original_cpu(root)
    checks = []
    def check(name, fn):
        fn()
        checks.append(dict(name=name, passed=True))
    def protocol():
        assert PROTOCOL['AT_gate']['enabled'] is False
        assert ORIGINAL['AT_gate']['preservation_aggregate'] == .95
        for field in ('sites', 'rank', 'updates', 'sizes', 'learning_rate', 'optimizer', 'batch', 'changing', 'preservation',
                      'standard_loss', 'gradual_AP_loss', 'gradual_C_loss', 'C_auxiliary_states', 'readout', 'AT_target', 'AP_target',
                      'atol', 'rtol', 'bootstrap', 'superiority', 'controls', 'alternatives', 'total_gpu_seconds', 'reserve_seconds', 'allowance'):
            assert PROTOCOL[field] == ORIGINAL[field], field
    check('only_authorized_AT_qualification_amendment_scientific_settings_unchanged', protocol)
    def dispatch():
        engine = SimpleNamespace(session=SimpleNamespace(transition=Mock()))
        teachers = {(variable, seed): dict(test_teacher=(variable, seed)) for variable in ('AP', 'C') for seed in SEEDS}
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'gate', return_value={}))
            stack.enter_context(patch.object(runner, 'validate_teachers', side_effect=lambda r,e,c,v,s: teachers[v,s]))
            fit = stack.enter_context(patch.object(runner.training, 'fit'))
            head = stack.enter_context(patch.object(runner.training, 'readout'))
            lock = stack.enter_context(patch.object(runner.training, 'lock'))
            at_gate = stack.enter_context(patch.object(runner.training, 'gate_at', side_effect=AssertionError('AT gate must not be called')))
            runner.train_apc(root, engine, object(), {})
        expected = [(v,m,s) for v in ('AP','C') for s in SEEDS for m in METHODS]
        assert [(c.args[3],c.args[4],c.args[5]) for c in fit.call_args_list] == expected
        assert fit.call_count == 12 and head.call_count == 6 and lock.call_count == 1
        at_gate.assert_not_called()
        for call in fit.call_args_list:
            variable, method, seed = call.args[3:6]
            assert call.args[6] == (teachers[variable, seed] if method == 'gradual' else None)
        assert [c.args[0] for c in engine.session.transition.call_args_list] == ['AP','C']
    check('twelve_AP_C_fits_no_AT_fit_or_gate_correct_teacher_and_order', dispatch)
    def count_representation():
        a = dict(correct=18, count=20, accuracy=.9)
        auditing.compare(a, dict(correct=18, count=20, accuracy=float(np.float32(.9))), 'boundary')
        for b in (dict(correct=17, count=20, accuracy=.9), dict(correct=18, count=20, accuracy=.900001)):
            try:
                auditing.compare(a,b,'must reject')
            except ValueError:
                pass
            else:
                raise AssertionError('Count/scoring discrepancy accepted')
        try:
            auditing.compare(dict(difference=.1),dict(difference=.10001),'bootstrap unchanged tolerance')
        except ValueError:
            pass
        else:
            raise AssertionError('Bootstrap discrepancy accepted')
    check('integer_counts_exact_FP32_representation_only_no_tolerance_relaxation', count_representation)
    def timestamp():
        scratch=Path('.tmp')/('apca_'+uuid.uuid4().hex[:8])
        path=scratch/'independent_verification.json'
        write(path,dict(time=1.,passed=True,rows=243))
        writer=auditing.stable_writer(write,read,scratch)
        writer(path,dict(time=2.,passed=True,rows=243))
        assert read(path)['time']==1.
        try:
            writer(path,dict(time=2.,passed=False,rows=243))
        except ValueError:
            pass
        else:
            raise AssertionError('Changed audit result accepted')
    check('full_finalization_reaudit_timestamp_only_idempotency', timestamp)
    def imported():
        from experiments.mcqa_c_at_it_das.training import tensor_digest
        for seed in SEEDS:
            q=runner.training.basis(root,'AT','standard',seed,'cpu')
            head=load(root/'readouts'/f'AT_S{seed}.pt')
            receipt=read(root/'readouts'/f'AT_S{seed}.json')
            integrity=read(root/'readouts'/f'AT_S{seed}.receipt.json')
            assert q.shape==(2304,128) and head['weight'].shape==(26,128)
            assert integrity['file']==f'AT_S{seed}.pt'
            assert receipt['q_digest']==tensor_digest(q)
            assert all(not v.requires_grad for v in head.values() if isinstance(v,torch.Tensor))
    check('all_three_exact_rank128_teachers_readouts_imported_and_frozen', imported)
    def budget():
        import time
        from experiments.mcqa_ap_gradual_chain_20261025.execution import Session
        session=object.__new__(Session)
        session.previous=PROTOCOL['inherited_seconds']
        session.profile_previous=PROTOCOL['inherited_profile_seconds']
        session.started=session.stage_started=time.perf_counter()
        session.active,session.stage=True,'resume_validation'
        value=session.ledger()
        assert value['total_limit_seconds']==43200 and value['seconds']>=8769.316136500012
        assert value['profile_seconds']==853.1099996000121
    check('same_cumulative_12h_cap_prior_time_and_profile_retained', budget)
    def inventory():
        inv=runner.evaluation.inventory()
        from experiments.mcqa_ap_gradual_chain_20261025.protocol import BANK_SIZES
        assert len(inv)==243 and sum(BANK_SIZES[b] for b in inv.values())==434304
    check('complete_original_final_control_inventory_unchanged', inventory)
    assert not torch.cuda.is_initialized()
    write(root/'engineering/cpu_tests.json',dict(passed=True,original_tests=base,new_checks=checks,
          original_test_count=base['tests'],additional_test_count=len(checks),cuda_initialized=False),immutable=False)
