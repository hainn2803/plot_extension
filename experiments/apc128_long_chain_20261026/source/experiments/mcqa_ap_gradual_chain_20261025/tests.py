"""CPU regressions; no model load, CUDA context, or scientific updates."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import io
import time
import unittest
from unittest.mock import patch
import torch

from experiments.mcqa_at_anchor_comparison.storage import read, write, BudgetStop
from experiments.mcqa_ap_localization_sweep.sites import gather, scatter
from . import data, training
from .engine import Engine, Subspace, projection, total_loss
from .protocol import PRIOR, BANK_SIZES, SEEDS, fit_names
from .execution import Session


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.groups = read(PRIOR/'groups.json')

    def test_bank_sizes(self):
        for bank, count in BANK_SIZES.items():
            slots = data.slots(bank)
            rows = [r for slot in slots for r in data.proposal(slot, 0, self.groups)]
            self.assertEqual(len(rows), count, bank)
            self.assertEqual([s['slot'] for s in slots], list(range(len(slots))))
            self.assertEqual(len(set(r['id'] for r in rows)), count)
            for r in rows:
                data.validate(r)
            if bank.startswith('readout_'):
                variable = bank.split('_')[1]
                counts = Counter(data.answer(r['base']) if variable == 'AT' else data.ap(r['base']) for r in rows)
                self.assertEqual(len(set(counts.values())), 1)

    def test_AP_target_not_donor_letter(self):
        slots = [s for s in data.slots('AP_train') if s['b'] != s['s']]
        for slot in slots[::64]:
            for row in data.proposal(slot, 0, self.groups):
                self.assertNotEqual(row['target'], data.answer(row['source']))
                self.assertEqual(row['target'], row['base']['labels'][row['metadata']['to_AP']])

    def test_AT_letters_and_cross_question(self):
        rows = [r for s in data.slots('AT_train') for r in data.proposal(s, 0, self.groups)]
        for row in rows:
            self.assertEqual(row['target'], data.answer(row['source']))
            self.assertNotEqual(row['base']['question'], row['source']['question'])
        self.assertEqual(len({r['target'] for r in rows}), 26)
        self.assertTrue(any(r['target'] not in r['base']['labels'] for r in rows))

    def test_paired_orders(self):
        for variable in ('AP', 'C', 'AT'):
            rows = [r for s in data.slots(variable+'_train') for r in data.proposal(s, 0, self.groups)]
            for seed in SEEDS:
                for step in (0, 1, 31, 99, 500, 999):
                    a = training.order(rows, variable, seed, step)
                    self.assertEqual(a, training.order(rows, variable, seed, step))
                    self.assertEqual(len(set(a)), 32)
                    self.assertEqual(sum(rows[i]['metadata']['kind'] == 'changing' for i in a), 24)

    def test_split_objects_disjoint(self):
        sets = {k: {o['id'] for o in v} for k, v in self.groups.items()}
        self.assertEqual([len(sets[k]) for k in ('train', 'calibration', 'final')], [96, 48, 108])
        self.assertFalse(sets['train'] & sets['calibration'] or sets['train'] & sets['final'] or sets['final'] & sets['calibration'])

    def test_joint_projection_split_and_identity(self):
        g = torch.Generator().manual_seed(1)
        hidden = torch.randn(2, 9, 8, generator=g)
        positions = torch.tensor([[1, 3, 5, 7], [1, 3, 5, 7]])
        values = gather(hidden, positions)
        self.assertEqual(values.shape, (2, 32))
        q = Subspace(1, width=32, rank=2).basis()
        self.assertTrue(torch.equal(projection(values, values, q), values))
        self.assertTrue(torch.equal(scatter(hidden, positions, values, torch.ones(2, dtype=torch.bool)), hidden))
        changed = scatter(hidden, positions, values+1, torch.tensor([True, False]))
        self.assertTrue(torch.equal(changed[1], hidden[1]))

    def test_auxiliary_weights_and_zero_equivalence(self):
        x = torch.tensor(3., requires_grad=True)
        terms = dict(direct=x, AT_replay=2*x, AT_readout=3*x, AP_replay=4*x, AP_readout=5*x)
        self.assertEqual(float(total_loss(terms, 'standard').detach()), 3.)
        self.assertEqual(float(total_loss(terms, 'gradual').detach()), 87.)
        zeros = {k: v*0 if k != 'direct' else v for k, v in terms.items()}
        self.assertTrue(torch.equal(total_loss(zeros, 'gradual'), total_loss(terms, 'standard')))
        self.assertEqual(float(torch.autograd.grad(total_loss(terms, 'gradual'), x)[0]), 29.)

    def test_frozen_teacher_loss_gradient_paths(self):
        torch.manual_seed(1)
        e = Engine.__new__(Engine)
        e.device, e.batch = torch.device('cpu'), 2
        e.pad = lambda x: x
        e.session = SimpleNamespace(check=lambda: None)
        head = torch.randn(8, 26)
        e.tail = lambda x: x.float()@head
        e.targets = lambda rows: (torch.tensor([0, 1]), torch.tensor([0, 1]))
        base_c, base_ap = torch.randn(2, 8), torch.randn(2, 32)
        c_to_ap, ap_to_at = torch.randn(8, 32), torch.randn(32, 8)
        def forward(contexts, donor=None, q=None, c_donor=None, c_q=None, at_donor=None, at_q=None, **kw):
            c = base_c if c_donor is None else projection(base_c, c_donor, c_q)
            ap = base_ap+c@c_to_ap
            if donor is not None:
                ap = projection(ap, donor['values'], q)
            at = ap@ap_to_at
            if at_donor is not None:
                at = projection(at, at_donor, at_q)
            return e.tail(at), dict(site=ap, at=at, available=torch.ones(2, dtype=torch.bool))
        e.forward = forward
        teachers = dict(AT_q=Subspace(7, width=8, rank=2).basis().detach(), AP_q=Subspace(8, width=32, rank=2).basis().detach(),
                        AT_head=dict(weight=torch.randn(26, 2), bias=torch.zeros(26)), AP_head=dict(weight=torch.randn(4, 2), bias=torch.zeros(4)))
        rows = [dict(base={}, metadata=dict(to_AP=i)) for i in (0, 1)]
        states = dict(base_AT=torch.randn(2, 8), source_AT=torch.randn(2, 8), source_AP=torch.randn(2, 32), source_C=torch.randn(2, 8))
        for variable in ('AP', 'C'):
            sub = Subspace(1, width=32 if variable == 'AP' else 8, rank=2)
            terms, _ = e.losses(variable, 'gradual', rows, states, sub, teachers)
            self.assertEqual(set(terms), {'direct', 'AT_replay', 'AT_readout'} | ({'AP_replay', 'AP_readout'} if variable == 'C' else set()))
            grad = torch.autograd.grad(total_loss(terms, 'gradual'), sub.raw)[0]
            self.assertTrue(torch.isfinite(grad).all() and bool(grad.norm() > 0))
            self.assertTrue(all(v.grad is None and not v.requires_grad for v in (teachers['AT_q'], teachers['AP_q'])))
            standard, _ = e.losses(variable, 'standard', rows, states, sub, teachers)
            self.assertEqual(set(standard), {'direct'})

    def test_inventory_fixed(self):
        from .evaluation import inventory
        values = inventory()
        self.assertEqual(len(fit_names()), 15)
        self.assertEqual(len(values), 243)
        self.assertEqual(Counter(values.values()), {'AP_final': 90, 'C_final': 144, 'AT_final': 9})

    def test_full_vocabulary_CE_not_restricted_letters(self):
        from .training import records
        letters = tuple('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        engine = SimpleNamespace(device=torch.device('cpu'), label_ids=tuple(range(26)),
                                 decode=lambda token: letters[int(token)] if int(token) < 26 else 'OTHER')
        logits = torch.zeros(1, 27)
        logits[0, 26] = 20
        arrays = records(engine, logits, [dict(target='A')])
        self.assertGreater(float(-arrays['target_log_probability'][0]), 19.)
        self.assertFalse(bool(arrays['correct'][0]))

    def test_independent_paired_bootstrap(self):
        import numpy as np
        from .evaluation import bootstrap
        from .verification import independent_bootstrap, compare
        rows = [dict(base=dict(question='object'+str(i%9)), metadata={}) for i in range(90)]
        a = np.array([i%3 == 0 for i in range(90)])
        b = np.array([i%2 == 0 for i in range(90)])
        mask = np.array([i%5 != 0 for i in range(90)])
        compare(bootstrap(rows, a, b, mask), independent_bootstrap(rows, a, b, mask), 'CPU bootstrap')

    def test_budget_reserve_allowance_and_failure(self):
        from experiments.mcqa_c_at_localization import runtime
        root = Path('diagnostics/ap_gradual_chain_20261025/cpu_test_scratch')/str(time.time_ns())
        root.mkdir(parents=True, exist_ok=False)
        if root:
            with patch.object(runtime, 'configure'), patch.object(runtime, 'guard'), patch.object(runtime, 'close'):
                s = Session(root, 'run')
                self.assertEqual(s.ledger()['total_limit_seconds'], 43200)
                self.assertTrue(s.allow(1., 'test')['allowed'])
                with patch.object(s, 'ledger', return_value=dict(seconds=42900, profile_seconds=0)):
                    with self.assertRaises(BudgetStop):
                        s.check()
                with self.assertRaises(BudgetStop):
                    s.allow(43200., 'must reject')
                s.close()
                self.assertFalse(read(root/'engineering/gpu_budget.json')['active'])
                write(root/'pause.request', dict(reason='test'))
                with self.assertRaises(runtime.Paused):
                    Session(root, 'run')

    def test_checkpoint_order_validation_contract(self):
        from experiments.mcqa_at_anchor_comparison.training import checkpoint, recover, restore_rng
        from experiments.mcqa_c_at_it_das.training import tensor_digest
        a, b = (Subspace(1, width=32, rank=2).state_dict()['raw'] for _ in range(2))
        self.assertTrue(torch.equal(a, b))
        sub = Subspace(1, width=32, rank=2)
        optimizer = torch.optim.Adam(sub.parameters(), lr=.01)
        (sub.raw*torch.arange(32)[:, None]).sum().backward()
        optimizer.step()
        folder = Path('diagnostics/ap_gradual_chain_20261025/cpu_test_scratch')/str(time.time_ns())
        state = dict(identity={'test': True}, step=1, updates=[dict(indices=list(range(32)))], initialization=tensor_digest(a), evaluations={})
        checkpoint(folder, state, sub, optimizer)
        restored, _ = recover(folder, state['identity'])
        self.assertEqual(restored['updates'], state['updates'])
        self.assertEqual(int(restored['optimizer']['state'][0]['step']), 1)
        restore_rng(restored)
        self.assertTrue(torch.equal(sub.raw.detach(), restored['subspace']['raw']))

    def test_evaluation_dispatch_never_retrains_and_closes_failure(self):
        from contextlib import ExitStack, nullcontext
        from . import runner
        closed = []
        class FakeSession:
            def __init__(self, *args):
                pass
            def transition(self, stage):
                pass
            def close(self):
                closed.append(True)
        class FakeWatch:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        root = Path('diagnostics/ap_gradual_chain_20261025/cpu_test_scratch')/str(time.time_ns())
        root.mkdir(parents=True)
        with ExitStack() as stack:
            for owner, attr, value in ((runner.data, 'verify', lambda *a, **k: {}), (runner, 'stamp', lambda *a: dict(pid=1)),
                                      (runner, 'Session', FakeSession), (runner, 'use_scoped_counter', lambda *a: nullcontext()),
                                      (runner, 'initialize', lambda *a: None), (runner, 'Engine', lambda *a: object()),
                                      (runner, 'verify_live_counter', lambda: {}), (runner, 'NaturalCache', lambda *a: None),
                                      (runner.profiling, 'profile', lambda *a: {}), (runner, 'gate', lambda *a: None),
                                      (runner, 'MemoryWatch', FakeWatch), (runner.banks, 'locked', lambda *a: None),
                                      (torch.cuda, 'synchronize', lambda: None), (torch.cuda, 'is_initialized', lambda: False)):
                stack.enter_context(patch.object(owner, attr, value))
            fit = stack.enter_context(patch.object(runner.training, 'fit'))
            stack.enter_context(patch.object(runner.evaluation, 'evaluate_all', side_effect=RuntimeError('forced evaluator failure')))
            with self.assertRaisesRegex(RuntimeError, 'forced evaluator failure'):
                runner.gpu(root, 'evaluate')
            fit.assert_not_called()
            self.assertEqual(closed, [True])


def cpu(root):
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight must precede CUDA initialization')
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    print(stream.getvalue(), flush=True)
    receipt = dict(passed=result.wasSuccessful(), tests=result.testsRun, output=stream.getvalue(), cuda_initialized=False)
    write(Path(root)/'engineering/cpu_tests.json', receipt, immutable=False)
    if not result.wasSuccessful():
        raise RuntimeError('CPU regressions failed')
    return receipt


if __name__ == '__main__':
    unittest.main()
