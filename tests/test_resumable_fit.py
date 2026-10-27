import copy
import io
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import numpy as np
    import torch
    from ctqs.resumable_fit import ResumableFit
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'research dependencies required')
class TestResumableFit(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(39); np.random.seed(39); random.seed(39)

    def model(self):
        m = torch.nn.Linear(2, 1, bias=False).double()
        with torch.no_grad(): m.weight.copy_(torch.tensor([[2.1, -.7]], dtype=torch.float64))
        return m

    def objective(self, model):
        x, y = model.weight.flatten()
        return (x - .3).square() + 3 * (y + .2).square() + .05 * x.pow(4)

    def assert_nested_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for k in a: self.assert_nested_equal(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            self.assertEqual(type(a), type(b)); self.assertEqual(len(a), len(b))
            for x, y in zip(a, b): self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_adam_split_matches_uninterrupted_parameters_and_memory(self):
        a = ResumableFit(self.model(), 'adam', lr=.03)
        b = ResumableFit(self.model(), 'adam', lr=.03)
        whole = a.advance(self.objective, 23)
        left = b.advance(self.objective, 7); right = b.advance(self.objective, 16)
        self.assert_nested_equal(a.model.state_dict(), b.model.state_dict())
        self.assert_nested_equal(a.optimizer.state_dict(), b.optimizer.state_dict())
        self.assertEqual(whole['closure_evaluations'], left['closure_evaluations'] + right['closure_evaluations'])
        self.assertEqual(a.counters['closure_evaluations'], b.counters['closure_evaluations'])

    def test_lbfgs_weights_only_save_resume_same_block_schedule_and_rng(self):
        a = ResumableFit(self.model(), 'lbfgs', lr=.7, history_size=4)
        first = a.advance(self.objective, 9)
        self.assertEqual(first['closure_evaluations'], 9)
        snapshot = a.state_dict()
        # Safe torch serialization is part of the public checkpoint contract.
        stream = io.BytesIO(); torch.save(snapshot, stream); stream.seek(0)
        restored = torch.load(stream, weights_only=True, map_location='cpu')
        expected_rng = (torch.rand(3), np.random.random(3).tolist(), random.random())
        a.load_state_dict(snapshot)
        expected = a.advance(self.objective, 17)
        expected_state = a.state_dict()
        b = ResumableFit(self.model(), 'lbfgs', lr=.7, history_size=4)
        b.load_state_dict(restored)
        actual_rng = (torch.rand(3), np.random.random(3).tolist(), random.random())
        self.assert_nested_equal(expected_rng, actual_rng)
        # Restore again so the RNG probe cannot change the replay itself.
        b.load_state_dict(restored)
        actual = b.advance(self.objective, 17)
        self.assert_nested_equal(expected, actual)
        self.assert_nested_equal(expected_state, b.state_dict())
        self.assertTrue(b.optimizer.state_dict()['state'])

    def test_interrupted_trial_restores_model_optimizer_and_rng(self):
        fit = ResumableFit(self.model(), 'lbfgs', history_size=4)
        fit.advance(self.objective, 5)
        before = fit.state_dict()
        def speculative(opt, closure):
            closure()
            with torch.no_grad(): next(iter(opt.param_groups[0]['params'])).fill_(99.)
            opt.state[next(iter(opt.param_groups[0]['params']))]['fake_trial_state'] = torch.tensor(83.)
            torch.rand(3); np.random.random(); random.random()
            closure()
        with patch.object(torch.optim.LBFGS, 'step', speculative):
            result = fit.advance(self.objective, 1)
        self.assertEqual(result['closure_evaluations'], 1)
        self.assertEqual(result['optimizer_steps_completed'], 0)
        self.assertEqual(result['discarded_final_step_evaluations'], 1)
        self.assertFalse(result['history'][0]['accepted_step_completed'])
        after = fit.state_dict()
        for k in ['model_state', 'optimizer_state', 'rng_state']:
            self.assert_nested_equal(before[k], after[k])
        self.assertEqual(after['counters']['closure_evaluations'], before['counters']['closure_evaluations'] + 1)

    def test_snapshot_is_detached_and_invalid_settings_rejected(self):
        fit = ResumableFit(self.model(), 'adam', lr=.03)
        snapshot = fit.state_dict()
        with torch.no_grad(): fit.model.weight.fill_(11.)
        self.assertFalse(torch.equal(snapshot['model_state']['weight'], fit.model.weight))
        with self.assertRaises(ValueError): ResumableFit(self.model(), 'adam', lr=.04).load_state_dict(snapshot)
        with self.assertRaises(ValueError): fit.advance(self.objective, 0)

    def test_nonfinite_loss_and_gradient_raise(self):
        for kind in ['adam', 'lbfgs']:
            fit = ResumableFit(self.model(), kind)
            with self.assertRaises(ValueError):
                fit.advance(lambda m: m.weight.sum() * float('nan'), 2)
            self.assertEqual(fit.counters['optimizer_steps_completed'], 0)
            self.assertEqual(fit.counters['attempted_closures'], 1)
            # finite forward, infinite derivative
            m = self.model()
            with torch.no_grad(): m.weight.zero_()
            fit = ResumableFit(m, kind)
            with self.assertRaises(ValueError):
                fit.advance(lambda model: model.weight.sqrt().sum(), 2)
            self.assertEqual(fit.counters['optimizer_steps_completed'], 0)


if __name__ == '__main__': unittest.main()
