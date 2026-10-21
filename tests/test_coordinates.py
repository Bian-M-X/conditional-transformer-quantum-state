import copy
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
try:
    import torch
    from ctqs.models import ConditionalMLP, ConditionalTransformer
    from ctqs.conditioning import conditioned_state
    from ctqs.coordinates import compensate_initialization, canonical_adam_step, lambda_weight
    from ctqs.physics import basis, hamiltonian
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, 'requires research dependencies')
class TestCoordinates(unittest.TestCase):
    def test_multistep_physical_trajectory_and_moments(self):
        torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
        bits = torch.tensor(basis(4), dtype=torch.float64)
        for cls in [ConditionalMLP, ConditionalTransformer]:
            # Include a non-power-of-two scale and a large epsilon: compensation
            # must include the full Adam denominator, not just sign gradients.
            for scale, eps in [(.0625, 1e-8), (.04, .01)]:
                torch.manual_seed(137)
                a = cls(4).double(); b = copy.deepcopy(a)
                compensate_initialization(b, scale)
                oa = torch.optim.Adam(a.parameters(), lr=.003, eps=eps, foreach=False)
                ob = torch.optim.Adam(b.parameters(), lr=.003, eps=eps, foreach=False)
                for step in range(30):
                    for model, opt, s in [(a, oa, 1.), (b, ob, scale)]:
                        opt.zero_grad(); loss = 0
                        for lam in [-.0625, 0., .0625]:
                            psi, _ = conditioned_state(model, bits, .6, lam, s)
                            loss = loss + psi @ torch.tensor(hamiltonian(4, .6, lam)) @ psi / 12
                        loss.backward()
                    torch.testing.assert_close(lambda_weight(a).grad[:, -1],
                                               scale * lambda_weight(b).grad[:, -1], atol=1e-12, rtol=1e-10)
                    oa.step(); canonical_adam_step(ob, b, scale)
                    for pa, pb in zip(a.parameters(), b.parameters()):
                        expected = pb.detach().clone()
                        if pb is lambda_weight(b): expected[:, -1] /= scale
                        torch.testing.assert_close(pa, expected, atol=1e-11, rtol=1e-9)
                        for name in ['exp_avg', 'exp_avg_sq']:
                            torch.testing.assert_close(oa.state[pa][name], ob.state[pb][name], atol=1e-12, rtol=1e-9)
                a.eval(); b.eval()
                for lam in [-.05, .003, .04]:
                    torch.testing.assert_close(conditioned_state(a, bits, .6, lam)[0],
                                               conditioned_state(b, bits, .6, lam, scale)[0], atol=1e-11, rtol=1e-9)

    def test_invalid_scale_and_missing_gradient(self):
        model = ConditionalMLP(4).double()
        for scale in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError): compensate_initialization(model, scale)
        opt = torch.optim.Adam(model.parameters())
        with self.assertRaises(ValueError): canonical_adam_step(opt, model, .0625)

    def test_frozen_grid_and_job_count(self):
        cfg = json.loads((ROOT / 'experiments/configs/round4.json').read_text())
        points = sum([cfg[x] for x in ['train', 'validation', 'test']], [])
        self.assertEqual(len(points), len(set(points)))
        self.assertEqual(len(cfg['h_values'])*len(cfg['architectures'])*len(cfg['seeds'])*len(cfg['arms']), 80)
        self.assertEqual(cfg['scale'], max(cfg['train']))


if __name__ == '__main__': unittest.main()
