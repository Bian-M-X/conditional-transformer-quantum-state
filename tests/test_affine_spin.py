from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import numpy as np
    import torch
    from ctqs.affine_spin import SpinFamily, CachedRitz, chain_edges, ladder_edges
    from ctqs.physics import hamiltonian, normalized, ritz
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def kron_reference(n, edges, h, lam, field_sites):
    """Separate tensor-product construction, without indexed bit flips."""
    x = np.array([[0., 1.], [1., 0.]])
    z = np.diag([1., -1.])
    identity = np.eye(2)
    def term(sites):
        result = np.ones((1, 1))
        for i in range(n):
            result = np.kron(result, sites.get(i, identity))
        return result
    result = np.zeros((2**n, 2**n))
    for i, j in edges:
        result -= term({i: z, j: z})
    for i in range(n):
        result -= h * term({i: x})
    for i in field_sites:
        result += lam * term({i: z})
    return result


@unittest.skipUnless(AVAILABLE, 'research dependencies required')
class TestAffineSpin(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(284)
        torch.set_num_threads(1)

    def test_chain_actions_against_dense_vectors_and_columns(self):
        for n in (4, 6):
            family = SpinFamily(n, chain_edges(n))
            for h, lam in ((0., 0), (.43, -.06), (1.15, .07)):
                dense = hamiltonian(n, h, lam)
                for shape in ((2**n,), (2**n, 3)):
                    v = self.rng.normal(size=shape) + 1j * self.rng.normal(size=shape)
                    np.testing.assert_allclose(family.apply_numpy(v, h, lam), dense @ v,
                                               atol=1e-12, rtol=1e-12)

    def test_ladder_and_multisite_fields_against_kronecker(self):
        for n in (4, 6):
            edges = ladder_edges(n); fields = (0, n - 1)
            family = SpinFamily(n, edges, fields)
            reference = kron_reference(n, edges, .71, -.037, fields)
            v = self.rng.normal(size=(2**n, 2))
            np.testing.assert_allclose(family.apply_numpy(v, .71, -.037), reference @ v,
                                       atol=1e-12, rtol=1e-12)
        self.assertEqual(len(ladder_edges(8)), 10)

    def test_torch_state_and_parameter_gradients(self):
        n = 4; family = SpinFamily(n, ladder_edges(n), (0, 3))
        dense = kron_reference(n, ladder_edges(n), .61, -.03, (0, 3))
        v = torch.tensor(self.rng.normal(size=(16, 2)), dtype=torch.float64, requires_grad=True)
        h = torch.tensor(.61, dtype=torch.float64, requires_grad=True)
        lam = torch.tensor(-.03, dtype=torch.float64, requires_grad=True)
        applied = family.apply_torch(v, h, lam)
        loss = (v * applied).sum()
        grad_v, grad_h, grad_lam = torch.autograd.grad(loss, (v, h, lam))
        np.testing.assert_allclose(applied.detach().numpy(), dense @ v.detach().numpy(), atol=1e-12)
        np.testing.assert_allclose(grad_v.numpy(), 2 * dense @ v.detach().numpy(), atol=1e-12)
        _, hx, hz = family.components_numpy(v.detach().numpy())
        self.assertAlmostEqual(float(grad_h), float((v.detach().numpy() * hx).sum()), places=11)
        self.assertAlmostEqual(float(grad_lam), float((v.detach().numpy() * hz).sum()), places=11)

    def test_cached_query_matches_original_ritz_and_has_no_full_action(self):
        n = 6; family = SpinFamily(n, chain_edges(n))
        states = [normalized(self.rng.normal(size=64)) for _ in range(3)]
        cache = CachedRitz(family, states)
        self.assertEqual(cache.component_actions, 3 * cache.rank)
        with patch.object(family, 'apply_numpy', side_effect=AssertionError('query reapplied H')), \
             patch.object(family, 'components_numpy', side_effect=AssertionError('query rebuilt projections')):
            for h, lam in ((.43, -.06), (.6, 0.), (1.15, .07)):
                dense = hamiltonian(n, h, lam)
                psi, rank = ritz(dense, states)
                result = cache.solve(h, lam)
                self.assertEqual(rank, result['rank'])
                self.assertAlmostEqual(result['energy'], float(psi @ dense @ psi), places=11)
                self.assertAlmostEqual(abs(np.vdot(psi, result['state']))**2, 1., places=11)
                self.assertAlmostEqual(np.linalg.norm(result['state']), 1., places=12)
                self.assertAlmostEqual(float(result['state'] @ dense @ result['state']), result['energy'], places=11)
                compact = cache.solve(h, lam, reconstruct=False)
                self.assertNotIn('state', compact)
                self.assertAlmostEqual(compact['energy'], result['energy'], places=12)

    def test_rank_deficiency_complex_states_and_cached_observables(self):
        n = 4; family = SpinFamily(n, ladder_edges(n))
        a = normalized(self.rng.normal(size=16) + 1j * self.rng.normal(size=16))
        b = normalized(self.rng.normal(size=16) + 1j * self.rng.normal(size=16))
        cache = CachedRitz(family, [a, a.copy(), b])
        self.assertEqual(cache.rank, 2)
        result = cache.solve(.77, .04);p = abs(result['state'])**2
        z0 = float(p @ family.z[:, 0]);zn = float(p @ family.z[:, -1])
        corr = float(p @ (family.z[:, 0] * family.z[:, -1])) - z0 * zn
        self.assertAlmostEqual(result['magnetization'], z0, places=12)
        self.assertAlmostEqual(result['connected_correlation'], corr, places=12)
        cache.cache_diagonal('magnetization_average', family.z.mean(axis=1))
        self.assertAlmostEqual(cache.expectation('magnetization_average', result['coefficients']),
                               float(p @ family.z.mean(axis=1)), places=12)
        rank_one = CachedRitz(family, [a, a.copy()])
        self.assertEqual(rank_one.rank, 1)
        self.assertAlmostEqual(abs(np.vdot(a, rank_one.solve(.77, .04)['state']))**2, 1., places=12)

    def test_invalid_graph_states_and_dimensions(self):
        for edges in ([(0, 0)], [(0, 4)], [(0, 1), (1, 0)]):
            with self.assertRaises(ValueError): SpinFamily(4, edges)
        with self.assertRaises(ValueError): ladder_edges(5)
        family = SpinFamily(4, [])
        with self.assertRaises(ValueError): family.apply_numpy(np.zeros(15), .4)
        with self.assertRaises(ValueError): CachedRitz(family, [])
        with self.assertRaises(ValueError): CachedRitz(family, [np.ones(16)])


if __name__ == '__main__':
    unittest.main()
