import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
try:
    import numpy as np
    from ctqs.acquisition import covariant_controls,next_batch_size,score_states,select_unseen
    from ctqs.physics import hamiltonian,normalized,ritz
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'requires research numpy/scipy')
class TestAcquisition(unittest.TestCase):
    def test_covariance_and_norm_at_nonzero_field(self):
        def predict(lam):return normalized(np.exp(np.linspace(-1,1,16)*(lam+0.3)))
        for lam in [0.,.002,.03]:
            _,sign,average=covariant_controls(predict,lam)
            _,minus_sign,minus_average=covariant_controls(predict,-lam)
            np.testing.assert_allclose(sign,minus_sign[::-1],atol=1e-14)
            np.testing.assert_allclose(average,minus_average[::-1],atol=1e-14)
            self.assertAlmostEqual(np.linalg.norm(average),1.)
        _,_,average=covariant_controls(predict,.03)
        self.assertGreater(np.linalg.norm(average-average[::-1]),1e-5)

    def test_scores_against_reference_and_no_full_eigensolve(self):
        matrix=hamiltonian(4,.35,.02)
        psi=normalized(np.random.default_rng(4).uniform(size=16))
        variance,cost=score_states([matrix],[psi],'variance')
        energy=psi@matrix@psi
        self.assertAlmostEqual(variance[0],np.linalg.norm(matrix@psi-energy*psi)**2)
        self.assertEqual(cost,1)
        eigen=np.linalg.eigvalsh
        def limited(m):
            self.assertLessEqual(m.shape[0],2)
            return eigen(m)
        with patch('numpy.linalg.eigvalsh',side_effect=limited):
            gain,cost=score_states([matrix],[psi],'ritz_gain')
        corrected,rank=ritz(matrix,[psi,psi[::-1]])
        self.assertAlmostEqual(gain[0],energy-corrected@matrix@corrected)
        self.assertEqual(cost,1+rank)

    def test_selection_excludes_train_and_ties_are_stable(self):
        self.assertEqual(select_unseen([-.04,0,.04],[0],[1,9,1]),-.04)
        with self.assertRaises(ValueError):select_unseen([0],[0],[1])

    def test_exact_budget_including_odd_query_cost(self):
        for queries in [0,51,153]:
            training=0;steps=0
            while True:
                size=next_batch_size(2400,training,queries,2)
                if not size:break
                training+=size;steps+=1
            self.assertEqual(training+queries,2400)
            self.assertEqual(steps,(2400-queries+1)//2)
        with self.assertRaises(ValueError):next_batch_size(2,2,1,2)

    def test_config_rejects_test_leakage(self):
        import json
        from run_round2 import validate_acquisition_config
        cfg=json.loads((ROOT/'experiments/configs/round2_acquisition.json').read_text())
        validate_acquisition_config(cfg)
        cfg['candidate_lambda'].append(cfg['test_lambda'][0])
        with self.assertRaisesRegex(ValueError,'test leakage'):
            validate_acquisition_config(cfg)


if __name__=='__main__':unittest.main()
