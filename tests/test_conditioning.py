import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
try:
    import torch
    from ctqs.conditioning import conditioned_state
    from ctqs.models import ConditionalMLP,ConditionalTransformer
    from ctqs.physics import basis,hamiltonian
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'requires research torch/numpy/scipy')
class TestConditioning(unittest.TestCase):
    def test_covariance_normalization_scaling_and_gradient(self):
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
        bits=torch.tensor(basis(4),dtype=torch.float64);matrix=torch.tensor(hamiltonian(4,.5,.023))
        for cls in [ConditionalMLP,ConditionalTransformer]:
            torch.manual_seed(7);model=cls(4).double().eval()
            raw,calls=conditioned_state(model,bits,.5,.023,.04,False)
            torch.testing.assert_close(raw,model.psi(bits,[.5,.023/.04]));self.assertEqual(calls,1)
            for lam in [0.,.023]:
                left,calls=conditioned_state(model,bits,.5,lam,.04,True)
                right,_=conditioned_state(model,bits,.5,-lam,.04,True)
                torch.testing.assert_close(left,right.flip(0),atol=1e-14,rtol=1e-14)
                self.assertAlmostEqual(float(left.detach()@left.detach()),1.)
                self.assertEqual(calls,2 if lam else 1)
            psi,_=conditioned_state(model,bits,.5,.023,.04,True)
            (psi@matrix@psi).backward()
            parameter=next(model.parameters());analytic=float(parameter.grad.flatten()[0]);epsilon=1e-5
            with torch.no_grad():
                original=parameter.flatten()[0].item();energies=[]
                for sign in [1,-1]:
                    parameter.flatten()[0]=original+sign*epsilon
                    psi,_=conditioned_state(model,bits,.5,.023,.04,True);energies.append(float(psi@matrix@psi))
                parameter.flatten()[0]=original
            self.assertAlmostEqual(analytic,(energies[0]-energies[1])/(2*epsilon),delta=2e-7)

    def test_frozen_grid_and_factorial(self):
        from run_round3 import jobs,key,validate_config
        cfg=json.loads((ROOT/'experiments/configs/round3.json').read_text());validate_config(cfg)
        self.assertEqual(len({key(j) for j in jobs(cfg)}),100)
        changed=copy.deepcopy(cfg);changed['stage_b']['test'].append(changed['stage_b']['train'][0])
        with self.assertRaisesRegex(ValueError,'leakage'):validate_config(changed)


if __name__=='__main__':unittest.main()
