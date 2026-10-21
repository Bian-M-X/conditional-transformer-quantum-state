import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from ctqs.models import ConditionalMLP,ConditionalTransformer
    from ctqs.conditioning import conditioned_state
    from ctqs.physics import basis
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'requires research torch/numpy/scipy')
class TestScalingEquivalence(unittest.TestCase):
    def test_compensating_affine_column_preserves_all_amplitudes(self):
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
        bits=torch.tensor(basis(4),dtype=torch.float64)
        for cls in [ConditionalMLP,ConditionalTransformer]:
            torch.manual_seed(31);original=cls(4).double().eval();scaled=copy.deepcopy(original)
            with torch.no_grad():
                layer=scaled.net[0] if cls is ConditionalMLP else scaled.condition
                layer.weight[:,-1]*=.04
                for h in [.35,.5,1.1]:
                    for lam in [-.04,-.007,0.,.013,.04]:
                        for covariant in [False,True]:
                            a,_=conditioned_state(original,bits,h,lam,1.,covariant)
                            b,_=conditioned_state(scaled,bits,h,lam,.04,covariant)
                            torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)


if __name__=='__main__':unittest.main()
