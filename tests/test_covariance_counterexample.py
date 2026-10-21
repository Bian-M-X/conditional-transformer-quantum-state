"""Check the analytic two-level illustration independently against an eigensolve."""
import unittest
try:
    import numpy as np
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'requires numpy')
class TestCovarianceCounterexample(unittest.TestCase):
    def test_exact_formulas_and_nonuniform_limit(self):
        plus=np.array([1.,1.])/np.sqrt(2)
        for t in [.1,.01,.001]:
            delta=t*t;lam=t
            matrix=np.array([[lam,-delta/2],[-delta/2,-lam]])
            values,vectors=np.linalg.eigh(matrix)
            fidelity=abs(plus@vectors[:,0])**2
            energy=plus@matrix@plus
            variance=np.linalg.norm(matrix@plus-energy*plus)**2
            self.assertAlmostEqual(fidelity,(1+delta/np.sqrt(delta**2+4*lam**2))/2)
            self.assertAlmostEqual(energy-values[0],np.sqrt(delta**2/4+lam**2)-delta/2)
            self.assertAlmostEqual(variance,lam**2)
            self.assertLess(fidelity,.53)
        self.assertLess(variance,1.1e-6)
        self.assertGreater(1-fidelity,.499)


if __name__=='__main__':unittest.main()
