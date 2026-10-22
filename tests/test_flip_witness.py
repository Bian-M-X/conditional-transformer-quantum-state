from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import numpy as np
    from ctqs.physics import basis,hamiltonian,normalized
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'research dependencies required')
class TestFlipWitness(unittest.TestCase):
    def test_identity_bound_and_incomplete_certificate(self):
        rng=np.random.default_rng(89)
        for n in [4,6]:
            z=1-2*basis(n)[:,0]
            for lam in [-.0625,0,.0625]:
                H=hamiltonian(n,.4,lam);e0=np.linalg.eigvalsh(H)[0]
                for complex_state in [False,True]:
                    psi=normalized(rng.normal(size=2**n)+(1j*rng.normal(size=2**n) if complex_state else 0))
                    energy=np.vdot(psi,H@psi).real;m=(abs(psi)**2)@z
                    flipped=np.vdot(psi[::-1],H@psi[::-1]).real
                    self.assertAlmostEqual(energy-flipped,2*lam*m,places=12)
                    self.assertLessEqual(max(0,2*lam*m),energy-e0+1e-12)
        # Zero lambda means zero witness even for an exact excited eigenstate.
        H=hamiltonian(4,.4,0);e,v=np.linalg.eigh(H);psi=v[:,-1]
        self.assertLess(abs(np.vdot(v[:,0],psi))**2,1e-20)
        self.assertEqual(max(0.,2*0.*float((abs(psi)**2)@(1-2*basis(4)[:,0]))),0.)


if __name__=='__main__':unittest.main()
