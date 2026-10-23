from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from ctqs.exact_sr import amplitude_jacobian,tangent_system,damped_direction,fit_sr
    from ctqs.models import ConditionalMLP,ConditionalTransformer
    from ctqs.physics import basis,hamiltonian
    AVAILABLE=True
except ImportError: AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'research dependencies required')
class TestExactSR(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64);torch.set_num_threads(1);torch.manual_seed(37)

    def test_dual_primal_and_gradient_factor(self):
        A=torch.randn(11,7);b=torch.randn(11);shift=.003
        d,r=damped_direction(A,b,shift)
        ref=torch.linalg.solve(A.T@A+shift*torch.eye(7),A.T@b)
        torch.testing.assert_close(d,ref,atol=1e-10,rtol=1e-10)
        self.assertLess(r,1e-9)
        with self.assertRaises(ValueError): damped_direction(A,b,0.)

    def test_models_jacobian_qgt_and_force(self):
        n=3;bits=torch.tensor(basis(n),dtype=torch.float64);c=[.4,.0625]
        H=torch.tensor(hamiltonian(n,*c))
        for model in [ConditionalMLP(n,width=4),ConditionalTransformer(n,width=4,heads=2)]:
            model.train();p,J=amplitude_jacobian(model,bits,c)
            torch.testing.assert_close(p,model.psi(bits,c),atol=1e-12,rtol=1e-12)
            self.assertLess(float(torch.linalg.vector_norm(p@J)),1e-12)
            A,b,loss=tangent_system([p],[J],[H],n)
            log_j=J/p[:,None];prob=p*p;mean=prob@log_j
            qgt=(log_j-mean).T@(prob[:,None]*(log_j-mean))
            torch.testing.assert_close(A.T@A,qgt,atol=1e-12,rtol=1e-10)
            psi=model.psi(bits,c);energy=psi@H@psi/n
            grad=torch.cat([v.reshape(-1) for v in torch.autograd.grad(energy,tuple(model.parameters()))])
            torch.testing.assert_close(2*A.T@b,grad,atol=1e-11,rtol=1e-9)
            # An independent central finite-difference directional derivative.
            vector=torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            tangent=torch.randn_like(vector);tangent/=torch.linalg.vector_norm(tangent)
            values=[]
            for sign in [-1,1]:
                torch.nn.utils.vector_to_parameters(vector+sign*1e-5*tangent,model.parameters())
                values.append(model.psi(bits,c).detach())
            torch.nn.utils.vector_to_parameters(vector,model.parameters())
            torch.testing.assert_close((values[1]-values[0])/2e-5,J@tangent,atol=1e-9,rtol=1e-6)

    def test_one_spin_imaginary_time_and_budget(self):
        # psi=(cos t,sin t), H=-X: S=1 and force=-cos(2t).
        t=torch.tensor(.31);p=torch.stack([t.cos(),t.sin()]);J=torch.stack([-t.sin(),t.cos()])[:,None]
        H=torch.tensor([[0.,-1.],[-1.,0.]])
        A,b,loss=tangent_system([p],[J],[H],1)
        d,_=damped_direction(A,b,.002)
        torch.testing.assert_close(d[0],-torch.cos(2*t)/1.002)
        n=3;model=ConditionalMLP(n,width=4);bits=torch.tensor(basis(n),dtype=torch.float64)
        c=[[.4,-.0625],[.4,.0625]];m=[torch.tensor(hamiltonian(n,*x)) for x in c]
        result=fit_sr(model,bits,c,m,n,steps=4)
        self.assertEqual(result['jacobian_rows'],4*2*8)
        self.assertEqual(result['training_h_actions'],2*(4+sum(x['trials'] for x in result['history'])))
        self.assertTrue(all(x['energy_after']<=x['energy_before']+1e-13 for x in result['history']))


if __name__=='__main__': unittest.main()
