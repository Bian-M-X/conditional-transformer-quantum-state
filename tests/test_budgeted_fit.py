from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
try:
    import torch
    from ctqs.budgeted_fit import fit
    AVAILABLE=True
except ImportError: AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'research dependencies required')
class TestBudgetedFit(unittest.TestCase):
    def model(self):
        m=torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad():m.weight.fill_(3.)
        return m

    def test_adam_and_real_lbfgs_exact_closure_cap(self):
        for kind in ['adam','lbfgs']:
            m=self.model(); count=[]
            def objective(model):
                count.append(1);return (model.weight-1.).square().sum()
            r=fit(m,objective,kind,17)
            self.assertEqual(len(count),17);self.assertEqual(r['closure_evaluations'],17)
            self.assertEqual(len(r['history']),17)
            self.assertLess(float((m.weight-1).square().sum().detach()),4.)

    def test_budget_exhausted_trial_restores_last_accepted(self):
        m=self.model()
        def fake_step(opt,closure):
            closure()
            with torch.no_grad(): next(iter(opt.param_groups[0]['params'])).fill_(99.)
            closure()  # exceeds budget before evaluating trial
        with patch.object(torch.optim.LBFGS,'step',fake_step):
            r=fit(m,lambda model:model.weight.square().sum(),'lbfgs',1)
        self.assertEqual(m.weight.item(),3.)
        self.assertEqual(r['optimizer_steps_completed'],0)
        self.assertEqual(r['discarded_final_step_evaluations'],1)
        self.assertFalse(r['history'][0]['accepted_step_completed'])

    def test_nonfinite_is_failure_not_false_completion(self):
        with self.assertRaises(ValueError):fit(self.model(),lambda m:m.weight.sum()*float('nan'),'adam',2)


if __name__=='__main__':unittest.main()
