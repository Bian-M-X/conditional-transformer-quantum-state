from pathlib import Path
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
try:
    import torch
    from ctqs.budgeted_fit import fit
    from ctqs.models import ConditionalMLP
    from ctqs.physics import basis,hamiltonian
    r7=load('round7_runner',ROOT/'scripts/run_round7.py')
    AVAILABLE=True
except ImportError:AVAILABLE=False

@unittest.skipUnless(AVAILABLE,'research dependencies required')
class TestRound7(unittest.TestCase):
    def test_disjoint_regions_and_sixty_jobs(self):
        c=json.loads((ROOT/'experiments/configs/round7.json').read_text(encoding='utf-8'))
        r7.validate_config(c);self.assertEqual(len(r7.jobs(c)),60)
        bad=copy.deepcopy(c);bad['test_regions']['common_extrapolation'].append(.8)
        with self.assertRaises(ValueError):r7.validate_config(bad)
        bad=copy.deepcopy(c);bad['test_regions']['coverage_extension'][0]=.625
        with self.assertRaises(ValueError):r7.validate_config(bad)

    def test_validation_uses_no_oracle_and_restores_mode(self):
        model=ConditionalMLP(4).double();model.train()
        bits=torch.tensor(basis(4),dtype=torch.float64)
        mats={.55:torch.tensor(hamiltonian(4,.55),dtype=torch.float64)}
        with patch.object(r7,'eigh',side_effect=AssertionError('ED leaked into validation')):
            score=r7.accepted_validation(model,bits,4,'raw',[.55],mats)
        self.assertTrue(torch.isfinite(torch.tensor(score)))
        self.assertTrue(model.training)

    def test_callback_never_sees_rejected_line_search_trial(self):
        model=torch.nn.Linear(1,1,bias=False).double()
        with torch.no_grad():model.weight.fill_(3.)
        observed=[];step=[0]
        def fake_step(opt,closure):
            step[0]+=1;closure()
            with torch.no_grad():model.weight.fill_(2. if step[0]==1 else 99.)
            if step[0]>1:closure()
        with patch.object(torch.optim.LBFGS,'step',fake_step):
            result=fit(model,lambda m:m.weight.square().sum(),'lbfgs',2,
                       on_accepted=lambda m,c,s:observed.append((m.weight.item(),c,s)))
        self.assertEqual(observed,[(2.,1,1)])
        self.assertEqual(model.weight.item(),2.)
        self.assertEqual(result['discarded_final_step_evaluations'],1)

class TestLocalManifest(unittest.TestCase):
    def test_local_delivery_requires_registered_task(self):
        validator=load('round7_validator',ROOT/'scripts/validate_workspace.py')
        m={k:None for k in validator.REQUIRED_FIELDS}
        m.update(schema_version=2,run_id='test-run',seed=1,status='planned',issue=None,
                 publication='local-only',local_task_id='local-round7-course-generalization')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'test-run'/'manifest.json';path.parent.mkdir()
            path.write_text(json.dumps(m));self.assertEqual(validator.check_manifest(path),[])
            m['local_task_id']='local-missing';path.write_text(json.dumps(m))
            self.assertTrue(validator.check_manifest(path))
            m['issue']=0;path.write_text(json.dumps(m));self.assertTrue(validator.check_manifest(path))

if __name__=='__main__':unittest.main()
