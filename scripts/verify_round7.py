"""Verify stored round7 evidence, full factorial, costs and no-oracle declarations."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from ctqs.physics import hamiltonian,lookup,ritz
from run_round7 import validate_config,jobs

def main():
    c=json.loads((ROOT/'experiments/configs/round7.json').read_text(encoding='utf-8'));validate_config(c)
    idx=json.loads((ROOT/'experiments/round7_index.json').read_text(encoding='utf-8'))
    expected=set(jobs(c));seen=[];total=0;norm=causal=0.;minimum_energy=0.;recalc=0.;hashes=defaultdict(set)
    frozen=set();failures=[]
    for rid in idx['runs']:
        folder=ROOT/'experiments/runs'/rid
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        assert m['status']=='completed'
        assert m['config_sha256']==hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
        assert json.loads((folder/'config.json').read_text(encoding='utf-8'))==c
        a=m['model'];key=(a['architecture'],a['optimizer'],a['training_mode'],a['coverage'],m['seed']);seen.append(key)
        assert key in expected
        assert m['issue'] is None and m['publication']=='local-only'
        assert m['local_task_id']==c['local_task_id']
        assert not m['oracle_access']['training'] and not m['oracle_access']['checkpoint_selection']
        frozen.add(m['git_commit']);hashes[(a['architecture'],m['seed'])].add(a['initial_parameter_hash'])
        budget=m['budget'];assert budget['closure_evaluations']==1200
        assert budget['training_forward_h_actions']==3600 and budget['training_backward_h_actions']==3600
        history=json.loads((folder/'training.json').read_text(encoding='utf-8'))
        assert len(history['closures'])==1200
        assert budget['validation_h_actions']==2*len(history['validation'])
        assert budget['evaluation_ed_calls']==25
        selected=min(history['validation'],key=lambda x:x['validation_energy_per_spin'])
        assert selected['closure']==a['selected_closure']
        checks=json.loads((folder/'checks.json').read_text(encoding='utf-8'))
        for v in checks.values():
            norm=max(norm,v['raw_normalization_error']);causal=max(causal,v['causality_max_logit_difference'])
            assert v['raw_normalization_error']<=1e-10 and v['causality_max_logit_difference']<=1e-10
        amps=json.loads((folder/'amplitudes.json').read_text(encoding='utf-8'))
        rows=json.loads((folder/'metrics.json').read_text(encoding='utf-8'));total+=len(rows)
        assert len(rows)==150
        train=m['parameter_split']['train'];oracle={}
        for endpoint in ['selected','final']:
            anchors=[np.array(amps[endpoint][str(h)]) for h in train]
            for h in map(float,amps[endpoint]):
                H=hamiltonian(c['n'],h)
                if h not in oracle:oracle[h]=np.linalg.eigh(H)
                values,vectors=oracle[h]
                candidates={'direct':np.array(amps[endpoint][str(h)]),
                            'lookup':lookup(h,train,anchors),'train_span_ritz':ritz(H,anchors)[0]}
                for method,psi in candidates.items():
                    row=next(r for r in rows if r['endpoint']==endpoint and r['h']==h and r['method']==method)
                    assert -.000000001<=row['fidelity']<=1.000000001
                    minimum_energy=min(minimum_energy,row['energy_error_per_spin'])
                    assert row['energy_error_per_spin']>=-1e-9
                    assert abs(np.vdot(psi,psi).real-1)<1e-9
                    energy=float(np.vdot(psi,H@psi).real)
                    fidelity=float(abs(np.vdot(vectors[:,0],psi))**2)
                    recalc=max(recalc,abs(energy-row['energy']),abs(fidelity-row['fidelity']))
                    assert abs(energy-row['energy'])<1e-9
                    assert abs(fidelity-row['fidelity'])<1e-6
                    if h in train and method=='lookup':
                        np.testing.assert_allclose(psi,candidates['direct'],atol=1e-12,rtol=1e-12)
    assert len(seen)==len(set(seen))
    failed_keys=[]
    for rid in idx['failed_runs']:
        m=json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text(encoding='utf-8'))
        assert m['status']=='failed';a=m['model']
        failed_keys.append((a['architecture'],a['optimizer'],a['training_mode'],a['coverage'],m['seed']))
    assert set(seen)|set(failed_keys)==expected
    assert all(len(v)==1 for v in hashes.values())
    assert len(frozen)==1
    # Exact content of all scientific source/config files still matches frozen commit.
    sha=next(iter(frozen))
    paths=['scripts/run_round7.py','src/ctqs/models.py','src/ctqs/physics.py','src/ctqs/budgeted_fit.py',
           'experiments/configs/round7.json','docs/round7_preregistration.md']
    for path in paths:
        original=subprocess.check_output(['git','show',sha+':'+path],cwd=ROOT)
        now=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        assert original.replace(b'\r\n',b'\n')==now,path
    report=dict(status='passed',completed_jobs=len(seen),failed_jobs=len(failed_keys),metric_rows=total,
                frozen_source_commit=sha,paired_initial_hash_groups=len(hashes),
                maximum_raw_norm_error=norm,maximum_causality_logit_difference=causal,
                minimum_energy_error_per_spin=minimum_energy,maximum_independent_numpy_recalculation_difference=recalc,
                evaluation='stored amplitudes re-scored by NumPy eigh; training not rerun',
                statistical_unit='seed',independent_human_review=False)
    out=ROOT/'docs/results/round7/evidence_audit.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
