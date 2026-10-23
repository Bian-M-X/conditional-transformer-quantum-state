"""Audit all frozen cells, exact SR work/descent, initial controls and replay."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def audit(replay=True):
    cfg=read(ROOT/'experiments/configs/round6.json');idx=read(ROOT/'experiments/round6_index.json')
    assert idx['status']=='completed' and len(idx['runs'])==60 and not idx['failed_runs']
    expected=set(itertools.product(cfg['h_values'],cfg['architectures'],cfg['seeds'],cfg['arms']))
    old={}
    for rid in read(ROOT/'experiments/round5_index.json')['runs']:
        m=read(ROOT/'experiments/runs'/rid/'manifest.json');j=m['job']
        if j['sharing']=='shared' and j['optimizer']=='adam':old[(j['h'],j['architecture'],j['seed'])]=m
    seen=set();commits=set();count=initial=0;worst_residual=0.;failed_steps=0
    for rid in idx['runs']:
        p=ROOT/'experiments/runs'/rid;m=read(p/'manifest.json');j=m['job'];b=m['budget']
        key=tuple(j[k] for k in ['h','architecture','seed','arm']);assert key in expected and key not in seen;seen.add(key)
        assert m['status']=='completed' and m['issue']==11 and m['seed']==j['seed']
        assert read(p/'config.json')==cfg and m['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if m['git_commit'] not in commits:
            assert json.loads(subprocess.check_output(['git','show',m['git_commit']+':experiments/configs/round6.json'],cwd=ROOT,text=True))==cfg
            commits.add(m['git_commit'])
        assert not m['oracle_access']['training'] and not m['oracle_access']['selection']
        assert m['parameter_split']==dict(train=cfg['train'],validation=[],test=cfg['test'])
        shared=j['sharing']=='shared';k=1 if shared else 3;sr=j['optimizer']=='sr'
        assert m['model']['model_count']==k and m['model']['total_parameters']==k*m['model']['parameters_per_model']
        assert len(m['initial_state_sha256'])==len(m['final_state_sha256'])==k and len(set(m['initial_state_sha256']))==1
        if sr:
            assert m['initial_state_sha256']==old[key[:3]]['initial_state_sha256']*k;initial+=1
        else: assert m['model']['total_parameters']==(4621 if j['architecture']=='mlp' else 6973)
        histories=read(p/'training.json');assert len(histories)==k
        work=jacrows=accepted=0
        for i,r in enumerate(histories):
            assert r['model_index']==i and r['train_lambda']==(cfg['train'] if shared else [cfg['train'][i]])
            hh=r['history'];nconditions=len(r['train_lambda'])
            if sr:
                assert r['outer_steps']==cfg['sr_steps'] and [x['step'] for x in hh]==list(range(1,cfg['sr_steps']+1))
                assert all(all(math.isfinite(v) for v in x.values()) for x in hh)
                assert all(1<=x['trials']<=cfg['sr_max_trials'] and x['energy_after']<=x['energy_before']+1e-12 for x in hh)
                assert all(abs(a['energy_after']-bb['energy_before'])<1e-10 for a,bb in zip(hh,hh[1:]))
                assert r['training_h_actions']==nconditions*(cfg['sr_steps']+sum(x['trials'] for x in hh))
                assert r['training_model_calls']==r['training_h_actions']
                assert r['jacobian_rows']==cfg['sr_steps']*nconditions*2**cfg['n']
                assert r['accepted_steps']==sum(x['accepted_rate']>0 for x in hh)
                assert all(x['accepted_rate']*x['tangent_velocity']<=cfg['sr_radius']+1e-10 for x in hh)
                worst_residual=max(worst_residual,max(x['linear_residual'] for x in hh))
                failed_steps+=cfg['sr_steps']-r['accepted_steps'];work+=r['training_h_actions'];jacrows+=r['jacobian_rows'];accepted+=r['accepted_steps']
            else:
                assert r['closure_evaluations']==1200 and len(hh)==1200
                assert [x['evaluation'] for x in hh]==list(range(1,1201))
                assert all(math.isfinite(x['loss']) for x in hh)
                assert sum(not x['accepted_step_completed'] for x in hh)==r['discarded_final_step_evaluations']
        if sr:
            assert b['training_forward_h_actions']==work and b['training_backward_h_actions']==0
            assert b['jacobian_rows']==jacrows==57600 and b['accepted_sr_steps']==accepted
            assert b['max_solve_dimension']==(192 if shared else 64)
            assert b['closure_evaluations']==0 and b['optimizer_steps_completed']==300*k
            assert work<=6300 and worst_residual<=1e-6
        else:
            assert b['training_forward_h_actions']==b['training_backward_h_actions']==3600
            assert b['closure_evaluations']==1200 and b['jacobian_rows']==0
        assert b['training_model_calls']==b['training_forward_h_actions']
        assert b['total_training_h_actions']==b['training_forward_h_actions']+b['training_backward_h_actions']
        assert b['validation_h_actions']==0 and b['exact_diagonalizations']==13 and b['wall_seconds']>0
        rows=read(p/'metrics.json');amps=read(p/'amplitudes.json');count+=len(rows)
        assert len(rows)==(42 if shared else 29) and b['evaluation_h_actions']==2*len(rows)
        assert b['evaluation_model_calls']==(13 if shared else 3)
        assert len({(r['split'],r['method'],r['x']) for r in rows})==len(rows)
        for r in rows:
            assert r['x'] in cfg[r['split']] and r['h']==j['h']
            assert all(math.isfinite(x) for x in r.values() if isinstance(x,(float,int)))
            assert -1e-10<=r['fidelity']<=1+1e-10 and r['energy_error']>=-1e-9 and r['norm_error']<1e-9 and r['variance']>=-1e-12
            psi=amps[str(r['x'])][r['method']];assert len(psi)==64 and abs(sum(x*x for x in psi)-1)<1e-9
        for label,s in m['metrics'].items():
            split,method=label.split('/');rr=[r for r in rows if (r['split'],r['method'])==(split,method)]
            assert len(rr)==len(cfg[split])
            assert abs(sum(1-r['fidelity'] for r in rr)/len(rr)-s['mean_infidelity'])<1e-12
            assert max(1-r['fidelity'] for r in rr)==s['worst_infidelity']
    assert seen==expected and initial==40
    if replay:
        info=read(ROOT/'docs/results/round6/replay_check.json');assert info['original'] in idx['runs'] and info['replay'] not in idx['runs']
        a,b=[ROOT/'experiments/runs'/info[k] for k in ['original','replay']]
        for f in ['config.json','training.json','metrics.json','amplitudes.json']:assert read(a/f)==read(b/f)
        am,bm=read(a/'manifest.json'),read(b/'manifest.json')
        for k in ['job','git_commit','initial_state_sha256','final_state_sha256']:assert am[k]==bm[k]
        assert bm['status']=='completed'
    return dict(jobs=len(seen),metric_rows=count,initial_matches_to_round5=initial,max_relative_solve_residual=worst_residual,
                rejected_sr_outer_steps=failed_steps,replay='exact' if replay else 'not_checked',commits=sorted(commits))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--without-replay',action='store_true');a=ap.parse_args()
    print('PASS:',json.dumps(audit(not a.without_replay)))
