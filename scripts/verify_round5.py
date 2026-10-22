"""Audit closure budgets, anchor endpoints, old Adam trajectories and replay."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def audit(replay=True):
    cfg=read(ROOT/'experiments/configs/round5.json');idx=read(ROOT/'experiments/round5_index.json')
    assert idx['status']=='completed' and len(idx['runs'])==80 and not idx['failed_runs']
    expected=set(itertools.product(cfg['h_values'],cfg['architectures'],cfg['seeds'],cfg['sharing'],cfg['optimizers']))
    seen=set();commits=set();rows_count=legacy=0;initial={};discarded=0
    refs={}
    for rid in read(ROOT/'experiments/round4_index.json')['runs']:
        p=ROOT/'experiments/runs'/rid;m=read(p/'manifest.json');j=m['job']
        if j['arm']=='physical':refs[(j['h'],j['architecture'],j['seed'])]=(p,m)
    for rid in idx['runs']:
        p=ROOT/'experiments/runs'/rid;m=read(p/'manifest.json');j=m['job'];b=m['budget']
        key=tuple(j[k] for k in ['h','architecture','seed','sharing','optimizer']);assert key in expected and key not in seen;seen.add(key)
        assert m['status']=='completed' and m['issue']==9 and m['seed']==j['seed']
        assert read(p/'config.json')==cfg and m['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if m['git_commit'] not in commits:
            assert json.loads(subprocess.check_output(['git','show',m['git_commit']+':experiments/configs/round5.json'],cwd=ROOT,text=True))==cfg
            commits.add(m['git_commit'])
        assert not m['oracle_access']['training'] and not m['oracle_access']['selection']
        assert m['parameter_split']==dict(train=cfg['train'],validation=[],test=cfg['test'])
        shared=j['sharing']=='shared';k=1 if shared else 3
        assert m['model']['model_count']==k and m['model']['total_parameters']==k*m['model']['parameters_per_model']
        assert len(m['initial_state_sha256'])==len(m['final_state_sha256'])==k
        assert len(set(m['initial_state_sha256']))==1
        initial.setdefault(key[:3],set()).update(m['initial_state_sha256'])
        assert b['training_forward_h_actions']==b['training_backward_h_actions']==b['training_model_calls']==3600
        assert b['total_training_h_actions']==7200 and b['validation_h_actions']==0
        assert b['closure_evaluations']==1200*k and b['evaluation_model_calls']==(13 if shared else 3)
        assert b['exact_diagonalizations']==13 and b['wall_seconds']>0
        histories=read(p/'training.json');assert len(histories)==k
        for i,record in enumerate(histories):
            assert record['model_index']==i and record['closure_evaluations']==1200
            assert record['train_lambda']==(cfg['train'] if shared else [cfg['train'][i]])
            hh=record['history'];assert [r['evaluation'] for r in hh]==list(range(1,1201))
            assert all(math.isfinite(r['loss']) for r in hh)
            steps=record['optimizer_steps_completed'];lost=record['discarded_final_step_evaluations']
            assert {r['optimizer_call'] for r in hh if r['accepted_step_completed']}==set(range(1,steps+1))
            assert sum(not r['accepted_step_completed'] for r in hh)==lost
            assert all(r['optimizer_call']==steps+1 for r in hh if not r['accepted_step_completed'])
            if j['optimizer']=='adam':assert steps==1200 and lost==0
        assert b['optimizer_steps_completed']==sum(r['optimizer_steps_completed'] for r in histories)
        assert b['discarded_final_step_evaluations']==sum(r['discarded_final_step_evaluations'] for r in histories)
        discarded+=b['discarded_final_step_evaluations']
        rows=read(p/'metrics.json');amps=read(p/'amplitudes.json');rows_count+=len(rows)
        assert len(rows)==(42 if shared else 29) and b['evaluation_h_actions']==2*len(rows)
        keys=set()
        for r in rows:
            ident=(r['split'],r['method'],r['x']);assert ident not in keys;keys.add(ident)
            assert r['x'] in cfg[r['split']] and r['h']==j['h']
            assert all(math.isfinite(x) for x in r.values() if isinstance(x,(float,int)))
            assert -1e-10<=r['fidelity']<=1+1e-10 and r['energy_error']>=-1e-9 and r['norm_error']<1e-9 and r['variance']>=-1e-12
            psi=amps[str(r['x'])][r['method']];assert len(psi)==64 and abs(sum(x*x for x in psi)-1)<1e-9
        for label,summary in m['metrics'].items():
            split,method=label.split('/');rr=[r for r in rows if (r['split'],r['method'])==(split,method)]
            assert len(rr)==len(cfg[split])
            assert abs(sum(1-r['fidelity'] for r in rr)/len(rr)-summary['mean_infidelity'])<1e-12
            assert max(1-r['fidelity'] for r in rr)==summary['worst_infidelity']
        if shared and j['optimizer']=='adam':
            old,om=refs[key[:3]];oldrows=read(old/'metrics.json');oldamps=read(old/'amplitudes.json')
            assert m['final_state_sha256'][0]==om['final_state_sha256']
            for x in cfg['train']:
                assert amps[str(x)]['anchor']==oldamps['final'][str(x)]
                a=next(r for r in rows if r['split']=='train' and r['method']=='anchor' and r['x']==x)
                bb=next(r for r in oldrows if r['checkpoint']=='final' and r['split']=='train' and r['method']=='direct' and r['x']==x)
                for name,value in a.items():
                    if name not in ['method']:assert value==bb[name]
            legacy+=1
    assert seen==expected and all(len(v)==1 for v in initial.values())
    if replay:
        info=read(ROOT/'docs/results/round5/replay_check.json');assert info['original'] in idx['runs'] and info['replay'] not in idx['runs']
        a,b=[ROOT/'experiments/runs'/info[k] for k in ['original','replay']]
        for f in ['config.json','training.json','metrics.json','amplitudes.json']:assert read(a/f)==read(b/f)
        am,bm=read(a/'manifest.json'),read(b/'manifest.json')
        for k in ['job','git_commit','initial_state_sha256','final_state_sha256']:assert am[k]==bm[k]
        assert bm['status']=='completed'
    return dict(jobs=len(seen),metric_rows=rows_count,exact_legacy_final_anchor_matches=legacy,
                discarded_final_step_evaluations=discarded,replay='exact' if replay else 'not_checked',commits=sorted(commits))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--without-replay',action='store_true');ap.add_argument('--write',action='store_true');args=ap.parse_args()
    result=audit(not args.without_replay)
    if args.write:
        p=ROOT/'docs/results/round5/evidence_audit.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('PASS:',json.dumps(result))
