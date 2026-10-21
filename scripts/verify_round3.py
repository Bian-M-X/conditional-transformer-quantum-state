"""Audit factorial evidence, exact legacy trajectories, budgets, and replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def identity(j):return tuple(j[k] for k in ['stage','n','architecture','scale','mode','seed'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--without-replay',action='store_true');args=ap.parse_args()
    cfg=read(ROOT/'experiments/configs/round3.json');index=read(ROOT/'experiments/round3_index.json')
    assert index['status']=='completed' and len(index['runs'])==100
    old=read(ROOT/'experiments/round2_index.json');references={}
    for rid in old['stage_a_new']:
        directory=ROOT/'experiments/runs'/rid;m=read(directory/'manifest.json')
        if m['budget']['gradient_steps']==2400:
            references[(m['model']['architecture'],m['model']['training_mode'],m['seed'])]=directory
    seen=set();initial={};commits=set();count=matches=0
    for rid in index['runs']:
        directory=ROOT/'experiments/runs'/rid;m=read(directory/'manifest.json');j=m['job'];b=cfg['stage_'+j['stage']]
        ident=identity(j);assert ident not in seen;seen.add(ident)
        assert read(directory/'config.json')==cfg
        assert m['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        assert m['status']=='completed' and not m['oracle_access']['training'] and not m['oracle_access']['selection']
        if m['git_commit'] not in commits:
            subprocess.run(['git','cat-file','-e',m['git_commit']+'^{commit}'],cwd=ROOT,check=True);commits.add(m['git_commit'])
        initial.setdefault((j['n'],j['architecture'],j['seed']),set()).add(m['initial_state_sha256'])
        history=read(directory/'training.json');rows=read(directory/'metrics.json');count+=len(rows)
        assert len(history)==b['steps']//cfg['validation_every']
        assert m['model']['selected_step']==min(history,key=lambda r:r['validation_energy_per_spin'])['step']
        budget=m['budget'];assert budget['gradient_steps']==b['steps']
        assert budget['training_forward_h_actions']==budget['training_backward_h_actions']==3*b['steps']
        assert budget['validation_h_actions']==96
        assert budget['total_training_selection_h_actions']==(14496 if j['stage']=='a' else 7296)
        assert budget['training_model_calls']==b['steps']*(5 if j['mode']=='covariant' else 3)
        assert budget['validation_model_calls']==96*(2 if j['mode']=='covariant' else 1)
        assert budget['evaluation_h_actions']==2*len(rows)
        expected_methods={'direct','lookup','train_span_ritz','even_projection','parity_ritz'} if j['stage']=='a' else {'direct','lookup','train_span_ritz','post_covariant'}
        assert len(rows)==(150 if j['stage']=='a' else 136)
        assert len({(r['checkpoint'],r['method'],r['x']) for r in rows})==len(rows)
        for r in rows:
            assert r['method'] in expected_methods and r['checkpoint'] in ['selected','final']
            assert r['x'] in m['parameter_split'][r['split']]
            assert r['norm_error']<1e-10 and -1e-10<=r['fidelity']<=1+1e-10
            assert r['energy_error']>=-1e-10 and r['variance']>=0
        for checkpoint in ['selected','final']:
            for x in {r['x'] for r in rows}:
                at={r['method']:r for r in rows if r['checkpoint']==checkpoint and r['x']==x}
                assert at['train_span_ritz']['energy']<=at['lookup']['energy']+1e-9
                if j['mode']=='covariant':
                    assert abs(at['direct']['fidelity']-at['post_covariant']['fidelity'])<1e-10
                    reverse=next(r for r in rows if r['checkpoint']==checkpoint and r['x']==-x and r['method']=='direct')
                    assert abs(at['direct']['energy']-reverse['energy'])<1e-10
                    assert abs(at['direct']['fidelity']-reverse['fidelity'])<1e-9
            for split in m['parameter_split']:
                for method in expected_methods:
                    rr=[r for r in rows if (r['checkpoint'],r['split'],r['method'])==(checkpoint,split,method)]
                    assert abs(sum(1-r['fidelity'] for r in rr)/len(rr)-m['metrics'][f'{checkpoint}/{split}/{method}']['mean_infidelity'])<1e-14
        if j['stage']=='a':
            reference=references[(j['architecture'],j['mode'],j['seed'])]
            assert history==read(reference/'training.json')
            former={(r['h'],r['method']):r for r in read(reference/'metrics.json')}
            for row in rows:
                if row['checkpoint']=='selected':
                    prior=former[(row['h'],row['method'])]
                    for k in prior.keys()&row.keys():assert row[k]==prior[k],(rid,k,row[k],prior[k])
            matches+=1
    expected_a={('a',6,a,1.,mode,s) for a in cfg['architectures'] for mode in ['raw','even'] for s in cfg['seeds']}
    expected_b={('b',n,a,scale,mode,s) for n in [4,6] for a in cfg['architectures'] for scale in [1.,.04] for mode in ['raw','covariant'] for s in cfg['seeds']}
    assert seen==expected_a|expected_b
    assert all(len(hashes)==1 for hashes in initial.values())
    if not args.without_replay:
        replay=read(ROOT/'docs/results/round3/replay_check.json')
        a=ROOT/'experiments/runs'/replay['original'];b=ROOT/'experiments/runs'/replay['replay']
        for name in ['metrics.json','training.json']:assert read(a/name)==read(b/name)
        for field in ['initial_state_sha256','selected_state_sha256','final_state_sha256']:
            assert read(a/'manifest.json')[field]==read(b/'manifest.json')[field]
    print(f'PASS: 100 factorial jobs, {count} metric rows, {matches} exact legacy trajectory/selected-metric matches; replay '+('NOT CHECKED' if args.without_replay else 'exact'))


if __name__=='__main__':main()
