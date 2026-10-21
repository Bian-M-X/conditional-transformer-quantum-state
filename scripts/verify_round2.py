"""Verify saved round-two evidence, budgets, splits, pairing and deterministic replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--without-replay',action='store_true',help='Initial audit before replay exists; never implies replay passed')
    args=parser.parse_args()
    index=read(ROOT/'experiments/round2_index.json')
    assert index['status']=='completed'
    assert len(index['stage_a_reused'])==12 and len(index['stage_a_new'])==48 and len(index['stage_b'])==60
    seeds=[17,29,43,61,73];architectures=['mlp','transformer']
    seen_a,seen_b,commits=set(),set(),set()
    prefixes={};row_count=0
    cfg_b=read(ROOT/'experiments/configs/round2_acquisition.json')
    for stage,ids in [('a',index['stage_a_reused']+index['stage_a_new']),('b',index['stage_b'])]:
        for rid in ids:
            directory=ROOT/'experiments/runs'/rid
            m=read(directory/'manifest.json');cfg=read(directory/'config.json')
            rows=read(directory/'metrics.json')
            assert m['status']=='completed' and m['run_id']==rid
            assert m['oracle_access']['training'] is False and m['oracle_access']['selection'] is False
            assert m['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
            if m['git_commit'] not in commits:
                subprocess.run(['git','cat-file','-e',m['git_commit']+'^{commit}'],cwd=ROOT,check=True)
                commits.add(m['git_commit'])
            for r in rows:
                assert r['norm_error']<1e-10 and -1e-10<=r['fidelity']<=1+1e-10
                assert r['energy_error']>=-1e-10 and r['variance']>=0
            row_count+=len(rows)
            if stage=='a':
                key=(m['model']['architecture'],m['model']['training_mode'],m['seed'],cfg['steps'])
                assert key not in seen_a;seen_a.add(key)
                assert m['budget']['gradient_steps']==cfg['steps']
                assert m['budget']['training_hamiltonian_actions']==3*cfg['steps']
                assert m['budget']['validation_hamiltonian_actions']==2*(cfg['steps']//50)
                reference=read(ROOT/f"experiments/configs/round2_budget_{cfg['steps']}.json")
                # Three original seeds at 600 updates predate stage/data-status metadata.
                for k in cfg:assert cfg[k]==reference[k] or k=='seeds'
                assert len(rows)==75
                for r in rows:assert r['h'] in cfg[r['split']]
                history=read(directory/'training.json')
                assert m['model']['selected_step']==min(history,key=lambda r:r['validation_energy_per_spin'])['step']
            else:
                assert cfg==cfg_b
                key=(m['model']['architecture'],m['model']['policy'],m['seed'])
                assert key not in seen_b;seen_b.add(key)
                group=(key[0],key[2]);prefixes.setdefault(group,set()).add(m['common_prefix_hash'])
                budget=m['budget'];events=read(directory/'acquisition.json')
                assert budget['total_training_selection_h_actions']==cfg['total_h_actions']==2400
                assert budget['training_h_actions']+budget['acquisition_h_actions']==2400
                assert budget['backward_h_actions']==budget['training_h_actions']
                assert budget['total_with_backward_h_actions']==4800-budget['acquisition_h_actions']
                assert budget['acquisition_h_actions']==sum(e['h_actions'] for e in events)
                assert budget['gradient_steps']==(budget['training_h_actions']+1)//2
                assert budget['training_model_calls']==budget['training_h_actions']
                assert len(events)==(0 if key[1]=='continue' else 3)
                pool=list(cfg['initial_lambda'])
                for event in events:
                    assert event['step'] in cfg['acquisition_steps']
                    chosen=event['selected_lambda']
                    assert chosen in cfg['candidate_lambda'] and chosen not in pool
                    if key[1] in ['variance','ritz_gain']:
                        assert event['candidates']==cfg['candidate_lambda']
                        available=[i for i,x in enumerate(event['candidates']) if x not in pool]
                        best=max(available,key=lambda i:(event['scores'][i],-i))
                        assert chosen==event['candidates'][best]
                        if key[1]=='variance':
                            assert event['h_actions']==17
                        else:
                            assert 34<=event['h_actions']<=51
                    elif key[1] in ['uniform','log']:
                        assert chosen==cfg[key[1]+'_additions'][cfg['acquisition_steps'].index(event['step'])]
                    pool.append(chosen)
                assert sorted(pool)==m['parameter_split']['train']
                assert not set(pool)&set(cfg['test_lambda'])
                assert len(rows)==5*(len(pool)+16)
                expected_query_calls=51 if key[1] in ['variance','ritz_gain'] else 0
                assert budget['acquisition_model_calls']==expected_query_calls
                for r in rows:
                    assert r['lam'] in (pool if r['split']=='train' else cfg['test_lambda'])
                for method in ['direct','lookup','sign_map','covariant_average','train_span_ritz']:
                    subset=[r for r in rows if r['method']==method and r['split']=='test']
                    average=sum(1-r['fidelity'] for r in subset)/16
                    assert abs(average-m['metrics'][method]['mean_infidelity'])<1e-14
                for lam in cfg['test_lambda']:
                    for method in ['sign_map','covariant_average']:
                        left=next(r for r in rows if r['lam']==lam and r['method']==method)
                        right=next(r for r in rows if r['lam']==-lam and r['method']==method)
                        assert abs(left['energy']-right['energy'])<1e-10
                        assert abs(left['fidelity']-right['fidelity'])<1e-9
            grid_key='h' if stage=='a' else 'lam'
            for value in {r[grid_key] for r in rows}:
                at={r['method']:r for r in rows if r[grid_key]==value}
                assert at['train_span_ritz']['energy']<=at['lookup']['energy']+1e-9
    assert seen_a=={(a,m,s,b) for a in architectures for m in ['raw','even'] for s in seeds for b in [600,1200,2400]}
    assert seen_b=={(a,p,s) for a in architectures for p in cfg_b['policies'] for s in seeds}
    assert all(len(v)==1 for v in prefixes.values()) and len(prefixes)==10
    replay_status='NOT CHECKED (--without-replay)'
    if not args.without_replay:
        replay=read(ROOT/'docs/results/round2/replay_check.json')
        original=ROOT/'experiments/runs'/replay['original']
        repeated=ROOT/'experiments/runs'/replay['replay']
        for name in ['training.json','metrics.json']:
            assert read(original/name)==read(repeated/name)
        clean=lambda xs:[{k:v for k,v in x.items() if k!='seconds'} for x in xs]
        assert clean(read(original/'acquisition.json'))==clean(read(repeated/'acquisition.json'))
        assert read(original/'manifest.json')['final_state_sha256']==read(repeated/'manifest.json')['final_state_sha256']
        replay_status='exact metrics/history/acquisition/final-weight hash'
    print(f'PASS: {len(seen_a)} stage-A + {len(seen_b)} stage-B jobs, {row_count} rows, budgets/splits/paired prefixes/physical bounds; replay: {replay_status}')


if __name__=='__main__':main()
