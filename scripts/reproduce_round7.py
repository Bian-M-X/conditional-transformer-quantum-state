"""Audit saved evidence or rerun frozen jobs without replacing the original index."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from compare_round7_replay import compare


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['audit','replay','full'],default='audit')
    args=p.parse_args()
    (ROOT/'tmp').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(ROOT/'tmp'),TMP=str(ROOT/'tmp'),
        MPLCONFIGDIR=str(ROOT/'tmp/mpl'),OPENBLAS_NUM_THREADS='1')
    def run(*parts):
        subprocess.run([sys.executable,'-X','utf8',*parts],cwd=ROOT,env=env,check=True)
    if args.mode=='audit':
        run('-m','unittest','discover','-s','tests','-v')
        run('scripts/validate_workspace.py')
        run('scripts/verify_round7.py')
        for name in ['baseline_replay','round7_replay']:
            d=json.loads((ROOT/'docs/results/round7'/f'{name}.json').read_text(encoding='utf-8'))
            command=['scripts/compare_round7_replay.py','--original',d['original'],'--replay',d['replay'],
                '--output',f'tmp/{name}_audit.json']
            if d['atol']==d['rtol']==0:command.append('--exact')
            run(*command)
        print('Audit passed. This re-scores saved evidence; it does not rerun training.')
        return
    # New run tags exclude all replay jobs from the original 60-job index.
    tag='round7-user-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    command=['scripts/run_round7.py','--tag',tag]
    if args.mode=='replay':command+=['--arch','transformer','--optimizer','lbfgs','--mode','raw','--coverage','wide','--seed','101']
    run(*command)
    index=json.loads((ROOT/'experiments/round7_index.json').read_text(encoding='utf-8'))
    def key(m):return tuple(m['model'][k] for k in ['architecture','optimizer','training_mode','coverage'])+(m['seed'],)
    originals={}
    for rid in index['runs']:
        originals[key(json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text(encoding='utf-8')))]=rid
    checks=[]
    for folder in sorted((ROOT/'experiments/runs').glob('*-'+tag+'-*')):
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        result={'run_id':folder.name,'completed':m['status']=='completed','comparisons':{}}
        if result['completed']:
            original=ROOT/'experiments/runs'/originals[key(m)];result['original']=original.name
            for name in ['metrics.json','training.json','amplitudes.json']:
                a=json.loads((original/name).read_text(encoding='utf-8'))
                b=json.loads((folder/name).read_text(encoding='utf-8'))
                result['comparisons'][name]=compare(a,b,atol=1e-8,rtol=1e-6)
        checks.append(result)
    expected=1 if args.mode=='replay' else 60
    passed=len(checks)==expected and all(q['completed'] and all(v['within_declared_tolerance'] for v in q['comparisons'].values()) for q in checks)
    out=ROOT/'tmp'/f'{tag}_comparison.json'
    out.write_text(json.dumps(dict(mode=args.mode,expected=expected,atol=1e-8,rtol=1e-6,
        passed=passed,checks=checks),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(checks)}/{expected} jobs; comparison passed={passed}; {out}')
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
