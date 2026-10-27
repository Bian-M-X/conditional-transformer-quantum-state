"""Compare stored numeric outputs, preserving timing/environment differences."""
from pathlib import Path
import argparse
import json
import math

ROOT=Path(__file__).resolve().parents[1]
def compare(a,b,atol=1e-8,rtol=1e-6):
    diffs=[];errors=[];count=0
    def walk(x,y,path=''):
        nonlocal count
        if isinstance(x,dict):
            if set(x)!=set(y):errors.append(path+': keys');return
            for k in x:walk(x[k],y[k],path+'/'+str(k))
        elif isinstance(x,list):
            if len(x)!=len(y):errors.append(path+': length');return
            for i,(xx,yy) in enumerate(zip(x,y)):walk(xx,yy,path+'/'+str(i))
        elif isinstance(x,(int,float)) and not isinstance(x,bool):
            count+=1;d=abs(x-y);diffs.append((d,path))
            if not math.isfinite(y) or d>atol+rtol*abs(x):errors.append(path)
        elif x!=y:errors.append(path+': value')
    walk(a,b)
    largest=max(diffs,default=(0.,''))
    return dict(numeric_values=count,max_absolute_difference=largest[0],max_difference_path=largest[1],
                within_declared_tolerance=not errors,discrepancies=errors[:20])

def main():
    p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--replay',required=True)
    p.add_argument('--output',required=True);p.add_argument('--exact',action='store_true');args=p.parse_args()
    out=dict(original=args.original,replay=args.replay,atol=0. if args.exact else 1e-8,
             rtol=0. if args.exact else 1e-6,scope='agent-run reproduction; not independent human review',checks={})
    files=['metrics.json','training.json']
    if (ROOT/'experiments/runs'/args.original/'amplitudes.json').exists():files.append('amplitudes.json')
    for file in files:
        a=json.loads((ROOT/'experiments/runs'/args.original/file).read_text(encoding='utf-8'))
        b=json.loads((ROOT/'experiments/runs'/args.replay/file).read_text(encoding='utf-8'))
        out['checks'][file]=compare(a,b,out['atol'],out['rtol'])
    out['passed']=all(v['within_declared_tolerance'] for v in out['checks'].values())
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False,indent=2))
    if not out['passed']:raise SystemExit(1)

if __name__=='__main__':main()
