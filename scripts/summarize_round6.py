"""Export all cells, baseline comparisons and costs without postselection."""
import csv
import itertools
import json
import statistics as st
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from verify_round6 import ROOT,read,audit
OUT=ROOT/'docs/results/round6'


def write_csv(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)


def stats(v):return dict(mean=st.mean(v),sd=st.stdev(v),minimum=min(v),maximum=max(v),values=v)


def main():
    OUT.mkdir(parents=True,exist_ok=True);evidence=audit('--without-replay' not in sys.argv)
    (OUT/'evidence_audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    cfg=read(ROOT/'experiments/configs/round6.json');data=[];endpoints=[];costs=[];allrows=[]
    for rnd in [5,6]:
        for rid in read(ROOT/f'experiments/round{rnd}_index.json')['runs']:
            p=ROOT/'experiments/runs'/rid;m=read(p/'manifest.json');j=m['job']
            arm=j.get('arm',j['sharing']+'_'+j['optimizer']);meta=dict(round=rnd,run_id=rid,h=j['h'],architecture=j['architecture'],seed=j['seed'],arm=arm)
            data.append(dict(**meta,value=m['metrics']['train/anchor']['worst_infidelity']))
            costs.append(dict(**meta,parameters=m['model']['total_parameters'],**m['budget']))
            for label,values in m['metrics'].items():
                split,method=label.split('/');endpoints.append(dict(**meta,split=split,method=method,**values))
            if rnd==6:
                for r in read(p/'metrics.json'):allrows.append({**meta,**r})
    arms=['shared_adam','shared_lbfgs','independent_adam','independent_lbfgs','shared_sr','independent_sr','wide_lbfgs']
    gates=[];aggregates=[];pairs=[]
    for h,arch,arm in itertools.product(cfg['h_values'],cfg['architectures'],arms):
        rr=sorted([r for r in data if (r['h'],r['architecture'],r['arm'])==(h,arch,arm)],key=lambda r:r['seed'])
        v=[r['value'] for r in rr];success=sum(x<=.01 for x in v)
        gates.append(dict(h=h,architecture=arch,arm=arm,reliable_seeds=success,pass_gate=success>=4,**stats(v)))
        ee=[r for r in endpoints if (r['h'],r['architecture'],r['arm'])==(h,arch,arm)]
        for split,method in sorted({(r['split'],r['method']) for r in ee}):
            qq=[r for r in ee if (r['split'],r['method'])==(split,method)]
            for metric in ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','worst_correlation_error','mean_magnetization_error']:
                aggregates.append(dict(h=h,architecture=arch,arm=arm,split=split,method=method,metric=metric,**stats([r[metric] for r in qq])))
    contrasts=[('shared_lbfgs','shared_sr'),('independent_lbfgs','independent_sr'),('shared_sr','independent_sr'),('shared_lbfgs','wide_lbfgs'),('independent_lbfgs','wide_lbfgs')]
    for h,arch,(base,alt) in itertools.product(cfg['h_values'],cfg['architectures'],contrasts):
        vals={r['arm']:r['values'] for r in gates if r['h']==h and r['architecture']==arch}
        delta=[b-a for a,b in zip(vals[base],vals[alt])]
        pairs.append(dict(h=h,architecture=arch,base=base,alternative=alt,improved=sum(x<0 for x in delta),**stats(delta)))
    summary=dict(config=cfg,gates=gates,paired_primary=pairs,aggregates=aggregates,main_jobs=60,metric_rows=len(allrows),
      new_seed_endpoints=sum(r['round']==6 for r in endpoints),main_wall_seconds=sum(r['wall_seconds'] for r in costs if r['round']==6),
      limitations='Development data. Five seeds per cell. Approximate capacity, different update/initialization and Jacobian costs. No significance claim.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv('all_metrics.csv',allrows);write_csv('seed_endpoints.csv',endpoints);write_csv('costs.csv',costs)
    table=['# Round6 and all round5 controls','','Worst training-anchor infidelity; mean ± sample SD; five seeds. Reliability <=.01 in>=4/5.','',
        '| h / architecture | arm | mean ± SD | reliable seeds |','|---|---|---|---|']
    for r in gates:table.append(f"| {r['h']} / {r['architecture']} | {r['arm']} | {r['mean']:.7g} ± {r['sd']:.3g} | {r['reliable_seeds']}/5 |")
    table+=['','## All endpoints','','| h / architecture / arm | split / method / metric | mean ± SD |','|---|---|---|']
    for r in aggregates:table.append(f"| {r['h']} / {r['architecture']} / {r['arm']} | {r['split']} / {r['method']} / {r['metric']} | {r['mean']:.7g} ± {r['sd']:.3g} |")
    (OUT/'tables.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    shown=['shared_lbfgs','independent_lbfgs','shared_sr','independent_sr','wide_lbfgs']
    fig,axes=plt.subplots(2,2,figsize=(11,6.6),sharex=True,sharey=True)
    for ax,(h,arch) in zip(axes.flat,itertools.product(cfg['h_values'],cfg['architectures'])):
        for i,arm in enumerate(shown):
            r=next(r for r in gates if (r['h'],r['architecture'],r['arm'])==(h,arch,arm))
            ax.scatter(i+np.linspace(-.08,.08,5),np.maximum(r['values'],1e-12),s=24)
            ax.plot([i-.16,i+.16],[r['mean']]*2,color='black')
        ax.axhline(.01,ls='--',c='gray');ax.set(yscale='log',ylim=(1e-8,1.4),title=f'N=6, h={h}, {arch}',ylabel='Worst training-anchor infidelity')
        ax.set_xticks(range(5),['shared\nL-BFGS','independent\nL-BFGS','shared\nSR','independent\nSR','wide shared\nL-BFGS']);ax.tick_params(labelsize=8);ax.grid(alpha=.15)
    fig.suptitle('Development gate: individual seeds and means; distinct costs and capacities')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(OUT/'sr_capacity_gate.png',dpi=180);plt.close(fig)
    print('Wrote',len(allrows),'rows;',summary['new_seed_endpoints'],'new endpoints; seconds',summary['main_wall_seconds'])
    for r in gates:
        if r['arm'] in cfg['arms']:print(r)


if __name__=='__main__':main()
