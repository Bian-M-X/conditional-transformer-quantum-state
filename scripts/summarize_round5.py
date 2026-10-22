"""Full descriptive factorial and frozen route gates; no significance claims."""
import csv
import itertools
import json
from pathlib import Path
import statistics as st
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from verify_round5 import read,audit,ROOT
OUT=ROOT/'docs/results/round5'


def csvwrite(name,rows):
    with (OUT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):return dict(mean=st.mean(values),sd=st.stdev(values),minimum=min(values),maximum=max(values),values=values)


def main():
    OUT.mkdir(parents=True,exist_ok=True);evidence=audit('--without-replay' not in sys.argv)
    (OUT/'evidence_audit.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf-8')
    cfg=read(ROOT/'experiments/configs/round5.json');idx=read(ROOT/'experiments/round5_index.json')
    endpoint=[];allrows=[];costs=[];primary={};aggregate=[];pairs=[];gates=[];anchors=[]
    for rid in idx['runs']:
        p=ROOT/'experiments/runs'/rid;m=read(p/'manifest.json');j=m['job'];k=tuple(j[x] for x in ['h','architecture','sharing','optimizer','seed'])
        primary[k]=m['metrics']['train/anchor']['worst_infidelity']
        costs.append(dict(run_id=rid,**j,model_count=m['model']['model_count'],total_parameters=m['model']['total_parameters'],**m['budget']))
        for label,metrics in m['metrics'].items():
            split,method=label.split('/');endpoint.append(dict(run_id=rid,**j,split=split,method=method,**metrics))
        for r in read(p/'metrics.json'):
            allrows.append({'run_id':rid,**j,**r})
            if r['method']=='anchor':anchors.append(dict(run_id=rid,**j,lam=r['x'],infidelity=1-r['fidelity'],magnetization=r['magnetization'],variance=r['variance'],energy_error_per_spin=r['energy_error_per_spin']))
    for h,arch,sharing,opt in itertools.product(cfg['h_values'],cfg['architectures'],cfg['sharing'],cfg['optimizers']):
        vals=[primary[(h,arch,sharing,opt,s)] for s in cfg['seeds']]
        passed=sum(x<=cfg['reliable_worst_infidelity'] for x in vals)
        gates.append(dict(h=h,architecture=arch,sharing=sharing,optimizer=opt,reliable_seeds=passed,pass_gate=passed>=cfg['required_seeds'],**stats(vals)))
        selected=[r for r in endpoint if (r['h'],r['architecture'],r['sharing'],r['optimizer'])==(h,arch,sharing,opt)]
        for split,method in sorted({(r['split'],r['method']) for r in selected}):
            rr=[r for r in selected if (r['split'],r['method'])==(split,method)]
            for metric in ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','worst_correlation_error','mean_magnetization_error']:
                aggregate.append(dict(h=h,architecture=arch,sharing=sharing,optimizer=opt,split=split,method=method,metric=metric,**stats([r[metric] for r in rr])))
    contrasts=[(('shared','adam'),('independent','adam')),(('shared','lbfgs'),('independent','lbfgs')),(('shared','adam'),('shared','lbfgs')),(('independent','adam'),('independent','lbfgs'))]
    for h,arch,(base,alt) in itertools.product(cfg['h_values'],cfg['architectures'],contrasts):
        aa=[primary[(h,arch,*base,s)] for s in cfg['seeds']];bb=[primary[(h,arch,*alt,s)] for s in cfg['seeds']]
        delta=[b-a for a,b in zip(aa,bb)];rescues=sum(a>0 and b<=a/cfg['relative_rescue_factor'] for a,b in zip(aa,bb))
        pairs.append(dict(h=h,architecture=arch,base='/'.join(base),alternative='/'.join(alt),improved=sum(x<0 for x in delta),
            tenfold_rescue_seeds=rescues,pass_relative_gate=rescues>=cfg['required_seeds'],**stats(delta)))
    summary=dict(config=cfg,gates=gates,paired_primary=pairs,aggregates=aggregate,main_wall_seconds=sum(r['wall_seconds'] for r in costs),
       jobs=len(costs),metric_rows=len(allrows),seed_endpoints=len(endpoint),interpretation='Five paired seeds per stratum; development diagnostic; mean and SD, no significance claim.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    csvwrite('all_metrics.csv',allrows);csvwrite('seed_endpoints.csv',endpoint);csvwrite('costs.csv',costs);csvwrite('anchors.csv',anchors)
    csvwrite('paired_primary.csv',[{**{k:v for k,v in r.items() if k!='values'},**{f'seed_{s}':v for s,v in zip(cfg['seeds'],r['values'])}} for r in pairs])
    table=['# Round 5 results','','Worst training-anchor infidelity, mean ± sample SD; five paired seeds. Reliability gate: <=0.01 in >=4/5. No significance claim.','',
      '| h / model | sharing / optimizer | mean ± SD | reliable seeds | pass |','|---|---|---|---|---|']
    for r in gates:table.append(f"| {r['h']} / {r['architecture']} | {r['sharing']} / {r['optimizer']} | {r['mean']:.7g} ± {r['sd']:.3g} | {r['reliable_seeds']}/5 | {r['pass_gate']} |")
    table+=['','## Paired route contrasts','','Alternative minus base, on the worst-anchor primary endpoint. Relative rescue >=10x in >=4/5 seeds.','',
      '| h / model | contrast | mean difference ± SD | improved | >=10x |','|---|---|---|---|---|']
    for r in pairs:table.append(f"| {r['h']} / {r['architecture']} | {r['alternative']} - {r['base']} | {r['mean']:.7g} ± {r['sd']:.3g} | {r['improved']}/5 | {r['tenfold_rescue_seeds']}/5 |")
    table+=['','## All endpoints','','| h / model / sharing / optimizer | split / method / metric | mean ± SD |','|---|---|---|']
    for r in aggregate:table.append(f"| {r['h']} / {r['architecture']} / {r['sharing']} / {r['optimizer']} | {r['split']} / {r['method']} / {r['metric']} | {r['mean']:.7g} ± {r['sd']:.3g} |")
    (OUT/'tables.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True,sharey=True)
    arms=[('shared','adam'),('shared','lbfgs'),('independent','adam'),('independent','lbfgs')]
    for ax,(h,arch) in zip(axes.flat,itertools.product(cfg['h_values'],cfg['architectures'])):
        for i,(sharing,opt) in enumerate(arms):
            vals=[primary[(h,arch,sharing,opt,s)] for s in cfg['seeds']]
            ax.scatter(i+np.linspace(-.08,.08,5),np.maximum(vals,1e-12),s=25)
            ax.plot([i-.18,i+.18],[max(st.mean(vals),1e-12)]*2,color='black',lw=1.5)
        ax.axhline(.01,color='gray',ls='--');ax.set(title=f'N=6, h={h}, {arch}',yscale='log',ylabel='Worst training-anchor infidelity',ylim=(1e-8,1.4))
        ax.set_xticks(range(4),['shared\nAdam','shared\nL-BFGS','independent\nAdam','independent\nL-BFGS']);ax.grid(alpha=.15)
    fig.suptitle('Frozen route gate: five seed points and means; dashed threshold = 0.01')
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'route_gate.png',dpi=180);plt.close(fig)
    print('Wrote',len(allrows),'metric rows;',len(endpoint),'seed endpoints; main wall',summary['main_wall_seconds'])
    for r in gates:print(r)


if __name__=='__main__':main()
