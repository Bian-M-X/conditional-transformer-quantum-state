"""Seed-level factorial summaries; no grid-point pseudoreplication."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/round3'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(name,value):(OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    index=read(ROOT/'experiments/round3_index.json');assert index['status']=='completed'
    OUT.mkdir(parents=True,exist_ok=True);endpoints=[];low=[];raw=[];costs=[]
    for rid in index['runs']:
        directory=ROOT/'experiments/runs'/rid;m=read(directory/'manifest.json');j=m['job'];rr=read(directory/'metrics.json')
        for name,values in m['metrics'].items():
            checkpoint,split,method=name.split('/')
            endpoints.append(dict(run_id=rid,**j,checkpoint=checkpoint,split=split,method=method,selected_step=m['model']['selected_step'],**values))
        for r in rr:
            raw.append(dict(run_id=rid,**j,**r))
            if j['stage']=='a' and r['h']==.2 and r['method']=='direct':
                low.append(dict(run_id=rid,architecture=j['architecture'],mode=j['mode'],seed=j['seed'],checkpoint=r['checkpoint'],fidelity=r['fidelity'],selected_step=m['model']['selected_step']))
        costs.append(dict(run_id=rid,**j,**m['budget']))
    csv_write('per_seed.csv',endpoints);csv_write('all_metrics.csv',raw);csv_write('low_h_checkpoints.csv',low);csv_write('costs.csv',costs)
    keys=['stage','n','architecture','scale','mode','checkpoint','split','method'];summary=[]
    for identity in sorted({tuple(r[k] for k in keys) for r in endpoints}):
        rr=sorted([r for r in endpoints if tuple(r[k] for k in keys)==identity],key=lambda r:r['seed']);assert len(rr)==5
        out=dict(zip(keys,identity));out['seeds']=[r['seed'] for r in rr]
        for name in ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','mean_magnetization_error','worst_correlation_error']:
            values=[r[name] for r in rr];out[name]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),min=min(values),max=max(values),values=values)
        summary.append(out)
    paired=[]
    def compare(label,left,right,metadata):
        a={r['seed']:r['mean_infidelity'] for r in left};b={r['seed']:r['mean_infidelity'] for r in right};assert a.keys()==b.keys() and len(a)==5
        delta=[a[s]-b[s] for s in sorted(a)]
        paired.append(dict(comparison=label,**metadata,seeds=sorted(a),deltas=delta,mean_delta=float(np.mean(delta)),sd_delta=float(np.std(delta,ddof=1)),wins=sum(d<0 for d in delta)))
    test=[r for r in endpoints if r['stage']=='b' and r['split']=='test']
    for n in [4,6]:
        for arch in ['mlp','transformer']:
            for checkpoint in ['selected','final']:
                rr=[r for r in test if (r['n'],r['architecture'],r['checkpoint'])==(n,arch,checkpoint)]
                meta=dict(n=n,architecture=arch,checkpoint=checkpoint)
                for mode in ['raw','covariant']:
                    compare('scaled minus physical',[r for r in rr if r['mode']==mode and r['scale']==.04 and r['method']=='direct'],
                        [r for r in rr if r['mode']==mode and r['scale']==1. and r['method']=='direct'],dict(**meta,mode=mode))
                for scale in [1.,.04]:
                    left=[r for r in rr if r['mode']=='covariant' and r['scale']==scale and r['method']=='direct']
                    for method in ['direct','post_covariant']:
                        compare('covariant training minus raw '+method,left,[r for r in rr if r['mode']=='raw' and r['scale']==scale and r['method']==method],dict(**meta,scale=scale))
            for scale in [1.,.04]:
                for mode in ['raw','covariant']:
                    rr=[r for r in test if (r['n'],r['architecture'],r['scale'],r['mode'],r['method'])==(n,arch,scale,mode,'direct')]
                    compare('final minus selected',[r for r in rr if r['checkpoint']=='final'],[r for r in rr if r['checkpoint']=='selected'],dict(n=n,architecture=arch,scale=scale,mode=mode))
    write('summary.json',dict(aggregates=summary,paired=paired))
    lines=['# Round 3 descriptive results','','Five paired seeds. Mean +/- sample SD, not confidence intervals. Negative paired deltas favor the first condition.','',
           '## A: low-field training fidelity','','| Architecture | Training | Checkpoint | Mean F at h=0.2 | Seed range |','|---|---|---|---:|---:|']
    for arch in ['mlp','transformer']:
        for mode in ['raw','even']:
            for checkpoint in ['selected','final']:
                v=[r['fidelity'] for r in low if (r['architecture'],r['mode'],r['checkpoint'])==(arch,mode,checkpoint)]
                lines.append(f'| {arch} | {mode} | {checkpoint} | {np.mean(v):.7g} +/- {np.std(v,ddof=1):.3g} | {min(v):.7g} to {max(v):.7g} |')
    lines+=['','## B: validation-selected test infidelity','','| N | Architecture | Scale | Training | Predictor | Mean infidelity | Mean worst infidelity |','|---:|---|---:|---|---|---:|---:|']
    for r in summary:
        if (r['stage'],r['checkpoint'],r['split'])!=('b','selected','test'):continue
        v,w=r['mean_infidelity'],r['worst_infidelity']
        lines.append(f"| {r['n']} | {r['architecture']} | {r['scale']} | {r['mode']} | {r['method']} | {v['mean']:.7g} +/- {v['sd']:.3g} | {w['mean']:.7g} +/- {w['sd']:.3g} |")
    lines+=['','## Paired changes in mean test infidelity','','| Comparison | N | Architecture | Checkpoint | Scale/mode | Mean delta | Wins/5 |','|---|---:|---|---|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['comparison']} | {r['n']} | {r['architecture']} | {r.get('checkpoint','paired')} | {r.get('scale','')}/{r.get('mode','')} | {r['mean_delta']:.7g} | {r['wins']} |")
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,arch in zip(axes,['mlp','transformer']):
        groups=[('raw','selected'),('raw','final'),('even','selected'),('even','final')]
        for i,(mode,checkpoint) in enumerate(groups):
            values=[r['fidelity'] for r in low if (r['architecture'],r['mode'],r['checkpoint'])==(arch,mode,checkpoint)]
            ax.scatter(np.arange(5)*.025+i-.05,values,s=25,alpha=.7)
            ax.plot([i-.13,i+.13],[np.mean(values)]*2,color='black',lw=2)
        ax.set(title=arch,xticks=range(4),xticklabels=['raw\nselected','raw\nfinal','even\nselected','even\nfinal'],ylabel='Fidelity at training h=0.2',ylim=(0,1.03));ax.grid(alpha=.2)
    fig.suptitle('A: same 2400-update trajectories; five seed points and means')
    fig.savefig(OUT/'checkpoint_audit.png',dpi=170);plt.close(fig)
    groups=[(1.,'raw'),(.04,'raw'),(1.,'covariant'),(.04,'covariant')]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,n in enumerate([4,6]):
        for j,arch in enumerate(['mlp','transformer']):
            ax=axes[i,j]
            for k,(checkpoint,method,color) in enumerate([('selected','direct','#23648b'),('final','direct','#bf5835'),('selected','post_covariant','#2c8c65'),('selected','lookup','#8d62a3')]):
                rr=[next(r for r in summary if (r['stage'],r['n'],r['architecture'],r['scale'],r['mode'],r['checkpoint'],r['split'],r['method'])==('b',n,arch,scale,mode,checkpoint,'test',method)) for scale,mode in groups]
                v=[r['mean_infidelity'] for r in rr];mean=np.array([r['mean'] for r in v])
                ax.errorbar(np.arange(4)+(k-1.5)*.13,mean,yerr=[mean-[r['min'] for r in v],[r['max'] for r in v]-mean],fmt='o',ms=4,capsize=2,color=color,label=checkpoint+'/'+method)
            ax.set(title=f'N={n}, {arch}',xticks=range(4),xticklabels=['physical\nraw','scaled\nraw','physical\ncovariant','scaled\ncovariant'],yscale='log',ylabel='Mean test infidelity');ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',fontsize=8)
    fig.suptitle('B: h=0.5; 7296 training/selection H actions including backward; mean and seed range')
    fig.savefig(OUT/'factorial_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for i,n in enumerate([4,6]):
        for j,arch in enumerate(['mlp','transformer']):
            ax=axes[i,j]
            for (scale,mode),color in zip(groups,['#23648b','#bf5835','#2c8c65','#8d62a3']):
                rr=[r for r in raw if (r['stage'],r['n'],r['architecture'],r['scale'],r['mode'],r['checkpoint'],r['method'],r['split'])==('b',n,arch,scale,mode,'selected','direct','test')]
                xs=sorted({r['x'] for r in rr});values=[[r['fidelity'] for r in rr if r['x']==x] for x in xs]
                ax.plot(xs,[np.mean(v) for v in values],'.-',color=color,label=f'{scale:g}/{mode}')
                ax.fill_between(xs,[min(v) for v in values],[max(v) for v in values],color=color,alpha=.1)
            ax.set(title=f'N={n}, {arch}',xlabel='lambda / J',ylabel='Ground-state fidelity',ylim=(0,1.03))
            ax.set_xscale('symlog',linthresh=.0002);ax.set_xticks([-.01,-.001,0,.001,.01],['-1e-2','-1e-3','0','1e-3','1e-2']);ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc='outside lower center',fontsize=8)
    fig.suptitle('B: validation-selected direct prediction; mean and range across five seeds')
    fig.savefig(OUT/'signed_field.png',dpi=170);plt.close(fig)
    print(f'Wrote {len(endpoints)} seed endpoints, {len(raw)} metric rows, {len(paired)} paired comparisons')


if __name__=='__main__':main()
