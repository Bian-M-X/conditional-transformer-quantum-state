"""Rebuild round-two descriptive statistics; seeds, not grid points, are replicates."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/round2'
POLICIES=['continue','uniform','log','random','variance','ritz_gain']
METHODS=['direct','lookup','sign_map','covariant_average','train_span_ritz']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def csv_write(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    index=read(ROOT/'experiments/round2_index.json')
    assert index['status']=='completed', 'Do not summarize an incomplete suite'
    OUT.mkdir(parents=True,exist_ok=True)
    a,b,raw_b,costs=[],[],[],[]
    for stage,rids in [('a',index['stage_a_reused']+index['stage_a_new']),('b',index['stage_b'])]:
        for rid in rids:
            directory=ROOT/'experiments/runs'/rid
            m=read(directory/'manifest.json');rows=read(directory/'metrics.json')
            assert m['status']=='completed'
            if stage=='a':
                low=next(r for r in rows if r['method']=='direct' and r['h']==.2)
                for method in ['direct','lookup','even_projection','parity_ritz','train_span_ritz']:
                    for split in ['train','interpolation','extrapolation']:
                        a.append(dict(run_id=rid,architecture=m['model']['architecture'],mode=m['model']['training_mode'],
                            seed=m['seed'],steps=m['budget']['gradient_steps'],selected_step=m['model']['selected_step'],
                            method=method,split=split,low_h_direct_fidelity=low['fidelity'],
                            **m['metrics'][split+'/'+method]))
            else:
                for method in METHODS:
                    subset=[r for r in rows if r['split']=='test' and r['method']==method]
                    b.append(dict(run_id=rid,architecture=m['model']['architecture'],policy=m['model']['policy'],
                        seed=m['seed'],method=method,**m['metrics'][method],
                        negative_mean_infidelity=float(np.mean([1-r['fidelity'] for r in subset if r['lam']<0])),
                        positive_mean_infidelity=float(np.mean([1-r['fidelity'] for r in subset if r['lam']>0]))))
                for row in rows:raw_b.append(dict(run_id=rid,**row))
                budget=m['budget']
                costs.append(dict(run_id=rid,architecture=m['model']['architecture'],policy=m['model']['policy'],seed=m['seed'],
                    gradient_steps=budget['gradient_steps'],training_h_actions=budget['training_h_actions'],
                    acquisition_h_actions=budget['acquisition_h_actions'],
                    total_h_actions=budget['total_training_selection_h_actions'],
                    backward_h_actions=budget['backward_h_actions'],total_with_backward_h_actions=budget['total_with_backward_h_actions'],
                    training_model_calls=budget['training_model_calls'],acquisition_model_calls=budget['acquisition_model_calls'],
                    initialization_training_seconds=budget['initialization_training_seconds'],
                    acquisition_seconds=budget['acquisition_seconds'],evaluation_seconds=budget['evaluation_seconds'],
                    wall_seconds=budget['wall_seconds'],selected_lambda=json.dumps(m['parameter_split']['train'])))
    csv_write('stage_a_per_seed.csv',a);csv_write('stage_b_per_seed.csv',b)
    csv_write('stage_b_all_metrics.csv',raw_b);csv_write('stage_b_costs.csv',costs)
    def aggregate(rows,keys,metrics):
        result=[]
        for key in sorted({tuple(r[k] for k in keys) for r in rows}):
            subset=sorted([r for r in rows if tuple(r[k] for k in keys)==key],key=lambda r:r['seed'])
            assert len(subset)==5 and len({r['seed'] for r in subset})==5
            out=dict(zip(keys,key));out['seeds']=[r['seed'] for r in subset]
            for metric in metrics:
                values=[r[metric] for r in subset]
                out[metric]=dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),
                                 min=min(values),max=max(values),values=values)
            result.append(out)
        return result
    sa=aggregate(a,['architecture','mode','steps','method','split'],
        ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','low_h_direct_fidelity'])
    sb=aggregate(b,['architecture','policy','method'],
        ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','mean_magnetization_error',
         'negative_mean_infidelity','positive_mean_infidelity'])
    paired=[]
    for arch in ['mlp','transformer']:
        for method in METHODS:
            for policy in ['random','variance','ritz_gain']:
                for reference in ['continue','uniform','log']:
                    left={r['seed']:r['mean_infidelity'] for r in b if (r['architecture'],r['policy'],r['method'])==(arch,policy,method)}
                    right={r['seed']:r['mean_infidelity'] for r in b if (r['architecture'],r['policy'],r['method'])==(arch,reference,method)}
                    delta=[left[s]-right[s] for s in sorted(left)]
                    paired.append(dict(architecture=arch,method=method,policy=policy,reference=reference,
                        seeds=sorted(left),paired_deltas=delta,mean_delta=float(np.mean(delta)),
                        sd_delta=float(np.std(delta,ddof=1)),wins=sum(d<0 for d in delta),ties=sum(d==0 for d in delta)))
    write('summary.json',dict(stage_a=sa,stage_b=sb,paired=paired))
    lines=['# Round 2 descriptive results','','Five paired seeds; mean +/- sample SD. No confidence intervals or significance claims.',
           'Grid values within one seed are correlated. Stage A is development data.','','## A: direct low-field training fidelity','',
           '| Architecture | Mode | Budget | F at h=0.2 | Seed range |','|---|---|---:|---:|---:|']
    for r in sa:
        if r['method']=='direct' and r['split']=='train':
            v=r['low_h_direct_fidelity']
            lines.append(f"| {r['architecture']} | {r['mode']} | {r['steps']} | {v['mean']:.6g} +/- {v['sd']:.3g} | {v['min']:.6g} to {v['max']:.6g} |")
    lines+=['','## B: held-out mean infidelity','','| Architecture | Policy | Predictor | Mean infidelity | Mean worst infidelity |','|---|---|---|---:|---:|']
    for r in sb:
        v,w=r['mean_infidelity'],r['worst_infidelity']
        lines.append(f"| {r['architecture']} | {r['policy']} | {r['method']} | {v['mean']:.6g} +/- {v['sd']:.3g} | {w['mean']:.6g} +/- {w['sd']:.3g} |")
    lines+=['','## Paired comparisons','','Negative delta is better for the candidate policy. Wins count seeds; no p-values.','',
            '| Architecture | Predictor | Policy | Reference | Mean delta | Wins / 5 |','|---|---|---|---|---:|---:|']
    for r in paired:
        lines.append(f"| {r['architecture']} | {r['method']} | {r['policy']} | {r['reference']} | {r['mean_delta']:.6g} | {r['wins']} |")
    lines+=['','## Measured costs','','Training plus acquisition = 2400 FORWARD H-vector actions in every stage-B run. Backward H work, gradient steps and wall time differ (ADR 0003).',
            'Costs for each seed, selected points, and model calls are in stage_b_costs.csv.','',
            '| Architecture | Policy | Updates (range) | Query H (range) | Wall seconds (mean) |','|---|---|---:|---:|---:|']
    for arch in ['mlp','transformer']:
        for policy in POLICIES:
            rr=[r for r in costs if (r['architecture'],r['policy'])==(arch,policy)]
            lines.append(f"| {arch} | {policy} | {min(r['gradient_steps'] for r in rr)}-{max(r['gradient_steps'] for r in rr)} | {min(r['acquisition_h_actions'] for r in rr)}-{max(r['acquisition_h_actions'] for r in rr)} | {np.mean([r['wall_seconds'] for r in rr]):.3f} |")
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    colors=['#1f638b','#bb5835','#2b8a66','#9362a7','#b18d22']
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    for ax,arch in zip(axes,['mlp','transformer']):
        for mode,color in zip(['raw','even'],colors):
            rr=[r for r in sa if (r['architecture'],r['mode'],r['method'],r['split'])==(arch,mode,'direct','train')]
            rr.sort(key=lambda r:r['steps'])
            x=[r['steps'] for r in rr];v=[r['low_h_direct_fidelity'] for r in rr]
            ax.plot(x,[t['mean'] for t in v],'o-',color=color,label=mode)
            ax.fill_between(x,[t['min'] for t in v],[t['max'] for t in v],color=color,alpha=.15)
        ax.set(title=arch,xlabel='Full optimization updates',ylabel='Fidelity at training h=0.2',ylim=(0,1.02),xticks=[600,1200,2400])
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Stage A: N=6, validation-selected checkpoint; mean and range over 5 seeds')
    fig.savefig(OUT/'budget_fidelity.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,arch in zip(axes,['mlp','transformer']):
        for j,(method,color) in enumerate(zip(METHODS,colors)):
            rr=[next(r for r in sb if (r['architecture'],r['policy'],r['method'])==(arch,p,method)) for p in POLICIES]
            y=np.array([r['mean_infidelity']['mean'] for r in rr])
            lo=np.array([r['mean_infidelity']['min'] for r in rr]);hi=np.array([r['mean_infidelity']['max'] for r in rr])
            ax.errorbar(np.arange(6)+(j-2)*.13,y,yerr=[y-lo,hi-y],fmt='o',ms=4,capsize=2,label=method,color=color)
        ax.set(title=arch,xticks=range(6),xticklabels=POLICIES,ylabel='Mean test infidelity',yscale='log')
        ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.2)
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=8,loc='outside lower center',ncol=5)
    fig.suptitle('Stage B: 2400 forward training + acquisition H actions; mean and seed range (n=5)')
    fig.savefig(OUT/'acquisition_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for i,arch in enumerate(['mlp','transformer']):
        for j,method in enumerate(['direct','covariant_average']):
            ax=axes[i,j]
            for policy,color in zip(['continue','uniform','variance','ritz_gain'],colors):
                rr=[r for r in raw_b if (r['architecture'],r['policy'],r['method'],r['split'])==(arch,policy,method,'test')]
                grid=sorted({r['lam'] for r in rr})
                vals=[[r['fidelity'] for r in rr if r['lam']==lam] for lam in grid]
                ax.plot(grid,[np.mean(v) for v in vals],'.-',label=policy,color=color)
                ax.fill_between(grid,[min(v) for v in vals],[max(v) for v in vals],color=color,alpha=.09)
            ax.set(xlabel='Boundary field lambda / J',ylabel='Ground-state fidelity',
                   title=f'{arch}, {method}',ylim=(0,1.02))
            ax.set_xscale('symlog',linthresh=6e-5)
            ax.set_xticks([-.01,-.001,-.0001,0,.0001,.001,.01],
                          ['-1e-2','-1e-3','-1e-4','0','1e-4','1e-3','1e-2'])
            ax.tick_params(axis='x',labelsize=8)
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Stage B: held-out signed field, mean and seed range (n=5)')
    fig.savefig(OUT/'signed_field.png',dpi=170);plt.close(fig)
    print(f'Summarized {len(a)} stage-A seed endpoints, {len(b)} stage-B seed endpoints, {len(raw_b)} stage-B metric rows')


if __name__=='__main__':main()
