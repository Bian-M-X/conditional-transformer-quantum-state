"""Rebuild all round7 tables and figures from the frozen statistical index."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics as st
import os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/mpl'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/results/round7'
REGIONS=['common_interpolation','coverage_extension','common_extrapolation','all_test']
METHODS=['direct','lookup','train_span_ritz']
COLORS={'narrow':'#126782','wide':'#D2693C'}

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def describe(values):
    return dict(mean=st.mean(values),sample_sd=st.stdev(values) if len(values)>1 else 0.,
                minimum=min(values),maximum=max(values),n_seeds=len(values),values=values)
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.png'),dpi=600)
    plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    c=json.loads((ROOT/'experiments/configs/round7.json').read_text(encoding='utf-8'))
    idx=json.loads((ROOT/'experiments/round7_index.json').read_text(encoding='utf-8'))
    if len(idx['runs'])+len(idx['failed_runs'])!=60:raise ValueError('Wait for all 60 attempted jobs')
    rows=[];manifests=[];endpoints=[];costs=[]
    for rid in idx['runs']:
        folder=ROOT/'experiments/runs'/rid
        m=json.loads((folder/'manifest.json').read_text(encoding='utf-8'));manifests.append(m)
        local=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
        for r in local:rows.append(dict(run_id=rid,**r))
        meta=dict(run_id=rid,architecture=m['model']['architecture'],optimizer=m['model']['optimizer'],
                  training_mode=m['model']['training_mode'],coverage=m['model']['coverage'],seed=m['seed'])
        costs.append(dict(**meta,**m['budget']))
        for ep in ['selected','final']:
            for region in ['train','validation']+REGIONS:
                for method in METHODS:
                    rr=[r for r in local if r['endpoint']==ep and r['method']==method and
                        (r['region']==region if region!='all_test' else r['region'] in REGIONS[:-1])]
                    endpoints.append(dict(**meta,endpoint=ep,region=region,method=method,n_points=len(rr),
                        mean_infidelity=st.mean(1-r['fidelity'] for r in rr),worst_infidelity=max(1-r['fidelity'] for r in rr),
                        mean_energy_error_per_spin=st.mean(r['energy_error_per_spin'] for r in rr),
                        mean_abs_mz_error=st.mean(r['abs_mz_error'] for r in rr),
                        worst_correlation_error=max(r['correlation_error'] for r in rr)))
    write_csv(OUT/'all_metrics.csv',rows);write_csv(OUT/'seed_endpoints.csv',endpoints);write_csv(OUT/'costs.csv',costs)
    grouped=defaultdict(list)
    for r in endpoints:grouped[(r['architecture'],r['optimizer'],r['training_mode'],r['coverage'],r['endpoint'],r['region'],r['method'])].append(r)
    summaries=[];gate=[]
    for key,rr in sorted(grouped.items()):
        rr=sorted(rr,key=lambda x:x['seed'])
        record=dict(zip(['architecture','optimizer','training_mode','coverage','endpoint','region','method'],key))
        record.update({metric:describe([r[metric] for r in rr]) for metric in
                       ['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','mean_abs_mz_error','worst_correlation_error']})
        summaries.append(record)
        if key[4:]==('selected','train','direct'):
            vals=[r['worst_infidelity'] for r in rr]
            gate.append(dict(architecture=key[0],optimizer=key[1],training_mode=key[2],coverage=key[3],
                             pass_count=sum(v<=c['reliable_anchor_infidelity'] for v in vals),
                             passes=sum(v<=c['reliable_anchor_infidelity'] for v in vals)>=4,
                             worst_anchor_infidelity=describe(vals),seeds=[r['seed'] for r in rr]))
    coverage=[];lookup_pairs=[]
    for arch in c['architectures']:
        for opt in c['optimizers_by_architecture'][arch]:
            for mode in c['training_modes']:
                for region in REGIONS:
                    parts={cover:sorted(grouped[(arch,opt,mode,cover,'selected',region,'direct')],key=lambda x:x['seed']) for cover in c['train_sets']}
                    diffs=[w['mean_infidelity']-n['mean_infidelity'] for n,w in zip(parts['narrow'],parts['wide'])]
                    coverage.append(dict(architecture=arch,optimizer=opt,training_mode=mode,region=region,
                                         wide_minus_narrow=describe(diffs),wide_wins=sum(d<0 for d in diffs)))
                    for cover in c['train_sets']:
                        dd=sorted(grouped[(arch,opt,mode,cover,'selected',region,'direct')],key=lambda x:x['seed'])
                        ll=sorted(grouped[(arch,opt,mode,cover,'selected',region,'lookup')],key=lambda x:x['seed'])
                        diffs=[d['mean_infidelity']-l['mean_infidelity'] for d,l in zip(dd,ll)]
                        lookup_pairs.append(dict(architecture=arch,optimizer=opt,training_mode=mode,coverage=cover,region=region,
                                                 direct_minus_lookup=describe(diffs),direct_wins=sum(d<0 for d in diffs)))
    primary=next(x for x in coverage if x['architecture']=='transformer' and x['optimizer']=='lbfgs' and x['training_mode']=='raw' and x['region']=='common_extrapolation')
    pg=[x for x in gate if x['architecture']=='transformer' and x['optimizer']=='lbfgs' and x['training_mode']=='raw']
    primary['training_gate_passes_both_coverages']=all(x['passes'] for x in pg)
    primary['prespecified_improvement_gate_passes']=(primary['training_gate_passes_both_coverages'] and
                    primary['wide_wins']>=c['required_paired_wins'] and primary['wide_minus_narrow']['mean']<0)
    summary=dict(completed_jobs=len(idx['runs']),failed_jobs=len(idx['failed_runs']),metric_rows=len(rows),
        endpoint_rows=len(endpoints),statistical_unit='seed; five seeds per cell; descriptive summaries',
        primary=primary,anchor_gates=gate,coverage_pairs=coverage,lookup_pairs=lookup_pairs,cells=summaries,
        wall_seconds_sum=sum(m['budget']['wall_seconds'] for m in manifests),
        training_seconds_sum=sum(m['budget']['training_seconds'] for m in manifests),
        training_h_actions_sum=sum(m['budget']['training_forward_h_actions']+m['budget']['training_backward_h_actions'] for m in manifests),
        max_norm_error=max(r['norm_error'] for r in rows),minimum_energy_error_per_spin=min(r['energy_error_per_spin'] for r in rows),
        source_commits=sorted(set(m['git_commit'] for m in manifests)))
    dump(OUT/'summary.json',summary)
    write_csv(OUT/'paired_coverage.csv',[dict(architecture=x['architecture'],optimizer=x['optimizer'],training_mode=x['training_mode'],region=x['region'],
         mean_difference=x['wide_minus_narrow']['mean'],wide_wins=x['wide_wins'],**{f'seed_{s}':v for s,v in zip(c['seeds'],x['wide_minus_narrow']['values'])}) for x in coverage])
    lines=['# 第七轮完整统计','', '统计单位为种子，均值与样本SD为描述量。详见CSV与summary.json。','',
           '|模型|优化器|训练|覆盖|训练达标数|最差锚点1-F均值|','|---|---|---|---|---|---|']
    for g in gate:lines.append(f"|{g['architecture']}|{g['optimizer']}|{g['training_mode']}|{g['coverage']}|{g['pass_count']}/5|{g['worst_anchor_infidelity']['mean']:.8g}|")
    lines+=['','## 各区域与方法','', '|模型/优化器/训练/覆盖|区域|方法|平均1-F（均值 ± SD）|','|---|---|---|---|']
    for x in summaries:
        if x['endpoint']=='selected' and x['region'] in REGIONS:
            d=x['mean_infidelity'];lines.append(f"|{x['architecture']}/{x['optimizer']}/{x['training_mode']}/{x['coverage']}|{x['region']}|{x['method']}|{d['mean']:.8g} ± {d['sample_sd']:.4g}|")
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
        'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'savefig.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.17})
    mm=1/25.4
    fig,ax=plt.subplots(figsize=(170*mm,65*mm),layout='constrained')
    for y,(cover,train) in enumerate(c['train_sets'].items()):
        ax.plot([train[0],train[-1]],[y,y],lw=13,alpha=.12,color=COLORS[cover])
        ax.scatter(train,[y]*3,c=COLORS[cover],s=45,marker='^',label=cover+' training')
    for j,(region,pts) in enumerate(c['test_regions'].items()):
        ax.scatter(pts,[-1-j*.45]*len(pts),s=15,label=region.replace('_',' '))
    ax.scatter(c['validation'],[1.65]*2,c='#333333',marker='x',s=35,label='validation')
    ax.set(xlim=(.03,1.9),xlabel='Transverse field h / J',yticks=[0,1,1.65,-1,-1.45,-1.9],
           yticklabels=['narrow','wide','validation','common interior','extension','common exterior'])
    ax.grid(axis='y',visible=False)
    save(fig,'parameter_split')
    fig,axes=plt.subplots(1,2,figsize=(170*mm,90*mm),layout='constrained',sharey=True)
    for ax,opt in zip(axes,['adam','lbfgs']):
        for j,(mode,cover) in enumerate([(m,r) for m in ['raw','even'] for r in ['narrow','wide']]):
            g=next(x for x in gate if x['architecture']=='transformer' and x['optimizer']==opt and x['training_mode']==mode and x['coverage']==cover)
            vals=g['worst_anchor_infidelity']['values'];ax.scatter(j+np.linspace(-.12,.12,5),np.maximum(vals,1e-16),color=COLORS[cover],s=22)
            ax.plot([j-.2,j+.2],[max(st.mean(vals),1e-16)]*2,c='black',lw=1.4)
        ax.axhline(.01,c='#444444',ls='--',lw=.9);ax.set_yscale('log')
        ax.set(title=opt.upper(),xticks=range(4),xticklabels=['raw\nnarrow','raw\nwide','even\nnarrow','even\nwide'])
    axes[0].set_ylabel('Worst training-anchor infidelity')
    save(fig,'anchor_reliability')
    fig,axes=plt.subplots(2,3,figsize=(174*mm,140*mm),layout='constrained')
    for ii,mode in enumerate(['raw','even']):
        for jj,(field,ylabel) in enumerate([('fidelity','Infidelity 1 - F'),('energy_error_per_spin','Energy error / spin [J]'),('abs_mz_error','Absolute |m_z| error')]):
            ax=axes[ii,jj]
            for cover in ['narrow','wide']:
                for method in ['direct','lookup']:
                    part=[r for r in rows if r['architecture']=='transformer' and r['optimizer']=='lbfgs' and r['training_mode']==mode and
                          r['coverage']==cover and r['endpoint']=='selected' and r['method']==method and r['region'] in REGIONS[:-1]]
                    hs=sorted({r['h'] for r in part});vv=np.array([[((1-r[field]) if field=='fidelity' else r[field]) for r in sorted(part,key=lambda x:x['seed']) if r['h']==h] for h in hs])
                    ax.plot(hs,np.maximum(vv.mean(1),1e-16),color=COLORS[cover],ls='-' if method=='direct' else '--',lw=1.05,
                            marker=('o' if cover=='narrow' else 's') if method=='direct' else None,ms=3.5,label=cover+' '+method)
                    if method=='direct':ax.fill_between(hs,np.maximum(vv.min(1),1e-16),np.maximum(vv.max(1),1e-16),color=COLORS[cover],alpha=.12,lw=0)
                for h in c['train_sets'][cover]:ax.axvline(h,color=COLORS[cover],alpha=.2,lw=.6)
            ax.set_yscale('log');ax.set_xlabel('h / J');ax.set_ylabel(ylabel);ax.set_title(mode+' training')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    save(fig,'transformer_generalization')
    fig,axes=plt.subplots(1,2,figsize=(170*mm,90*mm),layout='constrained')
    short=['common\ninterior','coverage\nextension','common\nexterior']
    for ax,kind in zip(axes,['coverage','lookup']):
        for mode,offset in [('raw',-.1),('even',.1)]:
            for j,region in enumerate(REGIONS[:-1]):
                if kind=='coverage':
                    x=next(v for v in coverage if v['architecture']=='transformer' and v['optimizer']=='lbfgs' and v['training_mode']==mode and v['region']==region)
                    vals=x['wide_minus_narrow']['values']
                else:
                    x=next(v for v in lookup_pairs if v['architecture']=='transformer' and v['optimizer']=='lbfgs' and v['training_mode']==mode and v['coverage']=='wide' and v['region']==region)
                    vals=x['direct_minus_lookup']['values']
                color='#126782' if mode=='raw' else '#7356A1'
                ax.scatter(j+offset+np.linspace(-.04,.04,5),vals,s=20,color=color,marker='o' if mode=='raw' else 's',label=mode if j==0 else None)
                ax.plot([j+offset-.08,j+offset+.08],[st.mean(vals)]*2,c='black',lw=1)
        ax.axhline(0,c='#555555',lw=.8);ax.set_yscale('symlog',linthresh=1e-4)
        ax.set(xticks=range(3),xticklabels=short,ylabel='Paired infidelity difference')
        ax.set_title('Wide - narrow' if kind=='coverage' else 'Direct - lookup (wide)');ax.legend()
    save(fig,'paired_comparisons')
    fig,ax=plt.subplots(figsize=(130*mm,78*mm),layout='constrained')
    labels=[]
    for j,(mode,cover) in enumerate([(m,r) for m in ['raw','even'] for r in ['narrow','wide']]):
        labels.append(mode+'\n'+cover)
        for k,method in enumerate(METHODS):
            vals=[x['mean_infidelity'] for x in grouped[('transformer','lbfgs',mode,cover,'selected','all_test',method)]]
            color=['#126782','#D2693C','#7356A1'][k];x=j+(k-1)*.19
            ax.scatter(x+np.linspace(-.025,.025,5),np.maximum(vals,1e-16),c=color,s=17,label=method if j==0 else None)
            ax.plot([x-.055,x+.055],[max(st.mean(vals),1e-16)]*2,c='black',lw=.9)
    ax.set_yscale('log');ax.set(xticks=range(4),xticklabels=labels,ylabel='Mean held-out infidelity (20 h values)');ax.legend()
    save(fig,'strong_baselines')
    fig,ax=plt.subplots(figsize=(170*mm,70*mm),layout='constrained');ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=color,ec='#44616d',lw=.7))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    box(.015,.65,.17,.22,'h condition\nBOS, s1, ..., s5','#dcecf2')
    box(.235,.65,.20,.22,'7 x 16 embeddings\n+ positions','#dcecf2')
    box(.495,.55,.25,.42,'1 causal encoder block\n2 attention heads\nFFN 16 - 32 - 16\nGELU + residual/norm','#e9e2f3')
    box(.805,.65,.18,.22,'6 Bernoulli logits\nconditional p_i','#f5e5d8')
    box(.43,.1,.35,.20,'psi(s; h) = sqrt[ product_i p_i ]\n64 configurations, mean energy / 6','#deecdf')
    for x1,y1,x2,y2 in [(.185,.76,.235,.76),(.435,.76,.495,.76),(.745,.76,.805,.76),(.89,.65,.74,.31)]:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1,color='#44616d'))
    ax.text(.02,.32,'p(s_i | h, s_<i)\nCausal mask + right shift\nDropout = 0',fontsize=9,va='center')
    ax.text(.80,.18,'Even control:\nnormalize\n(psi + P psi)',fontsize=9,va='center')
    save(fig,'model_correspondence')
    print(json.dumps({'completed':summary['completed_jobs'],'rows':len(rows),'primary':primary,'gates':gate,
                      'wall_seconds':summary['wall_seconds_sum']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
