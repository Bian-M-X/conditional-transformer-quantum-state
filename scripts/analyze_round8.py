"""Complete frozen R/A pilot analysis; hindsight headroom is not a policy speedup.

Core functions are pure and tested on synthetic arrays. This entrypoint requires
the complete 40-job factorial. It never trains or changes the frozen protocol.
"""
import csv
import itertools
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results/round8'
ACTION_ORDER = ['stop', 'refine_left', 'refine_right', 'acquire_center']
POLICY_ORDER = ACTION_ORDER + ['anchor_residual', 'snapshot_residual']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def clipped_infidelity(fidelity):
    """Only negative roundoff in 1-F is clipped; preserve raw rows separately."""
    return max(0., 1. - float(fidelity))


def describe(values):
    return dict(mean=st.mean(values), sd=st.stdev(values) if len(values) > 1 else 0.,
                minimum=min(values), maximum=max(values), values=list(values))


def policy_losses(losses, choices, policy_order):
    """losses[action][seed index], choices[adaptive policy][seed index]."""
    if not losses or any(not values for values in losses.values()):
        raise ValueError('nonempty action loss arrays required')
    size = len(next(iter(losses.values())))
    if any(len(v) != size for v in losses.values()):
        raise ValueError('action arrays must have the same seed count')
    if any(not math.isfinite(x) or x < 0 for values in losses.values() for x in values):
        raise ValueError('finite nonnegative losses required')
    result = {}
    for name in policy_order:
        if name in losses:
            result[name] = list(losses[name])
        else:
            selected = choices[name]
            if len(selected) != size or any(action not in losses for action in selected):
                raise ValueError('policy choices must select one recorded action per seed')
            result[name] = [losses[action][i] for i, action in enumerate(selected)]
    return result


def compute_headroom(losses, choices=None, policy_order=None, seeds=None):
    """Recompute the strongest empirical simple policy inside every LOO subset.

    B=min_policy mean_seed loss; O=mean_seed min_action loss. Both comparator
    selection and O are hindsight pilot summaries, never deployed policies.
    """
    choices = choices or {}
    policy_order = list(policy_order or losses)
    policies = policy_losses(losses, choices, policy_order)
    n = len(next(iter(losses.values())))
    if n < 2:
        raise ValueError('at least two seeds required for leave-one-out analysis')
    seeds = list(range(n)) if seeds is None else list(seeds)
    if len(seeds) != n or len(set(seeds)) != n:
        raise ValueError('one unique identifier per paired seed required')
    actions = list(losses)

    def subset(indices):
        means = {name: st.mean(values[i] for i in indices)
                 for name, values in policies.items()}
        best = min(policy_order, key=lambda name: means[name])
        oracle = [min(losses[a][i] for a in actions) for i in indices]
        B, O = means[best], st.mean(oracle)
        gap = max(0., B - O)
        return dict(best_simple_policy=best, baseline_mean=B, oracle_mean=O,
                    absolute_headroom=gap, relative_headroom=gap/B if B > 0 else 0.,
                    simple_policy_means=means)

    result = subset(list(range(n)))
    best_values = policies[result['best_simple_policy']]
    oracle_values = [min(losses[a][i] for a in actions) for i in range(n)]
    oracle_actions = [min(actions, key=lambda a: losses[a][i]) for i in range(n)]
    loo = [dict(omitted_seed=seeds[omitted], **subset([i for i in range(n) if i != omitted]))
           for omitted in range(n)]
    result.update(seeds=seeds, baseline_values=best_values, oracle_values=oracle_values,
                  oracle_actions=oracle_actions,
                  paired_improvements=[a-b for a, b in zip(best_values, oracle_values)],
                  leave_one_out=loo,
                  minimum_leave_one_out_headroom=min(r['relative_headroom'] for r in loo),
                  maximum_leave_one_out_headroom=max(r['relative_headroom'] for r in loo))
    return result


def refine_acquire_winners(losses, seeds, margin):
    """Require a type to beat BOTH the other training type and STOP strictly."""
    rows = []
    for i, seed in enumerate(seeds):
        left, right = losses['refine_left'][i], losses['refine_right'][i]
        R = min(left, right); A = losses['acquire_center'][i]; S = losses['stop'][i]
        r_margin = min(A-R, S-R); a_margin = min(R-A, S-A)
        winner = 'refine' if r_margin > margin else 'acquire' if a_margin > margin else 'none'
        rows.append(dict(seed=seed, best_refine=R, acquire=A, stop=S,
                         best_refine_action='refine_left' if left <= right else 'refine_right',
                         refine_advantage_against_A_and_STOP=r_margin,
                         acquire_advantage_against_R_and_STOP=a_margin,
                         winner_type=winner))
    return dict(refine_winner_seed_ids=[r['seed'] for r in rows if r['winner_type']=='refine'],
                acquire_winner_seed_ids=[r['seed'] for r in rows if r['winner_type']=='acquire'],
                rows=rows)


def gate_stratum(record, gate):
    eligible = (record['baseline_mean'] >= gate['minimum_baseline_error'] and
                record['relative_headroom'] >= gate['minimum_relative_headroom'] and
                record['minimum_leave_one_out_headroom'] >= gate['minimum_leave_one_out_headroom'])
    reasons = []
    if record['stop_mean'] < gate['minimum_baseline_error']:
        reasons.append('STOP already below baseline-error floor')
    if record['baseline_mean'] < gate['minimum_baseline_error']:
        reasons.append('strongest simple baseline below error floor')
    if record['relative_headroom'] < gate['minimum_relative_headroom']:
        reasons.append('relative headroom below threshold')
    if record['minimum_leave_one_out_headroom'] < gate['minimum_leave_one_out_headroom']:
        reasons.append('leave-one-out support below threshold')
    winner_actions = set(record['oracle_actions'])
    if {'refine_left','refine_right'} <= winner_actions and not record['acquire_winner_seed_ids']:
        reasons.append('left/right refinement selection does not establish R/A switching')
    return dict(eligible=eligible, failure_or_caveat_categories=reasons)


def summarize_primary_gate(records, gate):
    """Records supplied here must already be primary maturity/variant cells."""
    eligible = [r for r in records if r['eligible']]
    families = sorted({r['family'] for r in eligible})
    r_ids = sorted({s for r in eligible for s in r['refine_winner_seed_ids']})
    a_ids = sorted({s for r in eligible for s in r['acquire_winner_seed_ids']})
    enough_cells = len(eligible) >= gate['minimum_eligible_strata']
    enough_families = len(families) >= gate['required_families']
    enough_R = len(r_ids) >= gate['minimum_winner_seed_ids_per_type']
    enough_A = len(a_ids) >= gate['minimum_winner_seed_ids_per_type']
    return dict(pass_gate=enough_cells and enough_families and enough_R and enough_A,
                eligible_strata=len(eligible), eligible_families=families,
                eligible_keys=[{k:r[k] for k in ['family','radius','architecture']} for r in eligible],
                refine_winner_seed_ids=r_ids, acquire_winner_seed_ids=a_ids,
                enough_eligible_strata=enough_cells, enough_families=enough_families,
                enough_refine_seed_ids=enough_R, enough_acquire_seed_ids=enough_A,
                interpretation='Prospective investment gate only; no significance or deployable-speedup claim.')


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def endpoint(rows):
    if len(rows) != 8:
        raise ValueError('eight predeclared query rows required')
    inf = [clipped_infidelity(r['fidelity']) for r in rows]
    return dict(mean_infidelity=st.mean(inf), worst_infidelity=max(inf),
                mean_energy_error_per_spin=st.mean(r['energy_error_per_spin'] for r in rows),
                mean_energy_error=st.mean(r['energy_error'] for r in rows),
                mean_variance=st.mean(r['variance'] for r in rows),
                mean_magnetization_error=st.mean(r['magnetization_error'] for r in rows),
                mean_correlation_error=st.mean(r['correlation_error'] for r in rows),
                worst_correlation_error=max(r['correlation_error'] for r in rows),
                mean_magnetization=st.mean(r['magnetization'] for r in rows),
                mean_parity=st.mean(r['parity'] for r in rows),
                maximum_norm_error=max(r['norm_error'] for r in rows))


def collect(root, cfg):
    idx = read(root/'experiments/round8_index.json')
    if idx['status'] != 'completed' or len(idx['runs']) != 40 or idx['failed_runs']:
        raise ValueError('analysis requires all 40 completed frozen jobs, with no omitted failures')
    expected = set(itertools.product(cfg['families'], cfg['radii'], cfg['architectures'], cfg['seeds']))
    seen = set(); all_rows=[]; endpoints=[]; anchors=[]; choices={}; action_costs=[]; research_costs=[]; commits=set()
    component_totals = dict(base_fit_seconds=0., action_fit_seconds=0., policy_probe_seconds=0.,
                            acquisition_warmstart_seconds=0., projection_seconds=0.,
                            small_query_seconds=0., evaluation_reconstruction_seconds=0.)
    for rid in idx['runs']:
        folder=root/'experiments/runs'/rid; manifest=read(folder/'manifest.json'); job=manifest['job']
        key=tuple(job[k] for k in ['family','radius','architecture','seed'])
        if key not in expected or key in seen or manifest['status']!='completed':
            raise ValueError('missing, duplicate or non-completed factorial cell')
        seen.add(key); commits.add(manifest['git_commit'])
        rows=read(folder/'metrics.json'); decisions=read(folder/'decisions.json')
        if read(folder/'config.json') != cfg or len(decisions)!=len(cfg['maturities']):
            raise ValueError('config or maturity mismatch')
        meta=dict(run_id=rid,**job)
        research_costs.append(dict(**meta,**manifest['budget']))
        for r in rows:
            rr=dict(**meta,**r,raw_infidelity=1-r['fidelity'],clipped_infidelity=clipped_infidelity(r['fidelity']))
            all_rows.append(rr)
            if r['split']=='anchor': anchors.append(rr)
        cumulative_base_fit=0.
        for decision in decisions:
            maturity=decision['maturity']; features=decision['features']
            incremental=sum(decision['base_fit_seconds']);cumulative_base_fit+=incremental
            component_totals['base_fit_seconds']+=incremental
            component_totals['policy_probe_seconds']+=features['probe_seconds']
            component_totals['acquisition_warmstart_seconds']+=features['warmstart_seconds']
            for policy,action in features['choices'].items():
                choices[(key,maturity,policy)]=action
            for variant in cfg['basis_variants']:
                component_totals['projection_seconds']+=decision['base_projection_seconds'][variant]
                for action in cfg['actions']:
                    rr=[r for r in rows if (r['maturity'],r['variant'],r['action'],r['split'])==(maturity,variant,action,'query')]
                    endpoints.append(dict(**meta,maturity=maturity,variant=variant,action=action,**endpoint(rr)))
                    a=decision['actions'][action];v=a['variants'][variant]
                    component_totals['projection_seconds']+=v['projection_seconds']
                    component_totals['small_query_seconds']+=v['small_query_seconds']
                    component_totals['evaluation_reconstruction_seconds']+=v['full_reconstruction_seconds']
                    action_costs.append(dict(**meta,maturity=maturity,variant=variant,action=action,
                        base_fit_increment_seconds=incremental,base_fit_cumulative_seconds=cumulative_base_fit,
                        base_closure_evaluations=2*maturity,additional_closure_evaluations=a['closure_evaluations'],
                        discarded_action_closures=a['discarded_closures'],action_fit_seconds=a['fit_seconds'],
                        base_projection_seconds=decision['base_projection_seconds'][variant],
                        action_projection_seconds=v['projection_seconds'],action_projection_component_actions=v['component_actions'],
                        base_projection_component_actions=decision['base_geometry'][variant]['component_actions'],
                        small_query_seconds=v['small_query_seconds'],evaluation_reconstruction_seconds=v['full_reconstruction_seconds'],
                        rank=v['rank'],policy_probe_seconds=features['probe_seconds'],policy_probe_h_actions=features['probe_h_actions'],
                        acquisition_warmstart_seconds=features['warmstart_seconds'] if action=='acquire_center' else 0.,
                        acquisition_warmstart_h_actions=features['acquire_warmstart_h_actions'] if action=='acquire_center' else 0,
                        counterfactual_job_wall_seconds=manifest['budget']['wall_seconds'],
                        cost_note='Shared base/probe costs repeat in rows; do not sum this table as research wall or oracle-policy cost.'))
            component_totals['action_fit_seconds']+=sum(a['fit_seconds'] for a in decision['actions'].values())
    if seen != expected:
        raise ValueError('incomplete factorial')
    return dict(all_rows=all_rows,endpoints=endpoints,anchors=anchors,choices=choices,
                action_costs=action_costs,research_costs=research_costs,commits=sorted(commits),
                component_totals=component_totals)


def build_summary(data,cfg):
    strata=[]; action_aggregates=[]; policy_endpoints=[]; mechanisms=[]
    for family,radius,arch,maturity,variant in itertools.product(cfg['families'],cfg['radii'],cfg['architectures'],cfg['maturities'],cfg['basis_variants']):
        meta=dict(family=family,radius=radius,architecture=arch,maturity=maturity,variant=variant)
        selected=[r for r in data['endpoints'] if all(r[k]==v for k,v in meta.items())]
        losses={a:[next(r['mean_infidelity'] for r in selected if r['seed']==s and r['action']==a) for s in cfg['seeds']] for a in cfg['actions']}
        policy_choices={name:[data['choices'][((family,radius,arch,s),maturity,name)] for s in cfg['seeds']] for name in cfg['simple_policies'] if name not in cfg['actions']}
        result=compute_headroom(losses,policy_choices,cfg['simple_policies'],cfg['seeds'])
        types=refine_acquire_winners(losses,cfg['seeds'],cfg['gate']['minimum_absolute_action_advantage'])
        record=dict(**meta,**result,stop_mean=st.mean(losses['stop']),
                    action_losses=losses,policy_choices=policy_choices,
                    refine_winner_seed_ids=types['refine_winner_seed_ids'],acquire_winner_seed_ids=types['acquire_winner_seed_ids'])
        record.update(gate_stratum(record,cfg['gate']));strata.append(record)
        for row in types['rows']:mechanisms.append(dict(**meta,**row))
        policies=policy_losses(losses,policy_choices,cfg['simple_policies'])
        for name,values in policies.items():
            for seed,value in zip(cfg['seeds'],values):
                i=cfg['seeds'].index(seed)
                policy_endpoints.append(dict(**meta,seed=seed,policy=name,selected_action=name if name in cfg['actions'] else policy_choices[name][i],mean_infidelity=value))
        metrics=['mean_infidelity','worst_infidelity','mean_energy_error_per_spin','mean_variance',
                 'mean_magnetization_error','mean_correlation_error','worst_correlation_error']
        for action in cfg['actions']:
            rr=sorted([r for r in selected if r['action']==action],key=lambda r:cfg['seeds'].index(r['seed']))
            for metric in metrics:
                action_aggregates.append(dict(**meta,action=action,metric=metric,**describe([r[metric] for r in rr])))
    primary=[r for r in strata if r['maturity']==cfg['primary_maturity'] and r['variant']==cfg['primary_variant']]
    primary_gate=summarize_primary_gate(primary,cfg['gate'])
    return dict(strata=strata,primary_gate=primary_gate,action_aggregates=action_aggregates,
                policy_endpoints=policy_endpoints,mechanisms=mechanisms)


def figures(summary,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    primary=[r for r in summary['strata'] if r['maturity']==cfg['primary_maturity'] and r['variant']==cfg['primary_variant']]
    fig,axes=plt.subplots(2,4,figsize=(14,6.5),sharex=True,sharey=True)
    colors=['#7f7f7f','#2c7fb8','#e07b39','#3f9665','#8856a7','#5b247a']
    for ax,r in zip(axes.flat,primary):
        groups=[r['action_losses'][a] for a in cfg['actions']]+[r['baseline_values'],r['oracle_values']]
        for i,values in enumerate(groups):
            plotted=np.maximum(values,1e-12)
            ax.scatter(i+np.linspace(-.08,.08,len(values)),plotted,s=18,c=colors[i])
            ax.plot([i-.16,i+.16],[max(st.mean(values),1e-12)]*2,c='black',lw=1.3)
        ax.axhline(cfg['gate']['minimum_baseline_error'],ls='--',c='gray',lw=.8)
        ax.set(yscale='log',ylim=(1e-6,1e-2),title=f"{r['family']}, r={r['radius']}, {r['architecture']}")
        ax.set_xticks(range(6),['STOP','R-left','R-right','A','B','O']);ax.tick_params(axis='x',labelsize=8)
        ax.grid(alpha=.15)
    for ax in axes[:,0]:ax.set_ylabel('Eight-query mean infidelity')
    fig.suptitle('Primary: 1200 closures, flip-closed basis; five paired seeds and arithmetic means')
    fig.text(.5,.015,'B = strongest empirical simple policy; O = one-step hindsight minimum, not a deployable policy. All primary values are shown.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(out/'primary_headroom.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8));handles=[]
    palette={'chain':'#2878b5','ladder':'#cf5b42'};markers={'mlp':'o','transformer':'s'}
    for family,radius,arch in itertools.product(cfg['families'],cfg['radii'],cfg['architectures']):
        def pick(variant):return next(r for r in summary['strata'] if (r['family'],r['radius'],r['architecture'],r['maturity'],r['variant'])==(family,radius,arch,cfg['primary_maturity'],variant))
        raw,closed=pick('raw'),pick('flip_closed');label=f'{family} / {radius} / {arch}'
        h=axes[0].scatter(raw['relative_headroom'],closed['relative_headroom'],s=50,c=palette[family],marker=markers[arch],label=label,
                         edgecolors='black' if radius==max(cfg['radii']) else 'none',linewidths=.7)
        handles.append(h)
        axes[1].scatter(max(raw['stop_mean'],1e-12),max(closed['stop_mean'],1e-12),s=50,c=palette[family],marker=markers[arch],
                        edgecolors='black' if radius==max(cfg['radii']) else 'none',linewidths=.7)
    axes[0].plot([0,.23],[0,.23],ls='--',c='gray',lw=.8)
    axes[0].axhline(.2,ls=':',c='gray',lw=.8);axes[0].axvline(.2,ls=':',c='gray',lw=.8)
    axes[0].set(xlim=(-.01,.23),ylim=(-.01,.23),xlabel='Raw-basis relative headroom G',ylabel='Flip-closed relative headroom G',title='Hindsight headroom sensitivity')
    axes[1].plot([1e-6,1e-2],[1e-6,1e-2],ls='--',c='gray',lw=.8)
    for name in ['axhline','axvline']:getattr(axes[1],name)(cfg['gate']['minimum_baseline_error'],ls=':',c='gray',lw=.8)
    axes[1].set(xscale='log',yscale='log',xlim=(1e-6,1e-2),ylim=(1e-6,1e-2),xlabel='Raw STOP mean infidelity',ylabel='Flip-closed STOP mean infidelity',title='Zero-additional-training quality')
    for ax in axes:ax.grid(alpha=.15)
    fig.legend(handles=handles,labels=[h.get_label() for h in handles],loc='lower center',ncol=4,fontsize=7,frameon=False)
    fig.suptitle('Strong maturity only: same frozen snapshots, different retained basis')
    fig.tight_layout(rect=(0,.13,1,.94));fig.savefig(out/'raw_closed_sensitivity.png',dpi=180);plt.close(fig)


def export_tables(summary,out):
    lines=['# Round8: complete one-step headroom analysis','',
      'All means/SD are descriptive over five paired seeds. Query points and maturities are not independent replicates. O and best-simple selection are hindsight comparators, not deployable speedups.','',
      '## Primary investment gate','',json.dumps(summary['primary_gate'],indent=2),'',
      '## Every stratum (no primary switching)','',
      '| family / radius / architecture | maturity / basis | best simple | B | O | B-O | G | LOO G min..max | STOP mean | eligible |',
      '|---|---|---|---|---|---|---|---|---|---|']
    for r in summary['strata']:
        lines.append(f"| {r['family']} / {r['radius']} / {r['architecture']} | {r['maturity']} / {r['variant']} | {r['best_simple_policy']} | {r['baseline_mean']:.7g} | {r['oracle_mean']:.7g} | {r['absolute_headroom']:.7g} | {r['relative_headroom']:.4g} | {r['minimum_leave_one_out_headroom']:.4g} .. {r['maximum_leave_one_out_headroom']:.4g} | {r['stop_mean']:.7g} | {r['eligible']} |")
    lines+=['','## All action metrics','','| stratum | action / metric | mean ± seed SD |','|---|---|---|']
    for r in summary['action_aggregates']:
        lines.append(f"| {r['family']} / {r['radius']} / {r['architecture']} / {r['maturity']} / {r['variant']} | {r['action']} / {r['metric']} | {r['mean']:.7g} ± {r['sd']:.3g} |")
    lines+=['','## Cost interpretation','',
       'research_costs.csv counts each completed job once and includes execution of all counterfactuals. action_costs.csv repeats shared base/probe metadata intentionally; summing it double-counts shared costs. Projection, small-matrix online queries and full-vector evaluation reconstruction are separate. Matched additional closure allowances do not imply equal wall time or total compute.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    cfg=read(ROOT/'experiments/configs/round8.json');data=collect(ROOT,cfg);result=build_summary(data,cfg)
    OUT.mkdir(parents=True,exist_ok=True)
    research_wall=sum(r['wall_seconds'] for r in data['research_costs'])
    summary=dict(config=cfg,main_jobs=40,metric_rows=len(data['all_rows']),query_seed_endpoints=len(data['endpoints']),
                 frozen_source_commits=data['commits'],research_wall_seconds=research_wall,
                 measured_research_components=data['component_totals'],
                 other_research_seconds=research_wall-sum(data['component_totals'].values()),
                 primary_gate=result['primary_gate'],strata=result['strata'],action_aggregates=result['action_aggregates'],
                 limitations='N8 development; one decision; fixed finite menu; exact projection; no resampling action, learned scheduler, significance, independent-human replication or deployable speedup.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write_csv(OUT/'all_metrics.csv',data['all_rows']);write_csv(OUT/'query_endpoints.csv',data['endpoints'])
    write_csv(OUT/'anchors.csv',data['anchors']);write_csv(OUT/'action_costs.csv',data['action_costs'])
    write_csv(OUT/'research_costs.csv',data['research_costs']);write_csv(OUT/'simple_policy_endpoints.csv',result['policy_endpoints'])
    write_csv(OUT/'refine_acquire_mechanism.csv',result['mechanisms'])
    flat=[];loo=[];paired=[]
    for r in result['strata']:
        meta={k:r[k] for k in ['family','radius','architecture','maturity','variant']}
        flat.append(dict(**meta,**{k:r[k] for k in ['best_simple_policy','baseline_mean','oracle_mean','absolute_headroom','relative_headroom',
            'minimum_leave_one_out_headroom','maximum_leave_one_out_headroom','stop_mean','eligible']},
            refine_winner_seed_ids=','.join(map(str,r['refine_winner_seed_ids'])),acquire_winner_seed_ids=','.join(map(str,r['acquire_winner_seed_ids'])),
            failure_or_caveat_categories='; '.join(r['failure_or_caveat_categories'])))
        for a in r['leave_one_out']:
            loo.append(dict(**meta,**{k:v for k,v in a.items() if k!='simple_policy_means'}))
        for i,s in enumerate(r['seeds']):
            paired.append(dict(**meta,seed=s,best_simple_policy=r['best_simple_policy'],simple_loss=r['baseline_values'][i],
                               oracle_loss=r['oracle_values'][i],paired_improvement=r['paired_improvements'][i],oracle_action=r['oracle_actions'][i]))
    write_csv(OUT/'headroom_strata.csv',flat);write_csv(OUT/'leave_one_out.csv',loo);write_csv(OUT/'paired_headroom.csv',paired)
    export_tables(summary,OUT);figures(summary,cfg,OUT)
    print(json.dumps(dict(main_jobs=40,metric_rows=summary['metric_rows'],query_seed_endpoints=summary['query_seed_endpoints'],
                         primary_gate=summary['primary_gate'],research_wall_seconds=research_wall),indent=2))


if __name__ == '__main__': main()
