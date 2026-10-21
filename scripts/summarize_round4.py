"""Descriptive seed-level summaries; all arms, no significance or winner selection."""
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
from verify_round4 import audit, read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results/round4'


def csv_write(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def stats(values):
    return dict(mean=st.mean(values), sd=st.stdev(values), minimum=min(values), maximum=max(values), values=values)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    control = audit(with_replay='--without-replay' not in sys.argv)
    (OUT / 'control_audit.json').write_text(json.dumps(control, indent=2)+'\n', encoding='utf-8')
    cfg = read(ROOT / 'experiments/configs/round4.json'); index = read(ROOT / 'experiments/round4_index.json')
    endpoints = []; allrows = []; costs = []; initial = []; lookup = {}; histories = {}
    for rid in index['runs']:
        directory = ROOT / 'experiments/runs' / rid; m = read(directory / 'manifest.json'); j = m['job']
        histories[(j['h'], j['architecture'], j['seed'], j['arm'])] = read(directory / 'training.json')
        costs.append(dict(run_id=rid, **j, **m['budget']))
        initial.append(dict(run_id=rid, **j, max_abs_difference=m['initial_function']['max_abs_difference'], selected_step=m['model']['selected_step']))
        for key, values in m['metrics'].items():
            checkpoint, split, method = key.split('/')
            row = dict(run_id=rid, **j, checkpoint=checkpoint, split=split, method=method, **values)
            endpoints.append(row); lookup[(j['h'], j['architecture'], j['seed'], j['arm'], checkpoint, split, method)] = values
        for row in read(directory / 'metrics.json'): allrows.append({'run_id': rid, **j, **row})
    aggregates = []; pairs = []
    for h, arch, arm, checkpoint, split, method in itertools.product(cfg['h_values'], cfg['architectures'], cfg['arms'],
            ['selected', 'final'], ['train', 'validation', 'test'], ['direct', 'lookup', 'train_span_ritz', 'post_covariant']):
        values = [lookup[(h, arch, seed, arm, checkpoint, split, method)]['mean_infidelity'] for seed in cfg['seeds']]
        aggregates.append(dict(h=h, architecture=arch, arm=arm, checkpoint=checkpoint, split=split, method=method, **stats(values)))
    for h, arch, checkpoint, (base, alternative) in itertools.product(cfg['h_values'], cfg['architectures'], ['selected','final'],
            [('physical','scaled'),('physical','matched'),('matched','scaled'),('physical','canonical')]):
        deltas = [lookup[(h, arch, seed, alternative, checkpoint, 'test','direct')]['mean_infidelity']-
                  lookup[(h, arch, seed, base, checkpoint, 'test','direct')]['mean_infidelity'] for seed in cfg['seeds']]
        pairs.append(dict(h=h, architecture=arch, checkpoint=checkpoint, base=base, alternative=alternative,
                          improved=sum(x < -1e-12 for x in deltas), worsened=sum(x > 1e-12 for x in deltas), **stats(deltas)))
    summary = dict(config=cfg, main_jobs=len(index['runs']), metric_rows=len(allrows), seed_endpoints=len(endpoints),
                   wall_seconds=sum(c['wall_seconds'] for c in costs), aggregates=aggregates, paired_test_direct=pairs,
                   interpretation='Descriptive five-seed statistics; SD is not a CI; lambda points are correlated; no independent replication.')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    csv_write('seed_endpoints.csv', endpoints); csv_write('all_metrics.csv', allrows); csv_write('costs.csv', costs); csv_write('initialization.csv', initial)
    csv_write('paired_differences.csv', [{**{k:v for k,v in r.items() if k != 'values'}, **{f'seed_{s}':v for s,v in zip(cfg['seeds'],r['values'])}} for r in pairs])
    table = ['# Round 4 complete descriptive results', '', 'Mean ± sample SD across five seeds. Lower infidelity is better; no confidence or significance claim.', '',
             'P=physical; U=scaled unchanged weights; I=scaled matched initial function; C=matched function + physical-coordinate Adam.', '',
             '| h / architecture / checkpoint / split / method | P | U | I | C |','|---|---|---|---|---|']
    for h, arch, checkpoint, split, method in itertools.product(cfg['h_values'],cfg['architectures'],['selected','final'],['train','validation','test'],['direct','lookup','train_span_ritz','post_covariant']):
        rows = [next(r for r in aggregates if (r['h'],r['architecture'],r['arm'],r['checkpoint'],r['split'],r['method']) == (h,arch,arm,checkpoint,split,method)) for arm in cfg['arms']]
        table.append(f'| {h} / {arch} / {checkpoint} / {split} / {method} | '+' | '.join(f"{r['mean']:.7g} ± {r['sd']:.3g}" for r in rows)+' |')
    table += ['', '## Paired test/direct differences', '', 'Alternative minus base; negative is improvement. All five paired deltas are available in CSV/JSON.', '',
              '| h / model / checkpoint | Alternative - base | Mean ± SD | Improved / 5 | Worsened / 5 |','|---|---|---|---|---|']
    for r in pairs: table.append(f"| {r['h']} / {r['architecture']} / {r['checkpoint']} | {r['alternative']} - {r['base']} | {r['mean']:.7g} ± {r['sd']:.3g} | {r['improved']} | {r['worsened']} |")
    (OUT / 'tables.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#2475a8','#df6c24','#2a9770','#8c68ad']; labels=['P: physical','U: scaled','I: matched init','C: canonical Adam']
    fig, axes = plt.subplots(2,2,figsize=(10,6), sharex=True)
    for ax,(h,arch) in zip(axes.flat,itertools.product(cfg['h_values'],cfg['architectures'])):
        for i,arm in enumerate(cfg['arms']):
            values = [lookup[(h,arch,s,arm,'selected','test','direct')]['mean_infidelity'] for s in cfg['seeds']]
            ax.scatter(i+np.linspace(-.08,.08,5),values,color=colors[i],s=28)
            ax.plot([i-.2,i+.2],[st.mean(values)]*2,color='black',lw=1.8)
        ax.set(title=f'N=6, h={h}, {arch}',yscale='log',ylabel='Mean test infidelity')
        ax.set_ylim((.3,.6) if h==.4 else (.0003,.003))
        ax.set_xticks(range(4),['P','U','I','C']); ax.grid(alpha=.2)
    fig.suptitle('Validation-selected direct prediction: five paired seeds; black bars are means')
    fig.text(.5,.015,'P physical | U scaled | I matched initial function | C matched function + canonical Adam',ha='center')
    fig.tight_layout(rect=(0,.04,1,.96)); fig.savefig(OUT/'factorial.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
    for ax,(h,arch) in zip(axes.flat,itertools.product(cfg['h_values'],cfg['architectures'])):
        for i,arm in enumerate(cfg['arms']):
            values=np.array([[lookup[(h,arch,s,arm,'selected','test',method)]['mean_infidelity'] for s in cfg['seeds']]
                             for method in ['direct','lookup','post_covariant','train_span_ritz']])
            means=values.mean(axis=1)
            ax.plot(range(4),means,marker='o',label=labels[i],color=colors[i],ls='--' if arm=='canonical' else '-')
        ax.set(title=f'h={h}, {arch}',yscale='log',ylabel='Mean test infidelity')
        ax.set_ylim((.2,.6) if h==.4 else (.00002,.006))
        ax.set_xticks(range(4),['direct','lookup','post-cov','span Ritz']);ax.grid(alpha=.2)
    handles,ls=axes[0,0].get_legend_handles_labels();fig.legend(handles,ls,loc='lower center',ncol=4,frameon=False)
    fig.suptitle('Secondary postprocessing controls: five-seed means; Ritz incurs extra H work')
    fig.tight_layout(rect=(0,.05,1,.96));fig.savefig(OUT/'postprocessing.png',dpi=180);plt.close(fig)
    print(f"Wrote {len(allrows)} metric rows, {len(endpoints)} endpoints, complete tables and 2 figures; main wall {summary['wall_seconds']:.3f}s")


if __name__ == '__main__': main()
