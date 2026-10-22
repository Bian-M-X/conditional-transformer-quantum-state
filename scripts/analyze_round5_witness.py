"""Post hoc algebraic diagnostic from saved observables; no new H/ED calls."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results/round5'


def main():
    index=json.loads((ROOT/'experiments/round5_index.json').read_text());assert index['status']=='completed'
    rows=[]
    for rid in index['runs']:
        p=ROOT/'experiments/runs'/rid;m=json.loads((p/'manifest.json').read_text())
        for r in json.loads((p/'metrics.json').read_text()):
            if r['method']!='anchor':continue
            signed=2*r['lam']*r['magnetization'];w=max(0.,signed)
            assert w<=r['energy_error']+1e-9
            rows.append(dict(run_id=rid,**m['job'],lam=r['lam'],magnetization=r['magnetization'],
               signed_flip_energy_descent=signed,witness=w,infidelity=1-r['fidelity'],energy_error=r['energy_error'],variance=r['variance']))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'posthoc_flip_witness.csv').open('w',encoding='utf-8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    summary=dict(status='post_hoc_descriptive_not_preregistered',rows=len(rows),extra_h_actions=0,extra_ed=0,
       equation='E(P psi)-E(psi)=-2 lambda <Z1>',bound='max(0,2 lambda <Z1>) <= E(psi)-E0',
       positive_witness_rows=sum(r['witness']>0 for r in rows),zero_field_rows=sum(r['lam']==0 for r in rows),
       limitations='No tuned threshold, no validated classifier; zero witness does not certify fidelity. Ground energy used only to verify inequality after training.')
    (OUT/'posthoc_flip_witness.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(summary)


if __name__=='__main__':main()
