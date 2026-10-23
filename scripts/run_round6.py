"""Exact SR and approximate-capacity development controls."""
import argparse
import copy
import datetime as dt
import hashlib
import itertools
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.models import ConditionalMLP,ConditionalTransformer
from ctqs.physics import basis,hamiltonian,lookup,metrics,ritz
from ctqs.budgeted_fit import fit
from ctqs.exact_sr import fit_sr
from run_round1 import dump
from run_round2 import state_hash


def jobs(cfg):
    for h,arch,seed,arm in itertools.product(cfg['h_values'],cfg['architectures'],cfg['seeds'],cfg['arms']):
        sharing='independent' if arm=='independent_sr' else 'shared'
        opt='lbfgs' if arm=='wide_lbfgs' else 'sr'
        width=cfg['wide_'+arch] if arm=='wide_lbfgs' else (32 if arch=='mlp' else 16)
        yield dict(h=h,architecture=arch,seed=seed,sharing=sharing,optimizer=opt,arm=arm,width=width)


def key(j):return tuple(j[k] for k in ['h','architecture','seed','sharing','optimizer'])


def run(cfg,j,tag):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):raise RuntimeError('commit tracked changes first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    h,arch,seed,sharing,opt=key(j);n=cfg['n']
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rid=f'{stamp}-{tag}-n{n}-h{h:g}-{arch}-{sharing}-{opt}-s{seed}'
    out=ROOT/'experiments/runs'/rid;out.mkdir(parents=True)
    m=dict(schema_version=1,run_id=rid,created_at_utc=stamp,status='running',owner='Bian-M-X/Codex',issue=11,
      git_commit=commit,seed=seed,question='Does a verified SR baseline or approximate capacity control alter training reliability?',job=j,
      model=dict(architecture=arch,sharing=sharing,optimizer=opt,condition_scale=1.,checkpoint='final accepted'),
      hamiltonian=dict(n=n,h=h,lambda_value='variable',boundary='open',J=1,basis='MSB left; Z(0)=+1'),
      parameter_split=dict(train=cfg['train'],validation=[],test=cfg['test']),
      command=f'python scripts/run_round6.py --h {h} --arch {arch} --seed {seed} --sharing {sharing} --optimizer {opt} --tag {tag}',
      environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),device='cpu',dtype='float64',torch_threads=1),
      budget={},oracle_access=dict(training=False,selection=False,evaluation='ED after final state freeze',postprocessing=False),metrics={},
      artifacts=['config.json','training.json','metrics.json','amplitudes.json'],notes='Development diagnostic. SR Jacobians add substantial cost; H work is not total compute. Wide models approximately match independent total parameters, not initial functions.')
    m['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    dump(out/'config.json',cfg);dump(out/'manifest.json',m)
    start=time.monotonic();training=[];counts=dict(forward=0,backward=0);rows=[]
    try:
        torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
        base=(ConditionalMLP(n,width=j['width']) if arch=='mlp' else ConditionalTransformer(n,width=j['width'])).double()
        models=[copy.deepcopy(base) for _ in range(1 if sharing=='shared' else 3)]
        m['initial_state_sha256']=[state_hash(x) for x in models]
        m['model'].update(parameters_per_model=sum(p.numel() for p in base.parameters()),model_count=len(models),
            total_parameters=sum(p.numel() for model in models for p in model.parameters()))
        bits=torch.tensor(basis(n),dtype=torch.float64);mats={x:torch.tensor(hamiltonian(n,h,x)) for x in cfg['train']}
        for i,model in enumerate(models):
            xs=cfg['train'] if sharing=='shared' else [cfg['train'][i]]
            def objective(current):
                energies=[]
                for x in xs:
                    psi=current.psi(bits,[h,x]);counts['forward']+=1
                    energies.append(psi@mats[x]@psi/n)
                # Successful backward is verified by fit; one H action per term.
                return torch.stack(energies).mean()
            if opt=='sr':
                result=fit_sr(model,bits,[[h,x] for x in xs],[mats[x] for x in xs],n,
                    steps=cfg['sr_steps'],shift=cfg['sr_shift'],rate=cfg['sr_rate'],radius=cfg['sr_radius'],
                    max_trials=cfg['sr_max_trials'],deadline=start+cfg['timeout_seconds_per_run'])
                counts['forward']+=result['training_h_actions']
            else:
                result=fit(model,objective,opt,cfg['lbfgs_closures'],lbfgs_lr=1.,history_size=20,
                           deadline=start+cfg['timeout_seconds_per_run'])
                counts['backward']+=result['closure_evaluations']*len(xs)
            result['model_index']=i;result['train_lambda']=xs;training.append(result);dump(out/'training.json',training)
        fit_end=time.monotonic();m['final_state_sha256']=[state_hash(x) for x in models]
        for model in models:model.eval()
        # No ED or test predictions until all models have finished training.
        oracles={x:eigh(hamiltonian(n,h,x)) for x in cfg['train']+cfg['test']}
        amps={};eval_calls=post_h=0
        with torch.no_grad():
            anchors=[]
            for i,x in enumerate(cfg['train']):
                model=models[0] if sharing=='shared' else models[i]
                anchors.append(model.psi(bits,[h,x]).numpy());eval_calls+=1
            for split in ['train','test']:
                for x in cfg[split]:
                    mat=hamiltonian(n,h,x);span,rank=ritz(mat,anchors);post_h+=rank
                    predictions=dict(lookup=lookup(x,cfg['train'],anchors),train_span_ritz=span)
                    if split=='train':predictions['anchor']=anchors[cfg['train'].index(x)]
                    if sharing=='shared':
                        predictions['direct']=anchors[cfg['train'].index(x)] if split=='train' else models[0].psi(bits,[h,x]).numpy()
                        if split=='test':eval_calls+=1
                    amps[str(x)]={name:psi.tolist() for name,psi in predictions.items()}
                    for method,psi in predictions.items():
                        row=metrics(mat,psi,n,oracles[x]);row.update(split=split,method=method,x=x,h=h,lam=x);rows.append(row)
        for split,method in sorted({(r['split'],r['method']) for r in rows}):
            rr=[r for r in rows if (r['split'],r['method'])==(split,method)]
            m['metrics'][f'{split}/{method}']=dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),worst_infidelity=max(1-r['fidelity'] for r in rr),
               mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),worst_correlation_error=max(r['correlation_error'] for r in rr),
               mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in rr])))
        m['budget']=dict(training_forward_h_actions=counts['forward'],training_backward_h_actions=counts['backward'],
           total_training_h_actions=counts['forward']+counts['backward'],training_model_calls=counts['forward'],
           closure_evaluations=sum(x.get('closure_evaluations',0) for x in training),optimizer_steps_completed=sum(x.get('optimizer_steps_completed',x.get('outer_steps',0)) for x in training),
           jacobian_rows=sum(x.get('jacobian_rows',0) for x in training),accepted_sr_steps=sum(x.get('accepted_steps',0) for x in training),
           max_solve_dimension=max(x.get('max_solve_dimension',0) for x in training),
           discarded_final_step_evaluations=sum(x.get('discarded_final_step_evaluations',0) for x in training),validation_h_actions=0,
           evaluation_model_calls=eval_calls,evaluation_h_actions=2*len(rows),postprocessing_h_actions=post_h,exact_diagonalizations=len(oracles),
           fit_seconds=fit_end-start,evaluation_seconds=time.monotonic()-fit_end,wall_seconds=time.monotonic()-start)
        dump(out/'metrics.json',rows);dump(out/'amplitudes.json',amps);m['status']='completed'
        print(rid,'completed',round(m['budget']['wall_seconds'],2),'seconds',flush=True)
    except Exception as exc:
        m['status']='failed';m['notes']+=f' Failure {type(exc).__name__}: {exc}'
        m['budget']=dict(completed_model_fits=len(training),training_forward_h_actions=counts['forward'],wall_seconds=time.monotonic()-start)
        raise
    finally:
        dump(out/'training.json',training);dump(out/'manifest.json',m)
    return rid


def main():
    ap=argparse.ArgumentParser()
    for a,choices in [('arch',['mlp','transformer']),('sharing',['shared','independent']),('optimizer',['sr','lbfgs'])]:ap.add_argument('--'+a,choices=choices)
    ap.add_argument('--h',type=float);ap.add_argument('--seed',type=int);ap.add_argument('--tag',default='round6');args=ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+',args.tag):ap.error('tag must be a slug')
    cfg=json.loads((ROOT/'experiments/configs/round6.json').read_text())
    assert len(set(cfg['train']+cfg['test']))==len(cfg['train']+cfg['test'])
    path=ROOT/'experiments/round6_index.json';index=json.loads(path.read_text()) if path.exists() else dict(runs=[],failed_runs=[],status='running')
    seen={key(json.loads((ROOT/'experiments/runs'/r/'manifest.json').read_text())['job']) for r in index['runs']};start=time.monotonic()
    try:
        for j in jobs(cfg):
            if any(getattr(args,a) is not None and j[b]!=getattr(args,a) for a,b in [('h','h'),('arch','architecture'),('seed','seed'),('sharing','sharing'),('optimizer','optimizer')]):continue
            if args.tag=='round6' and key(j) in seen:continue
            if time.monotonic()-start>cfg['timeout_seconds_per_invocation']:raise TimeoutError('invocation cap exceeded')
            rid=run(cfg,j,args.tag)
            if args.tag=='round6':index['runs'].append(rid);dump(path,index)
        index['status']='completed' if len(index['runs'])==60 else 'partial'
    except Exception:index['status']='partial';raise
    finally:
        if args.tag=='round6':
            index['failed_runs']=[p.parent.name for p in (ROOT/'experiments/runs').glob('*-round6-*/manifest.json') if json.loads(p.read_text())['status']=='failed']
            dump(path,index)


if __name__=='__main__':main()
