"""Frozen factorial controls with final and validation-selected endpoints."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.conditioning import conditioned_state
from ctqs.models import ConditionalMLP,ConditionalTransformer,state
from ctqs.physics import basis,hamiltonian,lookup,metrics,normalized,project_even,ritz
from run_round1 import dump
from run_round2 import state_hash


def jobs(cfg):
    for stage in ['a','b']:
        block=cfg['stage_'+stage]
        for n in [block['n']] if stage=='a' else block['sizes']:
            for arch in cfg['architectures']:
                for scale in [1.] if stage=='a' else block['scales']:
                    for mode in block['modes']:
                        for seed in cfg['seeds']:
                            yield dict(stage=stage,n=n,architecture=arch,scale=scale,mode=mode,seed=seed)


def key(job):
    return tuple(job[k] for k in ['stage','n','architecture','scale','mode','seed'])


def validate_config(cfg):
    for stage in ['a','b']:
        block=cfg['stage_'+stage]
        names=['train','validation','interpolation','extrapolation'] if stage=='a' else ['train','validation','test']
        values=[x for name in names for x in block[name]]
        if len(values)!=len(set(values)):raise ValueError('split leakage or duplicate points')
        if block['train']!=sorted(block['train']):raise ValueError('unsorted train grid')
        if block['steps']%cfg['validation_every']:raise ValueError('steps must include final validation')
    if any(s<=0 for s in cfg['stage_b']['scales']):raise ValueError('nonpositive scale')


def run(cfg,job,tag='round3'):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):
        raise RuntimeError('commit tracked changes before scientific runs')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stage,n,arch,scale,mode,seed=key(job);block=cfg['stage_'+stage]
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rid=f'{stamp}-{tag}{stage}-n{n}-{arch}-{mode}-scale{scale:g}-s{seed}'
    out=ROOT/'experiments/runs'/rid;out.mkdir(parents=True)
    m=dict(schema_version=1,run_id=rid,created_at_utc=stamp,status='running',owner='Bian-M-X/Codex',issue=5,
        git_commit=commit,seed=seed,question='Do conditioning, training covariance and checkpoint selection explain the observed failures?',
        job=job,model=dict(architecture=arch,training_mode=mode,condition_scale=scale),
        hamiltonian=dict(n=n,h=block.get('h','variable'),lambda_value=0 if stage=='a' else 'variable',boundary='open',J=1,basis='MSB left; Z(0)=+1'),
        parameter_split={k:block[k] for k in ['train','validation','interpolation','extrapolation','test'] if k in block},
        command=f'python scripts/run_round3.py --stage {stage} --n {n} --arch {arch} --scale {scale} --mode {mode} --seed {seed} --tag {tag}',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),device='cpu',dtype='float64',torch_threads=1),
        budget={},oracle_access=dict(training=False,selection=False,evaluation='dense ED after checkpoint freeze',postprocessing=False),
        metrics={},artifacts=['config.json','training.json','metrics.json'],notes='Screening; H work including backward matched within stage, network calls/wall differ. No independent-member replication.')
    m['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    dump(out/'config.json',cfg);dump(out/'manifest.json',m)
    start=time.monotonic();history=[];training_h=validation_h=training_calls=validation_calls=steps=0
    try:
        torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
        model=(ConditionalMLP(n) if arch=='mlp' else ConditionalTransformer(n)).double()
        m['initial_state_sha256']=state_hash(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        bits=torch.tensor(basis(n),dtype=torch.float64)
        def predict(x):
            if stage=='a':return state(model,bits,x,mode=='even'),1
            return conditioned_state(model,bits,block['h'],x,scale,mode=='covariant')
        def matrix(x):return hamiltonian(n,x,0.) if stage=='a' else hamiltonian(n,block['h'],x)
        tensors={x:torch.tensor(matrix(x)) for x in block['train']+block['validation']}
        best=float('inf');selected=None;selected_step=0
        for step in range(1,block['steps']+1):
            if time.monotonic()-start>cfg['timeout_seconds_per_run']:raise TimeoutError('prespecified run timeout')
            model.train();optimizer.zero_grad();energies=[]
            for x in block['train']:
                psi,calls=predict(x);training_calls+=calls;training_h+=1
                energies.append(psi@tensors[x]@psi/n)
            loss=torch.stack(energies).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();optimizer.step();steps=step
            if step%cfg['validation_every']==0:
                model.eval();validation=[]
                with torch.no_grad():
                    for x in block['validation']:
                        psi,calls=predict(x);validation_calls+=calls;validation_h+=1
                        validation.append(float(psi@tensors[x]@psi/n))
                score=float(np.mean(validation))
                history.append(dict(step=step,train_energy_per_spin=float(loss.detach()),validation_energy_per_spin=score))
                if score<best:best,selected_step,selected=score,step,copy.deepcopy(model.state_dict())
                dump(out/'training.json',history)
        final=copy.deepcopy(model.state_dict());m['final_state_sha256']=state_hash(model)
        m['model'].update(parameters=sum(p.numel() for p in model.parameters()),selected_step=selected_step)
        fit_end=time.monotonic();rows=[];eval_calls=post_h=eval_h=0
        splits=m['parameter_split']
        # No ED or test prediction occurs until both endpoint states are frozen.
        oracles={x:eigh(matrix(x)) for values in splits.values() for x in values}
        for checkpoint,weights in [('selected',selected),('final',final)]:
            model.load_state_dict(weights);model.eval();m[checkpoint+'_state_sha256']=state_hash(model)
            cache={}
            def cached(x):
                nonlocal eval_calls
                if x not in cache:
                    value,calls=predict(x);eval_calls+=calls;cache[x]=value.detach().numpy()
                return cache[x]
            with torch.no_grad():
                train_states=[cached(x) for x in block['train']]
                for split,values in splits.items():
                    for x in values:
                        mat=matrix(x);direct=cached(x);span,rank=ritz(mat,train_states);post_h+=rank
                        predictions=dict(direct=direct,lookup=lookup(x,block['train'],train_states),train_span_ritz=span)
                        if stage=='a':
                            parity,prank=ritz(mat,[direct,direct[::-1]]);post_h+=prank
                            predictions.update(even_projection=project_even(direct),parity_ritz=parity)
                        else:predictions['post_covariant']=normalized(direct+cached(-x)[::-1])
                        for method,psi in predictions.items():
                            row=metrics(mat,psi,n,oracles[x]);eval_h+=2
                            row.update(checkpoint=checkpoint,method=method,split=split,x=x,h=x if stage=='a' else block['h'],lam=0. if stage=='a' else x)
                            rows.append(row)
        for checkpoint in ['selected','final']:
            for split in splits:
                for method in predictions:
                    rr=[r for r in rows if (r['checkpoint'],r['split'],r['method'])==(checkpoint,split,method)]
                    m['metrics'][f'{checkpoint}/{split}/{method}']=dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),
                        worst_infidelity=max(1-r['fidelity'] for r in rr),mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),
                        mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in rr])),worst_correlation_error=max(r['correlation_error'] for r in rr))
        m['budget']=dict(gradient_steps=steps,training_forward_h_actions=training_h,training_backward_h_actions=training_h,
            validation_h_actions=validation_h,total_training_selection_h_actions=2*training_h+validation_h,
            training_model_calls=training_calls,validation_model_calls=validation_calls,evaluation_model_calls=eval_calls,
            postprocessing_h_actions=post_h,evaluation_h_actions=eval_h,exact_diagonalizations=len(oracles),
            fit_selection_seconds=fit_end-start,evaluation_seconds=time.monotonic()-fit_end,wall_seconds=time.monotonic()-start)
        dump(out/'metrics.json',rows);m['status']='completed'
        print(rid,'completed',round(m['budget']['wall_seconds'],2),'seconds',flush=True)
    except Exception as exc:
        m['status']='failed';m['notes']+=f' Failure {type(exc).__name__}: {exc}'
        m['budget']=dict(gradient_steps=steps,training_forward_h_actions=training_h,validation_h_actions=validation_h,wall_seconds=time.monotonic()-start)
        raise
    finally:
        dump(out/'training.json',history);dump(out/'manifest.json',m)
    return rid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['a','b','all'],default='all')
    ap.add_argument('--arch',choices=['mlp','transformer']);ap.add_argument('--n',type=int);ap.add_argument('--scale',type=float)
    ap.add_argument('--mode',choices=['raw','even','covariant']);ap.add_argument('--seed',type=int);ap.add_argument('--tag',default='round3')
    args=ap.parse_args();cfg=json.loads((ROOT/'experiments/configs/round3.json').read_text());validate_config(cfg)
    path=ROOT/'experiments/round3_index.json'
    index=json.loads(path.read_text()) if path.exists() else dict(runs=[],failed_runs=[],status='running')
    seen={key(json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text())['job']) for rid in index['runs']}
    started=time.monotonic()
    try:
        for job in jobs(cfg):
            if args.stage!='all' and job['stage']!=args.stage:continue
            if any(getattr(args,a) is not None and job[b]!=getattr(args,a) for a,b in [('arch','architecture'),('n','n'),('scale','scale'),('mode','mode'),('seed','seed')]):continue
            if key(job) in seen and args.tag=='round3':continue
            if time.monotonic()-started>cfg['timeout_seconds_per_invocation']:raise TimeoutError('invocation budget exhausted')
            rid=run(cfg,job,args.tag)
            if args.tag=='round3':index['runs'].append(rid);dump(path,index)
        index['status']='completed' if len(index['runs'])==100 else 'partial'
    except Exception:
        index['status']='partial'
        index['failed_runs']=[p.parent.name for p in sorted((ROOT/'experiments/runs').glob('*round3*/manifest.json')) if json.loads(p.read_text())['status']=='failed']
        raise
    finally:
        dump(path,index)


if __name__=='__main__':main()
