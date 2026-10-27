"""Course-constrained, prerecord-frozen h generalization. No ED in training."""
import argparse
import copy
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.models import ConditionalMLP, ConditionalTransformer, state
from ctqs.physics import basis, hamiltonian, lookup, metrics, ritz
from ctqs.budgeted_fit import fit


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def weight_hash(model):
    digest=hashlib.sha256()
    for name,p in model.named_parameters():
        digest.update(name.encode());digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def validate_config(c):
    if c['n']!=6 or c['lambda']!=0: raise ValueError('round7 fixes N=6, lambda=0')
    test=[h for points in c['test_regions'].values() for h in points]
    if len(test)!=len(set(test)): raise ValueError('duplicate test values')
    for name,train in c['train_sets'].items():
        if train!=sorted(set(train)) or len(train)!=3: raise ValueError('three sorted training h required')
        if set(train)&set(c['validation']+test): raise ValueError('training/test leakage')
    if set(test)&set(c['validation']):raise ValueError('validation/test overlap')
    if min(test+c['validation']+sum(c['train_sets'].values(),[]))<=0: raise ValueError('h must be positive')
    for h in c['test_regions']['common_interpolation']:
        if not all(t[0]<h<t[-1] for t in c['train_sets'].values()):raise ValueError('bad common interpolation')
    for h in c['test_regions']['common_extrapolation']:
        if not all(h<t[0] or h>t[-1] for t in c['train_sets'].values()):raise ValueError('bad common extrapolation')
    for h in c['test_regions']['coverage_extension']:
        a,b=c['train_sets']['narrow'][0],c['train_sets']['narrow'][-1]
        w=c['train_sets']['wide']
        if not ((h<a or h>b) and w[0]<h<w[-1]):raise ValueError('bad coverage-extension region')


def jobs(c):
    return [(a,o,m,r,s) for a in c['architectures'] for o in c['optimizers_by_architecture'][a]
            for m in c['training_modes'] for r in c['train_sets'] for s in c['seeds']]


def accepted_validation(model,bits,n,train_mode,validation,mats):
    """Uses only supplied validation H; no spectrum or held-out target labels."""
    was_training=model.training
    model.eval()
    with torch.no_grad():
        energies=[]
        for h in validation:
            psi=state(model,bits,h,train_mode=='even')
            energies.append(psi@mats[h]@psi/n)
        result=float(torch.stack(energies).mean())
    model.train(was_training)
    return result


def check_model(model,bits,condition_h,tolerance):
    checks={};calls=0;model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout) and module.p!=0:raise ValueError('dropout must be zero')
        if isinstance(module,torch.nn.MultiheadAttention) and module.dropout!=0:raise ValueError('attention dropout must be zero')
    with torch.no_grad():
        # All 64 configurations; original probabilities, before any projection.
        psi=model.psi(bits,[condition_h,0.]);calls+=1
        checks['raw_normalization_error']=abs(float(psi@psi)-1)
        before=model.logits(bits,[condition_h,0.]);calls+=1
        errors=[]
        for i in range(bits.shape[1]):
            changed=bits.clone();changed[:,i:]=1-changed[:,i:]
            after=model.logits(changed,[condition_h,0.]);calls+=1
            errors.append(float((before[:,:i+1]-after[:,:i+1]).abs().max()))
        checks['causality_max_logit_difference']=max(errors)
        model.train();in_train=model.logits(bits,[condition_h,0.]);calls+=1
        checks['train_eval_max_logit_difference']=float((before-in_train).abs().max())
        model.eval()
        h_changed=model.logits(bits,[condition_h+.31,0.]);calls+=1
        checks['condition_change_max_logit_difference']=float((before-h_changed).abs().max())
    if checks['raw_normalization_error']>tolerance or checks['causality_max_logit_difference']>tolerance:
        raise ValueError('normalization/causality check failed')
    checks['raw_check_model_calls']=calls
    return checks


def run(c,arch,opt,mode,coverage,seed,tag):
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)
    if dirty:raise RuntimeError('Commit tracked source before scientific runs')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id=f'{stamp}-{tag}-{arch}-{opt}-{mode}-{coverage}-s{seed}'
    out=ROOT/'experiments/runs'/run_id;out.mkdir(parents=True,exist_ok=False)
    train=c['train_sets'][coverage];n=c['n']
    test=[h for points in c['test_regions'].values() for h in points]
    manifest=dict(schema_version=2,run_id=run_id,created_at_utc=stamp,status='running',
        owner='Codex-assisted local research; human contributions to be completed by members',issue=None,
        local_task_id=c['local_task_id'],publication='local-only',git_commit=commit,seed=seed,
        question='Does wider fixed-cardinality h coverage improve held-out prediction after training reliability is checked?',
        model=dict(architecture=arch,optimizer=opt,training_mode=mode,coverage=coverage),
        hamiltonian=dict(n=n,boundary='open',J=1,lambda_value=0.,hbar=1,basis='MSB left; Z(0)=+1'),
        parameter_split=dict(train=train,validation=c['validation'],test_regions=c['test_regions']),
        command=f'python scripts/run_round7.py --arch {arch} --optimizer {opt} --mode {mode} --coverage {coverage} --seed {seed} --tag {tag}',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         scipy=scipy.__version__,platform=platform.platform(),processor=platform.processor(),
                         device='cpu',threads=1,dtype='float64'),budget={},oracle_access=dict(training=False,
                         checkpoint_selection=False,evaluation='ED only after selected/final checkpoint freeze',lookup=False,ritz_basis=False),
        metrics={},artifacts=['config.json','metrics.json','training.json','amplitudes.json','checks.json'],
        config_sha256=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
        notes='Five seeds; full enumeration. Local prospective extension; same H work is not equal total compute. Even projection is a labeled physical control.')
    dump(out/'manifest.json',manifest);dump(out/'config.json',c)
    start=time.perf_counter()
    try:
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True);torch.manual_seed(seed);np.random.seed(seed)
        bits=torch.tensor(basis(n),dtype=torch.float64)
        model=(ConditionalMLP(n) if arch=='mlp' else ConditionalTransformer(n)).double()
        manifest['model']['parameters']=sum(p.numel() for p in model.parameters())
        manifest['model']['initial_parameter_hash']=weight_hash(model)
        mats={h:torch.tensor(hamiltonian(n,h),dtype=torch.float64) for h in train+c['validation']}
        training_calls=0
        def objective(current):
            nonlocal training_calls
            energies=[]
            for h in train:
                psi=state(current,bits,h,mode=='even');energies.append(psi@mats[h]@psi/n)
                training_calls+=1
            return torch.stack(energies).mean()
        selection=[];best=float('inf');selected=None;selected_at=0;next_check=c['validation_every_closures']
        def checkpoint(current,calls,steps,force=False):
            nonlocal best,selected,selected_at,next_check
            if not force and calls<next_check:return
            if selection and selection[-1]['closure']==calls:return
            score=accepted_validation(current,bits,n,mode,c['validation'],mats)
            selection.append(dict(closure=calls,accepted_optimizer_steps=steps,validation_energy_per_spin=score))
            if score<best:
                best=score;selected=copy.deepcopy(current.state_dict());selected_at=calls
            next_check=((calls//c['validation_every_closures'])+1)*c['validation_every_closures']
        result=fit(model,objective,opt,c['closure_evaluations'],adam_lr=c['adam_lr'],lbfgs_lr=c['lbfgs_lr'],
                   history_size=c['lbfgs_history_size'],deadline=time.monotonic()+c['timeout_seconds_per_run'],
                   on_accepted=checkpoint)
        checkpoint(model,result['closure_evaluations'],result['optimizer_steps_completed'],force=True)
        final=copy.deepcopy(model.state_dict())
        snapshots={'selected':selected,'final':final}
        frozen_hashes={}
        for name,weights in snapshots.items():model.load_state_dict(weights);frozen_hashes[name]=weight_hash(model)
        train_seconds=time.perf_counter()-start
        # No ED occurs above this point. Both endpoint states have now been fixed.
        all_h=sorted(train+c['validation']+test)
        matrices={h:hamiltonian(n,h) for h in all_h}
        oracles={h:eigh(matrices[h]) for h in all_h}
        rows=[];amplitudes={};checks={};post_h=0;eval_calls=0
        def location(h):
            if h in train:return 'train','train'
            if h in c['validation']:return 'validation','validation'
            region=next(k for k,v in c['test_regions'].items() if h in v)
            return ('interpolation' if train[0]<h<train[-1] else 'extrapolation'),region
        with torch.no_grad():
            for endpoint,weights in snapshots.items():
                model.load_state_dict(weights);model.eval()
                checks[endpoint]=check_model(model,bits,c['validation'][0],c['causality_tolerance'])
                states={h:state(model,bits,h,mode=='even').numpy() for h in all_h};eval_calls+=len(all_h)
                amplitudes[endpoint]={str(h):psi.tolist() for h,psi in states.items()}
                anchors=[states[h] for h in train]
                for h in all_h:
                    matrix=matrices[h];oracle=oracles[h]
                    span,rank=ritz(matrix,anchors);post_h+=rank
                    candidates=dict(direct=states[h],lookup=lookup(h,train,anchors),train_span_ritz=span)
                    split,region=location(h)
                    z=1-2*basis(n);abs_mz=np.abs(z.mean(axis=1))
                    exact_p=abs(oracle[1][:,0])**2
                    exact_abs_mz=float(exact_p@abs_mz)
                    for method,psi in candidates.items():
                        row=metrics(matrix,psi,n,oracle)
                        p=abs(psi)**2;p=p/p.sum()
                        mean_abs_mz=float(p@abs_mz)
                        row.update(endpoint=endpoint,h=h,split=split,region=region,method=method,
                            seed=seed,architecture=arch,optimizer=opt,training_mode=mode,coverage=coverage,
                            mean_abs_mz=mean_abs_mz,exact_mean_abs_mz=exact_abs_mz,
                            abs_mz_error=abs(mean_abs_mz-exact_abs_mz),nearest_train_distance=min(abs(h-t) for t in train),
                            outside_coverage_distance=max(train[0]-h,0.,h-train[-1]),ritz_rank=rank if method=='train_span_ritz' else 0)
                        rows.append(row)
        summary={}
        for endpoint in snapshots:
            for region in ['train','validation']+list(c['test_regions']):
                for method in ['direct','lookup','train_span_ritz']:
                    part=[r for r in rows if r['endpoint']==endpoint and r['region']==region and r['method']==method]
                    summary[f'{endpoint}/{region}/{method}']=dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in part])),
                        worst_infidelity=max(1-r['fidelity'] for r in part),mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in part])),
                        mean_abs_mz_error=float(np.mean([r['abs_mz_error'] for r in part])),worst_correlation_error=max(r['correlation_error'] for r in part))
        manifest['model'].update(selected_closure=selected_at,endpoint_parameter_hashes=frozen_hashes)
        validation_h=len(selection)*len(c['validation'])
        manifest['budget']=dict(closure_evaluations=result['closure_evaluations'],optimizer_steps_completed=result['optimizer_steps_completed'],
            discarded_final_step_evaluations=result['discarded_final_step_evaluations'],training_forward_h_actions=training_calls,
            training_backward_h_actions=training_calls,validation_h_actions=validation_h,
            training_network_calls=training_calls,validation_state_calls=validation_h,
            raw_network_calls_training=training_calls,raw_network_calls_validation=validation_h,
            training_projection_applications=training_calls if mode=='even' else 0,
            validation_projection_applications=validation_h if mode=='even' else 0,
            evaluation_state_calls=eval_calls,raw_check_model_calls=sum(v['raw_check_model_calls'] for v in checks.values()),
            postprocessing_h_actions=post_h,evaluation_h_actions=2*len(rows),evaluation_ed_calls=len(oracles),
            training_seconds=train_seconds,wall_seconds=time.perf_counter()-start,training_parameter_points=len(train),device='cpu')
        manifest['metrics']=summary;manifest['status']='completed'
        dump(out/'training.json',dict(closures=result['history'],validation=selection))
        dump(out/'metrics.json',rows);dump(out/'amplitudes.json',amplitudes);dump(out/'checks.json',checks)
        print(run_id,'completed',round(manifest['budget']['wall_seconds'],3),'seconds',flush=True)
    except Exception as exc:
        manifest['status']='failed';manifest['failure']=dict(type=type(exc).__name__,message=str(exc))
        manifest['budget']['wall_seconds']=time.perf_counter()-start
        print(run_id,'FAILED',type(exc).__name__,str(exc),flush=True)
    finally:
        dump(out/'manifest.json',manifest)
    return run_id,manifest['status']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='experiments/configs/round7.json')
    for arg in ['arch','optimizer','mode','coverage']:parser.add_argument('--'+arg)
    parser.add_argument('--seed',type=int);parser.add_argument('--tag',default='round7')
    args=parser.parse_args();c=json.loads((ROOT/args.config).read_text(encoding='utf-8'));validate_config(c)
    index_path=ROOT/'experiments/round7_index.json'
    index=json.loads(index_path.read_text(encoding='utf-8')) if index_path.exists() else dict(runs=[],failed_runs=[],status='running',local_task_id=c['local_task_id'])
    start=time.monotonic()
    for arch,opt,mode,coverage,seed in jobs(c):
        if any(v is not None and v!=actual for v,actual in [(args.arch,arch),(args.optimizer,opt),(args.mode,mode),(args.coverage,coverage),(args.seed,seed)]):continue
        if time.monotonic()-start>c['invocation_timeout_seconds']:raise TimeoutError('invocation cap')
        if args.tag=='round7':
            done=False
            for rid in index['runs']:
                m=json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text(encoding='utf-8'))
                if (m['model']['architecture'],m['model']['optimizer'],m['model']['training_mode'],m['model']['coverage'],m['seed'])==(arch,opt,mode,coverage,seed):done=True
            if done:continue
        rid,status=run(c,arch,opt,mode,coverage,seed,args.tag)
        if args.tag=='round7':
            index['runs' if status=='completed' else 'failed_runs'].append(rid)
            index['status']='completed' if len(index['runs'])==len(jobs(c)) else 'running'
            dump(index_path,index)
    print('Round7 index:',len(index['runs']),'completed;',len(index['failed_runs']),'failed',flush=True)


if __name__=='__main__':main()
