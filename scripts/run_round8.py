"""Checkpointed one-step allocation ceiling; ground labels enter only at the end."""
import argparse
import copy
import datetime as dt
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import re
import struct
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.affine_spin import SpinFamily,chain_edges,ladder_edges,CachedRitz
from ctqs.models import ConditionalMLP,ConditionalTransformer
from ctqs.physics import basis,metrics
from ctqs.resumable_fit import ResumableFit


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(value):
    h=hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            h.update(b'T'+str(x.dtype).encode()+str(tuple(x.shape)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'D')
            for key in sorted(x,key=repr):add(key);add(x[key])
        elif isinstance(x,(list,tuple)):
            h.update(b'L' if isinstance(x,list) else b'Q')
            for item in x:add(item)
            h.update(b';')
        elif isinstance(x,float):h.update(b'F'+struct.pack('>d',x))
        else:h.update((type(x).__name__+':'+str(x)+';').encode())
    add(value);return h.hexdigest()


def jobs(c):
    for family,radius,arch,seed in itertools.product(c['families'],c['radii'],c['architectures'],c['seeds']):
        yield dict(family=family,radius=radius,architecture=arch,seed=seed)


def key(j):return tuple(j[k] for k in ['family','radius','architecture','seed'])


def validate(c):
    if c['n']!=8 or c['maturities']!=[100,400,1200]:raise ValueError('fixed pilot dimensions/maturities')
    q=c['query_fractions']
    if len(q)!=len(set(q)) or any(not -1<x<1 or x==0 for x in q):raise ValueError('queries must exclude prospective anchors')
    if c['actions']!=['stop','refine_left','refine_right','acquire_center']:raise ValueError('fixed action menu')
    if c['primary_variant']!='flip_closed' or c['primary_maturity']!=1200:raise ValueError('fixed primary endpoint')


def predict(model,bits,h,lam):
    mode=model.training;model.eval()
    with torch.no_grad():p=model.psi(bits,[h,lam]).cpu().numpy()
    model.train(mode);return p


def residual(family,p,h,lam):
    hp=family.apply_numpy(p,h,lam);energy=float(np.vdot(p,hp).real)
    return float(np.linalg.norm(hp-energy*p))


def simple_choices(anchor_residuals,snapshot_residuals):
    i=int(np.argmax(anchor_residuals));j=int(np.argmax(snapshot_residuals))
    return dict(anchor_residual=['refine_left','refine_right'][i],
                snapshot_residual=['refine_left','acquire_center','refine_right'][j])


def run(c,j,tag):
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):raise RuntimeError('freeze tracked sources first')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    fam,radius,arch,seed=key(j);n=c['n'];h=c['families'][fam]['h'];anchors=[-radius,radius]
    queries=[radius*x for x in c['query_fractions']]
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rid=f'{stamp}-{tag}-{fam}-r{radius:g}-{arch}-s{seed}';out=ROOT/'experiments/runs'/rid;out.mkdir(parents=True)
    m=dict(schema_version=1,run_id=rid,created_at_utc=stamp,status='running',owner='Bian-M-X/Codex',issue=19,git_commit=commit,seed=seed,job=j,
      question='Does hindsight one-step R/A choice have robust headroom after retained snapshots and symmetry closure?',
      model=dict(architecture=arch,training='independent endpoint models; final accepted; append branches'),
      hamiltonian=dict(n=n,h=h,family=fam,lambda_value='variable',boundary='open',J=1),
      parameter_split=dict(train=anchors,prospective_acquisition=[0.],validation=[],test=queries),
      command=f'python scripts/run_round8.py --family {fam} --radius {radius} --arch {arch} --seed {seed} --tag {tag}',
      environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),device='cpu',dtype='float64',torch_threads=1),
      budget={},oracle_access=dict(training=False,selection=False,evaluation='after every maturity and branch of job frozen',scope='hindsight comparator only'),
      metrics={},artifacts=['config.json','training.json','decisions.json','predictions.json.gz','metrics.json','frozen_before_oracle.json'],
      notes='N8 development; exact projection, no sampling action; counterfactual research cost is not deployable policy cost. Local .pt checkpoints ignored by Git.')
    m['config_sha256']=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    dump(out/'manifest.json',m);dump(out/'config.json',c)
    start=time.monotonic();deadline=start+c['timeout_seconds_per_job'];traces=[];decisions=[];payload={};local_ckpts=[]
    counts=dict(training_forward_h_actions=0,training_backward_h_actions=0,training_model_sweeps=0)
    try:
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        family=SpinFamily(n,chain_edges(n) if fam=='chain' else ladder_edges(n));m['hamiltonian']['edges']=family.edges
        bits=torch.tensor(basis(n),dtype=torch.float64)
        base=(ConditionalMLP(n) if arch=='mlp' else ConditionalTransformer(n)).double()
        opt=c['optimizer'];sessions=[ResumableFit(copy.deepcopy(base),opt['name'],opt['lr'],opt['history_size']) for _ in anchors]
        m['initial_model_fingerprint']=fingerprint(base.state_dict());m['model']['parameters_per_model']=sum(p.numel() for p in base.parameters())
        def objective(x):
            def loss(model):
                p=model.psi(bits,[h,x]);counts['training_forward_h_actions']+=1;counts['training_model_sweeps']+=1
                return (p@family.apply_torch(p,h,x))/n
            return loss
        def train(session,x,budget,stage,label):
            t=time.monotonic();result=session.advance(objective(x),budget,deadline=deadline)
            seconds=time.monotonic()-t;counts['training_backward_h_actions']+=result['closure_evaluations']
            traces.append(dict(maturity=stage,label=label,train_lambda=x,**result))
            return result,seconds
        def save(session,name):
            s=session.state_dict();p=out/'checkpoints'/f'{name}.pt';p.parent.mkdir(exist_ok=True)
            torch.save(s,p);fp=fingerprint(s)
            local_ckpts.append(dict(path=str(p.relative_to(out)).replace('\\','/'),state_fingerprint=fp,bytes=p.stat().st_size))
            return s,fp
        previous=0
        for maturity in c['maturities']:
            if time.monotonic()>deadline:raise TimeoutError('job cap exceeded')
            base_times=[];snapshots=[];state_hashes=[]
            for i,x in enumerate(anchors):
                _,seconds=train(sessions[i],x,maturity-previous,maturity,f'base_{i}');base_times.append(seconds)
                s,fp=save(sessions[i],f'm{maturity}-base-{i}');snapshots.append(s);state_hashes.append(fp)
            old=[predict(s.model,bits,h,x) for s,x in zip(sessions,anchors)]
            base_cache={};base_geometry={};base_projection_seconds={}
            for variant in c['basis_variants']:
                states=old if variant=='raw' else old+[p[::-1] for p in old]
                t=time.monotonic();cache=CachedRitz(family,states,c['rank_rtol']);base_projection_seconds[variant]=time.monotonic()-t
                base_cache[variant]=cache;base_geometry[variant]=dict(rank=cache.rank,singular_values=cache.singular_values.tolist(),component_actions=cache.component_actions)
            t=time.monotonic();a_res=[residual(family,p,h,x) for p,x in zip(old,anchors)]
            probe_points=[anchors[0],0.,anchors[1]]
            p_res=[residual(family,base_cache['flip_closed'].solve(h,x)['state'],h,x) for x in probe_points]
            probe_seconds=time.monotonic()-t;choices=simple_choices(a_res,p_res)
            # Two honest training-energy queries choose a center warm start.
            warm_start=time.monotonic();warm=[predict(s.model,bits,h,0.) for s in sessions]
            warm_energy=[float(p@family.apply_numpy(p,h,0.))/n for p in warm];source=int(np.argmin(warm_energy))
            warm_seconds=time.monotonic()-warm_start
            d=dict(maturity=maturity,base_checkpoint_fingerprints=state_hashes,base_fit_seconds=base_times,
                   base_anchor_states=[p.tolist() for p in old],base_geometry=base_geometry,base_projection_seconds=base_projection_seconds,
                   features=dict(anchor_residuals=a_res,snapshot_residuals=p_res,choices=choices,probe_h_actions=5,
                                 probe_seconds=probe_seconds,center_initial_energies_per_spin=warm_energy,acquire_source=source,
                                 acquire_warmstart_h_actions=2,acquire_warmstart_model_sweeps=2,warmstart_seconds=warm_seconds),actions={})
            payload[str(maturity)]={variant:{} for variant in c['basis_variants']}
            for action in c['actions']:
                added=None;fit_seconds=0.;result=None;fp=None
                if action!='stop':
                    i=source if action=='acquire_center' else (0 if action=='refine_left' else 1)
                    model=copy.deepcopy(sessions[i].model)
                    session=ResumableFit(model,opt['name'],opt['lr'],opt['history_size'])
                    if action!='acquire_center':session.load_state_dict(snapshots[i])
                    x=0. if action=='acquire_center' else anchors[i]
                    result,fit_seconds=train(session,x,c['action_closures'],maturity,action)
                    added=predict(session.model,bits,h,x);_,fp=save(session,f'm{maturity}-{action}')
                action_info=dict(added_state=None if added is None else added.tolist(),train_lambda=None if added is None else x,
                    checkpoint_fingerprint=fp,fit_seconds=fit_seconds,closure_evaluations=0 if result is None else result['closure_evaluations'],
                    discarded_closures=0 if result is None else result['discarded_final_step_evaluations'],variants={})
                for variant in c['basis_variants']:
                    if added is None:cache=base_cache[variant];construction=0.
                    else:
                        ss=old+[added];ss=ss if variant=='raw' else ss+[p[::-1] for p in ss]
                        t=time.monotonic();cache=CachedRitz(family,ss,c['rank_rtol']);construction=time.monotonic()-t
                    # Time only small-matrix energy/observable queries. Reconstruct separately.
                    t=time.monotonic();solutions=[cache.solve(h,x,reconstruct=False) for x in queries];query_seconds=time.monotonic()-t
                    t=time.monotonic();pred=[cache.q@s['coefficients'] for s in solutions];reconstruction_seconds=time.monotonic()-t
                    payload[str(maturity)][variant][action]={str(x):p.tolist() for x,p in zip(queries,pred)}
                    action_info['variants'][variant]=dict(rank=cache.rank,singular_values=cache.singular_values.tolist(),
                        component_actions=0 if added is None else cache.component_actions,projection_seconds=construction,
                        small_query_seconds=query_seconds,full_reconstruction_seconds=reconstruction_seconds)
                d['actions'][action]=action_info
            decisions.append(d);previous=maturity
            dump(out/'training.json',traces);dump(out/'decisions.json',decisions)
        # Serialize all physical predictions BEFORE obtaining any ground oracle.
        compressed=gzip.compress(json.dumps(payload,sort_keys=True,allow_nan=False).encode(),mtime=0)
        (out/'predictions.json.gz').write_bytes(compressed)
        frozen=dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),source_commit=commit,
                    predictions_sha256=hashlib.sha256(compressed).hexdigest(),decision_count=len(decisions),
                    all_branch_checkpoint_fingerprints=[a['checkpoint_fingerprint'] for d in decisions for a in d['actions'].values() if a['checkpoint_fingerprint']])
        dump(out/'frozen_before_oracle.json',frozen);m['branch_freeze_time']=frozen['time_utc']
        m['oracle_start_time']=dt.datetime.now(dt.timezone.utc).isoformat();oracle_points=sorted(set(queries+anchors+[0.]))
        matrices={x:family.apply_numpy(np.eye(2**n),h,x) for x in oracle_points};oracles={x:eigh(H) for x,H in matrices.items()}
        rows=[]
        for d in decisions:
            maturity=d['maturity']
            for variant,aa in payload[str(maturity)].items():
                for action,qq in aa.items():
                    for xstr,p in qq.items():
                        x=float(xstr);r=metrics(matrices[x],np.asarray(p),n,oracles[x]);r.update(maturity=maturity,variant=variant,action=action,split='query',lam=x);rows.append(r)
            for i,p in enumerate(d['base_anchor_states']):
                x=anchors[i];r=metrics(matrices[x],np.asarray(p),n,oracles[x]);r.update(maturity=maturity,variant='state',action=f'base_{i}',split='anchor',lam=x);rows.append(r)
            for action,a in d['actions'].items():
                if a['added_state'] is not None:
                    x=a['train_lambda'];r=metrics(matrices[x],np.asarray(a['added_state']),n,oracles[x]);r.update(maturity=maturity,variant='state',action=action,split='anchor',lam=x);rows.append(r)
        for maturity,variant,action in itertools.product(c['maturities'],c['basis_variants'],c['actions']):
            rr=[r for r in rows if (r['maturity'],r['variant'],r['action'],r['split'])==(maturity,variant,action,'query')]
            m['metrics'][f'{maturity}/{variant}/{action}']=dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),
              worst_infidelity=max(1-r['fidelity'] for r in rr),mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),
              mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in rr])),worst_correlation_error=max(r['correlation_error'] for r in rr))
        m['budget']=dict(**counts,total_training_h_actions=counts['training_forward_h_actions']+counts['training_backward_h_actions'],
          closure_evaluations=sum(t['closure_evaluations'] for t in traces),discarded_closures=sum(t['discarded_final_step_evaluations'] for t in traces),
          projection_component_actions=sum(sum(g['component_actions'] for g in d['base_geometry'].values())+sum(v['component_actions'] for a in d['actions'].values() for v in a['variants'].values()) for d in decisions),
          policy_probe_h_actions=sum(d['features']['probe_h_actions'] for d in decisions),warmstart_h_actions=2*len(decisions),
          evaluation_ed_calls=len(oracles),evaluation_h_actions=2*len(rows),metric_rows=len(rows),wall_seconds=time.monotonic()-start,
          local_checkpoint_bytes=sum(x['bytes'] for x in local_ckpts))
        m['local_checkpoints']=local_ckpts;m['status']='completed';dump(out/'metrics.json',rows)
        print(rid,'completed',round(m['budget']['wall_seconds'],2),'seconds',flush=True)
    except Exception as exc:
        m['status']='failed';m['notes']+=f' FAILURE {type(exc).__name__}: {exc}'
        m['budget']=dict(**counts,wall_seconds=time.monotonic()-start);raise
    finally:
        dump(out/'training.json',traces);dump(out/'decisions.json',decisions);dump(out/'manifest.json',m)
    return rid


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['chain','ladder']);ap.add_argument('--radius',type=float)
    ap.add_argument('--arch',choices=['mlp','transformer']);ap.add_argument('--seed',type=int);ap.add_argument('--tag',default='round8');a=ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+',a.tag):ap.error('tag must be a slug')
    c=read(ROOT/'experiments/configs/round8.json');validate(c);path=ROOT/'experiments/round8_index.json'
    idx=read(path) if path.exists() else dict(runs=[],failed_runs=[],status='running')
    seen={key(read(ROOT/'experiments/runs'/r/'manifest.json')['job']) for r in idx['runs']};start=time.monotonic()
    try:
        for j in jobs(c):
            if any(getattr(a,x) is not None and getattr(a,x)!=j[y] for x,y in [('family','family'),('radius','radius'),('arch','architecture'),('seed','seed')]):continue
            if a.tag=='round8' and key(j) in seen:continue
            if time.monotonic()-start>c['timeout_seconds_per_invocation']:raise TimeoutError('invocation cap exceeded')
            rid=run(c,j,a.tag)
            if a.tag=='round8':idx['runs'].append(rid);dump(path,idx)
        idx['status']='completed' if len(idx['runs'])==len(list(jobs(c))) else 'partial'
    except Exception:idx['status']='partial';raise
    finally:
        if a.tag=='round8':
            idx['failed_runs']=[p.parent.name for p in (ROOT/'experiments/runs').glob('*-round8-*/manifest.json') if read(p)['status']=='failed']
            dump(path,idx)


if __name__=='__main__':main()
