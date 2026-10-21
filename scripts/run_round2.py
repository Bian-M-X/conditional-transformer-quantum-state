"""Budget diagnostic and cost-accounted acquisition screening. No ED in selection."""
import argparse
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
from ctqs.acquisition import covariant_controls, next_batch_size, score_states, select_unseen
from ctqs.models import ConditionalMLP, ConditionalTransformer
from ctqs.physics import basis, hamiltonian, lookup, metrics, ritz
from run_round1 import dump, run as run_h


def validate_acquisition_config(cfg):
    initial, candidates, test = map(set,(cfg['initial_lambda'],cfg['candidate_lambda'],cfg['test_lambda']))
    if (initial|candidates)&test:
        raise ValueError('test leakage: candidate/train/test overlap')
    if len(candidates)!=len(cfg['candidate_lambda']) or len(test)!=len(cfg['test_lambda']):
        raise ValueError('duplicate grid values')
    for name in ('uniform_additions','log_additions'):
        extra=cfg[name]
        if len(set(extra))!=len(cfg['acquisition_steps']) or not set(extra)<=candidates-initial:
            raise ValueError('invalid predeclared additions')
    if cfg['total_h_actions'] < 2*max(cfg['acquisition_steps'])+3*3*len(candidates):
        raise ValueError('insufficient budget for fixed acquisition schedule')


def state_hash(model):
    digest=hashlib.sha256()
    for key,tensor in sorted(model.state_dict().items()):
        digest.update(key.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_acquisition(cfg,arch,policy,seed,config_path,tag='round2b'):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True):
        raise RuntimeError('commit tracked changes before running')
    stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rid=f'{stamp}-{tag}-{arch}-{policy}-s{seed}'
    out=ROOT/'experiments/runs'/rid;out.mkdir(parents=True)
    m=dict(schema_version=1,run_id=rid,created_at_utc=stamp,status='running',owner='Bian-M-X/Codex',issue=3,
        git_commit=commit,seed=seed,question='Does acquisition beat continued training after charging selection cost and testing covariance controls?',
        model=dict(architecture=arch,policy=policy,training_mode='raw'),
        hamiltonian=dict(n=cfg['n'],h=cfg['h'],boundary='open',J=1,basis='MSB left, Z(0)=+1'),
        parameter_split=dict(train=cfg['initial_lambda'],validation=None,test=cfg['test_lambda'],candidates=cfg['candidate_lambda']),
        command=f'python scripts/run_round2.py --stage b --arch {arch} --policy {policy} --seed {seed} --tag {tag}',
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,
                         platform=platform.platform(),device='cpu',dtype='float64',torch_threads=1),
        budget={},oracle_access=dict(training=False,selection=False,evaluation='dense ED',postprocessing=False),metrics={},
        artifacts=['config.json','training.json','acquisition.json','metrics.json'],
        notes='Prespecified screening. H-action budget matched; gradient updates, forward costs and wall time differ. No novelty or independent replication claim.')
    m['config_sha256']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    dump(out/'manifest.json',m);dump(out/'config.json',cfg)
    start=time.monotonic();history=[];acquisition=[]
    train_actions=query_actions=steps=training_calls=query_calls=0
    acquisition_seconds=0.;pool=list(cfg['initial_lambda'])
    try:
        torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        model=(ConditionalMLP(cfg['n']) if arch=='mlp' else ConditionalTransformer(cfg['n'])).double()
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
        batch_rng=np.random.default_rng(seed)
        acquisition_rng=np.random.default_rng(seed+100000)
        bits=torch.tensor(basis(cfg['n']),dtype=torch.float64)
        candidates=cfg['candidate_lambda']
        matrices={lam:hamiltonian(cfg['n'],cfg['h'],lam) for lam in set(candidates)|set(pool)}
        tensors={lam:torch.tensor(mat) for lam,mat in matrices.items()}
        while True:
            if time.monotonic()-start>cfg['timeout_seconds_per_run']:
                raise TimeoutError('predeclared per-run timeout')
            batch=next_batch_size(cfg['total_h_actions'],train_actions,query_actions,cfg['batch_size'])
            if batch==0:break
            optimizer.zero_grad();model.train()
            lam_batch=batch_rng.choice(pool,size=batch,replace=True)
            losses=[]
            for lam in lam_batch:
                psi=model.psi(bits,[cfg['h'],float(lam)])
                losses.append(psi@tensors[float(lam)]@psi/cfg['n'])
            loss=torch.stack(losses).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite loss')
            loss.backward();optimizer.step()
            steps+=1;train_actions+=batch;training_calls+=batch
            if steps%100==0 or next_batch_size(cfg['total_h_actions'],train_actions,query_actions,cfg['batch_size'])==0:
                history.append(dict(step=steps,training_h_actions=train_actions,query_h_actions=query_actions,
                                    energy_per_spin=float(loss.detach()),pool=list(pool)))
            if steps==cfg['acquisition_steps'][0]:
                m['common_prefix_hash']=state_hash(model)
            if steps in cfg['acquisition_steps'] and policy!='continue':
                event=cfg['acquisition_steps'].index(steps)
                tic=time.monotonic();scores=None;cost=0
                if policy in ('variance','ritz_gain'):
                    model.eval()
                    with torch.no_grad():
                        states=[model.psi(bits,[cfg['h'],lam]).numpy() for lam in candidates]
                    query_calls+=len(candidates)
                    scores,cost=score_states([matrices[lam] for lam in candidates],states,policy)
                    chosen=select_unseen(candidates,pool,scores)
                elif policy=='random':
                    chosen=float(acquisition_rng.choice([v for v in candidates if v not in pool]))
                else:
                    chosen=cfg[policy+'_additions'][event]
                query_actions+=cost
                pool.append(chosen)
                elapsed=time.monotonic()-tic;acquisition_seconds+=elapsed
                acquisition.append(dict(step=steps,selected_lambda=chosen,candidates=candidates if scores is not None else [],
                                        scores=scores,h_actions=cost,seconds=elapsed))
                dump(out/'acquisition.json',acquisition)
            if steps%100==0:dump(out/'training.json',history)
        train_end=time.monotonic();model.eval()
        m['final_state_sha256']=state_hash(model)
        assert train_actions+query_actions==cfg['total_h_actions']
        # No oracle exists until training and acquisition have finished.
        rows=[];cache={};post_h=eval_h=eval_calls=0
        def predict(lam):
            nonlocal eval_calls
            if lam not in cache:
                cache[lam]=model.psi(bits,[cfg['h'],lam]).detach().numpy();eval_calls+=1
            return cache[lam]
        with torch.no_grad():
            train_lam=sorted(pool);train_states=[predict(lam) for lam in train_lam]
            for lam in train_lam+cfg['test_lambda']:
                matrix=hamiltonian(cfg['n'],cfg['h'],lam)
                oracle=eigh(matrix)
                direct,sign_map,average=covariant_controls(predict,lam)
                span,rank=ritz(matrix,train_states);post_h+=rank
                predictions=dict(direct=direct,lookup=lookup(lam,train_lam,train_states),sign_map=sign_map,
                                 covariant_average=average,train_span_ritz=span)
                for method,psi in predictions.items():
                    row=metrics(matrix,psi,cfg['n'],oracle);eval_h+=2
                    row.update(lam=lam,h=cfg['h'],method=method,seed=seed,architecture=arch,policy=policy,
                        split='train' if lam in pool else 'test',sign='negative' if lam<0 else 'positive' if lam>0 else 'zero',
                        relative_support='interpolation' if min(pool)<=lam<=max(pool) else 'extrapolation',
                        ritz_rank=rank if method=='train_span_ritz' else 0)
                    rows.append(row)
        for method in predictions:
            rr=[r for r in rows if r['split']=='test' and r['method']==method]
            m['metrics'][method]=dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),
                worst_infidelity=max(1-r['fidelity'] for r in rr),mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),
                worst_correlation_error=max(r['correlation_error'] for r in rr),mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in rr])))
        m['parameter_split']['train']=train_lam
        m['model']['parameters']=sum(p.numel() for p in model.parameters())
        m['budget']=dict(gradient_steps=steps,training_h_actions=train_actions,acquisition_h_actions=query_actions,
            total_training_selection_h_actions=train_actions+query_actions,training_model_calls=training_calls,acquisition_model_calls=query_calls,
            h_action_scope='forward applications; backward and evaluation counted separately (ADR 0003)',
            backward_h_actions=train_actions,total_with_backward_h_actions=2*train_actions+query_actions,
            evaluation_model_calls=eval_calls,postprocessing_h_actions=post_h,evaluation_h_actions=eval_h,
            exact_diagonalizations=len(train_lam)+len(cfg['test_lambda']),acquisition_seconds=acquisition_seconds,
            initialization_training_seconds=train_end-start-acquisition_seconds,fit_and_selection_seconds=train_end-start,
            evaluation_seconds=time.monotonic()-train_end,wall_seconds=time.monotonic()-start,device='cpu',
            prediction_forward_calls_nominal=dict(direct=1,lookup=0,sign_map=1,covariant_average=2,train_span_ritz=0))
        dump(out/'metrics.json',rows);dump(out/'training.json',history);dump(out/'acquisition.json',acquisition)
        m['status']='completed'
        print(rid,'completed',round(m['budget']['wall_seconds'],2),'seconds',flush=True)
    except Exception as exc:
        m['status']='failed';m['notes']+=f' Failure: {type(exc).__name__}: {exc}'
        m['budget']=dict(gradient_steps=steps,training_h_actions=train_actions,acquisition_h_actions=query_actions,wall_seconds=time.monotonic()-start)
        dump(out/'training.json',history);dump(out/'acquisition.json',acquisition)
        raise
    finally:dump(out/'manifest.json',m)
    return rid


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['a','b','all'],default='all')
    ap.add_argument('--arch',choices=['mlp','transformer'])
    ap.add_argument('--policy',choices=['continue','uniform','log','random','variance','ritz_gain'])
    ap.add_argument('--seed',type=int)
    ap.add_argument('--tag',default='round2b')
    args=ap.parse_args()
    index_path=ROOT/'experiments/round2_index.json'
    if index_path.exists():index=json.loads(index_path.read_text())
    else:index=dict(stage_a_reused=json.loads((ROOT/'experiments/round1_index.json').read_text())['training_runs'],stage_a_new=[],stage_b=[],status='running')
    started=time.monotonic()
    def check_time():
        if time.monotonic()-started>1800:
            raise TimeoutError('30-minute suite cap reached; keep incomplete index')
    if args.stage in ('a','all'):
        existing=set()
        for rid in index['stage_a_reused']+index['stage_a_new']:
            m=json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text())
            existing.add((m['model']['architecture'],m['model']['training_mode'],m['seed'],m['budget']['gradient_steps']))
        for budget in (600,1200,2400):
            config_path=f'experiments/configs/round2_budget_{budget}.json'
            cfg=json.loads((ROOT/config_path).read_text())
            for arch in [args.arch] if args.arch else cfg['architectures']:
                for mode in cfg['training_modes']:
                    for seed in [args.seed] if args.seed is not None else cfg['seeds']:
                        if (arch,mode,seed,budget) in existing:continue
                        check_time();rid=run_h(cfg,arch,mode,seed,config_path,3,'round2a')
                        index['stage_a_new'].append(rid);dump(index_path,index)
    if args.stage in ('b','all'):
        config_path='experiments/configs/round2_acquisition.json'
        cfg=json.loads((ROOT/config_path).read_text());validate_acquisition_config(cfg)
        existing=set()
        for rid in index['stage_b']:
            m=json.loads((ROOT/'experiments/runs'/rid/'manifest.json').read_text())
            existing.add((m['model']['architecture'],m['model']['policy'],m['seed']))
        for arch in [args.arch] if args.arch else cfg['architectures']:
            for policy in [args.policy] if args.policy else cfg['policies']:
                for seed in [args.seed] if args.seed is not None else cfg['seeds']:
                    if (arch,policy,seed) in existing and args.tag=='round2b':continue
                    check_time();rid=run_acquisition(cfg,arch,policy,seed,config_path,args.tag)
                    if args.tag=='round2b':index['stage_b'].append(rid);dump(index_path,index)
    index['status']='completed' if len(index['stage_a_new'])==48 and len(index['stage_b'])==60 else 'partial'
    dump(index_path,index)


if __name__=='__main__':
    try:
        main()
    except Exception:
        index_path=ROOT/'experiments/round2_index.json'
        if index_path.exists():
            index=json.loads(index_path.read_text())
            index['status']='partial'
            index['failed_runs']=[p.parent.name for p in sorted((ROOT/'experiments/runs').glob('*round2*/manifest.json'))
                                  if json.loads(p.read_text())['status']=='failed']
            dump(index_path,index)
        raise
