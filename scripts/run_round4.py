"""Frozen initial-function and Adam-coordinate controls. ED is evaluation-only."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.conditioning import conditioned_state
from ctqs.coordinates import compensate_initialization, canonical_adam_step
from ctqs.models import ConditionalMLP, ConditionalTransformer
from ctqs.physics import basis, hamiltonian, lookup, metrics, normalized, ritz
from run_round1 import dump
from run_round2 import state_hash


def jobs(cfg):
    for h, arch, seed, arm in itertools.product(cfg['h_values'], cfg['architectures'], cfg['seeds'], cfg['arms']):
        yield dict(h=h, architecture=arch, seed=seed, arm=arm)


def key(job):
    return tuple(job[k] for k in ['h', 'architecture', 'seed', 'arm'])


def amplitude_hash(values):
    return hashlib.sha256(np.stack(values).astype('<f8').tobytes()).hexdigest()


def run(cfg, job, tag):
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True):
        raise RuntimeError('commit tracked changes before experiments')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    h, arch, seed, arm = key(job); n = cfg['n']; scale = 1. if arm == 'physical' else cfg['scale']
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rid = f'{stamp}-{tag}-n{n}-h{h:g}-{arch}-{arm}-s{seed}'
    out = ROOT / 'experiments/runs' / rid; out.mkdir(parents=True)
    splits = {k: cfg[k] for k in ['train', 'validation', 'test']}
    m = dict(schema_version=1, run_id=rid, created_at_utc=stamp, status='running', owner='Bian-M-X/Codex', issue=7,
             git_commit=commit, seed=seed, question='Separate initial functions and Adam coordinates under domain scaling', job=job,
             model=dict(architecture=arch, arm=arm, condition_scale=scale, training_mode='raw'),
             hamiltonian=dict(n=n, h=h, lambda_value='variable', boundary='open', J=1, basis='MSB left; Z(0)=+1'),
             parameter_split=splits, command=f'python scripts/run_round4.py --h {h} --arch {arch} --seed {seed} --arm {arm} --tag {tag}',
             environment=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, scipy=scipy.__version__,
                              platform=platform.platform(), device='cpu', dtype='float64', torch_threads=1),
             budget={}, oracle_access=dict(training=False, selection=False, evaluation='ED after checkpoint freeze', postprocessing=False),
             metrics={}, artifacts=['config.json', 'training.json', 'metrics.json', 'amplitudes.json'],
             notes='Screening with paired seeds; canonical arm is a coordinate-equivalence control, not a new optimizer.')
    m['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    dump(out / 'config.json', cfg); dump(out / 'manifest.json', m)
    start = time.monotonic(); history = []; train_h = val_h = steps = 0
    try:
        torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64); torch.use_deterministic_algorithms(True)
        model = (ConditionalMLP(n) if arch == 'mlp' else ConditionalTransformer(n)).double()
        m['sampled_state_sha256'] = state_hash(model)
        bits = torch.tensor(basis(n), dtype=torch.float64)
        model.eval()
        with torch.no_grad():
            reference = [conditioned_state(model, bits, h, x)[0].numpy().copy() for x in cfg['train']]
        if arm in ['matched', 'canonical']: compensate_initialization(model, scale)
        m['initial_state_sha256'] = state_hash(model)
        def predict(x): return conditioned_state(model, bits, h, x, scale)[0]
        with torch.no_grad(): initial = [predict(x).numpy().copy() for x in cfg['train']]
        m['initial_function'] = dict(reference_hash=amplitude_hash(reference), actual_hash=amplitude_hash(initial),
                                    max_abs_difference=float(np.max(np.abs(np.stack(reference)-np.stack(initial)))))
        amplitude_records = dict(initial_train=initial, selected={}, final={})
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], eps=cfg['adam_epsilon'], foreach=False)
        matrices = {x: torch.tensor(hamiltonian(n, h, x)) for x in cfg['train']+cfg['validation']}
        best = float('inf'); selected = None; selected_step = 0
        for step in range(1, cfg['steps']+1):
            if time.monotonic()-start > cfg['timeout_seconds_per_run']: raise TimeoutError('frozen run cap exceeded')
            model.train(); optimizer.zero_grad(); energies = []
            for x in cfg['train']:
                psi = predict(x); energies.append(psi @ matrices[x] @ psi / n); train_h += 1
            loss = torch.stack(energies).mean()
            if not torch.isfinite(loss): raise ValueError('nonfinite training loss')
            loss.backward()
            if arm == 'canonical': canonical_adam_step(optimizer, model, scale)
            else: optimizer.step()
            steps = step
            if step % cfg['validation_every'] == 0:
                model.eval(); validation = []
                with torch.no_grad():
                    for x in cfg['validation']:
                        psi = predict(x); validation.append(float(psi @ matrices[x] @ psi / n)); val_h += 1
                score = float(np.mean(validation))
                history.append(dict(step=step, train_energy_per_spin=float(loss.detach()), validation_energy_per_spin=score))
                if score < best: best, selected_step, selected = score, step, copy.deepcopy(model.state_dict())
                dump(out / 'training.json', history)
        final = copy.deepcopy(model.state_dict()); fit_end = time.monotonic()
        m['model'].update(parameters=sum(p.numel() for p in model.parameters()), selected_step=selected_step)
        # Evaluation begins only after both checkpoints have been frozen.
        oracles = {x: eigh(hamiltonian(n, h, x)) for xs in splits.values() for x in xs}
        rows = []; eval_calls = post_h = eval_h = 0
        for checkpoint, weights in [('selected', selected), ('final', final)]:
            model.load_state_dict(weights); model.eval(); m[checkpoint+'_state_sha256'] = state_hash(model)
            cache = {}
            def cached(x):
                nonlocal eval_calls
                if x not in cache: cache[x] = predict(x).detach().numpy(); eval_calls += 1
                return cache[x]
            with torch.no_grad():
                train_states = [cached(x) for x in cfg['train']]
                for split, xs in splits.items():
                    for x in xs:
                        mat = hamiltonian(n, h, x); direct = cached(x)
                        span, rank = ritz(mat, train_states); post_h += rank
                        predictions = dict(direct=direct, lookup=lookup(x, cfg['train'], train_states), train_span_ritz=span,
                                           post_covariant=normalized(direct+cached(-x)[::-1]))
                        amplitude_records[checkpoint][str(x)] = direct
                        for method, psi in predictions.items():
                            row = metrics(mat, psi, n, oracles[x]); eval_h += 2
                            row.update(checkpoint=checkpoint, method=method, split=split, x=x, h=h, lam=x); rows.append(row)
        for checkpoint, split, method in itertools.product(['selected', 'final'], splits, predictions):
            rr = [r for r in rows if (r['checkpoint'], r['split'], r['method']) == (checkpoint, split, method)]
            m['metrics'][f'{checkpoint}/{split}/{method}'] = dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),
                worst_infidelity=max(1-r['fidelity'] for r in rr), mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),
                mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in rr])), worst_correlation_error=max(r['correlation_error'] for r in rr))
        m['budget'] = dict(gradient_steps=steps, training_forward_h_actions=train_h, training_backward_h_actions=train_h,
            validation_h_actions=val_h, total_training_selection_h_actions=2*train_h+val_h, initialization_model_calls=6,
            training_model_calls=train_h, validation_model_calls=val_h, evaluation_model_calls=eval_calls,
            coordinate_compensated_steps=steps if arm == 'canonical' else 0,
            postprocessing_h_actions=post_h, evaluation_h_actions=eval_h, exact_diagonalizations=len(oracles),
            fit_selection_seconds=fit_end-start, evaluation_seconds=time.monotonic()-fit_end, wall_seconds=time.monotonic()-start)
        dump(out / 'metrics.json', rows)
        dump(out / 'amplitudes.json', dict(initial_train=[x.tolist() for x in initial],
             **{c: {x: y.tolist() for x, y in amplitude_records[c].items()} for c in ['selected', 'final']}))
        m['status'] = 'completed'; print(rid, 'completed', round(m['budget']['wall_seconds'], 2), 'seconds', flush=True)
    except Exception as exc:
        m['status'] = 'failed'; m['notes'] += f' Failure {type(exc).__name__}: {exc}'
        m['budget'] = dict(gradient_steps=steps, training_forward_h_actions=train_h, validation_h_actions=val_h, wall_seconds=time.monotonic()-start)
        raise
    finally:
        dump(out / 'training.json', history); dump(out / 'manifest.json', m)
    return rid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float); ap.add_argument('--arch', choices=['mlp', 'transformer']); ap.add_argument('--seed', type=int)
    ap.add_argument('--arm', choices=['physical', 'scaled', 'matched', 'canonical']); ap.add_argument('--tag', default='round4')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]+', args.tag): ap.error('tag must be a lowercase slug')
    cfg = json.loads((ROOT / 'experiments/configs/round4.json').read_text())
    points = sum([cfg[k] for k in ['train', 'validation', 'test']], [])
    if len(points) != len(set(points)): raise ValueError('duplicate or leaking grid')
    path = ROOT / 'experiments/round4_index.json'
    index = json.loads(path.read_text()) if path.exists() else dict(runs=[], failed_runs=[], status='running')
    seen = {key(json.loads((ROOT / 'experiments/runs' / r / 'manifest.json').read_text())['job']) for r in index['runs']}
    started = time.monotonic()
    try:
        for job in jobs(cfg):
            if any(getattr(args, a) is not None and job[b] != getattr(args, a) for a, b in [('h','h'),('arch','architecture'),('seed','seed'),('arm','arm')]): continue
            if args.tag == 'round4' and key(job) in seen: continue
            if time.monotonic()-started > cfg['timeout_seconds_per_invocation']: raise TimeoutError('invocation cap exceeded')
            rid = run(cfg, job, args.tag)
            if args.tag == 'round4': index['runs'].append(rid); dump(path, index)
        index['status'] = 'completed' if len(index['runs']) == len(list(jobs(cfg))) else 'partial'
    except Exception:
        index['status'] = 'partial'
        raise
    finally:
        if args.tag == 'round4':
            index['failed_runs'] = [p.parent.name for p in sorted((ROOT / 'experiments/runs').glob('*-round4-*/manifest.json')) if json.loads(p.read_text())['status'] == 'failed']
            dump(path, index)


if __name__ == '__main__': main()
