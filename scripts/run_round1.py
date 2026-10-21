"""Frozen topic-12 screening. Every training job writes an independent manifest."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import scipy
from scipy.linalg import eigh
import torch
from ctqs.models import ConditionalMLP, ConditionalTransformer, state
from ctqs.physics import basis, hamiltonian, lookup, metrics, project_even, ritz


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def validate_config(cfg):
    groups = [cfg[x] for x in ('train', 'validation', 'interpolation', 'extrapolation')]
    flattened = [h for group in groups for h in group]
    if len(set(flattened)) != len(flattened) or min(flattened) <= 0:
        raise ValueError('positive, disjoint splits required')
    if sorted(cfg['train']) != cfg['train']:
        raise ValueError('train must be sorted')
    if not all(cfg['train'][0] < h < cfg['train'][-1] for h in cfg['interpolation']):
        raise ValueError('invalid interpolation split')
    if not all(h < cfg['train'][0] or h > cfg['train'][-1] for h in cfg['extrapolation']):
        raise ValueError('invalid extrapolation split')


def run(cfg, arch, mode, seed, config_path, issue, tag):
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_id = f'{stamp}-{tag}-{arch}-{mode}-s{seed}'
    out = ROOT / 'experiments' / 'runs' / run_id
    out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked_dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True)
    if tracked_dirty:
        raise RuntimeError('Commit tracked changes before scientific runs')
    manifest = dict(schema_version=1, run_id=run_id, created_at_utc=stamp, status='running',
                    owner='Bian-M-X/Codex', issue=issue, git_commit=commit, seed=seed,
                    question='Does direct conditional prediction outperform learned-state lookup, and does known symmetry remove low-field failure?',
                    model=dict(architecture=arch, training_mode=mode),
                    hamiltonian=dict(n=cfg['n'], boundary='open', J=1, lambda_value=0, basis='MSB left; Z(0)=+1'),
                    parameter_split={k:cfg[k] for k in ('train','validation','interpolation','extrapolation')},
                    command=f'python scripts/run_round1.py --config {config_path} --arch {arch} --mode {mode} --seed {seed} --issue {issue} --tag {tag}',
                    environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                                     torch=torch.__version__, platform=platform.platform(), processor=platform.processor(),
                                     device='cpu', threads=1, dtype='float64'),
                    budget={}, oracle_access=dict(training=False, selection=False, evaluation='dense ED', postprocessing=False),
                    metrics={}, artifacts=['metrics.json','training.json','config.json'],
                    notes='Screening only; exact enumeration. Same steps/H actions, NOT equal wall time across architectures. No independent reproduction yet.')
    manifest['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    dump(out / 'manifest.json', manifest)
    dump(out / 'config.json', cfg)
    start = time.monotonic()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float64)
        torch.use_deterministic_algorithms(True)
        n = cfg['n']
        bits = torch.tensor(basis(n), dtype=torch.float64)
        model = (ConditionalMLP(n) if arch == 'mlp' else ConditionalTransformer(n)).double()
        optim = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
        even = mode == 'even'
        # Training/selection only construct H at train and validation points.
        mats = {h:torch.tensor(hamiltonian(n, h)) for h in cfg['train'] + cfg['validation']}
        best, best_step, selected = float('inf'), 0, None
        history = []
        validation_calls = 0
        for step in range(1, cfg['steps'] + 1):
            if time.monotonic() - start > cfg['timeout_seconds_per_run']:
                raise TimeoutError('predeclared run timeout')
            model.train()
            optim.zero_grad()
            energies = []
            for h in cfg['train']:
                psi = state(model, bits, h, even)
                energies.append(psi @ mats[h] @ psi / n)
            loss = torch.stack(energies).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training energy')
            loss.backward()
            optim.step()
            if step % cfg['validation_every'] == 0 or step == cfg['steps']:
                model.eval()
                with torch.no_grad():
                    validation = []
                    for h in cfg['validation']:
                        psi = state(model, bits, h, even)
                        validation.append(float(psi @ mats[h] @ psi / n))
                    score = float(np.mean(validation))
                validation_calls += len(validation)
                history.append(dict(step=step, train_energy_per_spin=float(loss.detach()), validation_energy_per_spin=score))
                if score < best:
                    best, best_step, selected = score, step, copy.deepcopy(model.state_dict())
        training_seconds = time.monotonic() - start
        model.load_state_dict(selected)
        model.eval()
        # All parameters/checkpoint are now fixed; only now construct test ED oracles.
        rows = []
        postprocess_h_actions = 0
        evaluation_h_actions = 0
        with torch.no_grad():
            train_states = [state(model, bits, h, even).numpy() for h in cfg['train']]
            for split in ('train', 'validation', 'interpolation', 'extrapolation'):
                for h in cfg[split]:
                    matrix = hamiltonian(n, h)
                    oracle = eigh(matrix)
                    direct = state(model, bits, h, even).numpy()
                    span, rank = ritz(matrix, train_states)
                    parity, prank = ritz(matrix, [direct, direct[::-1]])
                    postprocess_h_actions += rank + prank
                    candidates = dict(direct=direct, lookup=lookup(h, cfg['train'], train_states),
                                      even_projection=project_even(direct), train_span_ritz=span, parity_ritz=parity)
                    for name, psi in candidates.items():
                        row = metrics(matrix, psi, n, oracle)
                        row.update(h=h, split=split, method=name, seed=seed, architecture=arch, training_mode=mode,
                                   ritz_rank=rank if name=='train_span_ritz' else prank if name=='parity_ritz' else 0)
                        rows.append(row)
                        evaluation_h_actions += 2
        summary = {}
        for split in ('train','validation','interpolation','extrapolation'):
            for method in candidates:
                rr = [r for r in rows if r['split']==split and r['method']==method]
                summary[f'{split}/{method}'] = dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in rr])),
                    worst_infidelity=max(1-r['fidelity'] for r in rr),
                    mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in rr])),
                    worst_correlation_error=max(r['correlation_error'] for r in rr))
        manifest['metrics'] = summary
        manifest['model'].update(parameters=sum(p.numel() for p in model.parameters()), selected_step=best_step)
        manifest['budget'] = dict(gradient_steps=cfg['steps'], training_hamiltonian_actions=cfg['steps']*len(cfg['train']),
                                  validation_hamiltonian_actions=validation_calls, postprocess_hamiltonian_actions=postprocess_h_actions,
                                  evaluation_hamiltonian_actions=evaluation_h_actions, training_seconds=training_seconds,
                                  wall_seconds=time.monotonic()-start, device='cpu', training_parameter_points=len(cfg['train']),
                                  exact_diagonalizations=sum(len(cfg[k]) for k in ('train','validation','interpolation','extrapolation')))
        dump(out / 'metrics.json', rows)
        dump(out / 'training.json', history)
        manifest['status'] = 'completed'
        print(run_id, 'completed', round(manifest['budget']['wall_seconds'],2), 'seconds', flush=True)
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['notes'] += f' FAILURE: {type(exc).__name__}: {exc}'
        manifest['budget']['wall_seconds'] = time.monotonic()-start
        raise
    finally:
        dump(out / 'manifest.json', manifest)
    return run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/configs/round1.json')
    ap.add_argument('--arch', choices=['mlp','transformer'])
    ap.add_argument('--mode', choices=['raw','even'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--issue', type=int, default=1)
    ap.add_argument('--tag', default='round1')
    args = ap.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    validate_config(cfg)
    for arch in [args.arch] if args.arch else cfg['architectures']:
        for mode in [args.mode] if args.mode else cfg['training_modes']:
            for seed in [args.seed] if args.seed is not None else cfg['seeds']:
                run(cfg, arch, mode, seed, args.config, args.issue, args.tag)


if __name__ == '__main__':
    main()
