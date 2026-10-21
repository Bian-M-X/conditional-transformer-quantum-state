"""Stdlib audit of complete factorial, budgets, coordinate control and replay."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def compare(a, b, atol, rtol):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([compare(a[k], b[k], atol, rtol) for k in a] or [0.])
    if isinstance(a, list):
        assert len(a) == len(b)
        return max([compare(x, y, atol, rtol) for x, y in zip(a, b)] or [0.])
    if isinstance(a, (int, float)):
        assert math.isfinite(a) and math.isfinite(b)
        assert math.isclose(a, b, abs_tol=atol, rel_tol=rtol), (a, b)
        return abs(a-b)
    assert a == b
    return 0.


def audit(with_replay=True):
    cfg = read(ROOT / 'experiments/configs/round4.json'); index = read(ROOT / 'experiments/round4_index.json')
    assert index['status'] == 'completed' and len(index['runs']) == 80 and not index['failed_runs']
    expected = set(itertools.product(cfg['h_values'], cfg['architectures'], cfg['seeds'], cfg['arms']))
    records = {}; commits = set(); count = 0
    for rid in index['runs']:
        directory = ROOT / 'experiments/runs' / rid; m = read(directory / 'manifest.json'); j = m['job']
        key = (j['h'], j['architecture'], j['seed'], j['arm'])
        assert key in expected and key not in records
        records[key] = (directory, m)
        assert read(directory / 'config.json') == cfg
        assert m['config_sha256'] == hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        assert m['status'] == 'completed' and m['issue'] == 7 and m['seed'] == j['seed']
        assert not m['oracle_access']['training'] and not m['oracle_access']['selection']
        assert m['parameter_split'] == {k: cfg[k] for k in ['train', 'validation', 'test']}
        if m['git_commit'] not in commits:
            raw = subprocess.check_output(['git', 'show', m['git_commit']+':experiments/configs/round4.json'], cwd=ROOT, text=True)
            assert json.loads(raw) == cfg; commits.add(m['git_commit'])
        b = m['budget']; assert b['gradient_steps'] == 1200
        assert b['training_forward_h_actions'] == b['training_backward_h_actions'] == b['training_model_calls'] == 3600
        assert b['validation_h_actions'] == b['validation_model_calls'] == 96
        assert b['initialization_model_calls'] == 6 and b['total_training_selection_h_actions'] == 7296
        assert b['coordinate_compensated_steps'] == (1200 if j['arm'] == 'canonical' else 0)
        assert b['exact_diagonalizations'] == 17 and b['evaluation_model_calls'] == 34
        assert b['wall_seconds'] > 0
        history = read(directory / 'training.json'); rows = read(directory / 'metrics.json')
        amplitudes = read(directory / 'amplitudes.json'); count += len(rows)
        assert [x['step'] for x in history] == list(range(50, 1201, 50))
        assert m['model']['selected_step'] == min(history, key=lambda r: r['validation_energy_per_spin'])['step']
        assert len(rows) == 136 and b['evaluation_h_actions'] == 272
        if j['arm'] != 'scaled': assert m['initial_function']['max_abs_difference'] <= cfg['control_atol']
        if j['arm'] == 'scaled': assert m['initial_function']['max_abs_difference'] > cfg['control_atol']
        rowkeys = set()
        for r in rows:
            ident = (r['checkpoint'], r['split'], r['method'], r['x'])
            assert ident not in rowkeys; rowkeys.add(ident)
            assert r['x'] in cfg[r['split']] and r['h'] == j['h']
            assert all(math.isfinite(v) for v in r.values() if isinstance(v, (float, int)))
            assert -1e-10 <= r['fidelity'] <= 1+1e-10 and r['norm_error'] < 1e-10
            assert r['energy_error'] >= -1e-9 and r['variance'] >= -1e-12
            if r['method'] == 'direct':
                psi = amplitudes[r['checkpoint']][str(r['x'])]
                assert len(psi) == 64 and abs(sum(x*x for x in psi)-1) < 1e-10
        assert len(amplitudes['initial_train']) == 3
        for checkpoint, split, method in itertools.product(['selected', 'final'], ['train', 'validation', 'test'], ['direct', 'lookup', 'train_span_ritz', 'post_covariant']):
            rr = [r for r in rows if (r['checkpoint'], r['split'], r['method']) == (checkpoint, split, method)]
            assert len(rr) == len(cfg[split])
            mean = sum(1-r['fidelity'] for r in rr)/len(rr)
            assert abs(mean-m['metrics'][f'{checkpoint}/{split}/{method}']['mean_infidelity']) < 1e-12
    assert set(records) == expected
    differences = []; exact = 0
    for h, arch, seed in itertools.product(cfg['h_values'], cfg['architectures'], cfg['seeds']):
        p, pm = records[(h, arch, seed, 'physical')]; c, cm = records[(h, arch, seed, 'canonical')]
        group = [records[(h, arch, seed, a)][1] for a in cfg['arms']]
        assert len({m['sampled_state_sha256'] for m in group}) == 1
        assert pm['model']['selected_step'] == cm['model']['selected_step']
        row = dict(h=h, architecture=arch, seed=seed)
        is_exact = True
        for file in ['training.json', 'amplitudes.json', 'metrics.json']:
            a, b = read(p/file), read(c/file)
            row[file] = compare(a, b, cfg['control_atol'], cfg['control_rtol'])
            is_exact = is_exact and a == b
        row['exact'] = is_exact; exact += is_exact; differences.append(row)
    if with_replay:
        info = read(ROOT / 'docs/results/round4/replay_check.json')
        assert info['original'] in index['runs'] and info['replay'] not in index['runs']
        a, b = [ROOT / 'experiments/runs' / info[k] for k in ['original', 'replay']]
        for name in ['config.json', 'training.json', 'metrics.json', 'amplitudes.json']: assert read(a/name) == read(b/name)
        am, bm = read(a/'manifest.json'), read(b/'manifest.json')
        assert bm['status'] == 'completed' and am['job'] == bm['job'] and am['git_commit'] == bm['git_commit']
        for name in ['sampled_state_sha256', 'initial_state_sha256', 'selected_state_sha256', 'final_state_sha256']: assert am[name] == bm[name]
    return dict(jobs=len(records), metric_rows=count, coordinate_pairs=20, exact_coordinate_pairs=exact, pairs=differences,
                replay='exact' if with_replay else 'not_checked', commits=sorted(commits))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--without-replay', action='store_true'); ap.add_argument('--write', action='store_true')
    args = ap.parse_args(); result = audit(not args.without_replay)
    if args.write:
        out = ROOT / 'docs/results/round4/control_audit.json'; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(f"PASS: {result['jobs']} jobs, {result['metric_rows']} metric rows, {result['exact_coordinate_pairs']}/20 exact coordinate-control pairs; replay {result['replay']}")
