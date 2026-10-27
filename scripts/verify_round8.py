"""Independent Kronecker/NumPy audit of the frozen round8 counterfactual pilot.

This re-scores saved states; it does not replay training (the separate replay
receipt is checked separately). Local ignored checkpoints are optional in CI.
Numerical tolerances fixed before recalculation: atol2e-8, rtol1e-7; norm1e-9;
variational nesting2e-8. No production Hamiltonian or metric routine is used.
"""
import argparse
from collections import defaultdict
import datetime as dt
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
ATOL, RTOL, NORM_TOL, NESTING_TOL = 2e-8, 1e-7, 1e-9, 2e-8


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def kron_sites(n, operators):
    result = np.ones((1, 1))
    for site in range(n):
        result = np.kron(result, operators.get(site, np.eye(2)))
    return result


def reference_components(n, family):
    z, x = np.diag([1., -1.]), np.array([[0., 1.], [1., 0.]])
    if family == 'chain':
        edges = [(i, i+1) for i in range(n-1)]
    elif family == 'ladder':
        length = n//2
        edges = ([(i, i+1) for i in range(length-1)] +
                 [(i+length, i+length+1) for i in range(length-1)] +
                 [(i, i+length) for i in range(length)])
    else:
        raise AssertionError(family)
    h0 = -sum(kron_sites(n, {i:z, j:z}) for i, j in edges)
    hx = -sum(kron_sites(n, {i:x}) for i in range(n))
    hz = kron_sites(n, {0:z})
    zz = np.diag(kron_sites(n, {0:z, n-1:z})).copy()
    return h0, hx, hz, edges, np.diag(hz).copy(), np.diag(kron_sites(n, {n-1:z})).copy(), zz


def projected_basis(states, tolerance):
    u, singular, _ = np.linalg.svd(np.column_stack(states), full_matrices=False)
    return u[:, singular > tolerance*singular[0]], singular


def reference_metrics(H, state, oracle, n, z0, zl, zz):
    norm_error = abs(float(np.vdot(state, state).real)-1.)
    p = state/np.linalg.norm(state)
    values, vectors = oracle
    prob, truth = abs(p)**2, abs(vectors[:, 0])**2
    def corr(v):
        return float(v@zz-(v@z0)*(v@zl))
    energy = float(np.vdot(p, H@p).real)
    residual = H@p-energy*p
    multiplicity = int(np.sum(values-values[0] < 1e-10))
    return dict(energy=energy, energy_error=energy-values[0],
                energy_error_per_spin=(energy-values[0])/n,
                fidelity=abs(np.vdot(vectors[:, 0], p))**2,
                ground_space_fidelity=np.sum(abs(vectors[:, :multiplicity].conj().T@p)**2),
                ground_multiplicity=multiplicity, gap=values[1]-values[0],
                variance=float(np.vdot(residual, residual).real),
                parity=float(np.vdot(p, p[::-1]).real), magnetization=float(prob@z0),
                magnetization_error=abs(float((prob-truth)@z0)),
                connected_correlation=corr(prob), correlation_error=abs(corr(prob)-corr(truth)),
                norm_error=norm_error)


def without_timing(value):
    if isinstance(value, dict):
        return {k:without_timing(v) for k, v in value.items()
                if not k.endswith('_seconds') and k not in {'base_fit_seconds'}}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def check_replay(write=False):
    path = ROOT/'docs/results/round8/replay_check.json'
    receipt = read(path)
    if not write:assert receipt['status'] == 'passed'
    original = receipt.get('original_run_id', receipt.get('original_run'))
    replay = receipt.get('replay_run_id', receipt.get('replay_run'))
    assert original and replay and original != replay
    first, second = [ROOT/'experiments/runs'/rid for rid in [original, replay]]
    for name in ['predictions.json.gz', 'metrics.json', 'training.json']:
        assert (first/name).read_bytes() == (second/name).read_bytes(), name
    assert without_timing(read(first/'decisions.json')) == without_timing(read(second/'decisions.json'))
    a, b = [read(p/'manifest.json') for p in [first, second]]
    for key in ['job', 'git_commit', 'config_sha256', 'initial_model_fingerprint', 'local_checkpoints', 'metrics']:
        assert a[key] == b[key], key
    for key in a['budget']:
        if key != 'wall_seconds':
            assert a['budget'][key] == b['budget'][key], key
    result = dict(original_run_id=original, replay_run_id=replay,
                  status='passed', mode='exact saved histories, amplitudes, metrics, checkpoints and physical decisions; wall times excluded',
                  predictions_sha256=hashlib.sha256((first/'predictions.json.gz').read_bytes()).hexdigest(),
                  metric_rows=len(read(first/'metrics.json')), checkpoint_fingerprints=len(a['local_checkpoints']),
                  replay_is_extra_statistical_seed=False)
    if write:path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-checkpoints', action='store_true')
    parser.add_argument('--without-replay', action='store_true')
    parser.add_argument('--write-replay-check', action='store_true')
    args = parser.parse_args()
    if args.without_replay and args.write_replay_check:parser.error('replay flags are mutually exclusive')
    c = read(ROOT/'experiments/configs/round8.json')
    index = read(ROOT/'experiments/round8_index.json')
    assert index['status'] == 'completed' and index['failed_runs'] == []
    expected = set(itertools.product(c['families'], c['radii'], c['architectures'], c['seeds']))
    assert len(index['runs']) == len(set(index['runs'])) == len(expected) == 40
    seen = set(); sources = set(); initial = defaultdict(set); oracle_cache = {}; components = {}
    differences = defaultdict(float); total_rows = checkpoints = closures = nesting = projections = feature_checks = 0
    max_norm = 0.; minimum_energy_error = 0.; maximum_energy_nesting_violation = 0.
    if args.local_checkpoints:
        import torch
        from run_round8 import fingerprint
        torch.set_num_threads(1)

    def close(actual, expected_value, label):
        difference = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected_value))))
        differences[label] = max(differences[label], difference)
        np.testing.assert_allclose(actual, expected_value, atol=ATOL, rtol=RTOL, err_msg=label)

    for rid in index['runs']:
        folder = ROOT/'experiments/runs'/rid
        m, decisions, traces, rows = [read(folder/name) for name in ['manifest.json', 'decisions.json', 'training.json', 'metrics.json']]
        job = m['job']; cell = tuple(job[k] for k in ['family', 'radius', 'architecture', 'seed'])
        assert cell in expected and cell not in seen; seen.add(cell)
        assert m['status'] == 'completed' and m['issue'] == 19 and m['seed'] == job['seed']
        assert read(folder/'config.json') == c
        assert m['config_sha256'] == hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
        sources.add(m['git_commit']); initial[(job['architecture'], job['seed'])].add(m['initial_model_fingerprint'])
        assert not m['oracle_access']['training'] and not m['oracle_access']['selection']
        frozen = read(folder/'frozen_before_oracle.json'); compressed = (folder/'predictions.json.gz').read_bytes()
        assert hashlib.sha256(compressed).hexdigest() == frozen['predictions_sha256']
        assert frozen['source_commit'] == m['git_commit'] and frozen['decision_count'] == 3
        assert frozen['time_utc'] == m['branch_freeze_time']
        assert dt.datetime.fromisoformat(m['branch_freeze_time']) <= dt.datetime.fromisoformat(m['oracle_start_time'])
        payload = json.loads(gzip.decompress(compressed))
        assert len(decisions) == 3 and [d['maturity'] for d in decisions] == c['maturities']
        assert set(payload) == set(map(str, c['maturities']))
        assert len(rows) == 207 and len(traces) == 15 and len(m['local_checkpoints']) == 15
        assert {(t['maturity'], t['label']) for t in traces} == set(itertools.product(c['maturities'], ['base_0', 'base_1', 'refine_left', 'refine_right', 'acquire_center']))
        assert frozen['all_branch_checkpoint_fingerprints'] == [a['checkpoint_fingerprint'] for d in decisions for a in d['actions'].values() if a['checkpoint_fingerprint']]
        budget = m['budget']
        for name, value in {'closure_evaluations':4200, 'training_forward_h_actions':4200,
                            'training_backward_h_actions':4200, 'training_model_sweeps':4200,
                            'total_training_h_actions':8400, 'policy_probe_h_actions':15,
                            'warmstart_h_actions':6, 'evaluation_ed_calls':11,
                            'evaluation_h_actions':414, 'metric_rows':207}.items():
            assert budget[name] == value, (rid, name)
        assert budget['closure_evaluations'] == sum(t['closure_evaluations'] for t in traces)
        assert budget['discarded_closures'] == sum(t['discarded_final_step_evaluations'] for t in traces)
        checkpoints_by_path = {x['path']:x for x in m['local_checkpoints']}
        assert len(checkpoints_by_path) == 15
        assert budget['local_checkpoint_bytes'] == sum(x['bytes'] for x in m['local_checkpoints'])
        for trace in traces:
            maturity, label = trace['maturity'], trace['label']
            prior = {100:0, 400:100, 1200:400}[maturity]
            count = maturity-prior if label.startswith('base_') else 200
            cumulative = maturity if label.startswith('base_') else 200 if label == 'acquire_center' else maturity+200
            history = trace['history']
            assert trace['closure_evaluations'] == trace['attempted_closures'] == len(history) == count
            assert trace['cumulative_counters']['closure_evaluations'] == cumulative
            assert [x['evaluation'] for x in history] == list(range(1, count+1))
            assert [x['cumulative_evaluation'] for x in history] == list(range(cumulative-count+1, cumulative+1))
            assert trace['discarded_final_step_evaluations'] == sum(not x['accepted_step_completed'] for x in history)
            assert trace['optimizer_steps_completed'] == len({x['optimizer_call'] for x in history if x['accepted_step_completed']})
            assert all(np.isfinite(x['loss']) for x in history)
            closures += len(history)
        if job['family'] not in components:
            components[job['family']] = reference_components(c['n'], job['family'])
        H0, Hx, Hz, edges, z0, zl, zz = components[job['family']]
        assert m['hamiltonian']['edges'] == [list(e) for e in edges]
        h, radius = c['families'][job['family']]['h'], job['radius']
        queries = [radius*x for x in c['query_fractions']]; anchors = [-radius, radius]
        assert m['parameter_split'] == dict(train=anchors, prospective_acquisition=[0.], validation=[], test=queries)
        points = sorted(set(queries+anchors+[0.]))
        matrices = {}
        for lam in points:
            H = H0+h*Hx+lam*Hz; matrices[lam] = H
            key = job['family'], lam
            if key not in oracle_cache: oracle_cache[key] = np.linalg.eigh(H)
        row_map = {(r['maturity'], r['variant'], r['action'], r['split'], r['lam']):r for r in rows}
        assert len(row_map) == 207
        assert len(m['metrics']) == 24
        for maturity, variant, action in itertools.product(c['maturities'], c['basis_variants'], c['actions']):
            selected = [row_map[maturity, variant, action, 'query', lam] for lam in queries]
            aggregate = dict(mean_infidelity=float(np.mean([1-r['fidelity'] for r in selected])),
                             worst_infidelity=max(1-r['fidelity'] for r in selected),
                             mean_energy_error_per_spin=float(np.mean([r['energy_error_per_spin'] for r in selected])),
                             mean_magnetization_error=float(np.mean([r['magnetization_error'] for r in selected])),
                             worst_correlation_error=max(r['correlation_error'] for r in selected))
            for name, value in aggregate.items():close(m['metrics'][f'{maturity}/{variant}/{action}'][name], value, 'manifest_'+name)
        job_projection_actions = 0
        for decision in decisions:
            maturity = decision['maturity']; old = [np.asarray(p) for p in decision['base_anchor_states']]
            for p in old:
                error = abs(float(np.vdot(p, p).real)-1.); max_norm = max(max_norm, error)
                assert error <= NORM_TOL
            for i in range(2):
                assert decision['base_checkpoint_fingerprints'][i] == checkpoints_by_path[f'checkpoints/m{maturity}-base-{i}.pt']['state_fingerprint']
            assert list(decision['actions']) == c['actions']
            geometries = {}
            for variant in c['basis_variants']:
                states = old if variant == 'raw' else old+[p[::-1] for p in old]
                q, singular = projected_basis(states, c['rank_rtol']); geometries[variant] = q
                geometry = decision['base_geometry'][variant]
                assert geometry['rank'] == q.shape[1] and geometry['component_actions'] == 3*q.shape[1]
                close(singular, geometry['singular_values'], 'basis_singular_values')
                job_projection_actions += geometry['component_actions']; projections += 1
            def residual_norm(p, lam):
                hp = matrices[lam]@p; return float(np.linalg.norm(hp-float(np.vdot(p, hp).real)*p))
            ar = [residual_norm(p, x) for p, x in zip(old, anchors)]
            sr = []
            q = geometries['flip_closed']
            for lam in [anchors[0], 0., anchors[1]]:
                _, v = np.linalg.eigh(q.T@matrices[lam]@q)
                sr.append(residual_norm(q@v[:, 0], lam))
            features = decision['features']; close(ar, features['anchor_residuals'], 'anchor_residuals')
            close(sr, features['snapshot_residuals'], 'snapshot_residuals')
            declared_choices = dict(anchor_residual=['refine_left', 'refine_right'][int(np.argmax(features['anchor_residuals']))], snapshot_residual=['refine_left', 'acquire_center', 'refine_right'][int(np.argmax(features['snapshot_residuals']))])
            assert features['choices'] == declared_choices
            # Nearly symmetric endpoint residuals can swap order at eigensolver
            # roundoff. Check that the recorded selected feature remains a
            # maximizer within the predeclared numerical tolerance independently.
            for name, values, menu in [('anchor_residual', ar, ['refine_left','refine_right']), ('snapshot_residual', sr, ['refine_left','acquire_center','refine_right'])]:
                chosen = menu.index(features['choices'][name])
                assert max(values)-values[chosen] <= ATOL+RTOL*max(values), (rid, maturity, name, values)
            assert features['acquire_source'] == int(np.argmin(features['center_initial_energies_per_spin']))
            assert features['probe_h_actions'] == 5 and features['acquire_warmstart_h_actions'] == features['acquire_warmstart_model_sweeps'] == 2
            feature_checks += 1
            action_energies = {}
            for action in c['actions']:
                a = decision['actions'][action]; added = None if a['added_state'] is None else np.asarray(a['added_state'])
                assert a['closure_evaluations'] == (0 if action == 'stop' else 200)
                if action == 'stop':
                    assert added is None and a['checkpoint_fingerprint'] is None and a['train_lambda'] is None
                else:
                    error = abs(float(np.vdot(added, added).real)-1.); max_norm = max(max_norm, error)
                    assert error <= NORM_TOL
                    assert a['train_lambda'] == (0. if action == 'acquire_center' else anchors[0 if action == 'refine_left' else 1])
                    assert a['checkpoint_fingerprint'] == checkpoints_by_path[f'checkpoints/m{maturity}-{action}.pt']['state_fingerprint']
                ss = old if added is None else old+[added]
                for variant in c['basis_variants']:
                    q, singular = projected_basis(ss if variant == 'raw' else ss+[p[::-1] for p in ss], c['rank_rtol'])
                    v = a['variants'][variant]
                    assert v['rank'] == q.shape[1] and v['component_actions'] == (0 if added is None else 3*q.shape[1])
                    close(singular, v['singular_values'], 'basis_singular_values')
                    job_projection_actions += v['component_actions']; projections += 1
                    assert set(payload[str(maturity)][variant][action]) == set(map(str, queries))
                    for lam in queries:
                        p = np.asarray(payload[str(maturity)][variant][action][str(lam)])
                        assert p.shape == (2**c['n'],) and np.isfinite(p).all()
                        values, vectors = np.linalg.eigh(q.T@matrices[lam]@q)
                        close(np.linalg.norm(p-q@(q.T@p)), 0., 'prediction_subspace_residual')
                        energy = float(p@matrices[lam]@p)
                        close(energy, values[0], 'projected_ground_energy')
                        action_energies[variant, action, lam] = energy
                        r = row_map[maturity, variant, action, 'query', lam]
                        expected_metric = reference_metrics(matrices[lam], p, oracle_cache[job['family'], lam], c['n'], z0, zl, zz)
                        for name, value in expected_metric.items():close(r[name], value, 'metric_'+name)
                        max_norm = max(max_norm, expected_metric['norm_error']); minimum_energy_error = min(minimum_energy_error, expected_metric['energy_error'])
                        assert expected_metric['norm_error'] <= NORM_TOL and expected_metric['energy_error'] >= -NESTING_TOL
                        total_rows += 1
            for i, p in enumerate(old):
                lam = anchors[i]; r = row_map[maturity, 'state', f'base_{i}', 'anchor', lam]
                for name, value in reference_metrics(matrices[lam], p, oracle_cache[job['family'], lam], c['n'], z0, zl, zz).items():close(r[name], value, 'metric_'+name)
                total_rows += 1
            for action, a in decision['actions'].items():
                if a['added_state'] is None:continue
                p, lam = np.asarray(a['added_state']), a['train_lambda'];r = row_map[maturity, 'state', action, 'anchor', lam]
                for name, value in reference_metrics(matrices[lam], p, oracle_cache[job['family'], lam], c['n'], z0, zl, zz).items():close(r[name], value, 'metric_'+name)
                total_rows += 1
            for lam in queries:
                for variant in c['basis_variants']:
                    stop = action_energies[variant, 'stop', lam]
                    for action in c['actions'][1:]:
                        violation = action_energies[variant, action, lam]-stop
                        maximum_energy_nesting_violation = max(maximum_energy_nesting_violation, violation)
                        assert violation <= NESTING_TOL; nesting += 1
                for action in c['actions']:
                    violation = action_energies['flip_closed', action, lam]-action_energies['raw', action, lam]
                    maximum_energy_nesting_violation = max(maximum_energy_nesting_violation, violation)
                    assert violation <= NESTING_TOL; nesting += 1
        assert budget['projection_component_actions'] == job_projection_actions
        if args.local_checkpoints:
            for record in m['local_checkpoints']:
                path = folder/record['path']; assert path.stat().st_size == record['bytes']
                state = torch.load(path, map_location='cpu', weights_only=True)
                assert fingerprint(state) == record['state_fingerprint']; checkpoints += 1
                assert state['optimizer_name'] == c['optimizer']['name'] and state['settings'] == {k:v for k, v in c['optimizer'].items() if k != 'name'}
                assert state['schema_version'] == 1 and set(state['rng_state']) == {'torch_cpu', 'python', 'numpy'}
    assert seen == expected and len(sources) == 1 and all(len(x) == 1 for x in initial.values())
    source = next(iter(sources))
    for path in ['scripts/run_round8.py', 'src/ctqs/affine_spin.py', 'src/ctqs/resumable_fit.py', 'src/ctqs/models.py', 'src/ctqs/physics.py', 'experiments/configs/round8.json', 'docs/round8_preregistration.md']:
        frozen_source = subprocess.check_output(['git', 'show', source+':'+path], cwd=ROOT).replace(b'\r\n', b'\n')
        assert frozen_source == (ROOT/path).read_bytes().replace(b'\r\n', b'\n'), path
    replay = {'status':'not_requested'} if args.without_replay else check_replay(args.write_replay_check)
    report = dict(status='passed', completed_jobs=len(seen), failed_jobs=0, metric_rows=total_rows,
                  frozen_source_commit=source, independent_reference='Kronecker Hamiltonians and NumPy eig; no production physics/cache/metric calls',
                  numerical_tolerances=dict(atol=ATOL, rtol=RTOL, norm=NORM_TOL, energy_nesting=NESTING_TOL),
                  maximum_differences=dict(differences), maximum_norm_error=max_norm,
                  minimum_energy_error=minimum_energy_error, maximum_energy_nesting_violation=maximum_energy_nesting_violation,
                  closure_history_rows=closures, cached_basis_geometry_checks=projections,
                  energy_nesting_comparisons=nesting, pre_oracle_feature_checks=feature_checks,
                  paired_initial_hash_groups=len(initial), local_checkpoints_verified=checkpoints,
                  residual_choice_check='Recorded argmax exact; independent chosen residual agrees with maximum within declared tolerance because symmetric endpoint ties can exchange at eigensolver roundoff.',
                  replay=replay, statistical_unit='seed', independent_human_review=False,
                  limitations='Saved timestamps/declarations verify recorded ordering, not an external tamper-proof oracle-access log. Local checkpoints are excluded from Git.')
    path = ROOT/'docs/results/round8/audit.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
