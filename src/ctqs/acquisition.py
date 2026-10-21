"""Oracle-free acquisition scores and symmetry-covariant prediction controls."""
import numpy as np
from .physics import normalized, project_even


def score_states(matrices, states, policy):
    """Only H and predicted vectors are accepted. H-action count is explicit."""
    if policy not in ('variance', 'ritz_gain'):
        raise ValueError('unknown acquisition score')
    if len(matrices) != len(states):
        raise ValueError('matrix/state counts differ')
    scores, actions = [], 0
    for matrix, state in zip(matrices, states):
        psi = normalized(state)
        hpsi = matrix @ psi
        energy = float(np.vdot(psi, hpsi).real)
        actions += 1
        if policy == 'variance':
            residual = hpsi - energy * psi
            scores.append(float(np.vdot(residual, residual).real))
        else:
            # ritz applies H to each retained column of an orthonormal basis.
            # Its small eigensolve has dimension <=2, not 2**N ED.
            u, singular, _ = np.linalg.svd(np.column_stack([psi, psi[::-1]]), full_matrices=False)
            q = u[:, singular > 1e-10 * singular[0]]
            projected = q.conj().T @ (matrix @ q)
            actions += q.shape[1]
            corrected_energy = float(np.linalg.eigvalsh(projected)[0])
            scores.append(max(0., energy-corrected_energy))
    if not np.all(np.isfinite(scores)):
        raise ValueError('nonfinite score')
    return scores, actions


def select_unseen(candidates, selected, scores):
    available = [i for i, value in enumerate(candidates) if value not in selected]
    if not available:
        raise ValueError('candidate pool exhausted')
    # Explicit stable tie-break by fixed grid order.
    return candidates[max(available, key=lambda i: (scores[i], -i))]


def covariant_controls(predict, lam):
    direct = predict(lam)
    opposite = predict(-lam) if lam != 0 else direct
    sign_map = (direct if lam > 0 else opposite[::-1]) if lam else project_even(direct)
    average = normalized(direct + opposite[::-1])
    return direct, normalized(sign_map), average


def next_batch_size(total, training, acquisition, requested):
    remaining = total-training-acquisition
    if remaining < 0:
        raise ValueError('H-action budget exceeded')
    return min(requested, remaining)
