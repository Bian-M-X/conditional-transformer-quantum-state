"""Independent dense reference. Site 0 is the leftmost (most significant) bit."""
import numpy as np
from scipy.linalg import eigh


def basis(n):
    if not isinstance(n, int) or n < 2:
        raise ValueError('n must be an integer >= 2')
    return ((np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1)


def hamiltonian(n, h, lam=0.0):
    z = 1 - 2 * basis(n)
    # Integer zero is a valid lambda; off-diagonal transverse fields are real.
    matrix = np.diag((-np.sum(z[:, :-1] * z[:, 1:], axis=1) + lam * z[:, 0]).astype(float))
    indices = np.arange(1 << n)
    for site in range(n):
        matrix[indices, indices ^ (1 << site)] -= h
    return matrix


def normalized(state):
    state = np.asarray(state)
    norm = np.linalg.norm(state)
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError('state has zero or invalid norm')
    return state / norm


def project_even(state):
    return normalized(state + state[::-1])


def ritz(matrix, states, rtol=1e-10):
    """Rank-revealing Rayleigh-Ritz; no exact eigenvectors enter this routine."""
    u, s, _ = np.linalg.svd(np.column_stack(states), full_matrices=False)
    if s[0] < 1e-14:
        raise ValueError('empty variational subspace')
    q = u[:, s > rtol * s[0]]
    values, vectors = eigh(q.conj().T @ matrix @ q)
    return normalized(q @ vectors[:, 0]), len(values)


def lookup(h, train_h, states):
    """Interpolate learned positive amplitudes; clamp outside training support."""
    xs = np.asarray(train_h)
    if np.any(np.diff(xs) <= 0):
        raise ValueError('training points must be strictly sorted')
    if h <= xs[0]:
        return normalized(states[0])
    if h >= xs[-1]:
        return normalized(states[-1])
    i = np.searchsorted(xs, h) - 1
    t = (h - xs[i]) / (xs[i + 1] - xs[i])
    return normalized((1 - t) * states[i] + t * states[i + 1])


def metrics(matrix, state, n, oracle=None):
    """Oracle used only here for evaluation; report degenerate-space overlap too."""
    norm_error = abs(float(np.vdot(state, state).real) - 1)
    state = normalized(state)
    values, vectors = eigh(matrix) if oracle is None else oracle
    ground = vectors[:, 0]
    z = 1 - 2 * basis(n)
    p, p0 = abs(state) ** 2, abs(ground) ** 2
    corr = lambda prob: float(prob @ (z[:, 0] * z[:, -1]) - (prob @ z[:, 0]) * (prob @ z[:, -1]))
    energy = float(np.vdot(state, matrix @ state).real)
    residual = matrix @ state - energy * state
    multiplicity = int(np.sum(values - values[0] < 1e-10))
    return dict(energy=energy, energy_error=energy - float(values[0]),
                energy_error_per_spin=(energy - float(values[0])) / n,
                fidelity=float(abs(np.vdot(ground, state)) ** 2),
                ground_space_fidelity=float(np.sum(abs(vectors[:, :multiplicity].conj().T @ state) ** 2)),
                ground_multiplicity=multiplicity, gap=float(values[1] - values[0]),
                variance=float(np.vdot(residual, residual).real),
                parity=float(np.vdot(state, state[::-1]).real),
                magnetization=float(p @ z[:, 0]),
                magnetization_error=abs(float((p - p0) @ z[:, 0])),
                connected_correlation=corr(p), correlation_error=abs(corr(p) - corr(p0)),
                norm_error=norm_error)
