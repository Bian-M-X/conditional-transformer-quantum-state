"""Matrix-free affine spin Hamiltonians and once-built Rayleigh--Ritz caches.

Site zero is the most significant bit, as in the independent dense reference.
The family is H(h, lam) = H0 + h Hx + lam Hz, with H0=-sum_edges ZiZj,
Hx=-sum_i Xi, and Hz=sum_field_sites Zi. No exact eigensystem is used here.
"""
import numpy as np
from scipy.linalg import eigh


def _valid_n(n):
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError('n must be an integer >= 2')


def chain_edges(n):
    _valid_n(n)
    return [(i, i + 1) for i in range(n - 1)]


def ladder_edges(n):
    """Two open rails followed by the rungs; top rail sites precede bottom."""
    _valid_n(n)
    if n < 4 or n % 2:
        raise ValueError('a two-leg ladder requires an even n >= 4')
    length = n // 2
    return ([(i, i + 1) for i in range(length - 1)]
            + [(length + i, length + i + 1) for i in range(length - 1)]
            + [(i, length + i) for i in range(length)])


class SpinFamily:
    def __init__(self, n, edges, field_sites=(0,)):
        _valid_n(n)
        canonical = []
        for edge in edges:
            if len(edge) != 2:
                raise ValueError('each edge has two sites')
            i, j = edge
            if (isinstance(i, bool) or isinstance(j, bool)
                    or not isinstance(i, (int, np.integer))
                    or not isinstance(j, (int, np.integer))
                    or not 0 <= i < n or not 0 <= j < n or i == j):
                raise ValueError('edge sites must be distinct and in range')
            canonical.append(tuple(sorted((int(i), int(j)))))
        if len(canonical) != len(set(canonical)):
            raise ValueError('duplicate undirected edge')
        fields = tuple(field_sites)
        if (any(isinstance(i, bool) or not isinstance(i, (int, np.integer))
                or not 0 <= i < n for i in fields)
                or len(fields) != len(set(fields))):
            raise ValueError('field sites must be unique sites in range')
        self.n = n
        self.dimension = 1 << n
        self.edges = tuple(canonical)
        self.field_sites = tuple(int(i) for i in fields)
        indices = np.arange(self.dimension, dtype=np.int64)
        bits = (indices[:, None] >> np.arange(n - 1, -1, -1)) & 1
        self.z = (1 - 2 * bits).astype(np.float64)
        self.diagonal_h0 = np.zeros(self.dimension)
        for i, j in self.edges:
            self.diagonal_h0 -= self.z[:, i] * self.z[:, j]
        self.diagonal_hz = self.z[:, self.field_sites].sum(axis=1)
        self.flip_indices = indices[None, :] ^ (1 << np.arange(n - 1, -1, -1))[:, None]
        self._torch_cache = {}

    def _check_shape(self, v):
        if v.ndim not in (1, 2) or v.shape[0] != self.dimension:
            raise ValueError('input must have shape (2**n,) or (2**n, r)')

    def components_numpy(self, v):
        v = np.asarray(v)
        self._check_shape(v)
        diagonal_shape = (self.dimension,) + (1,) * (v.ndim - 1)
        return (self.diagonal_h0.reshape(diagonal_shape) * v,
                -v[self.flip_indices].sum(axis=0),
                self.diagonal_hz.reshape(diagonal_shape) * v)

    def apply_numpy(self, v, h, lam=0.):
        constant, transverse, longitudinal = self.components_numpy(v)
        return constant + h * transverse + lam * longitudinal

    def apply_torch(self, v, h, lam=0.):
        """Apply using indexed flips; retain derivatives in v, h and lam."""
        import torch
        self._check_shape(v)
        if not (v.is_floating_point() or v.is_complex()):
            raise ValueError('torch amplitudes must have floating or complex dtype')
        key = (v.device, v.dtype)
        if key not in self._torch_cache:
            self._torch_cache[key] = (
                torch.as_tensor(self.diagonal_h0, dtype=v.dtype, device=v.device),
                torch.as_tensor(self.diagonal_hz, dtype=v.dtype, device=v.device),
                torch.as_tensor(self.flip_indices, dtype=torch.long, device=v.device))
        h0, hz, flips = self._torch_cache[key]
        diagonal_shape = (self.dimension,) + (1,) * (v.ndim - 1)
        return ((h0.reshape(diagonal_shape) + lam * hz.reshape(diagonal_shape)) * v
                - h * v[flips].sum(dim=0))


class CachedRitz:
    """SVD and affine projections once; target queries use only small matrices.

    ``component_actions`` counts three component actions per retained vector.
    It is an accounting quantity, not a claim of equal component operation cost.
    Reconstructing the full state is explicitly optional at query time.
    """
    def __init__(self, family, states, rtol=1e-10):
        if not np.isfinite(rtol) or not 0 < rtol < 1:
            raise ValueError('rtol must lie strictly between zero and one')
        states = list(states)
        if not states:
            raise ValueError('nonempty state list required')
        columns = []
        for state in states:
            state = np.asarray(state)
            if state.shape != (family.dimension,) or not np.isfinite(state).all():
                raise ValueError('states must be finite vectors with dimension 2**n')
            if abs(np.linalg.norm(state) - 1) > 1e-9:
                raise ValueError('cached basis states must be normalized')
            columns.append(state)
        matrix = np.column_stack(columns)
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        if s[0] < 1e-14:
            raise ValueError('empty variational subspace')
        self.family = family
        self.singular_values = s
        self.gram = matrix.conj().T @ matrix
        self.q = u[:, s > rtol * s[0]]
        self.rank = self.q.shape[1]
        self.component_actions = 3 * self.rank
        self.component_matrices = {}
        for name, applied in zip(('H0', 'Hx', 'Hz'), family.components_numpy(self.q)):
            projected = self.q.conj().T @ applied
            self.component_matrices[name] = (projected + projected.conj().T) / 2
        self.projected_observables = {}
        self.cache_diagonal('z0', family.z[:, 0])
        self.cache_diagonal('z_last', family.z[:, -1])
        self.cache_diagonal('zz_ends', family.z[:, 0] * family.z[:, -1])

    def cache_diagonal(self, name, diagonal):
        diagonal = np.asarray(diagonal)
        if diagonal.shape != (self.family.dimension,) or not np.isfinite(diagonal).all():
            raise ValueError('observable diagonal must be finite with dimension 2**n')
        if np.iscomplexobj(diagonal) and np.max(np.abs(diagonal.imag)) > 1e-12:
            raise ValueError('Hermitian diagonal observable must be real')
        projected = self.q.conj().T @ (diagonal.real[:, None] * self.q)
        self.projected_observables[name] = (projected + projected.conj().T) / 2

    def expectation(self, name, coefficients):
        c = np.asarray(coefficients)
        if c.shape != (self.rank,):
            raise ValueError('coefficients must use retained-q coordinates')
        return float(np.vdot(c, self.projected_observables[name] @ c).real)

    def small_hamiltonian(self, h, lam=0.):
        a = self.component_matrices
        return a['H0'] + h * a['Hx'] + lam * a['Hz']

    def solve(self, h, lam=0., reconstruct=True):
        energy, vectors = eigh(self.small_hamiltonian(h, lam), subset_by_index=(0, 0))
        c = vectors[:, 0]
        z0 = self.expectation('z0', c)
        z_last = self.expectation('z_last', c)
        result = dict(energy=float(energy[0]), coefficients=c, rank=self.rank,
                      magnetization=z0,
                      connected_correlation=self.expectation('zz_ends', c) - z0 * z_last)
        if reconstruct:
            result['state'] = self.q @ c
        return result
