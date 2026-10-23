# Exact normalized-state SR: convention and checks

Derived,2026-10-23. Real differentiable normalized states psi_k(theta), K training Hamiltonians H_k, objective L=mean_k psi_k^T H_k psi_k / N. Let J_k=d psi_k/d theta and T_k=(I-psi_k psi_k^T)J_k. Analytically psi_k^T J_k=0; explicit projection removes roundoff.

Stack A=vertical(T_k)/sqrt(K), b=vertical((H_k-E_k)psi_k/N)/sqrt(K). Then S=A^T A is the mean real-state quantum geometric tensor, f=A^T b=grad(L)/2. For strictly positive amplitudes O=J/psi, S equals Cov_p(O), p=psi^2. This equality requires normalization and applies to the current positive real model; no complex-state generalization is silently assumed.

For positive damping delta, d=(A^T A+delta I)^(-1)A^T b=A^T(AA^T+delta I)^(-1)b. Proof: (A^T A+delta I)A^T=A^T(AA^T+delta I); both shifted matrices are positive definite. This regularized identity does not require full rank. It is a known kernel formulation, not a new optimizer. Our update theta'=theta-eta*d uses the HALF energy gradient. It approaches an imaginary-time tangent step with time eta/N, not eta, because the objective is per spin.

For psi(t)=(cos t,sin t), H=-X and N=1, S=1, f=-cos(2t); damped d=-cos(2t)/(1+delta). This is tested analytically. In a complete state parameterization without damping, excited-to-ground coefficients evolve in continuous imaginary time as exp[-tau(E_i-E_0)]. Thus a small gap can cause slow convergence even for correct SR. Damping, finite step size and restricted neural tangents invalidate treating this relation as a measured convergence rate of our networks. ED gaps never set the training rate or stopping rule.

The implementation caps eta||Ad|| by radius.1, then uses train-energy-only Armijo backtracking. This is a first-order displacement cap, not a rigorous finite-step fidelity bound. All six possible trials count, accepted or rejected; an unsuccessful step leaves parameters unchanged and is recorded. The half-gradient convention is verified against autograd; switching objective from total to per-spin energy without compensating rate changes dynamics.

Tests cover both neural architectures, including tangent-vs-covariance equality, independent finite differences, and dual-vs-primal solves. Numerical roundoff and deterministic platform dependence remain. Full enumeration costs grow exponentially; the implementation explicitly forms Jacobians and dense kernels and does not establish scalable VMC efficiency.

Sources: [NetKet QGT/SR guide](https://netket.readthedocs.io/en/latest/user-guides/sr.html) for established SR and kernel background; [MinSR](https://arxiv.org/abs/2302.01941v3) for publication context. Derivation here fixes our own normalization/damping conventions, rather than claiming an exact implementation of either package or paper.
