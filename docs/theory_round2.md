# Round 2: covariance is not an accuracy certificate

Status: **derived**, elementary calculations; not a novelty claim. The two-level example is exact as a two-dimensional model, not an error-controlled reduction of the six-spin chain. Numerical chain results are separately **observed** in the round-two report.

## 1. The implemented covariance control

Let P be the global spin flip, P=P†=P⁻¹, and H(-λ)=P H(λ) P. For real, normalized, strictly positive amplitude predictions u(λ), define

v(λ) = [u(λ)+P u(-λ)] / ||u(λ)+P u(-λ)||.

Positivity prevents a zero denominator. Unitarity of P makes the normalization equal at ±λ, so v(-λ)=P v(λ). At λ=0 this reduces to the normalized even projection. At nonzero λ it does **not** generally impose P v(λ)=v(λ). This is a covariance construction, not a variational minimizer or a fidelity guarantee. It uses two predictions and, in our implementation, full-state normalization.

Our alternative sign map copies u(|λ|) to the negative side through P; at zero we use its even projection. It has covariance but can be discontinuous at zero if the unprojected u(0) is not even. Therefore it cannot establish accurate interpolation through an avoided crossing.

## 2. Exact counterexample: covariance plus vanishing variance can still fail

Consider H(λ)=-(Δ/2)σx+vλσz, Δ>0, v>0, with P=σx. Its spectral gap is sqrt(Δ²+4v²λ²). The constant prediction |+x> is covariant for all λ, and is unchanged by the amplitude averaging above. Direct diagonalization gives

- Fidelity F(λ) = [1+Δ/sqrt(Δ²+4v²λ²)]/2.
- Energy excess δE(λ) = sqrt(Δ²/4+v²λ²) - Δ/2.
- Variance Var(H) = v²λ².

Choose Δ=t², v=1, λ=t and take t→0⁺. Then δE→0 and Var(H)→0 while F→1/2. Thus exact parameter covariance, small energy excess and small variance together do not provide a **gap-uniform** accuracy certificate. At fixed positive Δ the limit λ→0 is different; exchanging these limits would invalidate the counterexample.

This explains why a covariant but weakly parameter-dependent predictor can miss branch polarization. It does not show that every covariance-aware architecture will fail, nor does it prove that this mechanism dominates the actual training trajectories.

## 3. Necessary sensitivity to resolve both branches

In the same model, choose normalized positive ground vectors g(±a). Their overlap is

<g(-a)|g(a)> = Δ / sqrt(Δ²+4v²a²).

Suppose normalized positive predictions u(±a) each have infidelity at most ε in [0,1]. Define dε = sqrt(2-2sqrt(1-ε)). Positivity fixes the relevant global phase and gives ||u(±a)-g(±a)||≤dε. The triangle inequality then yields

||u(a)-u(-a)|| ≥ max{0, sqrt[2-2Δ/sqrt(Δ²+4v²a²)] - 2dε}.

If u is differentiable on [-a,a], integrating u′ implies that at some point

||u′(λ)|| ≥ max{0, sqrt[2-2Δ/sqrt(Δ²+4v²a²)] - 2dε} / (2a).

Consequently, resolving both sides near a narrow avoided crossing requires large parameter sensitivity. Rescaling a small input feature can change optimization conditioning, but cannot remove the physical state variation. This bound applies to state vectors in the declared gauge, not raw logits, network weights, stochastic gradient convergence, or arbitrary complex phase conventions.

## 4. Falsifiable next experiments

**Hypothesized:** inadequate parameter sensitivity and weak optimization signals may confound acquisition comparisons. Round 2 intentionally keeps physical λ inputs unscaled; it does not test that hypothesis.

**Planned:** freeze new N/h and λ grids, compare physical λ against λ/0.04 (a domain scale, not an ED gap), learned covariant training against inference-time averaging, and Adam against an explicitly budgeted optimizer baseline. Keep selection cost and prediction cost separate. Measure sensitivity and training-point errors before attributing remaining failure to missing parameter coverage. If those controls remove the acquisition advantage, reject an acquisition novelty claim.

## Literature boundary

[Chen and Heyl, Nature Physics 20, 1476–1481 (2024); arXiv v3](https://arxiv.org/abs/2302.01941v3) develops minimum-step stochastic reconfiguration for deep neural quantum states. This supports considering optimization as a serious alternative explanation; our Adam budget scan is not a reproduction or comparison against that algorithm.

[Sarkar and Lee, Physical Review Research 4, 023214 (2022)](https://arxiv.org/abs/2107.13449v2) already studies self-learning emulators with eigenvector continuation. Neither active parameter selection nor Ritz correction is new here. The broader related-work exclusions from `literature_round1.md` remain in force.

Bibliographic metadata and abstracts above were checked against primary arXiv records on 2026-10-21. No claim of a comprehensive refreshed literature search is made.
