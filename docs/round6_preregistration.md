# Round 6: exact SR implementation gate and approximate-capacity controls

Frozen before scientific runs, 2026-10-23, [Issue #11](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/11). Material Passport: code experiment, Bian-M-X/Codex-assisted, prior rounds as development evidence, no independent human read/review claim. academic-research-suite execution/validation workflow used inline under the user's continuing research authorization.

## Scope and fixed design

N=6, h=.4/.6, MLP/Transformer, five seeds17/29/43/61/73. Three arms shared SR, independent SR (three copies), wide shared L-BFGS =60 jobs. Reuse round-five train AND test grids intentionally as development evidence. Numerical correctness tests precede frozen source commit. No ED, ground states, gaps or test values in training/line search. Freeze all models before ED evaluation. Final accepted parameters only, no validation selection.

SR:300 outer steps per model, exact normalized-amplitude Jacobian, additive QGT diagonal shift1e-4, nominal rate1, first-order state-displacement radius.1, up to6 Armijo trials (halving, coefficient1e-4). Objective mean per-spin energy. Direction solves half-gradient normal equations. Rejected trials are charged; if all fail, parameters revert and outer step still counts. A norm limit applies to the tangent prediction, not actual fidelity. This fixed setting is a tested SR control, not a tuned best SR/MinSR implementation. Initial weights match round5 for narrow SR arms; runtime math-attention differences can cause floating rounding, so do not require cross-optimizer full-trajectory equality.

Wide shared L-BFGS: MLP width60=4621 parameters vs independent narrow4707 (-1.83%); Transformer width28=6973 vs7347 (-5.09%). Two heads retained, all other architecture choices unchanged. This brackets neither exact capacity nor identical initial functions. It is an approximate parameter-count control and cannot uniquely identify a sharing mechanism. Same1200 closure, lr1, history20, strong-Wolfe, final-step rollback as round5. Compare with all relevant round5 arms, not only weak Adam.

## Budgets and endpoints

SR includes one H action per condition at Jacobian construction and one per line-search trial; H is not differentiated through in SR. At most300*3*7=6300 training H actions per job. Also report exact scalar amplitude-Jacobian row count, parameter count, kernel dimension, outer/accepted steps and measured wall time. Full dense Jacobian cost may dominate despite fewer H actions than L-BFGS. No equal-total-cost claim. All code uses full enumeration, float64 CPU, one Torch thread, positive real autoregressive states. Independent SR uses three times model storage but smaller solves; do not pool cost or h strata.

Primary per-seed worst training-anchor infidelity; reliable fit means <=.01 in>=4/5 seeds per cell, the same operating rule as round5. Show all five values and mean/sample SD, no significance or population success guarantee. Secondary direct/lookup/Ritz test metrics and all physical observables retained. No independent direct off-grid prediction. No choosing hyperparameters after examining these results. Same-machine replay: shared SR Transformer h=.4 seed17, excluded from statistics.

Failure records retained,600s/job and3600s/invocation. No seed substitution. If SR does not reliably fit this development task, do not infer all SR settings fail; inspect damping, finite imaginary time and representation before freezing N=8/10 validation. If wide L-BFGS removes the round5 gap, withdraw any inference that sharing alone explains it. If results remain heterogeneous, preserve the limited hypothesis and stop parameter-only causal claims. N=8/10, fresh h/seeds, stochastic sampling, another Hamiltonian and total training-plus-query costs remain future work, not implied by this implementation gate.

## Verification and literature boundary

Tests independently compare amplitude-tangent QGT with probability-weighted log-derivative covariance, half-gradient force with autograd, Jacobian directional differences with central finite differences, dual/primal damped solves, analytic one-spin imaginary-time direction, descent and trial-budget accounting. These are mathematical/code checks, not independent replication of published performance.

Primary sources checked2026-10-23: [NetKet SR/QGT guide](https://netket.readthedocs.io/en/latest/user-guides/sr.html), including normal equations, diagonal regularization and kernel formulation; [Chen and Heyl MinSR metadata/abstract](https://arxiv.org/abs/2302.01941v3); [PyTorch jacrev API](https://docs.pytorch.org/docs/2.14/generated/torch.func.jacrev.html). SR, its dual identity and backtracking are established techniques; no novelty claim and no stochastic MinSR reproduction.
