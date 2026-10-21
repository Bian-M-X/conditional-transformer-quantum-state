# Round 2 frozen screening protocol

Drafted 2026-10-19; frozen for execution 2026-10-21, Issue #3. Repository prerecord before running new experiments; not an external registration.

**Accounting correction before stage B:** ADR `0003-forward-budget-accounting.md` qualifies all "H-action budget" wording below as **forward** H actions. Backward H actions were omitted from the original cost wording; the preserved numerical protocol does not match total H work. This correction changes interpretation and metadata, not parameter grids, seeds or selection/training behavior.

## Stage A: optimization-budget alternative

Use original N=6 h task, architectures, raw/even training, learning rate, validation-based selection, and held-out grids. They are now **development diagnostics**, because round 1 has already revealed their outcomes. Seeds=17/29/43/61/73, full budgets=600/1200/2400 updates. The first 12 existing runs are retained at 600 steps; 48 missing jobs are newly executed. Budgets are separate from-scratch trajectories at matched initializations, not sequentially accumulated compute savings. Checkpoint selection stays based on variational validation energy, not exact labels. Report selected epoch to expose selection effects; lower-error later runs do not prove convergence.

Primary question: Does low-field training-point error remain at 4x budget, and does known-even-sector training still matter? Report every seed, fidelity at h=0.2 and mean/worst development-heldout errors. No significance test based on treating correlated h points as independent.

## Stage B: acquire points or continue fitting?

Fixed N=6,h=0.35 open TFIM plus lambda Z1. Same two model architectures. Raw energy-only training; input scaling unchanged to avoid an unregistered architecture change. Start at lambda={0,+0.04}. Exact candidate and held-out grids are in `experiments/configs/round2_acquisition.json`; they are disjoint. No test values enter acquisition, no ED states/energies/gaps enter training or acquisition.

Six policies: continue at initial points; fixed uniform additions {-0.04,-0.02,+0.02}; fixed log additions {-0.04,-0.001,+0.001}; random without replacement; maximum raw-state energy variance; maximum raw-state minus parity-span Ritz energy. These are baselines/diagnostics, not a new method or full reproduction of self-learning EC/FNEH.

All begin with identical first 400 updates for each architecture/seed. Add one point after updates 400/650/900. Each minibatch samples 2 parameter points with replacement from the current pool. Batch and acquisition randomness use separate generators. The named fixed grids specify the three additions, not an optimized quadrature.

Budget: exactly 2400 H-vector actions total for training plus acquisition. Variance needs one H action per candidate; Ritz gain needs one raw H action plus the retained subspace rank. Scan the complete fixed candidate pool (including current training points for transparent equal per-round search domain), but select only unseen candidates. Passive methods do not waste compute on scans. Adaptive methods receive correspondingly fewer updates; a final batch of size1 spends a remaining odd action. No ED cost is charged as training; all scoring/postprocessing/evaluation costs are separately recorded. All methods use final state, not metric-selected checkpoints.

This matches a **single compute axis**. Report gradient updates, model calls, acquisition time, training time, evaluation time and wall time; do not claim wall-time or gradient-count equivalence. Training lambda cardinality is 2 for continue and 5 otherwise; do not disguise that intentional resource allocation difference.

## Evaluation and decision rules

Evaluate raw direct prediction, same-network learned-state lookup, sign-canonical mapping, smooth covariant amplitude average, and training-state-span Ritz.
Sign mapping uses psi(|lambda|), flipped for negative lambda; at zero it uses even projection. It can have a zero-field seam, so it is a control, not a proposed continuous model. Smooth average normalizes psi(lambda)+P psi(-lambda), which obeys covariance without imposing even parity at nonzero lambda. Neither uses ED. Their extra forward costs must be shown.

Primary endpoint: mean test infidelity per seed; secondary: worst infidelity, Z1/correlation errors, energy error, and sign-stratified error. Report all five paired seeds and sample SD. No threshold or hyperparameter retuning on this grid. An acquisition policy that fails to consistently beat both fixed grids and continue, particularly after covariance controls, cannot support a superiority claim. Cheap physics/subspace baselines explaining gains is a negative result for novelty, not a reason to hide them.

## Remaining confounders and hard limits

Raw lambda features are small; input scaling, optimizer choice, learning rate and model width remain possible explanations. One h and N=6 cannot establish cross-system generality. Exact enumeration is exponential. Five seeds still provide screening evidence only. Continue is not symmetry-trained; the covariance baseline is inference-time. There is no claim of independent researcher replication.

Unit checks: covariance and normalization at signed lambda, reference variance and Ritz scores, no full-H eigensolver in acquisition, exact H-budget conservation, disjoint grids, deterministic selection. Global suite cap 30 minutes; per-run cap600s. Preserve failed manifests and incomplete suite indexes. Literature context remains active EC (Sarkar & Lee, PRR 2022) and FNEH (Zhang et al., arXiv:2608.14208); current results do not create a novelty exception.
