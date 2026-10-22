# Issue #9 handoff: route decision and sharing/optimizer gate

2026-10-22. Bian-M-X / Codex-assisted. Observed/derived, pending independent member review.

## Goal and dependency

[Issue #9](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/9): assess progress against course topic12 and ICLR ambitions, decide and execute a discriminating next round. Branch `issue/9-sharing-optimizer-gate` stacks on unreviewed PR #8 / `issue/7-coordinate-controls` at 6569db4cce86731258b5259523aa94c6ad95ec3e. ADR0006 records the exception. Do not merge automatically or push main.

## Completed evidence

- Pre-data assessment: `docs/strategy_after_round4.md`; frozen protocol/config and numerical code at `2acdc205d9fcb46335547216dc73d4d7c04abc0d`. 25 tests passed before training.
- Main command `.venv\Scripts\python.exe -u scripts/run_round5.py`: 80 completed jobs, zero failed, `experiments/round5_index.json`. First `20261022T144705865824Z-round5-n6-h0.4-mlp-shared-adam-s17`, last `20261022T145345586994Z-round5-n6-h0.6-transformer-independent-lbfgs-s73`. Main manifest wall sum393.68907560002845s.
- `docs/results/round5/`: 2840 raw metric rows, 480 seed endpoints, full tables, costs, paired contrasts, frozen gates and figure. Each arm uses7200 forward/backward H actions; independent has three times parameters, no claim of total compute/update/memory equality.
- 20 shared Adam final hashes and training-anchor amplitudes/metrics exactly match round4 physical final models.
- Replay original `20261022T144805914918Z-round5-n6-h0.4-transformer-shared-lbfgs-s17`; replay `20261022T145417650909Z-round5-replay-n6-h0.4-transformer-shared-lbfgs-s17`; actual command `.venv\Scripts\python.exe -u scripts/run_round5.py --h 0.4 --arch transformer --seed 17 --sharing shared --optimizer lbfgs --tag round5-replay`. Config/history/metrics/amplitudes and initial/final hashes exact; replay excluded from index/statistics.
- `docs/round5_report.md` and PDF consolidate route decision and results. `docs/theory_round5.md` and posthoc CSV/JSON give the flip competitor bound, explicitly exploratory and added after partial observations. No new algorithm claim.

## Findings and decision

Low-field Transformer reliable-fit counts (worst train-anchor infidelity<=.01) shared Adam0/5, shared L-BFGS3/5, independent Adam0/5, independent L-BFGS5/5. A cell requires4/5. Shared L-BFGS improves over shared Adam by>=10x in5/5; independent vs shared L-BFGS only3/5, failing that separate relative gate. Shared L-BFGS mean below.01 does not pass reliability.

Low-field MLP L-BFGS shared3/5 and independent2/5; independent Adam also0/5. All h=.6 cells pass5/5, but shared Transformer L-BFGS mean is worse than Adam. Sharing is not necessary for failure, and neither optimizer replacement nor splitting universally succeeds. Retain only a conditional sharing-associated reliability hypothesis with capacity/update confounds.

Low-field L-BFGS Transformer mean test infidelity shared direct.0520534, shared Ritz.00307111, independent Ritz.0000439689. These secondary endpoints make strong interpolation/postprocessing baselines and total query cost essential. No claim of new Ritz method, total-cost superiority or scalable VMC.

## Errors, unfinished items and risks

No scientific jobs failed. 52 closure evaluations consumed in interrupted final line searches were retained in cost, then parameters rolled back. This is the preregistered accepted-parameter rule, not seed exclusion.

The new posthoc physics test exposed integer lambda=0 dtype casting in the reference Hamiltonian. A later analysis commit casts the diagonal to float; all recorded lambda values were floats. Exact matrix equality checked on the round5 grid, so raw runs and frozen numerical source remain unchanged. First full26-test run failed on this edge case; final validation records the repaired run. No hidden training rerun.

Five seeds per stratum, N=6, full enumeration, positive states, reused development anchors and unreviewed PR stack. No independent-human replication, no uncertainty interval for population success rates, no final ICLR novelty claim. Witness W>0 supplies a variational competitor; W=0 says nothing about ground-state fidelity. STATE unchanged.

## First action and next gate

First run `.venv\Scripts\python.exe scripts/verify_round5.py`; then read report section7 and `docs/results/round5/validation.json`. Replay on the frozen source in a clean checkout using the original command with a new tag, and declare platform tolerances before comparison. Same-machine equality is not cross-member review.

Next issue should implement and analytically validate an exact SR baseline on development data, then freeze capacity/update controls and fresh N=8/10, h and seed evaluations. Report H-work-matched and update-matched comparisons separately and include training plus Q queries against independent-state Ritz. Do not run another cosmetic scale/acquisition sweep. Stop the shared-failure mechanism claim if strong/capacity-matched baselines remove it; retain negative benchmark findings if no amortization benefit survives. New protocol is not yet frozen or executed.
