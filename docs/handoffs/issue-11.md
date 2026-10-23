# Issue11 handoff: exact SR, approximate capacity and main integration

2026-10-23. Bian-M-X/Codex-assisted; independent member review remains outstanding.

## Goal / authorization

[Issue11](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/11) continues round5 with a validated SR baseline and approximate-capacity controls. User explicitly requests main catch up at this upload. ADR0007 records the one-time integration exception; no force push, history rewrite or fabricated member review. Branch `issue/11-exact-sr-controls` starts at ffab952, merges remote main19a8683 (three README additions preserved), then freezes numerical source.

## Completed

- Numerical source/config/protocol commit `d420a9df24eb69af8f8219fa46a0882592e321b8`;29 unit tests passed before scientific runs. Source `src/ctqs/exact_sr.py`, proof `docs/theory_round6.md`, protocol `docs/round6_preregistration.md`.
- Main command `.venv\Scripts\python.exe -u scripts/run_round6.py`;60 completed jobs, no failures or substituted seeds, `experiments/round6_index.json`. First `20261023T041204047255Z-round6-n6-h0.4-mlp-shared-sr-s17`; last `20261023T041805082457Z-round6-n6-h0.6-transformer-shared-lbfgs-s73`. Main wall sum357.6328228000003s. Full2260 metric rows,380 new seed endpoints; all round5 baseline endpoints also appear in summary.
- Initial weights match round5 for all40 SR jobs. SR max relative solve residual7.236127155278322e-9, all steps accepted at first trial,1800 H actions and57600 scalar Jacobian rows/job. Full-basis sweep count is not literal backend-call/FLOP equality; independent models have3x parameters and smaller kernels. Wide L-BFGS uses7200 H actions, different initialization, approximate total parameter match.
- Replay original `20261023T041257555039Z-round6-n6-h0.4-transformer-shared-sr-s17`; replay `20261023T041821187916Z-round6-replay-n6-h0.4-transformer-shared-sr-s17`. Command `.venv\Scripts\python.exe -u scripts/run_round6.py --h 0.4 --arch transformer --seed 17 --sharing shared --optimizer sr --tag round6-replay`. Config/history/amplitudes/metrics and hashes exact. Replay excluded; not independent-human or cross-platform evidence.
- Complete report/PDF `docs/round6_report.*`, all tables, costs, paired comparisons and validation under `docs/results/round6/`. STATE receives a navigation/provisional-status refresh, no confirmed advantage claim.

## Findings / route decision

Low h=.4 MLP reliable counts shared SR2/5, independent SR1/5, wide shared L-BFGS5/5. Transformer1/5,1/5,4/5 respectively. h=.6 all new cells5/5. Wide Transformer seed43 has worst-anchor error.101955: wide mean.023972 is worse than narrow shared L-BFGS.009247 despite more passing seeds. No universal widening or splitting benefit.

Stop elevating shared-specific failure to a proposed contribution. The approximate capacity control shows shared models can pass; neither capacity nor sharing is causally isolated. A correct fixed-setting SR implementation can remain inaccurate at finite time; this does not rule out other SR schedules. The per-spin objective implies nominal cumulative imaginary time<=50, a possible explanation requiring further evidence, not ED-guided training.

Wide low-field MLP/Transformer direct test means.067865/.065864 versus their Ritz.0001164/.0021823. Independent Transformer L-BFGS Ritz from round5 remains.00004397. Strong postprocessing and total training+query cost remain essential.

## Unfinished / risks / next action

Only N=6 full enumeration, reused development train/test grid and five seeds. No strict parameter/update/compute match, no tuned best SR, no stochastic MinSR reproduction, no new Hamiltonian or fresh-size validation. Independent member review still absent even after main integration.

First command `.venv\Scripts\python.exe scripts/verify_round6.py`; use pinned environment. For independent replay use frozen source above, clean tracked tree and a new tag. Declare platform tolerances before inspecting discrepancies. Rebuild summaries with `scripts/summarize_round6.py`.

Next issue: bounded SR convergence/regularization check on development data without ED-driven tuning; then freeze fresh N=8/10/h/seeds and training+Q-query costs against strong shared/independent L-BFGS and Ritz. Keep H-work- and update-matched arms distinct; withdraw shared-failure hypothesis if controls remove it. Do not resume acquisition/scale cosmetic sweeps or claim ICLR readiness from passing CI.
