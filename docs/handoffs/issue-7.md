# Issue #7 handoff: initial functions and Adam coordinates

2026-10-22. Owner: Bian-M-X / Codex-assisted. Completed local study; independent member review outstanding.

## Goal and dependency

[Issue #7](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/7) asks whether scaling changes initial functions or optimizer coordinates. Branch `issue/7-coordinate-controls` stacks on unreviewed PR #6 at ee0d7c2; incremental PR base is `issue/5-conditioning-controls`. ADR0005 records the dependency exception. Do not merge automatically or push main.

## Completed and provenance

- Numerical code/config/protocol frozen at `414e78594d752ff44a9ecbef8aed658e2c34348b` before every main run and replay. Twenty-two unit tests passed before execution.
- `experiments/round4_index.json`: 80 completed main jobs, no failed jobs. N=6, h=0.4/0.6, two architectures, five paired seeds, four arms. All 10880 metric rows and selected/final direct amplitudes preserved; `docs/results/round4/` has 1920 seed endpoints, complete tables, paired differences, costs, figures and verification results.
- Main command: `.venv\Scripts\python.exe -u scripts/run_round4.py`. Actual summed manifest wall time 335.1701083000153 seconds, excluding replay/testing/reporting.
- `src/ctqs/coordinates.py` performs plain Adam in physical coordinates for the transformed lambda column. `docs/theory_round4.md` derives all-step equivalence and states assumptions. No new optimizer claim.
- All 20 P-C pairs match histories, checkpoint choice, amplitude arrays and all metric values exactly. Main tolerance was predeclared as atol=1e-10, rtol=1e-8; it was not widened. C runs are independent executions, not copied records.
- Original `20261021T162611316855Z-round4-n6-h0.4-transformer-canonical-s17`; replay `20261021T163132928313Z-round4-replay-n6-h0.4-transformer-canonical-s17`. Config/history/metrics/amplitudes and sampled/initial/selected/final weight hashes match exactly. Replay excluded from index/statistics.

## Findings and limits

At h=0.6, MLP test infidelity P=0.00198420 and I=0.00113121; I improves all five seeds at matched initial function. Transformer I worsens four seeds and has mean 0.00072122 versus P=0.00066402. The h=0.4 direct failure remains near 0.5; the apparent U-MLP mean improvement is dominated by seed 17. Postprocessing and train-point errors are explicitly reported. No universal scale benefit, no function-class enlargement, no additive attribution of initialization versus coordinate contributions.

Every arm has 7296 forward/backward/validation H actions, 3600 training model calls, 96 validation calls and six initial-function calls. C adds coordinate-conversion arithmetic; wall time is measured, not matched. All runs use full enumeration, positive amplitudes and a single system size. Different h values are separate trained tasks. New grid is now development data.

Analysis-only anomaly: first summary export raised a duplicate `h` keyword error while combining metadata and metric rows. Fixed dictionary merge and reran the exporter; raw training outputs remained unchanged and no main experiment failed. No hidden retry or seed exclusion.

## Reproduce and continue

First action: `.venv\Scripts\python.exe scripts/verify_round4.py` in the pinned environment. Rebuild tables: `.venv\Scripts\python.exe scripts/summarize_round4.py`.

Independent replay command: `.venv\Scripts\python.exe -u scripts/run_round4.py --h 0.4 --arch transformer --seed 17 --arm canonical --tag independent-round4`. Compare the original above using `training.json`, `metrics.json`, `amplitudes.json` and hashes. Declare platform tolerances before inspecting differences. Tags produce new directories and leave the index unchanged. Same-machine equality is not independent-member replication.

Next research issue should freeze single-H versus shared-condition training at low h, with explicit optimizer and H-work controls, to test whether parameter sharing causes failure. Do not return to adaptive acquisition until training-point failure is understood. Add scalable dimensions/sampling later; no such experiments were executed here. Keep `docs/STATE.md` provisional until independent review, and preserve the course task's original multi-h/interpolation/extrapolation evidence and member-authored defense records.
