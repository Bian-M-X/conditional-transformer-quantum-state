# Issue #5 handoff — conditioning, covariance and checkpoint selection

Date: 2026-10-21. Owner: Bian-M-X / Codex-assisted. Status: completed local study; provisional conclusions pending independent member review.

## Goal and branch

Issue https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/5 separates the three round-two confounders: input scale, training-time covariance and final-vs-validation checkpoint. Branch `issue/5-conditioning-controls` stacks on PR #4's verified head4215bfb (`issue/3-budget-acquisition`), because that PR is still open without reviews. See ADR0004. Retarget the incremental PR after its dependency merges; no main push or automatic merge.

## Completed evidence

- Frozen code/config/protocol commit `ae5236a` precedes all new training. Eighteen tests passed before execution.
- `experiments/round3_index.json`: 100 unique main jobs, five paired seeds per cell, no failed main runs. Stage A20, stage B80. The separate replay is excluded from the index.
- A: N=6 original h task, 2400 updates, raw/even MLP/Transformer. All20 histories and validation-selected metrics match round two exactly, and final checkpoint metrics are newly recorded.
- B: N=4/6, h=0.5, raw/covariant, lambda or lambda/0.04, two architectures, five seeds; 1200 updates. Frozen splits and no ED in training/checkpoint selection. Every job uses7296 forward+backward+validation H applications; network calls and wall time differ.
- `docs/results/round3/`: 13880 metric rows, 2720 seed endpoints, paired differences, figures, cost tables and replay record. `docs/round3_report.md` / PDF contain interpretation and limitations.
- `docs/theory_round3.md` derives the exact function-class equivalence under compensating affine weights and the first-step Adam coordinate effect. An additional unit test verifies state equality for both networks and covariance modes.

## What the data say

Original raw MLP mean F at h=0.2 is0.528694 for validation-selected and0.503493 for final. Failure is not solely selection of an early checkpoint in this fixed trajectory; it does not exclude other optimizers or longer budgets.

On the new B grid, N=6 raw Transformer mean test infidelity changes0.018690→0.004909 with domain scaling (4/5 seed wins), while N=4 raw Transformer does not show the same improvement. N=6 scaled covariant Transformer is worse than scaled raw in all5 seeds (mean0.081112 vs0.004909). N=6 MLP benefits from covariance, but much of the improvement can already come from inference-time covariance. Preserve all cells, not just favorable results.

Same sampled weights under different input scales do not yield the same initial function. This study does not isolate optimizer conditioning from initialization. B changes h and sampling schedule relative to round two, so cross-round absolute improvements are not causal scale effects. H actions are matched, but covariant B training uses6000 forward network calls versus3600 raw; no equal wall-compute claim.

## Actual commands and replay

Main: `.venv\Scripts\python.exe -u scripts/run_round3.py`.

Audit: `.venv\Scripts\python.exe scripts/verify_round3.py --without-replay`, then `scripts/summarize_round3.py`.

Replay: `.venv\Scripts\python.exe -u scripts/run_round3.py --stage b --n 6 --arch transformer --scale 0.04 --mode covariant --seed 17 --tag round3-replay`.

Original `20261021T052933340386Z-round3b-n6-transformer-covariant-scale0.04-s17`; replay `20261021T053042877707Z-round3-replayb-n6-transformer-covariant-scale0.04-s17`. Training history, all selected/final metrics and initial/selected/final state hashes match exactly. Final full evidence verification passed. This is same-machine deterministic replay, not independent researcher replication.

## Next person's first action

Run `.venv\Scripts\python.exe scripts/verify_round3.py` in a clean pinned environment. To independently reproduce, use the replay command with `--tag independent-round3`; the new directory is not added to the existing statistical index. Compare the above original's history/metrics/hashes; declare platform-dependent tolerances before inspecting discrepancies. Do not overwrite original records.

Next research task: freeze a new grid, compensate the lambda affine weight column to preserve initial functions, and explicitly control the optimizer's parameter-coordinate step. Only then attribute differences to conditioning. Investigate diagnostics of failed branch learning without ED-based selection, and return to acquisition only after reliable training-point fits. A separate Issue/config/commit is required before new experiments.

## Remaining limits

N≤6, one h in the factorial, full enumeration and positive amplitudes. No demonstrated scalable new method, no universal symmetry/scale improvement, no independent member review. Stage B becomes development data after this round. `docs/STATE.md` remains unchanged. Human contributions and course defense preparation remain with the members; the reports are AI-assisted research records, not an ICLR submission-ready manuscript.
