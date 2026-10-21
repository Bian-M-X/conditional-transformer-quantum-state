# Issue #3 handoff — budget, acquisition, covariance

Date: 2026-10-21. Owner: Bian-M-X / Codex-assisted. Status: local experiments complete, conclusions provisional pending independent member review.

## Goal and branch

Evaluate whether more training or acquiring parameter points resolves low-field errors, while exposing checkpoint selection and known parameter covariance. Issue: https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/3.

Branch `issue/3-budget-acquisition` stacks on `issue/1-topic12-audit` at 68a8311 because PR #2 is still open. Review the incremental PR against that branch; retarget to main after #2 merges. No main push, automatic merge or independent human review has occurred.

## Completed evidence

- `60d18bb`: frozen implementation/configs/protocol; 15 tests passed before training.
- `b98f23f`: forward-budget accounting correction before stage B, analytic covariance counterexample, summaries and evidence verifier; 16 tests passed.
- `experiments/round2_index.json` selects exactly 60 stage-A configurations (12 reused, 48 new) and 60 stage-B configurations. Five paired seeds: 17/29/43/61/73. Do not glob all run directories into seed statistics.
- All stage-B results and replay use b98f23f. Most new A runs use 60d18bb, two resumed A runs use b98f23f; numerical A training implementation/configuration did not change.
- `docs/results/round2/` contains per-seed endpoints, all stage-B metric rows, cost tables, paired differences, regenerated figures and replay identity.
- `scripts/verify_round2.py` passes 10650 metric rows, expected factorial combinations, H counts, candidate/test separation, shared 400-step prefixes, physical bounds and exact replay.
- `docs/round2_report.md` / PDF explain observed findings, limits and course connection. `docs/theory_round2.md` proves the elementary two-level illustration and necessary state-sensitivity inequality without claiming novelty.

## Observed results, with limits

Raw MLP low-h selected F stays 0.528694 on average from 600 to 2400 maximum steps because validation selects step 550/600. This does not establish final-iterate failure. Known-even training gives F>0.9984 in every seed/model at h=0.2 in the 2400-budget comparison.

Stage-B direct mean test infidelity stays approximately 0.5. Covariant averaging reduces it to about 0.244 but leaves large worst-point errors. Variance/Ritz-gain acquisition does not consistently beat continue plus both fixed grids after covariance controls. Training-point failures also persist, so these cannot be attributed solely to generalization or missing coverage.

**Cost correction:** original "total H actions" wording counted only forward applications. Actual stage-B forward+backward training/selection H work is 4800 for passive/random, 4749 for variance, 4647 for Ritz gain. All share 2400 forward actions. See ADR 0003, profiler output and per-run budget fields. No equal-total-compute or algorithmic-efficiency claim is justified.

## Failure and repeat

Failed run `20261020T172651729101Z-round2a-transformer-even-s61` triggered the 600-second per-run check following a long wall-clock interruption; its manifest records 38641.260 seconds. The cause of the interruption is unknown. No failed metrics were fabricated or included in averages. The same frozen configuration succeeded in `20261021T041409153968Z-round2a-transformer-even-s61`. The index includes the failed ID separately.

Original `20261021T041644936691Z-round2b-transformer-variance-s17` was repeated as `20261021T041758982150Z-round2-replay-transformer-variance-s17`. Training, metrics, acquisition decisions/scores and final weight hash match exactly, excluding acquisition timing fields. The repeat is excluded from the statistical index.

Commands actually used: `.venv\Scripts\python.exe -u scripts/run_round2.py` (initial invocation timed out, a second invocation resumed remaining entries); `scripts/verify_round2.py --without-replay` before replay; `scripts/run_round2.py --stage b --arch transformer --policy variance --seed 17 --tag round2-replay`; `scripts/verify_round2.py`; `scripts/summarize_round2.py`.

## One-command handoff

First run `.venv\Scripts\python.exe scripts/verify_round2.py`.

Then independently reproduce one adaptive run with `.venv\Scripts\python.exe scripts/run_round2.py --stage b --arch transformer --policy variance --seed 17 --tag independent-round2`.

Use a clean tracked checkout and the pinned research dependencies. The nondefault tag creates a new directory but does not append it to the existing statistical index. Compare to the original above using `metrics.json`, `training.json`, acquisition entries excluding `seconds`, and the final state hash. Report environmental differences; a different platform need not be bitwise identical, so declare numerical tolerances before inspecting differences. Never replace the original records.

## Next research and acceptance gate

Before adding acquisition complexity, freeze a new grid and separate final-vs-validation checkpoint, physical-vs-domain-scaled λ, and raw-vs-covariant training. Preserve budget measures including backward work. Only revisit active selection after training-point accuracy is adequate and simpler controls fail to explain an advantage. New results must use new manifests and a separate Issue.

The course backbone remains round 1 plus `course_requirements.md`; human members must still supply actual contributions, independent checks and oral defense preparation. This report is not a submission-ready ICLR paper. Formal `docs/STATE.md` conclusions remain unchanged.
