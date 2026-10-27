# Issue19: checkpointed refinement/acquisition headroom pilot

2026-10-27. Bian-M-X/Codex-assisted. [Issue19](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/19). Branch `issue/19-allocation-headroom`, stacked on Issue17/draft PR18. User authorized beginning this gate and optional Astra advice. No new direct-main exception is inferred; no force push or scientific claim promotion.

## Goal

Before developing a scheduler, test whether hindsight one-step R/A choice leaves meaningful improvement over six strong simple strategies after known spin-flip closure. This is a development investment gate, not proof of novelty or deployable net-cost benefit. Freeze `d2595d75514862b09fb41d6ea004cac191c88015`; predecessor `d31b53399764c75bc10b548199b429cb71f06bd6` preserves Issue17 and main44e03c6.

## Completed and observed

- `src/ctqs/affine_spin.py`: matrix-free chain/ladder H and cached affine Ritz with small-matrix energy/observable queries. `src/ctqs/resumable_fit.py`: model/optimizer/RNG snapshots, exact closure caps and full-state rollback at incomplete line searches. New modules preserve round7 frozen science files.
- `experiments/configs/round8.json`, `docs/round8_preregistration.md`, ADR0010 frozen before all jobs. Actual gpt-6-astra/high advisory input and scope in `docs/results/round8/design_review.json`; no human-review or independent Astra source-audit claim.
- All40 factorial jobs complete, no failures or replacements; run index `experiments/round8_index.json`. Main8280 metric rows/960 query seed endpoints/168000 closure evaluations. Complete negative seeds, raw/closed variants and correlated maturities retained.
- Primary1200/closed:0/8 eligible. All B<1e-3;7/8 have O=B and fixed A strongest. Remaining narrow-ladder Transformer G=.0831696, B-O=2.0534e-6, minLOO G=.0248932. All48 sensitivity strata none eligible. Wide primary A meaningfully beats R and STOP on all5 seeds, no primary meaningful R win. Stop scheduler development for THIS menu/task/budget; this does not refute all allocation methods.
- `scripts/analyze_round8.py`, synthetic analysis tests, complete CSV/JSON/tables and two figures. Cost649.18s sequential research wall,336000 training H actions; full counterfactual cost never represented as feasible oracle-policy cost. Action-cost rows intentionally repeat shared metadata; do not sum them.
- `scripts/verify_round8.py`: independent Kronecker H/NumPy eig, all physics metrics,1200 cache geometries,9600 energy nesting checks,120 policy features, frozen-source/config checks; maxF difference1.90e-13/norm2.66e-15. Local safe `weights_only=True` checks600 full checkpoint fingerprints. DefaultCI excludes unavailable ignored binaries and rechecks recorded replay content.
- Independent-directory same-machine replay `20261027T135738696065Z-round8-replay-chain-r0.4-transformer-s107`, original `20261027T100350919750Z-round8-chain-r0.4-transformer-s107`. Predictions.gz/training/metrics byte-equal, physical decisions/15 checkpoint hashes equal. Not another statistical seed or independent-human replication. Exact timing excluded.
- Full50 tests pass; workspace551 manifests pass; round7 saved-evidence audit60/9000 passes. First full test invocation had2 sandbox temporary-directory permission errors, corrected by process-local TEMP/TMP to repo tmp without source changes. Final verification receipt `docs/results/round8/verification.json`; report `docs/round8_report.md/.pdf`.

## Commands and regeneration

Research Python: `.venv\Scripts\python.exe` (3.13.9, torch2.14.1+cpu, float64, one Torch thread; hardware details in Issue17 resource evidence).

```powershell
$env:TEMP = 'D:\Research\conditional-transformer-quantum-state\tmp'
$env:TMP = $env:TEMP
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe scripts/validate_workspace.py
.venv\Scripts\python.exe scripts/verify_round8.py --local-checkpoints
.venv\Scripts\python.exe scripts/analyze_round8.py
```

Remote fresh checkout lacks ignored `.pt`; omit `--local-checkpoints` there. Source comparison normalizes CRLF only. Residual-rule recorded argmax is checked exactly; independent residual maxima allow declared numerical tolerance because symmetric endpoint ordering can swap under eigensolver roundoff. No action choices or scientific data changed to make audits pass.

To reproduce a fresh job, first commit unrelated tracked changes or use a clean checkout containing the frozen scientific source. Run (creates a new directory, does not overwrite index or original):

```powershell
.venv\Scripts\python.exe -u scripts/run_round8.py --family chain --radius 0.4 --arch transformer --seed 107 --tag round8-independent
```

For all40 fresh jobs omit the four job filters and retain an independent tag. Do not append these replay jobs to the primary index. Actual original invocation was `python scripts/run_round8.py`; source tree was clean. Checkpoints use the exact same segmented schedule; this is not equivalence to uninterrupted L-BFGS. Full state hash semantics are defined by `run_round8.fingerprint`, not raw `.pt` byte identity.

PDF rebuild uses bundled ReportLab runtime and `scripts/build_round1_pdf.py --source docs/round8_report.md --output docs/round8_report.pdf --footer 'CTQS / Round 8 / 2026-10-27 / 冻结先导与止损' --title '精修还是加点：第八轮研究报告'`. Render all pages and visually check before delivery.

## Unfinished, limits and next action

Independent member reproduction/review, a naturally motivated new bottleneck and novelty/strong-control comparison remain. N10, learned policy, held-out-task transfer, Monte Carlo/resampling and GPU training are not run. R carries optimizer memory while A resets; fixed center, retained old vectors, finite one-step menu, two fixed-h N8 tasks and additional closure ceilings bound interpretation. An oracle-selected action cannot support an oracle-free policy claim. Saved ordering declarations/timestamps are not an externally tamper-proof access log.

First next action: another member runs `python scripts/verify_round8.py` and one fresh independent-tag configuration, checks the report's narrower decision, then drafts a mechanism-driven proposal before authorizing a new factorial. Do not advance directly to a complex scheduler, larger systems or MC solely to rescue this failed gate. Prior seven-round multi-h evidence continues to support course-topic12; this fixed-h independent-endpoint pilot does not replace it. Human contribution/defense and independent scientific review remain human tasks.
