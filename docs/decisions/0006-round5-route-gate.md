# ADR 0006: stop local preprocessing sweeps; gate on training attribution

2026-10-22, Issue #9. Accepted for bounded screening execution.

User requested an explicit assessment against the final objective followed by a research decision and execution. Rounds 1-4 established reproducible diagnostics, not a novel method. Pause acquisition/scale novelty work and run shared-versus-independent fitting under Adam and line-search L-BFGS. Use all closure calls as the optimizer-independent H-work budget and final accepted states to remove selection-rule ambiguity.

Known limitations: independent models triple parameter/storage resources; update counts and wall compute differ; L-BFGS is a deterministic full-enumeration control, not SR or scalable VMC. Anchors reuse development data; hypotheses and gates are frozen before this round's results. A strategic decision after results must be written rather than automatically extending the same sweep.

PR #8 remains open without reviews at 6569db4cce86731258b5259523aa94c6ad95ec3e. Continue the explicitly authorized stacked chain on `issue/9-sharing-optimizer-gate`, base `issue/7-coordinate-controls`; retarget when dependency merges. This is a recorded exception to default main-based branches/single-active-task preference, not permission to auto-merge, overwrite main or promote unreviewed STATE conclusions.
