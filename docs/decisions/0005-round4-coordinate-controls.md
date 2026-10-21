# ADR 0005: isolate initialization and optimizer coordinates

Date: 2026-10-22. Issue #7. Status: accepted for exploratory execution.

Round three proved elementary function-class equivalence but did not preserve initial functions in the factorial. Add two controls instead of interpreting the observed scaling improvement as expressivity or a universal conditioning benefit. Keep parameter-coordinate compensation local to the lambda affine column; a global learning-rate reduction would confound every other parameter.

Use scale 1/16 to avoid unnecessary multiply/divide roundoff in the exact-equivalence control. The training bound changes accordingly. New h values provide two independently trained tasks, not h generalization. Eighty configurations and one separately indexed replay are feasible within the existing CPU setup. Full results remain screening evidence.

PR #6 remains open, unreviewed, head ee0d7c205b6168e827a7ebddbe9ff5f797480a55. To honor the user's continuing research instruction without merging unreviewed changes, branch `issue/7-coordinate-controls` stacks on `issue/5-conditioning-controls`. This is the same documented dependency exception as earlier rounds to the default main-based branch convention and single-active-task preference. Retarget after dependency merges; do not push main, force-push or auto-merge. Independent review is outstanding across the chain.
