# ADR 0002: Second-round budget study on a stacked branch

Date: 2026-10-19. Status: implementation decision for Issue #3; not a scientific consensus update.

PR #2 is still open and unreviewed. The user requested continuation. Round 2 depends on its implementation, so `issue/3-budget-acquisition` branches from its verified head `68a8311` and targets `issue/1-topic12-audit` in a separate PR. This is a documented exception to the README's main-based branch convention so that the incremental review contains one new research objective. No main merge/push, history rewrite, or implicit human review occurs. Retarget after PR #2 is merged.

The reviewable objective is to distinguish optimization budget from acquisition benefit, with known parameter covariance and query costs included. Stage A is a controlled extension of the earlier h study. Stage B is a deliberately small signed-lambda screening. Neither proposes a new algorithm.

Stage B matches the sum of training and acquisition H-vector actions exactly, and reports gradient updates, forward calls and wall time separately. It does not artificially spend baseline compute on unused acquisition scans. Consequently it does not claim simultaneous matching of all resource measures. H-action matching is one resource axis, not a complete measure of algorithmic complexity.
