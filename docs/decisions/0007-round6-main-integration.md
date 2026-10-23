# ADR0007: SR implementation gate and user-authorized main integration

2026-10-23. User explicitly requested continuing research and bringing main up to date at the next GitHub upload. This supersedes the default no-main-update/independent-review-before-merge workflow for this integration only; it does not authorize fabricated human reviews or promotion of provisional research claims.

Round6 operates on Issue11 branch from ffab952. Remote main has README commit19a8683, absent from the stacked research branches. Merge it with history preserved before numerical freeze; keep its three documentation additions. Do not overwrite collaborators, force-push or rebase the chain. Re-fetch before publication and integrate any further changes rather than replacing them. Publish only after tests and all evidence audits; normal push must be a descendant of current main. Preserve original PR discussions and identify which commits were integrated.

STATE may receive an explicitly provisional navigation/status refresh so main readers find actual rounds, while confirmed scientific conclusions remain unpromoted. Main synchronization is repository integration, not independent scientific validation.

Scope: exact-enumeration SR implementation and60 development runs (40 narrow SR +20 wider L-BFGS). Full capacity/update and fresh-size validation remain separate future gates. The wider models are approximate parameter controls, not a capacity-matched causal experiment. Avoid turning this engineering/diagnostic stage into a claim of a novel optimizer or ICLR-ready contribution.
