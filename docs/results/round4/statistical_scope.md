# Statistical interpretation and scope audit

Material Passport: code-experiment descriptive validation, Issue #7, 2026-10-22, Bian-M-X / Codex-assisted. Inputs: frozen round-four config, manifests, seed endpoints, paired differences and control audit. Numerical/replay verdicts are recorded separately in `validation.json` and `control_audit.json`; this document cannot certify those runs by itself.

The unit of replication is a seed within a fixed h/architecture/arm, n=5. Report mean, sample SD, five individual values and paired deltas. No p-values, normality claims, population confidence intervals, power claims or effect-size labels. Infidelity varies strongly across configurations; avoid an overall improvement percentage pooled across h and architecture. Final checkpoints and postprocessing are secondary. Coordinate equality is an algebraic/control result, not a statistical noninferiority finding.

## Eleven interpretation checks

| Risk | Handling and remaining boundary |
|---|---|
| Simpson reversal | All h/model strata reported separately; no pooled direction claim. |
| Ecological inference | Group means are not claims about every seed; improvement counts and paired deltas retained. |
| Selection/Berkson bias | Frozen seeds and all four arms included; this hand-selected small-system grid is not representative of all quantum tasks. |
| Collider conditioning | No filtering by achieved fidelity or validation score; validation selects a checkpoint within every run by the same rule, and final results are also reported. |
| Base-rate neglect | No diagnostic sensitivity/precision or deployment-prevalence claim. |
| Regression to mean | New frozen grid and paired controls; do not infer scale benefit from comparing the worst prior-round cell with this round. |
| Survivorship | Index enumerates all 80 required identities and explicitly retains failure records; verifier fails if any cell is missing. |
| Look-elsewhere | Full tables include every arm, checkpoint and method; no significance testing or isolated-best-cell headline generalized to all tasks. |
| Forking paths | Code/config committed before training, primary endpoint fixed, no post hoc tolerances; prior rounds influenced the scientific question, so this is screening. |
| Causal overreach | P-I and U-I are controlled software interventions at fixed model/h/budget; they do not establish universal optimization mechanisms or an additive decomposition. |
| Reverse causality | Initializations and optimizer transformations precede training, and ED runs only after checkpoint freeze; no retrospective oracle-selected intervention. |

Coverage: 11/11 interpretation risks addressed, with explicit limits. This is an author-side audit, not independent peer review. An exact same-machine replay does not establish independent reproducibility or cross-platform bitwise determinism.
