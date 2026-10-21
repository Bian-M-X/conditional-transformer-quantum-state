# Round 4 preregistration: initial functions versus Adam coordinates

Frozen before main experiments, 2026-10-22. Issue [#7](https://github.com/Bian-M-X/conditional-transformer-quantum-state/issues/7).

## Material Passport

Mode: code experiment run; owner: Bian-M-X / Codex-assisted; input: round-three handoff and theory; output: frozen config, runner, per-run evidence, descriptive report. Status at registration: planned. No human-subject data or external model service. User authorized continued research, execution and GitHub publication.

## Question and four arms

The previous scale comparison changed both the initial function and the Adam coordinate system. Use the same sampled tensors for four arms: P (`physical`), U (`scaled`, unchanged tensors), I (`matched`, compensate the lambda column by s), C (`canonical`, same compensated initialization plus physical-coordinate Adam). All use raw positive autoregressive amplitudes; covariance is evaluation-only here. P versus I isolates the coordinate effect at matched initialization; U versus I isolates initial-function differences at the same scaled coordinates. P versus C is an equivalence control, not a proposed performance improvement. Effects may interact; no additive attribution of percentages.

## Frozen configuration

`experiments/configs/round4.json` is authoritative. N=6; h=0.4 and 0.6; MLP and Transformer; seeds 17,29,43,61,73; four arms: 80 jobs. h values index separate trained models, not cross-h generalization. 1200 full-batch Adam steps, lr=0.005, eps=1e-8, other defaults, foreach=False, float64 CPU, one Torch thread, no weight decay/clipping/schedule/dropout. Scale s=1/16, chosen as the training bound and an exactly representable binary scaling factor, not from a spectral oracle.

- Training lambda: -0.0625, 0, 0.0625.
- Validation lambda: -0.02, -0.004, 0.004, 0.02.
- Test lambda: ±0.0003, ±0.0015, ±0.008, ±0.027, ±0.052.

No point overlaps among the splits. The new grid is a frozen screening extension, not independent external validation. It becomes development data after this round.

## Endpoints and acceptance

Primary metric: per-seed mean test infidelity of the checkpoint selected by lowest validation variational energy, checked every 50 steps. Final checkpoint, worst point, energy/magnetization/connected-correlation errors and learned lookup, post-covariance and training-span Ritz are secondary. Exact diagonalization is called only after both checkpoints are frozen; no oracle for initialization, training or selection.

Report all cells, seed values, sample SD and paired differences within each h/architecture. No significance or population confidence claim from five seeds; correlated lambda points are not replicate samples. Primary P-C control tolerance: 1e-10 absolute plus 1e-8 relative on histories, direct amplitudes and metrics, with identical selected steps. Save amplitude arrays so this can be checked directly, not only through energy or hashes. Any discrepancy is a reported failure of the control; no post hoc tolerance changes. Exact hashes, when equal, are additional evidence.

## Accounting, monitoring and failure handling

Each job: 3600 forward training H actions + 3600 backward H actions + 96 validation H actions = 7296; 3600 training and 96 validation model calls. Six extra model calls compare initial training-grid functions for every arm. Evaluation calls, two H actions per metric, Ritz H work, ED and wall time are separate. C has 1200 coordinate conversions/optimizer steps. Equal H/model counts do not make wall time identical.

Hard cap 600 seconds per run; 1800 seconds per invocation; monitor process and training records. Failed records remain outside completed aggregates. No silent retries or replacing seeds; a code failure requires diagnosis and a documented corrective commit before rerun. Default run resumes only missing job identities. Nondefault tags produce independent replays without changing the statistical index. Training source/config must be committed with a clean tracked tree. Run one canonical Transformer replay at h=0.4, seed=17; exact same-machine history, metric and amplitude comparison, not independent-member replication.

## Limits and stop rule

If C reproduces P, it establishes implementation/coordinate equivalence under this control, not that all scaling gains disappear under ordinary Adam. If I differs from P, report coordinate-dependent optimization at fixed initialization; if U differs from I, report initialization dependence at fixed scaled coordinates. Do not generalize across arbitrary scales, dimensions, models, or optimizers. No new optimizer, scalable method or ICLR-readiness claim; full-enumeration N=6 remains a mechanism diagnostic. Independent member replication is still required before formal STATE promotion.
