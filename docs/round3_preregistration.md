# Round 3 frozen protocol

2026-10-21, Issue #5. Frozen in Git before new training; not an external registration. Configuration: `experiments/configs/round3.json`.

## A: directly audit the checkpoint confound

Twenty jobs: original N=6 lambda=0 task, MLP/Transformer, raw/even training, five paired seeds, 2400 updates. Preserve original initialization, optimizer, point order and validation rule. Evaluate both final and validation-best checkpoints only after training; match saved selected-checkpoint history/metrics to round two. The old h grids are development data. No stopping or selection based on fidelity. Saving both endpoints resolves the missing-final-iterate limitation without choosing the better endpoint after the fact.

## B: scaling by training-domain size and covariance during training

Eighty jobs: N=4/6 at h=0.5, two architectures, scales 1/0.04, raw/covariant training, five paired seeds. Train at lambda={-0.04,0,0.04}; four validation and ten test values in the config are disjoint. Positive and negative values are equally weighted. Full-batch Adam, 1200 updates, lr=0.005, validate every50 updates. The scale is a domain endpoint, never an oracle gap. Same weights at initialization do not mean the same initial function after input scaling; this confound must be disclosed.

Covariant training normalizes u(lambda)+P u(-lambda) using differentiable full-state enumeration. At zero, reuse the same forward call. It is a known construction, not a proposed new architecture or a scalable sampling implementation. Raw and covariant models use identical train/validation H inputs. Evaluate final and validation-best checkpoints, direct states, same-network learned lookup, inference-time covariance average and training-state-span Ritz. Never project nonzero-lambda states into even parity. No ED until all checkpoints are frozen.

Primary endpoint: per-seed mean test infidelity of **validation-selected direct** states. Secondary endpoints: final direct, worst-point error, training errors, Z1/correlation errors and paired postprocessing comparisons. Compare scaling and covariance within architecture/size/seed; report the full factorial, sample SD and paired deltas. Seed count is5, not the number of correlated grid values. No significance or broad superiority claim based on one favorable cell. A pattern appearing only at one size/architecture is limited to that setting. Test grids become development data after this round; no retuning within this study.

## Resource ledger and stopping

Count forward training, backward H applications, validation, network calls, evaluation/postprocessing and wall time separately. For A: 7200 forward +7200 backward +96 validation =14496 training/selection H actions. For B: 3600 forward +3600 backward +96 validation =7296. All B cells share H counts and gradient updates, but covariant training uses more network calls and normalization; no equal wall-compute claim. Postprocessing and ED are outside training budgets and separately recorded. One run cap600seconds, invocation cap1800seconds, preserve failed records and resume unchanged configs in new directories.

## Decision gate

If scaling/covariant training resolves failures, do not attribute gains to active acquisition. If final and validation-best differ, report the direction and size; do not claim validation energy universally fails. Require adequate training-point fidelity before future acquisition claims; carry forward negative results, no architecture/optimizer tuning on this grid. This is a mechanistic screening study, not an ICLR-ready method or independent-member replication.
