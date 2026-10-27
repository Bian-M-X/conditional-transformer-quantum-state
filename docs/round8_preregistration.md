# Round8: one-step refinement/acquisition headroom gate

2026-10-27, Issue19. Material Passport: code experiment / development diagnostic, Bian-M-X/Codex-assisted. Frozen source/config commit must precede all scientific runs. User authorized this gate and optional Astra advice; an actual gpt-6-astra/high subagent reviewed the proposed design before freezing. This is advisory AI review, not independent-human validation or a novel algorithm.

## Question and scope

Does hindsight choice among a few fully specified refinement/acquisition actions leave meaningful error-reduction headroom over strong simple strategies? A negative result closes this action-menu/task/budget pilot, not all possible allocation research. A positive result only justifies checking whether permitted information predicts value; it does not demonstrate deployable speedup, learnability, or ICLR novelty. The main endpoint is eight-query mean infidelity at a fixed additional training-closure allowance, not equal total compute.

Exact enumeration and exact projected matrices deliberately isolate R/A. There is no Monte Carlo noise or M/resampling action; this cannot validate the three-action proposal from Issue17. ED and true gaps are evaluation-only after ALL branches of one job are frozen. No ED-driven training, warm start, checkpoint choice, action choice or retries.

## Fixed factorial and actions

Authoritative config `experiments/configs/round8.json`: N8, open chain h=.55 and two-leg open ladder h=1.15, each radius r=.06/.4, MLP/Transformer, five new paired seeds107/223/311/419/521. Forty jobs, each containing three correlated maturity states100/400/1200 closures per endpoint model, thus120 decisions. Statistical replication is seed, not maturity or query point. A separate model is trained at each endpoint lambda=-r,+r, from identical sampled initial weights. Different tasks/families are not jointly learned.

Eight held-out query fractions ±{.08,.3,.65,.9} times r exclude old endpoints and prospective center. For each maturity, fork complete model/optimizer/RNG snapshots:

- STOP: retain current basis; zero additional training. It is a budget-ceiling action, not a fictitious200-closure job.
- R-left/R-right: continue the corresponding optimizer for200 closures and APPEND the resulting state while retaining both old states. No old snapshot is discarded.
- A-center: evaluate both old models at lambda0, choose the lower initial variational energy (left on exact tie), charge two H actions/model sweeps, clone those model weights, reset optimizer, then train200 closures at0 and APPEND. Ground labels never choose the warm start.

All non-STOP raw bases thus contain at most3 learned snapshots. Primary bases additionally include the spin-flipped copy of every snapshot, with rank at most6; STOP primary basis has at most4. H(-lambda)=P H(lambda)P, so this zero-training closure is a mandatory strong known-physics baseline, not a new method. Its projection/linear algebra cost is charged. Raw counterparts are secondary sensitivity analyses using the same trained states, not extra training.

## Safe budgets and provenance

New `ResumableFit` leaves frozen round7 sources unchanged. Every successful objective/gradient closure counts, including rejected trials. At a hard block boundary inside L-BFGS line search, restore the model, curvature memory and RNG to the start of that incomplete optimizer call; retain all consumed work in counters/history. This saves the last fully accepted state, never a partial line-search state. Same block schedule plus save/load must reproduce exactly. Segmentation can alter the trajectory relative to an uninterrupted call; no claim of equivalence to a single1200-closure run.

Base chunks are100,300,800 closures, preserving optimizer memory across maturities. A branch chunk is200. Tolerance settings are zero; repeated current-point evaluations can consume the allowance without changing weights, and completed optimizer-call counts are not physical-update counts. Rollback work is part of the action, not a pure physical-intervention effect.

Record forward/backward H actions, full-basis network sweeps, optimizer calls, accepted/discarded closure work, checkpoints, affine component actions, projection/online query/reconstruction time and wall time separately. These units are not summed as a single compute number. Cached affine Ritz uses one SVD and once-built Q^T H_j Q; online energy/observables need no full H or full-state reconstruction. Full vectors reconstructed only for declared evaluation. Offline execution of all counterfactuals is research cost, never the cost of a feasible oracle policy. Policy-probe overhead and cache invalidation are separately visible.

Full checkpoints are local ignored `.pt` files, safely loadable with weights_only=True; hashes and regeneration commands accompany tracked manifests. Compressed JSON amplitudes and full numerical metrics/history are retained. Per-job600s and invocation3600s cap; retain failure records, no seed replacement or hidden numerical retries.

## Simple strategies and hindsight gate

Before any oracle access, record two known-information choices. `anchor_residual` chooses R-left/right by larger current anchor-state energy-residual norm (left tie). `snapshot_residual` chooses among old endpoints and center using largest Ritz residual of the primary closed basis; endpoint means refine it, center means acquire. It is an uncertified NQS adaptation of residual/snapshot reasoning, not a claimed faithful reproduction of the PDE/active-EC papers. Three H probes plus two anchor probes and reconstruction costs are charged; not treated as free deployable decisions.

At each family/radius/architecture/maturity stratum, define e_s,a as the mean infidelity of a frozen action over the eight query points. Tiny negative1-F from floating roundoff is clamped to0 for loss aggregation only; raw metrics retained. The simple-policy set contains STOP, three constant training actions and the two residual rules. B=min_policy mean_seed e_s,policy(s); O=mean_seed min_action e_s,a; G=1-O/B (zero when B=0). The hindsight best fixed/simple policy is already optimistic and evaluated on this pilot; it is not a learned or out-of-task policy. O is an empirical one-step ceiling for THIS finite menu, not a bound over all algorithms. Include absolute B-O, five paired improvements and leave-one-seed-out G without interpreting those as confidence intervals.

Primary investment gate uses flip_closed and maturity1200 only. Eligible strata require B>=1e-3, G>=.20 and every leave-one-out G>=.10. At least2 eligible strata must collectively cover BOTH physical families. Additionally compare best-R=min(R-left,R-right) versus A: in eligible states, both R and A must each beat the other type AND STOP by>1e-4 for at least two distinct seed IDs. This prevents left/right selection alone from masquerading as a refine/acquire mechanism. No pooling of maturity states into independent samples, no statistical-significance claim, and no switching primary metric after observing results.

If primary gate fails, classify: already solved by STOP/closure, insufficient magnitude/support, left/right-only headroom, or no robust R/A switching. Early100/400 or raw-basis headroom remains exploratory. Energy, variance, Z0 magnetization, connected end correlation and training-anchor quality are secondary and complete. Apparent energy improvement does not certify fidelity. Any next N10/Monte Carlo experiment must be separately frozen; this pilot is not held-out validation.
