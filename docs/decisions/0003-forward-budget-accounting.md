# ADR 0003: Correct the H-action accounting interpretation

Date: 2026-10-21. Status: audit correction, before any stage-B run.

The frozen counters in ADR 0002 and the initial prerecord count **forward** training and acquisition H-vector applications. They do not count the H-vector application in automatic differentiation of `psi @ H @ psi`. Calling them total H actions without this qualification was inaccurate.

A CPU torch profiler on a 64-component quadratic energy and its backward pass records two `aten::mm` operations of shapes [1,64] by [64,64]. The implementation keeps H fixed, so one is the forward and one is the backward H action. Store the short profiler output under the round-two results; it is a computational-cost audit, not a model benchmark.

We preserve frozen training/selection behavior and the 2400-forward-action cap, rather than silently changing the design. For stage B report:

- forward training + acquisition H actions = 2400;
- backward H actions = number of training parameter samples;
- forward + backward training/selection H actions = 4800 - acquisition H actions;
- model calls, gradient updates, postprocessing/evaluation work, and wall time separately.

Thus adaptive policies receive **less** total H work than passive policies under this protocol. This is a qualified resource comparison, not an exactly matched total-compute experiment; it cannot support a claim of optimal cost efficiency. Names in historical JSON counters remain stable, with explicit scope and additional counters in new stage-B manifests. Plot labels, reports and handoffs must use the qualified wording.

The stage-A run `20261020T172651729101Z-round2a-transformer-even-s61` timed out following a long wall-clock interruption. Its failed manifest is retained, no metrics are inferred, and the configuration will be retried in a new run directory with unchanged numerical settings. The interruption cause is not established by the logs. Wall-clock timings spanning it are not CPU utilization measurements. Resume the remaining frozen suite as a separate invocation under the same per-invocation 30-minute cap.
