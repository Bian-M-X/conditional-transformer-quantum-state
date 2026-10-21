# Round 4: an exact coordinate control for Adam

Status: **derived**, elementary coordinate algebra applied to the implemented models. No novelty or convergence claim. Sources: [Adam, Kingma and Ba](https://arxiv.org/abs/1412.6980v9) and [PyTorch 2.14 Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html), verified 2026-10-22. Their gradient-rescaling statements must not be confused with invariance of physical trajectories under a parameter-coordinate change.

## Assumptions and proposition

Lambda enters the MLP or Transformer only in a known affine column w. Let s>0, lambda'=lambda/s and w'=s*w; leave every other parameter unchanged. Use the same loss, batch ordering, Adam hyperparameters and initial moments, with zero weight decay, no gradient clipping and no randomness during the forward pass. Let the scaled model compute gradients in its own coordinates. Before each Adam step, replace w' by w'/s and its gradient g' by s*g'; keep optimizer moments in these physical coordinates; run ordinary Adam; then map the updated column back by multiplying by s.

**Proposition.** In real arithmetic, with compensated initial weights, the physical and compensated models have identical predictions, losses and physical-coordinate Adam moments at every step. Validation scores are equal and a common deterministic tie rule selects the same checkpoint.

**Proof.** At initialization w'*(lambda/s)=w*lambda. All subsequent neural operations therefore agree. If physical parameters agree at step t, the loss functions agree; the chain rule gives g'=g/s for the transformed column and unchanged gradients for the remaining parameters. Converting s*g' recovers g. Moment updates, bias corrections, epsilon and the parameter update are consequently identical in physical coordinates. Mapping the column back preserves the induction hypothesis. Validation follows from identical predictions. This establishes all finite steps, not convergence. QED.

The implementation calls ordinary `torch.optim.Adam` in physical coordinates using an in-place no-grad coordinate conversion after backward. It restores both the column and its gradient afterward. Moments of this one column remain physical; moments of all other coordinates are already physical. Changing this convention midway through a run would invalidate the proposition unless moments were transformed too. Closures and differentiable optimizer steps are outside the implemented API.

## Equivalent per-coordinate hyperparameter transformation

If moments were instead stored in scaled coordinates, induction gives m'=m/s and v'=v/s² for the transformed column. Ordinary scaled-coordinate Adam reproduces physical updates when eta'=s*eta and epsilon'=epsilon/s on that column alone: the denominator is (sqrt(v)+epsilon)/s, and the update of w' is s times the physical update. Keep beta values and all other coordinates' hyperparameters unchanged. Our implementation avoids per-column parameter splitting by updating in physical coordinates directly.

With unchanged eta and epsilon, the first physical update at matched initialization is -(eta/s)*g/(|g|+s*epsilon), generally different from -eta*g/(|g|+epsilon). Even when epsilon is negligible, rescaling a gradient and rescaling a parameter are different operations. A one-dimensional nonzero-gradient loss already supplies a counterexample to unchanged-hyperparameter coordinate invariance.

## Numerical and causal boundaries

Real-arithmetic equality does not guarantee bitwise equality for arbitrary s, kernels or hardware. The main control uses s=1/16 so binary exponent shifts are exact away from overflow/underflow. Unit tests also cover s=0.04 and large epsilon with declared numerical tolerances. Main runs independently save trajectories and amplitudes for verification. They do not copy the physical arm's outputs.

P vs I (matched initialization, ordinary scaled Adam) tests the coordinate intervention at a common initial function. U vs I (same scaled optimizer, different initial functions) tests the initialization intervention. C vs P is an implementation-equivalence control. Neither comparison uniquely decomposes the U-P change into additive causal contributions; interaction and finite-budget path dependence remain possible. The proposition supplies no certificate of ground-state fidelity and does not cure near-degenerate energy/fidelity ambiguity.
