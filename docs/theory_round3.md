# Round 3: scaling changes optimization, not this function class

Status: **derived**, elementary reparameterization of the implemented models. Not a new-method or theorem-priority claim. These statements do not prove why a particular trained run succeeds or fails.

## Exact representational equivalence

For either implemented network, lambda enters only through an affine map. Write its contribution as w*lambda. If the input is changed to lambda/s with s>0, replacing the corresponding weight column by w'=s*w leaves that affine map unchanged. All later operations, all autoregressive probabilities, and hence all state amplitudes are identical for every bit string and lambda. For the MLP this is the last input column of `net[0].weight`; for the Transformer it is the last column of `condition.weight`. The reverse transformation is invertible.

Therefore the physical-input and scaled-input versions have exactly the same representable function class (in real arithmetic). This also holds after differentiable covariance averaging, since both unprojected predictions are unchanged. An observed scaling gain cannot demonstrate increased expressivity or a larger model class.

Our frozen factorial uses the same sampled weights, **without** the compensating column transformation. It therefore changes both the initial function and the optimization parameterization. Equal initial tensor hashes certify paired weights, not equal initial predicted states. Attributing all benefits specifically to better conditioning would exceed the evidence.

## Adam is not invariant to this change with a fixed learning rate

For a function-preserving initialization w'=s*w, let g be the gradient with respect to the physical coefficient w. The chain rule gives g'=g/s. With zero initial Adam moments and its bias corrections, the first physical update is

delta w = -eta*g/(|g|+epsilon).

The scaled-coordinate update, converted back to an effective physical coefficient, is

delta(w'/s) = -(eta/s)*g/(|g|+s*epsilon).

For nonzero g large relative to epsilon, the effective step differs approximately by 1/s. In the round-three s=0.04 condition this factor is25. This statement describes an individual affine weight's first update at matched functions. It is not a prediction of a25-fold speedup, and it does not imply subsequent gradient trajectories remain matched.

## Consequences for the next causal control

**Planned, not run in round three:** compare (1) physical inputs, (2) scaled inputs with unchanged sampled weights, (3) scaled inputs with the exact compensating initialization. A further optimizer control can compensate the relevant coordinate's learning rate and epsilon; it needs explicit implementation and gradient/trajectory tests, rather than a global learning-rate change that also alters unrelated weights. Keep a new frozen evaluation grid and include all extra calls in the ledger.

If the claimed effect disappears under these controls, describe a parameterization/initialization effect rather than a new conditional quantum-state representation. Separate this question from active sampling, which this factorial does not test.
