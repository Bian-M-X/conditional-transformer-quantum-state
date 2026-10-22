# A variational spin-flip witness, not a fidelity certificate

Status: **derived**, elementary consequence of the declared Hamiltonian. Added during round-five analysis planning after observing some low-field failures; not a preregistered endpoint or novelty claim. It does not change the frozen training design or route gates.

Let H(lambda)=H0+lambda Z1, with P=product X_i and P H0 P=H0, P Z1 P=-Z1. For any normalized state psi, define m=<psi|Z1|psi> and E=<psi|H(lambda)|psi>. Then

E(P psi;lambda)=<psi|P H(lambda) P|psi>=E-2 lambda m.

Consequently W=max(0,2 lambda m) is an exactly computable energy-descent witness using one known symmetry-related trial state. If lambda m>0, the original state is strictly above a constructible variational competitor by W and hence cannot be the ground state. In particular E-E0 >= W. No exact spectrum, gap or ground vector enters W. For a true ground state, lambda m<=0 is a necessary condition, including degenerate ground states.

This identity is not a gap-free fidelity certificate, a convergence theorem or a proof that gradient descent can reach P psi locally in network parameters. At lambda=0 it is identically zero, even for a highly inaccurate state. For nonzero lambda, correct-sign magnetization is still insufficient: a symmetric prediction with m=0 can have poor fidelity, as in the round-two two-level counterexample. Tiny positive witnesses require a declared numerical tolerance before empirical classification. This round reports raw signed 2 lambda m and checks the energy-error bound; no tuned detector threshold or deployable classifier is claimed.

The difference between simultaneous covariance H(-lambda)=P H(lambda) P and a variational orbit move at fixed lambda matters: the second compares two candidate states for the SAME Hamiltonian. Existing symmetry and Ritz methods already cover broad versions of orbit/subspace improvements. A future intervention must beat those baselines with full cost accounting; this identity alone is not a new algorithm.

Round-five analysis can compute W from already saved anchor magnetization without further training, H multiplication or ED. Exact energy errors and fidelities may be used afterward to evaluate the witness, clearly labeled oracle evaluation. This is a post hoc diagnostic of stored records, separate from the frozen sharing/optimizer gate. If useful, the next experiment must freeze thresholds/interventions on fresh data before claiming predictive or corrective value.
