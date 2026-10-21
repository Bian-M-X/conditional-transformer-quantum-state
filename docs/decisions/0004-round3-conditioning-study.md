# ADR 0004: Separate conditioning, covariance and checkpoint selection

2026-10-21. Issue #5. PR #4 is open with no reviews, verified head4215bfb. User explicitly requested continued research and GitHub synchronization. Branch `issue/5-conditioning-controls` therefore stacks on `issue/3-budget-acquisition`; this is an explicit exception to the main-based branch convention, following ADR0002. Submit an incremental draft PR, no main push/merge.

Round two cannot distinguish final-iterate improvement from validation selection and does not test input scaling or training-time covariance. Round three directly fills these gaps, freezes new signed-field points and adds N=4 as a second size. It does not add active-selection complexity after that hypothesis failed screening. Scaling and covariance are known controls, not new contributions. Correct backward H accounting is designed in from the start; differing model-call and wall costs stay visible.
