# FC-P010: train-only rear-lift amplitude supervision

Lead approval: implementation, unit tests, independent review, then one bounded CPU-cache diagnostic. No GPU candidate, formal evaluation, PPO or new CFD generation is authorized by this document.

## Evidence and hypothesis

P009 native diagnostic result `2dea49fae69339879ceda8fc21b2cabc5c2a8c5c06d9221eebc82a8065e0b1ef` shows that the saved head's native-versus-pooled numerical differences are much smaller than the rear-lift amplitude prediction errors on four selected training windows. The hypothesis is that adding explicit temporal amplitude supervision to the existing rear-lift readout improves the control-relevant training-phase holdout statistics.

## Fixed scope

- Reuse the immutable P009 train-only feature cache, targets, normalization, 1368 windows, 100 steps and existing source-phase mapping. No additional data or split changes.
- Freeze the FNO and every readout row except the existing rear-Cl row and its bias. This is a project-defined loss, not a new PhysicsNeMo API or architecture.
- Full-data initialization: actual saved P009 float32 rear-Cl row, represented in float64 for this diagnostic.
- Four-fold source-phase holdout initialization: fit the existing joint 50/50 alpha-zero readout on that fold's training phases only. Never initialize an OOF fold with the all-phase P009 fitted head.
- Use original window-step weights and equal H1/free-AR domain mass for the normalized rear-Cl pointwise mean squared loss.
- Add centered rear-Cl RMS supervision on the free-AR trailing 62 samples (relative steps 39 through 100), ddof=0. Normalize this RMS error by the weighted RMS of training-window truth RMS values computed on that fold's training phases only. Require a finite positive scale; fail rather than silently replacing it.
- Objective: `0.5 * pointwise_rear_cl_MSE + 0.5 * scaled_window_RMS_MSE`. No coefficient, mixture, window, budget or initialization search.
- Deterministic CPU float64 LBFGS, at most 200 optimizer iterations per fit; record actual iterations, function evaluations, convergence and final gradients. Configure bounded CPU threading and a 10-minute wall-time limit. No model checkpoint is saved.

## Review and interpretation

Before execution, independent review must verify phase isolation, gradient correctness including centered RMS, physical de-normalization, fixed non-rear-Cl rows and reproducibility. Tests must include constant bias invariance of centered RMS and known amplitude scaling.

Report the unchanged fold-specific baseline and optimized result on every held phase, both feature domains, and each family/profile: per-step forces and trailing-window total Cd, mean rear Cl and centered rear-Cl RMS. Report full-fit separately. Windows overlap and are not independent physical samples.

This is a mechanism diagnostic, not a new admission gate. Report all tradeoffs without requiring artificial dominance over every metric or claiming success from a pooled average. The Lead will decide whether evidence supports a candidate build; the original formal and real-CFD acceptance criteria remain unchanged.

Owners: Surrogate agent implements and runs after independent acceptance; Objective agent reviews and records evidence; Lead selects the next action. Stop this diagnostic on nonfinite values, identity mismatch, resource violation, or timeout, preserving partial failure evidence.
