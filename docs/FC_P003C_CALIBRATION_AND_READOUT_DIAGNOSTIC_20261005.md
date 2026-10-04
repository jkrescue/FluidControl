# FC-P003C calibration evidence and fixed-feature readout diagnostic

Status: the two bounded train-only calibrations are complete and rejected. Implementation and CPU tests for a fixed-feature readout diagnostic are authorized; feature extraction will require a separately approved GPU pass, followed by CPU fitting. It is not a model candidate, gate, or PPO authorization.

The delta calibration used the C epoch-2 parent, one continuous 128-step regular stream, 64 paired updates and eight complete dynamic8 passes. Rear-Cl action-minus-zero MAE improved only 3.02%/2.57% on prefix/late while absolute rear-Cl and every field channel regressed. The absolute calibration kept the same sample order and initial readout and changed only the paired loss. Action rear-Cl improved 5.68%/5.13%, but zero rear-Cl worsened 174.9%/145.0%, delta improved only 2.19%/1.82%, and `u/v/p` all regressed. Neither branch proceeds to formal validation or PPO.

The next diagnostic freezes the original C model and extracts the exact masked-mean 128-dimensional feature immediately before its existing four-force affine layer. It fits a 129-column affine readout in float64 with `lstsq(rcond=1e-10)` using only train8 targets 1–100. Each phase's zero feature is computed once and repeated only to give the same symmetric action/zero weight as its two action profiles. Targets 101–200 are read only after fitting as a same-trajectory temporal-transfer check. No model or fitted readout is saved; validation, frozen data and PPO are excluded.

Required audit fields are: original-readout reproduction within a declared floating-point tolerance (the first three model outputs remain unchanged by construction, and all model parameters/buffers must remain unchanged); train/late action, zero and action-minus-zero metrics for all four force channels; unique versus weighted row counts; matrix rank, retained singular spectrum/condition number and coefficient norm. Repeated zero rows reduce the number of independent observations and intentionally change weighting, so nominal row count is not effective sample size. The late panel shares trajectories and phases with the fit panel and is not independent generalization evidence.

Interpretation is limited:

- prefix and late improvement would show that the frozen feature contains force information accessible to an affine map, but would not prove that optimizer choice is the sole failure;
- prefix improvement without late improvement indicates temporal/state-distribution overfit or an ill-conditioned readout;
- failure on prefix indicates that this affine readout cannot recover the missing response, but does not distinguish insufficient features from a nonlinear mapping;
- any success is force-only diagnostic evidence: it does not improve the state rollout, satisfy the unchanged field/window gates, or authorize PPO.
