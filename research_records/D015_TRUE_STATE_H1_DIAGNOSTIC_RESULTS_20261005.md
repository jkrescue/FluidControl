# D015 true-state H1 diagnostic results

FC-P003C remains scientifically rejected. This diagnostic separates force
fitting on existing train trajectories from validation transfer; it does not
change any gate.

## Execution and evidence

- Fixed model/state SHA: `f78c2f…697eb4` / `a63186…8b4a`.
- Result SHA: `b311715724287c34aa496405f0381fb034089123d5dcf3bec51f80633f293b54`.
- Worker receipt SHA: `f8de7c01baf7ab73e63d11ad9d6c11186f3587ae092c1a2edb51bf918ef6049b`.
- Artifact: `artifacts/fcp003c_train_vs_validation_true_state_h1_20261005/`.
- Existing formal dynamic6 H1 absolute errors were reproduced over 2,424
  endpoint-channel values; maximum difference was `3.5762787e-7`.
- `optimizer_steps=0`; no model was saved; frozen/PPO were not accessed.

The first execution attempt failed before producing scientific output because
the HDF time coordinate has shape `[N,1]` and was passed directly to `float`.
Its journal and receipt are preserved under
`artifacts/fcp003c_d015_operational_failure_v1_20261005/`. V2 changed only
explicit singleton extraction and retained every numerical input and threshold.

## Results

The table reports micro MAE over the four force channels and then the
rear-cylinder lift channel alone. Absolute errors deduplicate a phase's zero
trajectory; delta errors exclude validation zero-vs-zero rows.

| panel | absolute endpoints | nonzero delta endpoints | 4-force absolute MAE | 4-force delta MAE | rear-Cl absolute MAE | rear-Cl delta MAE |
|---|---:|---:|---:|---:|---:|---:|
| train paired targets 1–100 | 1200 | 800 | 0.029630 | 0.042622 | 0.081124 | 0.118801 |
| train late targets 100–200 | 1212 | 808 | 0.029328 | 0.042297 | 0.073322 | 0.107526 |
| validation late targets 100–200 | 606 | 404 | 0.044413 | 0.064773 | 0.116464 | 0.171848 |

Target 100 belongs to both train windows and is explicitly marked as overlap;
it is not independent evidence. Targets 101–200 lack the extra paired term but
remain covered by regular train8 windows.

By channel, delta MAE for paired/late/validation is:

- front Cd: 0.000690 / 0.001276 / 0.001326
- front Cl: 0.007111 / 0.010416 / 0.020422
- rear Cd: 0.043887 / 0.049969 / 0.065495
- rear Cl: 0.118801 / 0.107526 / 0.171848

The absolute rear-Cl values in the main table include each deduplicated zero
branch. Action-only rear-Cl absolute MAE on the paired window is 0.118696, so it
must not be compared directly with the zero-inclusive 0.081124. As a descriptive
check, not a gate, the nonzero rear-Cl action-response correlations for
paired/late/validation are 0.452600/0.387430/0.542503. Predicted-versus-true
response RMS is 0.143484/0.153617, 0.110488/0.131581, and
0.159913/0.249629 respectively. The modest correlations show that the error is
not explained by one uniform amplitude scale alone; they do not identify a
cause or define a new threshold.

## Decision implication

The extra-paired training window itself still has large rear-Cl error, so the
failure cannot be attributed only to validation distribution shift or to using
the later half of each trajectory. Validation is nevertheless materially worse
than the same-index train-late panel, so a transfer gap also remains. The next
single-factor experiment, if approved, should test train-only rear-Cl fitting or
paired-update exposure while preserving the model, data, field objective and all
evaluation thresholds. This result does not justify a larger network, a looser
gate, PPO, or frozen-test access.
