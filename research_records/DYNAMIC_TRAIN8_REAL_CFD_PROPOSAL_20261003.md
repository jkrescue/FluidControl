# Dynamic train8 real-CFD proposal (not executed)

Status: **proposal only**. No case has been generated or solved, and no waveform
may be selected after inspecting the validation-only dynamic6 results.

The fixed-action full40 development gate fails at long horizon (e5 H100 pooled
total-Cd NRMSE 372.18, worse than persistence). Therefore PPO and closed-loop
claims remain blocked. The next useful experiment is new real OpenFOAM training
coverage for action transients, not reward tuning.

## Leakage boundary and matrix

- Use only train phases `b00`, `b02`, `b04`, and `b06` from exact real restart
  states. Never read the dynamic6 validation phases `b01`/`b05` to select a
  waveform, and never open frozen phases `b03`/`b07`.
- Predeclare two complementary, deterministic, near-zero-mean profiles per
  train phase: one smoothed persistently exciting binary sequence and one
  fixed multisine/chirp family. This gives eight 20-D/U trajectories.
- Freeze all knots, seeds, signs, amplitudes, and phase offsets in a
  machine-readable GitLab artifact before generating any case. Mirrored
  schedules must be frozen together; no result-dependent replacement is
  allowed.
- Retain `|omega| <= 0.75`, `|delta omega| <= 0.1` per 0.1-D/U decision,
  `omega(0)=omega(20)=0`, `dt=0.005`, field interval 0.1 and force interval
  0.005. Each case is 4,000 steps and 201 real field states.

## Provenance and processing

Each branch must bind the five source-state file SHA-256 values and both
baseline force-file SHA-256 values, verify the actual OpenFOAM omega table,
pin the existing OpenFOAM image, and retain solver/Courant/continuity logs.
Raw, VTK and HDF stages require separate atomic SHA receipts. Curate into a new
train-only profile with the official PhysicsNeMo Curator Source/Filter/Sink
pipeline; do not mutate full40 or dynamic6 and do not compute statistics from
validation/frozen data.

Retrain the official PhysicsNeMo FNO on the augmented train data. Keep the
existing `b01`/`b05` dynamic6 panel strictly validation-only and reuse the
unchanged H1/H10/H50/H100 protocol: pooled H100 total-Cd NRMSE at most 0.10
and strict common-start action-difference total-Cd MAE at most 0.023. Report
stability, persistence comparison, lift errors, and action ordering. Failure
stops model promotion; it does not trigger waveform selection on validation.

This experiment tests whether action-history coverage repairs the surrogate.
It is not evidence of closed-loop control, and PPO remains unauthorized until
the unchanged validation gates pass and a later paired real-CFD acceptance run
is completed.
