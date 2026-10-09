# v4 H20 validation-only protocol

Date: 2026-10-03. Entry point:
`scripts/run_control_gap_v4_validation_only_spark.sh`.

## Purpose and scope

This is the formal post-training **development-validation comparison** for
the five-epoch v4 H20/rear-drag FNO. It computes 1/10/50/100-step total-drag
NRMSE for:

1. the completed v4 candidate; and
2. its immutable v3 H20/rear-drag parent, re-evaluated on exactly the same v4
   validation data but with the v3 train-only normalization on which that
   checkpoint was trained.

The script cannot select another split and contains no frozen-test evaluation
command. A lower v4 100-step NRMSE is a development result only. It neither
changes the existing 10% accuracy target nor demonstrates closed-loop CFD
control benefit.

## Fail-closed checks

Before requesting the GPU, `scripts/audit_control_gap_v4_validation.py`
requires all of the following:

- the data root is exactly `tandem_cylinders_control_gap_v4`, whose manifest
  declares profile `control_gap_v4`, 28 training trajectories, four validation
  trajectories and action support `max_abs_omega=5.0`;
- both v4 and v3 normalizations contain the expected state and four force
  channels, finite positive standard deviations, and
  `computed_from: train split only`;
- all four v4 validation HDF5 files are byte-identical to the immutable v3
  validation files. Only the `validation/` directories are hashed;
- v4 training history contains exactly epochs 1 through 5 and a single matched
  official PhysicsNeMo `.mdlus`/optimizer `.pt` best-checkpoint pair exists;
- the parent has a single matched best-checkpoint pair;
- candidate and parent resolved configurations have the same FNO architecture,
  output seven channels and use force indices `[0,1,2,3]` in the same order;
- no `train_tandem_fno_rollout.py` process is running. This is checked before
  preflight and again before each of the two sequential GPU evaluations;
- host `MemAvailable` is at least 20 GiB and the pinned container image ID is
  unchanged.

Each evaluation uses the repository's existing `evaluate_tandem_fno.py`,
`conf/tandem_fno_total_drag.yaml`, observed actions, stride 25, batch four and
zero visualizations. The isolated official PhysicsNeMo 2.2.2 container has no
network, uses GPU 0, has a 64 GiB container limit, and is wrapped by
`spark_gpu_guard.py --min-free-gib 20 --allocator-fraction 0.20 --margin-gib 4`.

## Outputs

The wrapper refuses to overwrite a previous formal result and writes under:

`artifacts/tandem_cylinders/control_gap_v4_validation_only_20261003/`

- `preflight.json`: dataset, normalization, validation trajectories and both
  checkpoint SHA-256 provenance;
- `v4_candidate_validation.json`: unmodified evaluator report;
- `v3_parent_on_v4_validation.json`: parent report using its v3 training
  normalization on the byte-identical v4 validation trajectories;
- `v4_candidate_metric_integrity.json` and `v3_parent_metric_integrity.json`:
  strict profile/checkpoint/normalization audits and recomputed pooled, macro and
  worst-case 100-step terminal-force NRMSE;
- `decision.json`: 1/10/50/100-step macro NRMSE table, pooled 100-step metrics,
  worst cases, candidate-minus-parent deltas and the strict pooled development
  decision;
- `VALIDATION_COMPLETE` or `VALIDATION_FAILED`: terminal marker.

Normalization is part of each checkpoint's training contract. The v4 candidate
therefore uses v4 train-only statistics and the v3 parent uses v3 train-only
statistics. Forcing either model onto the other's scale would be scientifically
invalid. Comparability instead comes after each prediction is de-normalized to
physical force units: both evaluations use byte-identical raw validation force
targets, so total-drag NRMSE has the same physical target RMS denominator.
`decision.json` records both normalization roots and hashes. The comparison
must not be presented as an equal-training-budget ablation because the v4 run
is a five-epoch warm start.

The strict improvement decision uses pooled 100-step terminal total-drag NRMSE,
not the evaluator's arithmetic mean of per-case NRMSE values. The fixed 10%
Gate-B threshold is also reported without modification. Both pooled and macro
values remain terminal instantaneous-force errors; neither establishes accurate
mean `Cd_total` over the full 100-step window.

## Execution state

The wrapper and audit are prepared but have not been run. Do not start them
until the active v4 H20 training process has fully exited. No user systemd unit
is installed or enabled in this change; this avoids an accidental concurrent
GPU evaluation. After code review, automation may call this exact wrapper—the
wrapper's process, memory, image, split and completion checks remain mandatory.
