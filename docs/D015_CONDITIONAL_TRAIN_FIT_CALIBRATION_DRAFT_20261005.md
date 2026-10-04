# Conditional train-fit calibration after D015

Status: design only. This path is considered only if D015 shows that the FC-P003C checkpoint remains inaccurate on the exact `train_paired_window` targets 1..100. It is not approved for execution, is not a new admission gate, and does not authorize validation-driven tuning, PPO, or frozen-test access.

## What can be reused

Reuse `scripts/train_tandem_fno_paired_stats.py`, the official PhysicsNeMo FNO, `DynamicMatchedPairStatDataset`, `true_state_paired_optimizer_step`, the C best checkpoint, train20+train8+train16 regular data, dynamic8 action/zero pairs, train20 normalization, H100/chunk10, batch 1, AdamW settings, zero teacher forcing, lambda 10, force weights `[1,1,4,1]/7`, gradient clipping, and the interleaved schedule. Every paired update already evaluates the unchanged regular field/force rollout objective on its regular batch and adds the paired objective; therefore increasing paired exposure does not remove the regular field loss.

There is no honest config-only execution path today. Lines 369–383 of the trainer deliberately require exactly two dynamic8 passes and 16 paired updates for `true_state_step_force`. Merely adding epochs would keep the ratio at 16 paired updates per 1368 regular updates and would not isolate increased paired exposure. A later approved implementation should make only a narrow fail-closed guard extension for the exact calibration contract below; it must not add a model, loss, loader, optimizer, or training framework.

## Fixed bounded contract

Start from the immutable C epoch-2 model SHA `f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4` as `training.initial_checkpoint`, with a fresh optimizer exactly as the existing initialization path does.

Use one epoch with:

```yaml
training:
  epochs: 1
  max_train_batches: 128
  expected_regular_batches: 128
  paired_dataset_repetitions: 8
  paired_batches_per_epoch: 64
  paired_batch_schedule: interleaved
  paired_batch_size: 1
  paired_objective_kind: true_state_step_force
  paired_step_chunk_size: 10
  teacher_forcing_start: 0.0
  teacher_forcing_end: 0.0
```

All other resolved values must be byte/field identical to the C configuration. This is exactly 128 optimizer steps: 64 regular-only steps and 64 regular-plus-paired steps. All 128 steps retain the regular rollout loss; the 64 paired steps comprise eight complete deterministic passes over the eight dynamic pairs. Compared with completed C, the diagnostic supplies twice as many paired updates as C's full two-epoch total (64 versus 32), while bounding regular computation to 128 rather than another 1368-step epoch. The runner must record the 128 regular sample identities, eight ordered complete pair passes, per-step finite status, gradient clipping, memory floor, parent/output SHA, and zero validation/frozen access.

## Train-only readout and exit rule

Before and after calibration, evaluate the same checkpoint pair on the fixed D015 train panels without gradients:

- H1 absolute and action-minus-zero delta MAE/RMSE/bias for all four forces, reported per phase/profile and pooled;
- H1 `u`, `v`, and `p` field errors on the same targets and masks, reported per phase/profile and pooled with the existing physical normalization;
- targets 1..100 as the primary fit panel and targets 100..200 only as a non-selection transfer diagnostic; target 100 remains an explicit overlap check.

This small run is useful only if force and delta errors decrease consistently across the eight pair identities, including rear lift, while the same-sample `u/v/p` field errors do not regress. No new numeric admission threshold is introduced: the comparison determines whether paired exposure is a plausible mechanism, not whether the surrogate is deployable. If the paired panel does not improve consistently, or improves force by degrading fields, stop this branch—do not increase lambda, epochs, or network size. If it does improve, a separate Lead decision is still required before the unchanged validation10, Dynamic6, force-window, and development protocol; PPO remains blocked until every existing gate passes.
