# Conditional train-fit calibration after D015

Status: CPU implementation and nine focused tests complete in commit `ff34a1c`; GPU execution remains subject to separate review and approval. D015 result SHA `b311715724287c34aa496405f0381fb034089123d5dcf3bec51f80633f293b54` shows that FC-P003C remains inaccurate on the exact `train_paired_window`: rear-Cl action-minus-zero MAE is 0.118801 there, 0.107526 on train-late, and 0.171848 on validation-late. The corresponding absolute values 0.081124/0.073322/0.116464 include deduplicated zero branches and must not be compared as action-only errors. Thus train fit is already inadequate, with an additional validation gap. This implementation is not a new admission gate and does not authorize validation-driven tuning, PPO, or frozen-test access.

## What can be reused

Reuse `scripts/train_tandem_fno_paired_stats.py`, the official PhysicsNeMo FNO, `DynamicMatchedPairStatDataset`, `true_state_paired_optimizer_step`, the C best checkpoint, train20+train8+train16 regular data, dynamic8 action/zero pairs, train20 normalization, H100/chunk10, batch 1, AdamW settings, zero teacher forcing, lambda 10, force weights `[1,1,4,1]/7`, gradient clipping, and the interleaved schedule. Every paired update already evaluates the unchanged regular field/force rollout objective on its regular batch and adds the paired objective; therefore increasing paired exposure does not remove the regular field loss.

There is no honest invocation of the current full trainer for this diagnostic. It always constructs and evaluates validation data, while D015 requires a train-only calibration; `max_validation_batches=0` is false-like and would run the full validation loop. The trainer also locks each epoch to two dynamic8 passes and 16 paired updates. A separate small wrapper is therefore appropriate, but it must import the existing datasets and objective functions rather than copy their mathematics.

## Fixed bounded contract

Start from the immutable C epoch-2 model SHA `f78c2f3341663ed2f6e7f4c64a0bf6539a065d2e7f11ada8f19f493320697eb4` as `training.initial_checkpoint`, with a fresh optimizer exactly as the existing initialization path does.

Use one continuous train-only iterator for exactly 128 optimizer steps. Insert 64 paired updates at the fixed interleaved positions, including positions 0 and 127, so the eight dynamic pairs are each consumed exactly eight times. Do not reset the regular iterator, optimizer, or pair schedule at artificial short-epoch boundaries. Use a fresh AdamW optimizer at the fixed FC-P003C initial learning rate, with no learning-rate scheduler and no checkpoint selection.

```yaml
effective_calibration:
  optimizer_steps: 128
  regular_iterator_restarts: 0
  paired_steps: 64
  paired_complete_passes: 8
  paired_batch_schedule: interleaved
  paired_batch_size: 1
  paired_objective_kind: true_state_step_force
  paired_step_chunk_size: 10
  teacher_forcing_ratio: 0.0
  scheduler_steps: 0
  selection_performed: false
```

All other resolved values must be byte/field identical to the C configuration; the wrapper records the distinct effective 128/64 contract so the parent configuration's two-epoch declaration cannot be mistaken for this calibration schedule. Exactly 64 steps are regular-only and 64 are regular-plus-paired. All 128 steps retain the regular rollout loss; the 64 paired steps form eight complete deterministic passes over dynamic8. Compared with completed C, paired exposure doubles from 32 to 64 while regular computation is bounded to 128 rather than another 1368-step epoch. The wrapper imports `TandemRolloutDataset`/`compose_training_data`, `DynamicMatchedPairStatDataset`, PhysicsNeMo `DataLoader`, `regular_rollout_objective`, and `true_state_paired_optimizer_step`; it does not instantiate validation data. It records all regular sample identities, all eight ordered pair passes, per-channel paired losses, finite/clip telemetry, parent/output/config/source SHA, fixed learning rate, and zero validation/frozen access.

## Train-only readout and exit rule

Before and after calibration, evaluate the same checkpoint pair on the fixed D015 train panels without gradients. For each endpoint, call `true_state_step_input(q_t, ..., omega_t, omega_t+1)`, then existing `train_tandem_fno.predict`; compute `qhat_{t+1}=(q_t+delta)*mask`. Reuse `evaluate_tandem_fno.field_error_sums` and `relative_field_metrics` with `(qhat-q_target)*state_std` and `q_target*state_std+state_mean`, so pooled `u/v/p` and velocity relative-L2 have exactly the formal evaluator's meaning. The force output from the same call supplies the four physical-force errors after the unchanged force denormalization.

- H1 absolute and action-minus-zero delta MAE/RMSE/bias for all four forces, reported per phase/profile and pooled;
- H1 `u`, `v`, and `p` field errors on the same targets and masks, reported per phase/profile and pooled with the existing physical normalization;
- targets 1..100 as the primary fit panel and targets 100..200 only as a non-selection transfer diagnostic; target 100 remains an explicit overlap check.

This small run is useful only if force and delta errors decrease consistently across the eight pair identities, including rear lift, while the same-sample `u/v/p` field errors do not regress. No new numeric admission threshold is introduced: the comparison determines whether paired exposure is a plausible mechanism, not whether the surrogate is deployable. If the paired panel does not improve consistently, or improves force by degrading fields, stop this branch—do not increase lambda, epochs, or network size. If it does improve, a separate Lead decision is still required before the unchanged validation10, Dynamic6, force-window, and development protocol; PPO remains blocked until every existing gate passes.
