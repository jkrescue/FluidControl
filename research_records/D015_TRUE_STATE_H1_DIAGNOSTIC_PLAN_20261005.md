# D015 true-state H1 diagnostic execution contract

Diagnostic only, not a gate, selection run, PPO authorization, or frozen test.

- `train_paired_window`: dynamic8 targets 1..100; extra paired supervision plus regular training.
- `train_late_window`: targets 100..200. Target 100 overlaps the paired prefix; targets 101..200 do not. All occur in regular training, so this is not called unseen.
- `validation_late_window`: dynamic6 targets 100..200, retaining its actual validation split.

Each H1 prediction consumes recorded q_t and omega_t/omega_t+1. Output preserves signed absolute and action-minus-zero error for all four forces and reports MAE/RMSE/bias. Same-phase zero rows are summarized once. Existing formal H1 absolute errors are compared only as a reproduction check.

Lead approved and the one-shot Worker run completed under the pinned PhysicsNeMo
image, allocator .15, chunk 10, and continuous 20 GiB guard. It mounted only
train8, four train-phase zero HDFs, dynamic6 validation, train normalization, C
best checkpoint/config, and existing segments. No frozen path, optimizer, save,
or policy execution occurred. The first operational attempt failed before a
result because a singleton HDF time array was passed directly to `float`; that
failure is preserved. The approved v2 changed only explicit scalar extraction
and completed with the original numerical contract. See
`docs/D015_TRUE_STATE_H1_DIAGNOSTIC_RESULTS_20261005.md`.
