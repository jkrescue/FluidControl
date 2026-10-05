# FC-P011: paired decoder-representation experiment

Status: Lead approves implementation and CPU tests only. GPU preflight and training require separate approval after independent review. No new architecture, dataset, formal threshold, PPO authorization or replacement of the project goal.

## Hypothesis and comparison

The fixed linear force readout diagnostics have not repaired controlled lift amplitude. Test whether adapting the last existing nonlinear decoder layer improves this response compared with adapting only the rear-lift output row under otherwise identical native training.

Both arms start from P009 model `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` and training-state `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`. Use the pinned official PhysicsNeMo FNO and existing DataPipe/loader implementation. The training selection/masking and loss are project code, not claimed official APIs.

- Arm A: only `decoder_net.final_layer.linear` rear-Cl row 6 and bias 6 may change (129 effective trainable coefficients).
- Arm B: additionally allow `decoder_net.layers.1.linear.weight` and `.bias` (16512 coefficients).
- Freeze all other tensors. Mask final-layer gradients and preserve frozen rows against optimizer weight decay; test actual parameter values after updates, not just `requires_grad`. Record effective trainable coefficients separately from whole-tensor optimizer parameter counts.
- B changes a shared decoder layer: its field, Cd and front-Cl outputs can change even when their final rows stay fixed. Never claim those predictions are preserved. A preserves field and other force predictions.

## Common training protocol

Reuse the existing 1368 train-only H100 regular windows (base20/train8/train16; family counts 720/408/240), original fixed train normalization, batch size 1, one complete identical deterministic pass and seed 20261003. Do not add paired updates or use validation to select checkpoints.

Both arms use AdamW, learning rate 1e-5, weight decay 1e-4, global gradient clipping 1, no teacher forcing, existing default TF32/high precision, and equal step weights over H100. Objective:

`L = existing masked normalized field MSE + 0.2 * (0.5 * mean_four_normalized_force_MSE + 0.5 * normalized_rear_Cl_MSE)`.

This corresponds to force channel weights `(0.125, 0.125, 0.125, 0.625)` in front-Cd/front-Cl/rear-Cd/rear-Cl order. Both arms use the same loss, order, optimizer and budget: trainable scope is the only experimental difference. P010 is contextual evidence, not this experiment's control arm.

## Implementation and review

Reuse existing official loading/saving, rollout, regular loss, gradient checkpointing where already used, resource guards and train-only diagnostics. Do not invent a new training framework. CPU tests must cover exact loss arithmetic, gradient masks, weight-decay-safe frozen rows, per-scope confinement, initialization identity, data order equality and no held/frozen dataset inclusion.

Before GPU training, run a separately approved small forward/backward resource probe for each scope: full H100 gradient behavior can differ despite frozen encoder parameters because predicted states feed later steps. Do not infer memory requirements solely from the number of trainable weights. Preserve at least 20 GiB unified MemAvailable; estimate time and memory from the probe before authorizing the full pass.

Predeclare fixed train-only diagnostic windows before training; use the same windows for both arms at steps 0/32/128/512/1368. Report H1/free-AR rear-Cl errors, trailing62 centered RMS, mean lift, all forces, field u/v/p errors, per-group gradient norms and clipping. Diagnostics cannot select an early checkpoint or trigger scientific early stopping; nonfinite, identity/resource violations and wall-time expiry are operational stops with retained records.

Only the terminal checkpoint of the fixed pass may be proposed for the unchanged validation10, dynamic6, force-window and development-admission suite. Compare both arms; retain all regressions. Existing gates determine admissibility and real-CFD validation determines physical control benefit. A/B training completion alone does not authorize PPO.

Owners: Surrogate agent implements; Evaluation agent independently reviews; Compute agent executes only after explicit GPU approval; Lead chooses the next stage from evidence. Record exact code/config/data identities and isolated artifacts before launch.
