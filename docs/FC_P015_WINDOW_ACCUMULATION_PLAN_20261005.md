# FC-P015 — fixed eight-window accumulation protocol

Implementation/CPU tests approved 2026-10-05; GPU execution NOT approved here.
This declaration follows observed P013/P014 results and precedes P015 execution.

## Hypothesis and limits

P014 result SHA 5550140b818a3d9028073b913cf994da99d9d3e917ea24896e4a9d477b8e95a7
shows same-panel H1 and AR original objectives worse in all six windows. AR
centered residual MSE also worsens in all six H100 windows. P013 formal joint
admission is 0/6. P013 uses one H100 window per optimizer step; its mixed20 force
batch contains correlated endpoints, not twenty independently sampled windows.
All 1368 recorded gradients exceeded the fixed clip norm 1 (minimum 1.81845,
median 21.73668, maximum 98.28410). These observations motivate testing a different
update aggregation protocol; they do NOT identify noise, LR, or capacity as cause.

Hypothesis: equal aggregation of eight consecutive fixed-order windows before
clipping/updating can improve the actual fixed-panel objective and centered
waveform errors relative to the single-window protocol. This is falsifiable and
is not a claim that larger batches generally help.

## One intended protocol change

Start both official FNO instances from the exact P009 checkpoint, with a new
AdamW optimizer (not P013 continuation). Flow remains frozen/eval. Aerodynamic
scope remains the original 28 trainable tensors; the two official frozen lifting
biases stay frozen. No architecture, loss, precision, channel, data, seed, action,
geometry, or normalization changes. AdamW lr=1e-5, weight_decay=1e-4, clip=1.

Consume the same 1368 train H100 windows in seed 20261003 order SHA
177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f exactly once.
Force forwards retain the original ten mixed20 chunks per window. Accumulate raw
window gradients without clipping or stepping; divide every gradient by eight,
audit, clip once, and step once per group. This is mathematically equivalent to
backpropagating each window loss/8; CPU tests quantify float32 agreement rather
than claiming arbitrary bitwise equivalence. Exactly 171 optimizer steps.

The update count, Adam moments, clipping on aggregated directions and cumulative
weight decay all change with this protocol. Do not call it an isolated test of
gradient noise or compensate with an unapproved LR/weight-decay/epoch change.

## Identities and dependencies

P009 model/state SHA dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31 /
4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e.
Reuse the immutable P013 source tree artifacts/fcp013_training_source_1634c05_immutable;
seven numerical dependency SHAs are checked via the pinned P014 diagnostic module
SHA 849570afd814faeaa92af99b1cc26cf71182439aa5c4c42f76b9e3b90bb1c30d.
P013 objective SHA f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7;
P011 data/window SHA 9c761cfcb3d4f18dbe35aed1b3defe0614b29db867006055dae63d9fa94485a5.
Resolved numerical config remains SHA 07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9;
the separate P015 experiment identity records accumulation8/171steps explicitly.
Root's approved launcher must verify all 44 train HDF hashes once, bind the image
and source hashes, and mount no validation/frozen files during training.

## Fixed diagnostics, terminal and formal evaluation

At consumed-window counts 0,456,912,1368 (optimizer steps 0,57,114,171), evaluate
the exact original six train windows with pinned P014 capture_force_panel and
P013 frozen_flow_states/true_state_inputs. Record H1/AR balanced objective,
four normalized channel MSEs, H100/tail62 rear-Cl signed bias, bias squared,
centered residual MSE, physical residuals and analytic bias derivative.
Predictor train mode, frozen flow eval, float32/high/TF32 remain unchanged.
Diagnostics run no_grad, restore modes, do not mutate tensors or optimizer
gradients/RNG, do not select models, and are not extra training examples.

Save only the predetermined final checkpoint. Save/reload through official
PhysicsNeMo checkpoint utilities, verify both model tensor hashes and unchanged
P009 flow files. P015 experiment, metadata/status, 171 optimizer steps and 1368
windows must be explicit; no claim that it was trained by P013. Retain the dual
structural schema, but do not mislabel unsupported legacy dual-loader admission
as verified. Root owns the separately reviewed P015 runtime/formal profile.

Run the entire unchanged original formal suite on the fixed terminal under
separate approval, even if train diagnostics disappoint. No best-checkpoint
selection, early stopping, LR retry, validation-based tuning or automatic PPO.
Comparison controls: immutable P009 and P013 same-panel/P014 observations and
their original formal results; a historical control does not isolate all hardware
nondeterminism. In-run baseline and terminal use identical diagnostic code.

## Interpretation and next action

- H1/AR objectives and centered errors improve: supports this protocol on these
  train panels; formal remains authority, not a new six-window admission count.
- Objective improves only through bias: waveform hypothesis is not supported.
- Opposite H1/AR objective changes: domain tradeoff, no weight sweep.
- Both worsen: no support for this fixed-budget protocol; do not extend epochs.
- Formal fails any unchanged admission gate: candidate remains blocked from PPO.

Changing-window training loss is not a convergence curve. Intermediate fixed
panel values establish a limited trajectory, not whole-train stationarity or
population performance. Report zero/nonzero windows individually and all signs.

## Budget and failure handling

Exactly 1368 training window backwards, 171 updates and four six-window diagnostic
panels (24 frozen-flow H100 rollouts, 240 mixed20 force forwards). At most four
hours for training/diagnostics/save/reload; formal uses separate original budget.
One pinned official image/GPU, no concurrent formal/GPU job, inherited allocator
fraction and container limits; MemAvailable and MemFree each >=20 GiB throughout.
No GPU run authorized by this plan. Root supplies immutable execution approval,
launcher and external resource guard. Any identity, nonfinite, mutation, sequence,
resource or timeout failure is preserved and diagnosed, never auto-restarted.
