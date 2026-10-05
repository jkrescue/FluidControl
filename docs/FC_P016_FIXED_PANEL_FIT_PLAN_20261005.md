# FC-P016: fixed-panel local fitting probe

Preregistered 2026-10-05 after FC-P015 formal rejection. This is a project-owned
optimization probe using the existing official PhysicsNeMo FNO, not a new
architecture or deployment candidate. Preparation is approved; GPU execution
requires separate approval after implementation and independent review.

## Evidence and hypothesis

FC-P015 completed the unchanged formal protocol and failed: joint1/6,
Cd5/6, rear-Cl-prime RMS2/6, mean-Cl2/6. Its result receipt SHA is
`353004afa2aa5a35b912205751a100bc6d37d0572ce5431108dca3ed2ea95a5b`.
Fixed training panels show common signed mean-error shifts whose direction
changes during the pass. This motivates a fixed-distribution optimization
check, but does not prove catastrophic forgetting, gradient noise or inadequate
input representation. Earlier gradient and linear-readout diagnostics remain
valid; a further bias-only calibration cannot resolve fluctuation error.

Hypothesis: under a fixed training panel, the current nonlinear force FNO and
unchanged objective can simultaneously reduce objective, mean bias and waveform
error. This tests local fitting only, not generalization or controller quality.

## Fixed protocol

- Start from P009 parent model SHA
  `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31` and state SHA
  `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`.
- Keep original flow FNO frozen. The same 28 aerodynamic parameter tensors are
  trainable; the two official non-gradient lifting biases remain frozen.
- Reuse immutable P013 numerical source and P014 mixed-batch capture helpers.
  Original six windows, in this fixed order: global indices160,816,923,975,1077,1233.
  Index160 is the zero-action-history window; the other five form the nonzero group.
- H100, ten time points per chunk, actual mixed H1/AR batch20. Original four-force
  normalized objective, H1/AR each weighted one half. No loss-weight change.
- For each update, average the six complete raw window gradients, then clip norm1
  once and make one fresh AdamW update (lr1e-5, weight_decay1e-4).
- Exactly32 updates,192 window exposures. No checkpoint selection, extra updates,
  learning-rate sweep, architecture change or continuation from P015.
- Precision remains default TF32/high. Seed20261003. CPU caching of detached
  frozen-flow and true-state inputs is permitted without changing tensor values.
- Collect fixed no-gradient panels at updates0,8,16,32, restoring model mode,
  tensors, gradients and RNG state. Interpret terminal32 only; intermediate panels
  explain optimization trajectory and cannot select a model.

## Measurements and decision

Report per-window H1 and autoregressive normalized objective, four physical force
errors, rear-Cl signed residual mean, bias squared, centered residual MSE and the
absolute difference between predicted and true centered Cl RMS. Keep full100 and
tail62 definitions explicit. Centered residual error is NOT Cl-prime RMS error.

For each domain separately, objective is the equal-weight mean over all six
windows. For each domain separately, bias squared, centered residual MSE and
absolute Cl-prime RMS error are equal-weight means over the five nonzero windows
using the tail62 samples. Do not pool domains or hide per-window regressions.

The hypothesis receives local support only if terminal32 versus initial0 lowers
both domain objectives AND all three five-window statistics in each domain.
Report absolute and relative changes, including increases; strict lower values
are an interpretive rule, not a new formal admission threshold. A tiny difference
is not evidence of a practically useful gain.

- Only bias improves or domains trade off: this fixed optimization protocol does
  not support simultaneous repair; no automatic full training.
- Objectives fail to decrease: inspect same-batch loss/gradient/update mechanics
  before further single-pass training. Do not infer network capacity failure.
- All comparisons improve: local fitting is feasible; propose a separately
  reviewed controlled full-data training experiment. Memorization remains possible.

No outcome proves wake-only input sufficiency or identifies a unique cause.
No model save, candidate, validation/frozen access, formal evaluation or PPO.
The original full admission and final real-CFD acceptance remain unchanged.

## Resource, implementation and provenance

Main Spark GPU0 only, isolated pinned official image
`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
Proposed allocator cap0.15; both physical MemAvailable and MemFree >=20GiB;
startup physical free>=30GiB. No concurrent GPU task or bulk data transfer.
Inner30-minute limit with bounded external cleanup. Any NaN/Inf, incorrect
gradient scope, changed frozen flow or memory violation stops the probe as an
operational failure, not a scientific verdict.

Bind source/config/parent/normalization/train-HDF hashes and window identities;
save measured panels, update history and resource records, but no checkpoint.
Root owns plan/launch/integration; implementation and independent evaluation are
separate agents. Source and execution SHAs will be recorded after code review.
