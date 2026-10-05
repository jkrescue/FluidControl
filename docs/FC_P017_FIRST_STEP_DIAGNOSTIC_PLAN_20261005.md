# FC-P017 — first AdamW step directional diagnostic

Preregistered after P016; preparation and CPU testing only are approved.
GPU execution needs separate source/resource review. This is project-owned
numerical diagnosis around the unchanged official PhysicsNeMo FNO.

## Evidence and question

P016 result SHA `f760d2e79a9fc797501f6481f74808cb0ffe790149fdb67340ac9abeb3454248`
has operationally verified 32 updates and no simultaneous repair. Same-six-window
total objective rose from 0.006246136754 to 0.127936633925 after the first step
(20.48 times); panel0/8/16 and pre-update records1/9/17 agree exactly. No weighting
or sign defect has yet been identified. Test whether the first actual AdamW
direction is descending locally but its full finite displacement overshoots.

## Fixed protocol

Use exactly P016 initial P009 model/state, config, immutable P013 objective,
P014 helper, six training windows in original order, H100 mixed20 chunk10,
TF32/high, seed20261003, frozen flow and frozen two lifting biases. Pin P016
source SHA `18b210077a93bbd21a327ae6578a73fb371bf0d2f48d0541f6ca8bd87aa42bbb`.
No heldout mounts, architecture/loss/normalization changes or checkpoint saves.

1. Cache the same detached inputs and evaluate the initial six-window H1/AR/total
   objective twice with identical mode and restored RNG; retain both values.
2. Compute once the original raw six-window mean gradient g. Record its norm and
   clip1 factor. One fresh AdamW step: lr1e-5, betas(.9,.999), eps1e-8, wd1e-4.
3. Record actual displacement delta = theta_after - theta_before and g dot delta
   with float64 reductions, global and per named parameter. Independently compute
   first-step formula including epsilon and decoupled decay; report discrepancy,
   update norm, zero/disappearing FP32 displacements, and decay contribution.
   Formula comparison is observational unless a justified CPU-tested rounding
   bound is approved before execution; do not invent a PASS tolerance afterward.
4. From the same initial weights independently evaluate theta+delta and
   theta+delta/64 and theta-delta/64. Every trial is restored from the initial
   tensor copy, not sequentially stepped. These three fixed points are not an
   optimizer sweep, candidate selection or learning-rate recommendation.
5. Report central directional difference [L(plus)-L(minus)]/(2/64), g dot delta,
   and actual rounded displacement norms. Repeat initial evaluation after exact
   tensor restoration and require the initial tensor SHA to be restored exactly.

Report per-window/domain values, total means, finite checks, source/input hashes,
frozen tensor hashes and precision. Preserve objective evaluation RNG/mode.
Report forward repeat spread and numerical resolution alongside differences;
when sign is not resolved beyond observed repeat variation, mark interpretation
inconclusive. Repeat spread is not a rigorous bound on TF32/rounding error.

## Interpretation and next action

- Negative g dot delta, resolved small-positive-step decrease and full-step
  increase support first-step finite-step overshoot, not convergence or global
  validity. A subsequent controlled training intervention needs a new plan.
- Formula disagreement or ascent direction requires implementation analysis.
- Unresolved small-step or directional discrepancy requires numerical/gradient
  consistency analysis; do not conclude capacity/ROI failure or add epochs.

No model/policy saved, no validation/frozen/PPO/formal evaluation. Original
surrogate admission and paired real-CFD drag/lift criteria remain unchanged.

## Resources and ownership

Main Spark GPU0, pinned P016 official image, allocator0.15, CPU cached tensors,
same train-only mounts, both MemAvailable and MemFree >=20GiB, startup free>=30GiB.
Bound execution to 15 minutes with external watchdog and exact owned-container
cleanup; no concurrent GPU task. Preserve failure artifacts and no blind retry.
Lead owns plan/launcher; implementation agent owns new probe/tests; independent
evaluation reviews formulas, provenance and results before interpreting them.
