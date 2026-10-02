# Canonical research objective

Status: active
Objective version: 1.0
Locked on: 2026-10-02

This file is the project-level guard against objective drift. Experiments may
change the method, model horizon, data volume or controller, but must not silently
change the physical question.

## Physical question

Can bounded rotation of the downstream cylinder in a fixed tandem-cylinder flow
reduce the **mean total drag of the two-cylinder system** in real CFD, without
materially worsening lift fluctuations?

The current geometry contains fixed cylinders. Vortex-induced-vibration reduction
is therefore outside the present claim. It may become a separate later study only
after adding physically validated structural dynamics.

## Objective and constraints

The primary quantity is

```text
Cd_total = Cd_front + Cd_rear
```

The final claim is based on a phase-matched, zero-rotation OpenFOAM reference over
multiple shedding periods. The initial real-CFD acceptance criteria are:

- at least 2% reduction in mean `Cd_total`;
- no more than 5% increase in rear-cylinder `Cl RMS`;
- absolute mean rear `Cl` no greater than 10% of the uncontrolled rear `Cl RMS`;
- bounded actuator magnitude and rate, with action energy reported separately;
- no hidden solver failure, early termination or discarded unfavorable phase.

Front and rear forces must always be retained separately in the data and reports.
A rear-only drag improvement is not sufficient if the front-cylinder drag offsets
it. Lift is a safety/load constraint, not an optional visualization.

## Method roles

- OpenFOAM supplies the physical CFD evidence and phase-matched final replay.
- Official PhysicsNeMo APIs supply the action-conditioned flow/force surrogate.
- HydroGym supplies the auditable Gymnasium-compatible environment contract.
- CEM-MPC is the first controller baseline because it tests the world model before
  introducing policy-optimization confounders.
- PPO is allowed only after the surrogate and CEM-MPC gates pass.

PhysicsNeMo prediction accuracy is not itself evidence of drag reduction.
HydroGym execution is not itself evidence of CFD fidelity. Surrogate-only reward
improvement is always labelled as a screening result.

## Stage order

1. **Gate A — real data and one-step model:** four force channels, independent
   train/validation/test action histories, persistence comparison.
2. **Gate B — control-horizon model:** stable 1/10/50/100-step rollouts, genuine
   action sensitivity, independent shedding phases, and no more than 10% total-drag
   NRMSE at the reported full-period proxy horizon.
3. **Gate C — CEM-MPC/HydroGym screen:** normalized one-period reward window,
   explicit lift/action/rate ledger, OOD guard and short rollouts.
4. **Gate D — frozen real-CFD replay:** phase-matched OpenFOAM comparison for at
   least 5–10 shedding periods using the acceptance criteria above.
5. **Gate E — policy-distribution refresh:** add real CFD near the selected policy,
   retrain, repeat frozen replay, then study robustness.

Failure at a gate changes the next method, not the research claim. In particular,
Gate-B long-horizon drift routes to multi-step training and shorter CFD-anchored
rollouts; it does not permit lowering the accuracy threshold after seeing results.

## Evidence and change control

Every milestone records:

- Git commit and container image/digest;
- data profile, split provenance and checksums/manifest;
- random seed and resolved configuration;
- pass/fail thresholds fixed before evaluation;
- failed runs and the reason they were excluded;
- canonical artifact path on the primary Spark node.

Changing the physical objective requires a new version of this document with CFD
or literature evidence explaining why the old objective is invalid. Tuning a model
or controller does not justify changing the objective or evaluation denominator.
