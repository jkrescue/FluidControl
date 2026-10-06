# Canonical research objective

Status: active
Objective version: 1.0
Locked on: 2026-10-02

> **Current-status note (2026-10-06):** The physical objective and all
> thresholds below remain unchanged. Later Lead/user decisions authorized
> explicitly exploratory PPO training and paired real-CFD feedback before the
> full surrogate/CEM-MPC gates passed. Those runs are real development evidence,
> not a retrospective PASS of the gates in this version and not a relaxation of
> the final physical criteria. The current execution state and decision scope are
> maintained in [PROJECT_STATE](../PROJECT_STATE.md) and
> [DECISIONS](../DECISIONS.md); this file retains the original staged objective.

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

Metric clarification (2026-10-03; thresholds and objective unchanged): because
the physical question says **lift fluctuations**, the 5% criterion uses
`Cl'_rms = sqrt(mean((Cl - mean(Cl))²))` on the declared replay window.
The separate mean-lift bound uses `abs(mean(Cl))`. Also report
`Cl_total_rms = sqrt(mean(Cl²))`, but do not relabel it as fluctuating RMS.
The Stage-C surrogate screening code currently penalizes **total** lift RMS
as a conservative soft proxy; this is not a substitute for checking both
fixed Gate-D lift criteria on real CFD. This clarification follows a
raw-force audit of a historical table that mixed the two RMS definitions.

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

Gate-B clarification (2026-10-03; no physical threshold changed): the 10% drag
NRMSE is a minimum screening condition, **not sufficient evidence** that a
surrogate can select a 2% drag improvement. The uncontrolled total Cd is about
2.30, so the target difference is about 0.046, smaller than the absolute error
allowed by a 10% full-scale NRMSE. Before trusting Gate-C action rankings,
evaluate matched-start CFD action pairs for the *difference* in window-mean
total drag and their ordering, including independent low-magnitude action
histories over the full control horizon. Validation windows that contain no
such histories cannot establish low-action generalization. Keep the original
validation and frozen test cases intact; add a separately identified independent
profile when action coverage is missing.

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
