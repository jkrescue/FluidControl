# Real OpenFOAM phase-feedback pilots — 2026-10-03

## Scope and fixed physical case

These are short, paired OpenFOAM v2512 `pimpleFoam` diagnostics for the existing
two-dimensional tandem-cylinder case at `Re=100`, `L/D=5`.  Both cylinders are
fixed in space and only the rear cylinder rotates.  They are not PPO runs, do
not use the PhysicsNeMo surrogate, and are not final evidence of closed-loop
drag reduction.

Every feedback case is paired with a newly integrated segmented `omega=0` case.
Both members copy the exact `t=80` `U` and `p` restart, whose SHA-256 hashes must
match before integration.  Both use `dt=0.005`, 20 solver steps per control
interval, the same `0.1 D/U` segmentation, mesh and solver dictionaries.

The two-period simulation window is fixed at `t=80..92.4`.  The comparison
window was declared before solution as the final one-period interval
`t=86.2..92.4`; it is not selected after viewing results.  The original pilot
check was declared as:

- lower mean `Cd_front + Cd_rear` than the paired zero case;
- rear-cylinder total lift RMS no more than 5% above zero;
- absolute rear mean lift no more than 10% of zero-case lift-fluctuation RMS.

This pilot check was weaker than, and partly inconsistent with, the project's
canonical physical Gate-D: total drag reduction must be at least 2%, rear
**fluctuation** RMS `Cl'` must be no more than `1.05` times zero, and absolute
mean rear `Cl` must be no more than `0.1` times zero-case `Cl'` RMS.  The
original `predeclared_checks` are retained in the JSON as historical fact; a
separate `canonical_physical_checks` re-audit is added without pretending that
the corrected thresholds were registered before these pilots.  All three
controllers fail the canonical joint gate.

All force statistics use every `dt=0.005` OpenFOAM force sample in that window.
`omega` and `delta omega` are kinematic effort proxies.  Rear `CmPitch` is
retained separately; `-omega*CmPitch/Cd_total_zero` is reported only as a
fluid-torque work proxy, not electrical actuator power.

## Predeclared controllers

1. `k=0.75`, zero lag:
   `omega_target = clip(0.75 Cl_rear(t), -1, 1)`.
2. `k=0.20`, zero lag:
   `omega_target = clip(0.20 Cl_rear(t), -1, 1)`.
3. Exploratory phase-lag controller, registered after both proportional laws
   failed: `omega_target = clip(0.50 Cl_rear(t-1.5), -1, 1)`.  It holds
   `omega=0` until the causal 1.5-time-unit history exists.

Every controller also enforces `|delta omega| <= 0.1` per interval and
`|omega| <= 1`.  The lag is an engineering hypothesis of approximately one
quarter of the measured shedding period; it is **not** claimed to reproduce an
exact numerical parameter from Zhao et al. (2024).  The paper's reported
action/lift phase relationship motivates testing phase as a mechanism, but it
does not validate this particular law.

## Locked interpretation

The first two controllers are retained as negative results.  If the third also
fails the joint check, no further short-window gain/lag tuning is justified in
this campaign.  The next step must instead be mechanism diagnosis followed by
a longer and multi-phase validation design.

## Results recorded without changing the window

The zero-lag laws both failed the drag requirement:

| Controller | Feedback total Cd | Paired-zero total Cd | Drag reduction | Rear `Cl'` RMS ratio | Rear total RMS ratio | Rear mean Cl | Canonical Gate-D |
|---|---:|---:|---:|---:|---:|---:|---|
| `k=0.75` | 2.70659 | 2.30392 | -17.48% | 1.0192 | 1.0192 | -0.00732 | fail |
| `k=0.20` | 2.42008 | 2.30392 | -5.04% | 1.0086 | 1.0086 | -0.00360 | fail |
| `k=0.50`, lag 1.5 | 2.28949 | 2.30392 | +0.63% | 1.3782 | 1.3782 | +0.00115 | fail |

The negative reduction means increased drag.  Both laws kept mean lift close
to zero and remained inside the 5% rear-lift-RMS guard, but this does not
compensate for failure of the primary total-drag objective.  Lowering the gain
reduced the harm without reversing it, so further same-phase gain tuning was
stopped.

The phase-lag hypothesis changed the mechanism: rear mean drag fell from
`0.91453` to `0.90017`, while front mean drag was essentially unchanged.  But
rear total lift RMS rose from `1.17094` to `1.61381` (+37.8%).  It therefore
passed the drag-sign and mean-lift checks but decisively failed the lift-RMS
guard.  Per the pre-registration, no further short-window gain/lag search was
performed and this case was not extended to a longer horizon.

Structured results and raw audit material are under:

- `artifacts/tandem_cylinders/phase_feedback_pair_k075_20261003/`
- `artifacts/tandem_cylinders/phase_feedback_pair_k020_20261003/`
- `artifacts/tandem_cylinders/phase_feedback_pair_k050_l15_20261003/`

Each directory contains `result.json`, the 124-step `progress.json`,
`torque_audit.json`, and the retained `rear_cylinder_torque_timeseries.csv`.

## Read-only mechanism diagnosis

A single-frequency projection at the baseline shedding frequency
`f0=1/6.2` was applied to the fixed final-period window.  This was diagnostic,
not another selection criterion:

| Controller | omega-minus-Cl phase | corr(omega, rear Cl) | delta front Cd | delta rear Cd |
|---|---:|---:|---:|---:|
| `k=0.75` | -14.5 deg | 0.9666 | -0.00291 | +0.40558 |
| `k=0.20` | -5.8 deg | 0.9949 | -0.00059 | +0.11675 |
| `k=0.50`, lag 1.5 | -90.0 deg | 0.0012 | -0.00007 | -0.01436 |

The two same-phase controllers changed front drag negligibly and increased rear
drag strongly.  The causal lag produced the intended approximately 90-degree
phase relationship and slightly reduced rear drag, but amplified `Cl'` RMS by
37.8%.  This identifies a real drag/lateral-load trade-off rather than a valid
controller.

OpenFOAM reports `CmPitch` as the fluid moment coefficient about the rear
cylinder's z axis.  With this case's `D=U_inf=lRef=1`, the sign-sensitive
actuator work proxy normalized by matched zero drag power is
`-omega*CmPitch/Cd_total_zero`.  Its mean was negative for all three cases,
meaning the OpenFOAM sign convention indicates net fluid-to-prescribed-motion
work over these windows.  The positive-only values, assuming no recovery in
negative intervals, were 0.1407%, 0.0120%, and 0.1908% of zero drag power for
`k=0.75`, `k=0.20`, and the lagged case.  These are fluid-torque kinematic
proxies—not measured motor electrical power—and the raw `CmPitch` series is
retained so the sign convention remains auditable.

Paired numerical QC passed in every case: initial `U` and `p` hashes matched,
the independently segmented zero runs reproduced the same final-window force
statistics, and the maximum Courant number across all pairs was below `0.246`.
