# Existing CFD cohort versus the locked physical objective

Date: 2026-10-03 UTC. This is a retrospective **exploratory** check of the
35-case v3 real OpenFOAM cohort, restricted to its 26 training and four
validation trajectories. The five frozen test trajectories were not read.
`cfd/tandem_cylinders/audit_existing_open_loop_acceptance.py` checks each
source restart, split, 801 `U/p` snapshots, 16,000 force rows, solver ending
and Courant history, then measures the identical `t=120..160` window against
the paired zero-rotation OpenFOAM reference. It uses the clarified
**fluctuating** rear Cl RMS plus the separate mean-lift bound from
`RESEARCH_OBJECTIVE.md`; no thresholds were fitted to these trajectories.

| Locked check | Passing existing trajectories |
| :-- | --: |
| System-total mean Cd reduction ≥2% | 4/30 |
| Rear fluctuating Cl RMS ≤1.05 × zero | 0/30 |
| `|mean rear Cl| ≤0.10 × zero Cl'_rms` | 9/30 |
| All three simultaneously | **0/30** |

The four drag-only successes are `expanded_train_04` (9.79%), `_06` (6.79%),
`_07` (5.18%) and `_24` (4.77%). Their respective rear fluctuating Cl RMS
ratios are `1.92`, `3.27`, `2.62` and `3.25`, so none is a valid joint-control
result. The other 26 trajectories also fail the fluctuation criterion. This
does **not** prove no open-loop schedule could pass, nor that closed loop will;
the cohort is a finite set of mostly random/ramped actions at one phase and
one coarse mesh. It does show that the locked target is not already solved by
these existing controls and that optimizing drag alone would be misleading.

Canonical artifact:
`artifacts/tandem_cylinders/existing_open_loop_acceptance_audit_20261003.json`
(SHA-256 `f240bf848ac80cbbc740a6b1ba81f1bfb5517808d7bcf8de35daa4978676c809`).
The earlier `control_objective_v3_audit_20261002.json` used a different
aggregation and mean-absolute-lift diagnostic; its counts should not be
substituted for this exact locked three-criterion check. A final publishable
claim still requires a frozen closed-loop policy, paired independent phases,
medium-grid/time-step checks, actuator effort/work and CFD solver-hour budget.
