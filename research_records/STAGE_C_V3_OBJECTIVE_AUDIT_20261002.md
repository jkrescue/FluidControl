# Stage-C physical-objective audit on the v3 real-CFD cohort

At 2026-10-02 15:59 UTC, the existing force/action auditor read the 35
validated v3 OpenFOAM HDF5 trajectories (26 train, 4 validation, 5 test),
without loading flow fields or using an FNO prediction. The output is
`artifacts/tandem_cylinders/control_objective_v3_audit_20261002.json`
(SHA-256 `4a293456f11c379570c260c503b92b46104640838f4531e1543f3d0995a81b02`).
The phase-matched zero-action baselines are
`artifacts/tandem_cylinders/stage_c_phase_baselines_v3_20261002.json`
(SHA-256 `b7a259980e810eed56ec25b5722196e40df30ab67523964944b69586b9a90083`).
Both artifacts remain on the primary Spark.

The resulting baseline manifest has exactly 35 case keys. For the new,
untouched `expanded_test_05`, the `t=80..160` zero-action source has
time-averaged total Cd `2.299928`, front Cl RMS `0.312235`, and rear Cl RMS
`1.179124`. Its predeclared random high-rotation CFD trajectory has mean
total Cd `2.257377` (1.85% lower than zero action), but rear Cl RMS is
**3.4773 times** the zero-action value. This is an observed open-loop
trade-off, **not** a learned controller result or evidence that the current
surrogate passes Gate B. It supports keeping rear-lift and actuation
safeguards while targeting system-total-drag reduction.

The primary scientific outcome will be a frozen policy's *paired* OpenFOAM
comparison, with separate force and effort components. The earlier 32-step
PPO integration run used a rear-cylinder objective and worsened its matched
physical objective; it is not transferable evidence for this Stage-C target.
The Stage-C HydroGym adapter now rejects an episode shorter than 1.5
shedding periods: at `dt=0.1`, the old 32-step duration is 3.2 versus the
6.15-unit causal reward window, so it would provide no physical drag reward.
The 69-channel Stage-C observation now reads both cylinder forces at the
exact source `t=80` when a shared-restart frame is used. A CPU-only read of
the original OpenFOAM files returned front Cd `1.39119339` and rear Cd
`1.06314135`; all four force channels matched the newly curated
`expanded_train_24` frame-0 label exactly (maximum absolute difference `0`).
The old 67-channel rear-only reader remains unchanged. This fixes online
observation provenance without rewriting inherited training HDF5.

The read-only 69-channel parity audit now checks original OpenFOAM against
Curator HDF5 at validation case 00 frames 100/400/800 and fresh test case 05
frames 0/100/400/800. All seven comparisons passed the pre-existing
probe `0.02`, force `1e-4`, and action `1e-5` tolerances. The largest
probe discrepancy was `0.0042061`; front/rear force and action discrepancies
were exactly zero in these rows. The reports are
`artifacts/tandem_cylinders/observation_69d_validation00_v3.json` and
`artifacts/tandem_cylinders/observation_69d_test05_v3.json` on the primary.
This validates channel order and data provenance, not closed-loop benefit.

Reproduce the CPU-only audit in the pinned PhysicsNeMo 2.2.2 container with
`scripts/audit_tandem_control_objective.py --data
data/curated/tandem_cylinders_gate_b_aug_v3` and then
`scripts/build_stage_c_phase_baselines.py` on its output. Both scripts
refuse to overwrite existing artifacts and verify case counts, force schema,
finite values and phase-matched source alignment.
