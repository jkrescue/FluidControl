# Train-only signed-action CFD acquisition

The completed best FNO failed to rank `omega=+1` versus `-1` correctly on a
predeclared `t=80` validation panel. This motivates a small, **train-only**
real-CFD acquisition near signed unit rotation. It is a targeted data-design
experiment, not yet a measured active-learning improvement. Do not train on
the four validation action traces or the frozen test cases.

Before solving, two action histories are fixed from the same uncontrolled
`t=82` phase (already assigned to the training split). Their signed pulse
tables are exact mirrors. Four 20-time-unit blocks alternate amplitudes
`0.75, 1.25, 0.75, 1.25`; each block ramps over two units, holds for eight,
ramps back to zero over two, then rests for eight. Thus `|omega|≤1.25`,
`|domega/dt|≤0.625`. This is close to but **not identical with** the
validation schedules. Each run spans `t=82..162`, with 16,000 OpenFOAM
v2512 `pimpleFoam` steps at `Δt=0.005` and 801 `U/p` snapshots. Geometry,
mesh and force objects match the existing 19,290-cell Re=100, L/D=5 case.

```bash
python3 -m unittest tests.test_signed_pulse_training_pair
python3 cfd/tandem_cylinders/make_signed_pulse_training_pair.py \
  --audit artifacts/tandem_cylinders/signed_pulse_train_action_audit_20261003.json
bash cfd/tandem_cylinders/run_signed_pulse_training_case.sh train_signed_pulse_p_v4_20261003
bash cfd/tandem_cylinders/run_signed_pulse_training_case.sh train_signed_pulse_m_v4_20261003
```

After both cases pass solver and force-series QC, export with the existing
OpenFOAM→VTK→official PhysicsNeMo Curator pipeline into a **new versioned
training-only profile**; never overwrite the v3 dataset or its normalization.
The control-ranking validation panel and five-case frozen test remain
unchanged. Compare any v4 retraining against the same v3 baseline at equal
optimization budget. A claim of active-learning sample efficiency requires
an equal-cost random-acquisition comparator, which this pair alone does not
provide. No surrogate or physical-control benefit is assumed in advance.
