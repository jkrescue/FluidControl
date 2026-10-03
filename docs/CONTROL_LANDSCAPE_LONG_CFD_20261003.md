# Long matched-start rotation check

The exploratory `t=80..116` paired OpenFOAM panel found 3.62–4.08% lower
system-total mean Cd under `ω=±1`, but the short window's *fluctuating*
rear Cl RMS disagrees with the earlier long constant-rotation panel. Its
nonzero mean lateral force also raises the rear Cl **total** RMS by 26–30%.
The short panel therefore cannot settle a drag-versus-lift claim.

Before running new solver cases, this long panel fixes **three** schedules
from the same uncontrolled `t=80` restart: `ω=0`, and linear two-time-unit
ramps to `ω=+1` and `ω=−1`, then hold. All run to `t=160`; the predeclared
analysis window is `t=120..160` (40 time units, about 6.5 uncontrolled
shedding periods). Geometry, pinned OpenFOAM v2512 solver, 19,290-cell
coarse mesh, `Δt=0.005`, forces and field-output settings match the short
panel. Exact action tables, source `U/p` hashes and expected 801 snapshots
are recorded before solving. These are validation-only open-loop responses;
they are not appended to FNO training or the frozen five-case test.

```bash
python3 -m unittest tests.test_control_landscape_long_panel
python3 cfd/tandem_cylinders/make_control_landscape_long_panel.py --audit \
  artifacts/tandem_cylinders/control_landscape_long_action_audit_20261003.json
bash cfd/tandem_cylinders/run_control_landscape_case.sh landscape_long_val_zero_20261003
```

The other names are `landscape_long_val_p100_20261003` and
`landscape_long_val_m100_20261003`. The primary DGX Spark stores all raw
results. Report paired mean total/front/rear Cd, rear Cl fluctuating RMS,
rear Cl total RMS and absolute mean, mean Cl offset, action/rate proxies,
solver health and per-shedding-period variability. Do **not** call the
kinematic proxies physical actuator energy: wall torque is not measured.

Any apparent drag reduction must also be compared with the existing
constant-rotation coarse/medium-grid checks and the zero baseline. This
panel establishes feasibility and transient sensitivity, **not** a learned
feedback controller or broad Reynolds/spacing generalization.
