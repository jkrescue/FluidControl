# Matched-start OpenFOAM control-landscape panel

This is an **exploratory validation-only** physical feasibility and
surrogate-ranking panel, not training data, frozen test data, or a learned
controller. Four cases share the byte-identical validated uncontrolled
OpenFOAM state at `t=80`. They differ only in predeclared rear-cylinder
rotation; no FNO prediction or frozen test outcome was used to select them.

Scene: two fixed-centre equal-diameter tandem circular cylinders, Re=100,
L/D=5, `D=U∞=1`, `ν=0.01`; OpenFOAM v2512 `pimpleFoam` in the digest-pinned
read-only container. The existing coarse 19,290-cell mesh and backward
time scheme use `Δt=0.005`; force coefficients are written each solver step,
and `(u,v,p)` fields every `0.1` time units. Each case runs `t=80..116`
(36 time units, approximately 5.85 uncontrolled shedding periods).

| Case suffix | Rotation schedule, after a two-unit ramp |
| --- | --- |
| `zero` | `ω=0`, the paired reference |
| `p100` | `ω=+1` |
| `m100` | `ω=−1` |
| `sine` | `ω=sin(2π(t−80)/6.154)` |

All schedules begin at `ω=0` and are linearly tabulated every `0.1` units;
`|ω|≤1` and `|dω/dt|≤1.6`, inside the existing training action support.
The action-audit JSON and each `case_config.json` freeze the full table,
source `U/p` hashes, solver setup and analysis window before solving.

The prespecified force-analysis window is `t=92..116` (24 units, about 3.9
shedding periods), so the initial ramp and early transient are excluded.
Report mean front/rear/total Cd, front/rear Cl RMS, action RMS and rate,
solver health, and block variability. The short window and coarse mesh mean
this panel can screen feasibility and control ranking, **not** establish
final closed-loop benefit. A final policy would need a longer matched CFD
comparison and mesh/time-step sensitivity.

Generation and execution on the **primary DGX Spark** only:

```bash
python3 -m unittest tests.test_control_landscape_panel
python3 cfd/tandem_cylinders/make_control_landscape_panel.py --audit \
  artifacts/tandem_cylinders/control_landscape_panel_action_audit_20261003.json
bash cfd/tandem_cylinders/run_control_landscape_case.sh landscape_val_zero_20261003
```

The other three names are `landscape_val_p100_20261003`,
`landscape_val_m100_20261003`, and `landscape_val_sine_20261003`. The runner
refuses to overwrite solver logs or existing output times. Every case must
reach the OpenFOAM `End` marker and pass finite-force, continuity/Courant,
frame-count and source-provenance checks before comparison. Worker node is
compute-only for FNO; its CFD raw output is not used for durable storage.

This panel tests the central objective concern recorded in
`RESEARCH_VALUE_AUDIT_20261003.md`: total drag may decline while rear lift
or actuation costs rise. Comparisons must therefore report the trade-off
instead of a single scalar reward. For model ranking, score all four action
returns with the selected FNO from the *same source state* and compare their
ordering with these CFD outcomes, without using the frozen test split.
