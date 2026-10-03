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

After all three solver runs finish, a fail-closed analysis validates 801
snapshots and 16,000 force samples per case, the source `U/p` hashes,
solver ending, Courant and continuity, then reports the predeclared window
and six non-overlapping force blocks:

```bash
python3 -m unittest tests.test_control_landscape_long_analysis
python3 cfd/tandem_cylinders/analyze_control_landscape_long_panel.py \
  --output artifacts/tandem_cylinders/control_landscape_long_result_20261003.json
```

## Completed CFD result (2026-10-03 UTC)

All three OpenFOAM runs ended cleanly. The analyzer checked 801 field snapshots
and 16,000 force rows per case, identical restart `U/p` hashes, Courant below
0.261, and maximum per-step global continuity below `1.6e-12`. The result is
`artifacts/tandem_cylinders/control_landscape_long_result_20261003.json`
(SHA-256 `db09d044c5dab57cf581977ee73c15b3b64c69dea3e7ec94a6fa231fa30ea8b0`).

| Rear `omega` | Total mean Cd | Change vs paired zero | Front mean Cd | Rear mean Cd | Rear mean Cl | Rear fluctuating Cl RMS | Rear total Cl RMS ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2.299313 | — | 1.39126 | 0.90805 | -0.08005 | 1.17904 | 1.000 |
| +1 | 2.192260 | -4.656% | 1.39106 | 0.80120 | -1.06594 | 1.14280 | 1.322 |
| -1 | 2.218997 | -3.493% | 1.39049 | 0.82850 | +0.91422 | 1.14436 | 1.239 |

The total-drag reduction comes almost entirely from the rear cylinder; the
front-cylinder mean Cd is nearly unchanged. It appears in all six paired
non-overlapping drag blocks for each controlled case: `+1` ranges from
`-3.75%` to `-5.23%`, while `-1` ranges from `-4.32%` to `-3.03%`. These
blocks are correlated time segments, **not** six independent CFD replications
or a confidence interval. Their trends mean long-run stationarity is not yet
established.

The fluctuating rear Cl RMS is about 3% below the paired baseline in this
particular ramp-start, late-window panel. But mean lateral lift is large, so
the rear total RMS including its mean rises 24–32%, and mean absolute Cl rises
10–21%. This is a drag/side-load trade-off, not an unqualified improvement.
The earlier constant-rotation `t=80..160` pilot's ~1.517 statistic was
**total** Cl RMS including mean, not fluctuating RMS. A matched instant-vs-ramp
replication found fluctuating RMS ratios within `0.0021%` of unity. The
retained old `control_small_p100` force series independently gives
fluctuating RMS `1.137038` and total RMS `1.517437`; see
`CONTROL_ONSET_REPLICATION_20261003.md`. The seeming lift discrepancy was
primarily a metric-label inconsistency, not a demonstrated onset mechanism.
These cases have only a coarse-grid check at other conditions, no measured
wall torque, and no policy feedback. They justify studying a Pareto-constrained
controller, not claiming one has been achieved.

The pre-existing canonical acceptance rule in `RESEARCH_OBJECTIVE.md` also
requires absolute mean rear Cl to stay at or below **10% of the uncontrolled
rear fluctuating Cl RMS**. Here that fixed limit is `0.117904`, while the
measured magnitudes are `1.06594` (`+1`) and `0.91422` (`-1`): roughly 9.0
and 7.8 times the permitted value. Both rotations pass the predeclared 2%
total-drag reduction threshold, but **both fail the mean-lift constraint**.
They are not Gate-D control successes, even though their fluctuating Cl RMS
alone is slightly below baseline in this panel.
