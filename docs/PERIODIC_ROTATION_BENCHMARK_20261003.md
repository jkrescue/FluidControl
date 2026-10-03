# Zero-mean periodic rotation: fixed open-loop control benchmark

The matched constant `omega=±1` OpenFOAM cases lower system-total Cd by
3.49–4.66% but violate the locked mean-rear-lift limit by almost an order of
magnitude. A phase-varying controller should be compared against simple
zero-mean actuation, not only against no actuation. This validation-only
benchmark therefore fixes two smooth alternating rear-rotation schedules
**before** solving:

```text
omega(t) = tanh(3 sin(2π(t−80)/P)) / tanh(3)
P = 10 or 20 time units; |omega|≤1; |domega/dt|≤2
```

Both begin at zero action at the same uncontrolled `t=80` restart. The
OpenFOAM v2512 `pimpleFoam` configuration, Re=100, L/D=5, 19,290-cell mesh,
`Δt=0.005`, force outputs and field interval `0.1` match the completed
paired panel. Cases run through `t=160` with a predeclared `t=120..160`
analysis window containing four complete `P=10` cycles or two complete
`P=20` cycles. Source-field hashes and action tables are saved before CFD.
Neither case is added to model training or the frozen test set.

The three unchanged research-objective checks, relative to the paired zero
case, are: at least 2% lower mean total Cd; no more than 5% higher **rear
fluctuating** Cl RMS; and `|mean Cl_rear| ≤ 0.1 × zero Cl'_rms`. Report
front/rear drag separately, total lift RMS, action/rate proxies, numerical
health and six period-ish force blocks. Passing this one-phase coarse-grid
screen would not establish robust control; failing it helps define the
nontrivial task for feedback.

```bash
python3 -m unittest tests.test_periodic_rotation_benchmark
python3 cfd/tandem_cylinders/make_periodic_rotation_benchmark.py \
  --audit artifacts/tandem_cylinders/periodic_rotation_action_audit_20261003.json
bash cfd/tandem_cylinders/run_control_landscape_case.sh periodic_val_p10_20261003
bash cfd/tandem_cylinders/run_control_landscape_case.sh periodic_val_p20_20261003
```

## Completed CFD result (2026-10-03 UTC)

Both cases ended cleanly and passed the fixed 801-field, 16,000-force-sample,
identical-restart, Courant and continuity checks. The exact audited result is
`artifacts/tandem_cylinders/periodic_rotation_benchmark_result_20261003.json`
(SHA-256 `46b573daf70399c514c1422579c255d6ba32b8bdcee2936ddfe91e8de39a0d7f`).

| Case | Mean total Cd | Change vs zero | Rear fluctuating Cl RMS / zero | `|mean rear Cl| / zero Cl'_rms` | Fixed one-phase screen |
| :-- | --: | --: | --: | --: | :-- |
| Zero | 2.29931 | — | 1.000 | 0.0679 | reference |
| Period 10 | 2.31279 | +0.586% | 1.103 | 0.0703 | fail drag and fluctuation |
| Period 20 | 2.27819 | -0.919% | 1.138 | 0.0647 | fail drag and fluctuation |

Both periodic controls satisfy the locked mean-lift bound (`≤0.1`), unlike
the constant signed controls, but neither reaches the 2% total-drag reduction
or the `≤1.05` rear fluctuating-lift ratio. The period-10 force blocks vary
substantially because the generic six-block diagnostic is not aligned to its
ten-unit actuation period; the full analysis window contains four complete
periods. These two predeclared choices do not exhaust useful open-loop
schedules, so failure here supports a nontrivial phase-feedback hypothesis,
not a proof that closed loop is necessary or that it will succeed. No physical
actuator torque or power was measured; `omega` and slew are only kinematic
proxies. Final comparisons still require independent phases and grid checks.
