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
