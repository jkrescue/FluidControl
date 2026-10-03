# Low-action phase-94 validation-only CFD (2026-10-03)

## Scope and isolation

This profile contains two real OpenFOAM trajectories from the same uncontrolled
`t=94` restart.  They are an independent, validation-only low-action diagnostic
for fixed surrogate checkpoints.  They must not be merged into
`control_gap_v4`, used for model selection, or treated as frozen-test data.

The paired schedules have peak rear-cylinder rotation `omega=+/-0.75`, peak
rate `0.375`, run interval `[94,174]`, `delta_t=0.005`, and 801 saved frames.
Both cases share the exact initial fields:

- `U`: `1f321c10b7aa8f80d26440f938c21a390285a62a06b246e6076db5281c6427c3`
- `p`: `f20225035ffda5804a9b1fc7cf331ee265b76560f3fe4fad5a92fc72d9cafc27`

Raw cases live under `cfd/tandem_cylinders/cases/validation_signed_low_pulse_{p,m}_phase94_v1_20261003`.
The separated curated profile is
`data/curated/tandem_cylinders_low_action_phase94_validation_v1`.

The reproducible case generator is
`cfd/tandem_cylinders/generate_low_action_phase94_validation.py`. From
`cfd/tandem_cylinders`, the original workflow was:

```bash
python generate_low_action_phase94_validation.py
bash run_low_action_phase94_validation_case.sh validation_signed_low_pulse_p_phase94_v1_20261003
bash run_low_action_phase94_validation_case.sh validation_signed_low_pulse_m_phase94_v1_20261003
```

The runner delegates to `run_openfoam.sh`, which pins the digest below and
disables network access. Existing case directories deliberately cause refusal
instead of overwrite; reproduction should use a clean compute staging tree and
copy verified results back to Spark.

## Solver and numerical QC

The calculation used the pinned OpenCFD container
`opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319`
(image ID `sha256:24205c9677d39c95221eb903988094dd7a228fc41a2054df3eaa13f80e465fcb`),
the established 19,290-cell tandem-cylinder mesh, and the existing project
solver configuration.  Each case completed 16,000 steps and 16,000 front/rear
force samples.  Peak Courant numbers were 0.248259 and 0.247841; maximum
absolute global continuity error per step was below `1.28e-12`; both solver
logs ended cleanly.  Raw QC is in
`artifacts/distributed_runs/control_gap_low_action_phase94_validation_v1_worker78/low_action_phase94_validation_v1_cfd_qc.json`.

VTK-to-HDF5 curation is reproducible with the project copy of the exact script
that ran (SHA-256
`b11eee58a41ef8634c80ffb009a536fc966600c7b27b01ae890e334af8cd44d2`):

```bash
.venv-curator-py312/bin/python -u scripts/curate_low_action_phase94_validation.py \
  --profile low_action_phase94_validation_v1 \
  --output data/curated/tandem_cylinders_low_action_phase94_validation_v1 \
  --nx 256 --ny 128 \
  --cases validation_signed_low_pulse_p_phase94_v1_20261003 \
          validation_signed_low_pulse_m_phase94_v1_20261003 \
  --defer-finalize
```

It uses official PhysicsNeMo Curator `Source`, `Filter`, `Sink`, `VTKSource`,
and `run_pipeline` APIs plus PhysicsNeMo `Mesh`/`BVH` sampling. Final validation
is performed by `scripts/finalize_low_action_phase94_curated.py`. Float32
Curator timestamps are checked against the exact 801-point grid with `2e-5`
absolute tolerance (observed roundoff `6.11e-6`) and a separate uniform-step
check; a directed test rejects larger perturbations.

## Same-phase physical diagnostic

The comparison uses the existing `alternating_t94_zero_20261003` baseline and
the inherited `[114,174]` analysis window.  This window was fixed earlier in
`artifacts/tandem_cylinders/two_phase_alternating_predeclared_20261003.json`
(SHA-256 `34355adfe55a019c8615514c1634531c76bcc230940b9c0e80319746374e627b`,
mtime `2026-10-03T04:36:54.157373525Z`).  The low-action predeclaration itself
only states the full run interval `[94,174]`; it does not independently declare
the analysis window.  The inherited same-phase window was fixed before these
new low-action force results, but this limitation is recorded explicitly.

The canonical joint gate requires all three conditions: at least 2% lower mean
total Cd, rear `Cl'` RMS no greater than 1.05 times zero, and absolute mean rear
Cl no greater than 0.1 times zero-case rear `Cl'` RMS.

| case | mean total Cd change | rear Cl' RMS ratio | abs(mean rear Cl)/zero Cl' RMS | joint |
|---|---:|---:|---:|---|
| `+0.75` pulse | +0.4353% | 1.01875 | 0.35127 | fail |
| `-0.75` pulse | -1.6860% | 1.04675 | 0.28626 | fail |

The negative pulse shows a real drag-reduction signal and remains inside the
5% fluctuation allowance, but misses the 2% drag threshold and has excessive
mean-lift bias.  The positive pulse increases drag and also violates the
mean-lift bound.  Neither is a valid control win.

For completeness, the fluid-torque work proxy
`-omega*CmPitch/Cd_zero` (signed / positive-only / absolute mean) is
`-0.011567 / 0.000160 / 0.011886` for the positive pulse and
`-0.011176 / 0.000109 / 0.011393` for the negative pulse.  This is not
electrical motor energy; the positive-only quantity assumes no regenerative
recovery.

The authoritative result is
`low_action_phase94_physical_audit_v3_canonical.json`.  The retained v1 used a
non-canonical two-term gate.  The retained v2 corrected the gate but overstated
the low-action file's own analysis-window declaration and omitted torque
proxies.  Both are superseded and kept only for provenance.

## Curator incident and recovery

Two Curator processes were initially believed to target the same positive HDF5.
The younger process was conservatively stopped before producing an output file;
no raw or curated data were deleted.  Log inspection then showed the main
process was handling the negative case first.  Positive curation was restarted
in a distinct staging directory, so the processes never shared an HDF5 target.
The final profile is assembled only after each case completes and passes schema,
801-frame, time/action/force alignment, finite-value, pressure-gauge, and split
isolation checks.  This incident affects elapsed time, not CFD values.

The audit implementation is `scripts/audit_low_action_phase94_physics.py`; its
canonical three-term gate has a directed regression test in
`tests/test_low_action_phase94_physics.py`.

## Fixed-checkpoint surrogate diagnostic

After the pair passed final QC, the pinned PhysicsNeMo 2.2.2 image
`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`
evaluated the fixed v3 and v4 H20 checkpoints. Each model used its own immutable
training normalization. Only this profile's `validation` split was requested;
the frozen test was not accessed. Strict pooled endpoint total-drag NRMSE was:

| checkpoint | H1 | H10 | H50 | H100 |
|---|---:|---:|---:|---:|
| v3 H20 parent | 1.0339% | 0.8728% | 6.2206% | 16.2544% |
| v4 H20 candidate | 0.8752% | 0.5786% | 6.8923% | 17.8589% |

The v4 candidate improves H1/H10 but is worse at H50/H100. At H100 its pooled
NRMSE is 9.87% higher than v3; state MAE is `0.044852` versus `0.044270`, and
total-drag MAE is `0.373271` versus `0.350797`. This independent low-action
diagnostic therefore does not support promoting v4 for long-horizon closed-loop
work. It is diagnostic evidence, not a new tuning or model-selection split.
Both H100 drag errors are also worse than the same segments' persistence drag
MAE (`0.216900`); at H50 both models still beat persistence (`0.260562`). The
authoritative machine-readable decision is
`low_action_phase94_fixed_checkpoint_decision_v3.json` (SHA-256
`ab2722fd5e8bf18daf492e0fd991b3d0b8842f417b130656ea9c40835ff9ba5f`).
Its conclusion is derived from three saved H100 checks: NRMSE at most 10%,
total-drag MAE better than persistence, and NRMSE no worse than v3. All three
fail here. Even a counterfactual pass of all three checks yields only
`ELIGIBLE_FOR_PAIRED_ACTION_RANKING_AUDIT`: paired same-initial-state action
drag-difference/ranking evidence and matched real CFD are still required. It
does not pass Gate-C or authorize closed-loop promotion. The earlier non-v2
JSON and v2 JSON are retained for provenance but superseded; non-v2 hardcoded
the conclusion, while v2 overstated what a surrogate error-gate pass permits.

Commands and immutable normalization/checkpoint hashes are recorded by
`scripts/run_low_action_phase94_validation_spark.sh` in each
`evaluation_provenance.json`. Raw summaries, endpoint segments and strict pooled
audits are under
`artifacts/distributed_runs/control_gap_low_action_phase94_validation_v1_worker78/evaluations/`.
