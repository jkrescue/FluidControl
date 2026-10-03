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
