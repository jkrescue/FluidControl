# Numerical credibility audit: Re=100, L/D=5 tandem-cylinder OpenFOAM

Date: 2026-10-03 UTC

## Scope and decision

This is a read-only audit of the current OpenFOAM evidence. No CFD was
started. The purpose is to separate solver health, representative numerical
sensitivity, and claim-specific numerical validation.

The current 19,290-cell, `dt=0.005` calculations are suitable for dataset
generation, method development, and qualitative screening. Existing evidence
supports the direction of the long constant-rotation response, but it does not
establish formal grid convergence of a learned or time-varying controller.
The exact final action history still requires a matched discretization study.

## Reconstructed numerical setup

- Solver: OpenCFD OpenFOAM v2512 `pimpleFoam`, laminar incompressible flow.
- Geometry: `D=1`, cylinder centres `(10,7.5)` and `(15,7.5)`, hence `L/D=5`.
- Domain: `30D x 15D`, with `10D` upstream of the front cylinder, `15D`
  downstream of the rear cylinder, and slip boundaries `7.5D` above/below.
- Fluid: `U_inf=1`, `nu=0.01`, hence `Re=100`.
- Time: second-order `backward`, normally `dt=0.005 D/U_inf`.
- Space: `Gauss linearUpwind grad(U)` convection, linear gradients and
  interpolation, corrected Laplacian and surface-normal gradients.
- Coupling: one PIMPLE outer corrector, two pressure correctors and one
  non-orthogonal corrector. Pressure and velocity absolute tolerances are
  `1e-7` and `1e-8`; final solves set `relTol=0`.
- The reproducible initial perturbation is a 2% cross-stream velocity patch
  behind the front cylinder. It is an initial-condition seed, not actuation.

### Meshes

The mesh is a one-cell-thick 2-D hexahedral extrusion with a structured
background and a four-block O-grid around each cylinder. `simpleGrading` is
one in all directions.

| Quantity | Coarse | Medium |
| :-- | --: | --: |
| Cells | 19,290 | 77,160 |
| Faces per cylinder | 96 | 192 |
| Approximate first radial O-grid spacing | `0.03125D` | `0.015625D` |
| Maximum aspect ratio | 2.995 | 3.043 |
| Maximum non-orthogonality | 43.15 degrees | 44.08 degrees |
| Maximum skewness | 0.989 | 1.010 |
| `checkMesh` | Mesh OK | Mesh OK |

The medium grid doubles resolution in each in-plane direction, producing four
times as many cells. Only two levels exist, so Richardson extrapolation, an
observed order, and a grid-convergence index cannot be computed. These results
must be called a coarse/medium sensitivity check, not grid convergence.

## Force-coefficient definition

Front and rear forces are integrated separately with OpenFOAM `forceCoeffs`.
The configuration fixes `rhoInf=1`, `magUInf=1`, `lRef=1`, and `Aref=0.1`.
`Aref` is exactly `D` times the extruded span `0.1`, so the resulting `Cd/Cl`
are the standard per-cylinder coefficients based on
`0.5 rho U_inf^2 D span`. Total system drag is the sum of front and rear Cd.
Drag is positive in `+x` and lift in `+y`. Force coefficients are written at
every CFD step. The existing `forceRear` output includes `CmPitch`, a
hydrodynamic moment coefficient, but its sign/normalization as an actuator
power proxy has not been validated here; motor power or total actuator work
is not directly measured.

This definition is internally consistent and matches the documented
comparison with Zhao et al. It does not validate physical power consumption.

## Evidence already available

### Solver health

The baseline completes 32,000 steps to `t=160`; maximum Courant number is
0.266 and maximum absolute global continuity contribution per step is
`8.35e-9`. Expanded trajectories report maximum Courant below about 0.50 and
continuity errors near `1e-12`. The recent long `omega=0,+1,-1` panel has
maximum Courant below 0.262 and clean solver exits. This is strong evidence of
numerical stability, but stability alone is not accuracy.

### Existing spatial checks

For uncontrolled flow on coarse versus medium grids over `t=80..160`, the
recorded front/rear mean-Cd differences are 0.464%/0.048%, fluctuating-Cl-RMS
differences 1.557%/0.501%, and shedding-frequency differences about 0.8%.

For constant `omega=+1`, raw coarse and medium cases remain on this machine.
Their `t=80..160` differences are:

| Observable | Front | Rear |
| :-- | --: | --: |
| Mean Cd | 0.498% | 0.156% |
| Fluctuating Cl RMS | 2.464% | 1.062% |
| Zero-crossing shedding frequency | 0.795% | 0.793% |

The medium-grid maximum Courant number is 0.527 at the same `dt=0.005`.
These results support coarse-grid screening at this representative constant
action. They do not validate a different ramp, periodic schedule, or feedback
trajectory.

The uncontrolled medium raw case and the earlier high-rotation raw cases are
not present under the current server's `cfd/.../cases` directory. Their
summaries are documented, but a fully independent re-analysis from raw files
cannot presently be reproduced here.

### Existing time-step checks

Project records state that at `q=+-2.5` (`omega=+-5`) halving the coarse-grid
step from `0.005` to `0.0025` changed rear fluctuating Cl RMS by 0.33--0.45%.
The same records report coarse/medium long-window Cl-RMS differences of
2.723--3.051%. This is useful worst-action evidence, but the associated raw
cases and structured result JSON are absent from the current server. There is
no exact half-step replay of the current long panel or a frozen controller.

## Statistical convergence of `t=120..160`

With baseline `St` about 0.1611, the 40-time-unit window contains only about
6.4 shedding periods. Six equal time blocks in the recent matched panel give:

| Action | Mean total Cd | Block CV | Block range / mean | Last-half minus first-half |
| :-- | --: | --: | --: | --: |
| zero | 2.299313 | 0.391% | 1.112% | -0.544% |
| ramp to +1 | 2.192260 | 0.928% | 2.544% | -1.568% |
| ramp to -1 | 2.218997 | 0.213% | 0.656% | +0.265% |

The blocks are serially correlated and are not independent replications. The
`+1` trajectory still exhibits material drift; even zero narrowly exceeds a
0.5% half-window consistency target. Thus `t=120..160` is adequate for an
exploratory effect-size estimate, but stationarity and a sampling confidence
interval have not been demonstrated.

The observed constant-rotation drag changes of -4.66% and -3.49% are larger
than the representative grid differences and block variability, so their
direction is credible. Their exact magnitudes are not final. Periodic-control
changes of +0.59% and -0.92% are comparable with current statistical and
discretization uncertainty and should be treated as numerically unresolved.

## Minimum independent recomputation after CPU contention clears

Freeze one candidate action history before looking at the new results. Do not
reuse the four-action validation panel or frozen test trajectories to tune it.
For the candidate and its phase-matched zero-action reference, run this `2 x 2`
matrix from the identical physical restart:

| Pair | Grid | Time step | Purpose |
| :-- | --: | --: | :-- |
| Existing reference | 19,290 | 0.005 | Current result |
| Temporal pair | 19,290 | 0.0025 | Isolate time-step sensitivity |
| Spatial pair | 77,160 | 0.005 | Isolate grid sensitivity |

“Pair” means both candidate and zero, so the minimum is four new CFD cases.
Use the same fixed action table for an open-loop replay, or independently run
the same frozen policy on each discretization for a closed-loop robustness
claim. These are different questions and must not be mixed.

Run to at least `t=160`, retain `t=120..160` as the predeclared primary window,
and extend in increments of 20 time units without changing the action or gates
if the stationarity checks below fail. To stay within a two-hour triage window,
run at most two CPU cases concurrently and suppress dense full-field output;
forces, action, solver log and probes remain every step, while fields need only
be written every 2 time units. A two-hour timeout produces an incomplete
diagnostic, never a passed result.

An exact medium-grid restart should be generated on the medium grid. Mapping a
coarse `t=80` field onto the medium mesh is acceptable only if labelled as a
mapped-restart sensitivity test and followed by at least 40 time units of
washout; it is not equivalent to an independently developed medium baseline.

## Predeclared acceptance limits

All checks apply to both candidate and zero unless explicitly stated.

### Numerical health

- clean solver end, finite force/probe values and exact requested final time;
- maximum Courant number `<0.6` and never `>=1`;
- maximum absolute global continuity contribution per step `<1e-7`;
- identical coefficient definitions and action samples across comparisons.

### Statistical stationarity

- at least six complete shedding periods in the primary window;
- absolute first-half/last-half difference in mean total Cd `<=0.5%`;
- coefficient of variation of six equal-block total-Cd means `<=0.5%`;
- the last three cycle-aligned means each within `1%` of their joint mean.

If any stationarity condition fails, extend all paired cases equally before
performing the discretization decision.

### Time-step sensitivity (`0.005` versus `0.0025`, coarse grid)

- each cylinder mean Cd relative difference `<=1%`;
- front/rear fluctuating Cl RMS relative difference `<=1%`;
- candidate-minus-zero total-drag benefit differs by `<=0.5` percentage point;
- rear-Cl-RMS ratio differs by `<=1` percentage point.

### Grid sensitivity (19,290 versus 77,160 cells, `dt=0.005`)

- each cylinder mean Cd relative difference `<=3%`;
- front/rear fluctuating Cl RMS relative difference `<=3%`;
- shedding frequency relative difference `<=1%`;
- candidate-minus-zero total-drag benefit differs by `<=1` percentage point;
- rear-Cl-RMS ratio differs by `<=3` percentage points;
- the sign of every claimed benefit is unchanged.

For a final Stage-D claim, the medium-grid result itself must still satisfy the
locked physical criteria: at least 2% lower mean total drag, rear fluctuating
Cl RMS no more than 5% above zero, and absolute mean rear Cl no more than 10%
of the zero-action rear fluctuating Cl RMS. A coarse-grid pass that becomes a
medium-grid fail is a failed physical claim.

These limits are predeclared engineering tolerances for this project. Passing
two grid levels establishes bounded sensitivity, not asymptotic convergence.
A formal convergence claim requires a third systematically refined grid and a
reported observed order/GCI.

## Evidence paths

- `cfd/tandem_cylinders/make_baselines.py`
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/system/`
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/log.checkMesh`
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/analysis.json`
- `docs/results/constant_p100_grid_pair_numerical_qc.json`
- `docs/results/constant_p100_grid_pair_statistics.json`
- `artifacts/tandem_cylinders/control_landscape_long_result_20261003.json`
- `docs/CONSTANT_ROTATION_GRID_CHECK_20261002.md`
- `cfd/tandem_cylinders/CASE_SPEC.md`
