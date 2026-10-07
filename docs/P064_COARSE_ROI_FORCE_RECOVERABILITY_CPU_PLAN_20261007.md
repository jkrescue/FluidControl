# P064 coarse-ROI force recoverability CPU diagnostic plan — 2026-10-07

**Status:** proposal only; no execution, model inference, optimizer, CFD, or new
data is authorized.

## Non-duplication finding

FC-P008 and its FC-E016 precursor already tested a *learned* affine readout of
frozen FNO features. P008 used 19,648 train-only H1 endpoints and improved some
rotating lift-RMS errors while worsening drag/mean-force metrics; formal
admission failed. P009 additionally tested a joint H1/free-AR force head. These
experiments show that another fitted readout/regressor would be repetitive.
They did **not** separate OpenFOAM pressure and viscous force components, test
HDF-to-raw time alignment for those components, or test a geometry-defined
near-wall pressure integral. Repository search found no completed surface-force
or coarse-pressure-quadrature diagnostic.

## One bounded, no-fit diagnostic

Use only the fixed training cases
`matched_start_acquisition_train_b00_{m075,zero,p075}` and the mechanically
declared HDF frame indices `{0,100,200,300,400,500,600,700,800}`. These are
three same-phase action cases and nine uniformly spaced saved times; neither
case nor time is chosen from an observed error. For both cylinders and both
force directions, bind:

- physical HDF `u,v,gauge_pressure`, mask, `omega`, time, and total Cd/Cl;
- the same-time OpenFOAM `CdPressure`, `CdViscous`, `ClPressure`, and
  `ClViscous` from each saved
  `uniform/functionObjects/functionObjectProperties` dictionary;
- exact file paths and SHA-256 values in the result receipt.

Run three ordered checks and stop on a failed predecessor:

1. **Alignment/algebra.** Require exact case identity and mechanical time
   mapping. Report max/mean absolute differences for HDF total versus OpenFOAM
   total and for `pressure + viscous` versus total. A mismatch diagnoses label
   alignment/parsing, not model error.
2. **Component importance.** Without fitting, report signed pressure and
   viscous contributions, `|viscous|/(|pressure|+|viscous|)`, and cancellation
   `|pressure+viscous|/(|pressure|+|viscous|)` by cylinder, direction, action,
   time, and pooled. This measures whether a pressure-only explanation is even
   numerically plausible; it does not prove that velocity cannot encode shear.
3. **Fixed coarse-pressure proxy.** On the existing physical gauge-pressure
   grid, evaluate a preregistered 128-angle circular quadrature at radius
   `R + 2h`, where `R=0.5` and `h=max(dx,dy)`. Use bilinear interpolation only
   when all stencil cells are fluid; report coverage and fail closed rather
   than substituting nearest cells. Treat the offset values as estimates of
   wall pressure but retain the true-wall quadrature weight `R dtheta` (do not
   inflate area to the probe radius). Integrate `-p n` with the case's fixed
   `rho=1`, `U=1`, and `Aref=0.1` coefficient convention. Compare only
   with the OpenFOAM **pressure** Cd/Cl components using MAE, RMSE, signed bias,
   per-point error, and action-minus-zero error. Do not tune radius, angle
   count, scale, or offset from these results.

The ring is deliberately outside the wall and is therefore a coarse proxy, not
an exact traction reconstruction. Large error can reflect coarse sampling or
the fixed proxy itself; small error would show that a deterministic summary of
the saved pressure carries the instantaneous pressure-force signal. Neither
outcome alone identifies the cause of FNO H1/AR error.

## Reuse, budget, and decision

Reuse the official `HDF5Reader` invocation and schema from
`src/fluid_control/tandem_datapipe.py`, strict time/total-force parsing from
`src/fluid_control/openfoam_force_history.py` and
`scripts/audit_tandem_control_objective.py`, and the immutable train manifest,
normalization, and force-representation audit. Only a small parser for the four
named `functionObjectProperties` scalars and the fixed NumPy quadrature is new.

Proposed bound: CPU1, 2 GiB RAM, no swap, CUDA hidden, 120 seconds; 27 HDF frames
and their paired dictionaries only. No train/dev fitting and no validation or
frozen data access.

Decision use is diagnostic, not an admission gate: alignment failure sends work
to data provenance; material viscous/cancellation evidence limits a
pressure-only claim; a stable low-error pressure proxy weakens the hypothesis
that instantaneous coarse pressure is intrinsically unrecoverable. Otherwise
the result remains inconclusive between coarse resolution and proxy error. It
does not authorize a new channel, architecture, training run, PPO, or CFD.
