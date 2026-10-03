# Full40 time-varying-action validation protocol

## Status and purpose

This is a predeclared **real OpenFOAM** development-validation panel. It has
not been staged or solved. It uses only the train-excluded validation phases
`b01` and `b05`; it neither opens nor enumerates frozen-test HDF5 data. Its
purpose is to test whether the official PhysicsNeMo FNO resolves force
transients and action ordering under bounded action switches before a
canonical PPO policy can be trusted. It is not training data, a PPO rollout,
or evidence of control benefit.

## Frozen six-case matrix

Each phase has three byte-identical real restart branches: `plus`, `minus`,
and `zero`. The source times and five state-file hashes come directly from the
reviewed full40 predeclaration: `b01` starts at `t=130`, and `b05` uses its
separately predeclared real restart. No state is interpolated.

The two nonzero schedules are exact sign mirrors. They begin and end at zero,
last `20 D/U`, use knots only on the `0.1 D/U` decision grid, visit both
`±0.75` and `±0.375`, include holds, reversals and zero returns, and obey
`|omega| <= 0.75` and `|delta omega| <= 0.1` per decision. The zero branch is
advanced over the same solver and output grid. The machine-readable schedule
is the authority; no action may be chosen after seeing CFD or FNO results.

OpenFOAM is pinned to v2512 by digest, with the existing Re=100, L/D=5 mesh,
`dt=0.005`, force output every solver step and fields every `0.1 D/U`. Each
case is 4,000 solver steps and 201 physical field states including the real
restart.

## Predeclared measurements

Real-CFD QC reports front/rear Cd and Cl, total-Cd means and lift fluctuation
RMS on elapsed windows `[0,10]`, `[10,20]`, and `[0,20] D/U`, always against
the same-phase zero branch. It also records the true three-way total-Cd
ranking. Solver completion, 4,000-step count, Courant/continuity limits,
source-state hashes, action table, and immutable predeclaration SHA must all
pass.

The later FNO report must use H1/H10/H50/H100 on all six cases. H100 pooled
total-Cd NRMSE retains the unchanged 10% limit. At strict common-state
`start_frame=0`, the signed action-difference total-Cd MAE limit is `0.023`,
one half of the approximate 2% control-signal scale. The report must also show
the exact minus/zero/plus ordering for both fixed 10-D/U halves and the full
20-D/U window. Passing one metric cannot hide failure of another.

Only frame 0 within each phase is a strict counterfactual action comparison.
Later equal frame indices are useful transient diagnostics but their states
have already diverged under earlier actions. Endpoint force accuracy is not
the same as window-mean accuracy. This development panel cannot replace the
formal full40 validation gate, promotion check, frozen-policy discipline, or
paired real-CFD PPO acceptance run.

## Execution gates and resource estimate

The generator refuses a pre-existing case, requires the reviewed
predeclaration SHA and an explicit generation token, and stages one case per
invocation. The runner defaults to preflight, requires 40 GiB MemAvailable,
refuses existing solver output, and blocks while `pimpleFoam`, matched-start
Curator, or FNO training is active. Consequently this reviewed version is
strictly serial: a second panel case cannot start while the first solver is
running. Execution additionally requires a separate environment token. The
panel QC output is exclusive and cannot overwrite an earlier report.

Execution is deliberately restricted to the authoritative Spark repository
path. A Worker temporary-copy run is rejected: this protocol does not yet
define atomic raw transfer, a per-file transfer manifest/receipt, or
post-transfer source/action/force revalidation. Worker execution may be added
only as a separately reviewed protocol with those SHA-bound receipts. The
current panel therefore has no transfer step; its cases, solver logs, force
files, completion markers, and aggregate QC remain together on Spark.

Based on the measured full40 throughput of roughly 23 minutes per 80-D/U case,
one 20-D/U case is estimated at 6–8 minutes including startup and QC. Six
cases require approximately **36–48 minutes wall time in the enforced serial
mode**. This is a linear estimate, not a completed measurement. The panel must wait for current
Curator/training work to finish and must not delay the primary dev30 chain.

Files:

- `cfd/tandem_cylinders/make_full40_dynamic_validation_panel.py`
- `cfd/tandem_cylinders/run_full40_dynamic_validation_case.sh`
- `scripts/audit_full40_dynamic_validation_panel.py`
- `artifacts/tandem_cylinders/full40_dynamic_validation_predeclared_20261003.json`
