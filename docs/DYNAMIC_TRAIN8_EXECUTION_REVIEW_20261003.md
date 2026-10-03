# Dynamic train8 real-CFD and Curator execution review

Status: **implementation prepared; generation, solver and Curator execution are
disabled pending review of the immutable predeclaration SHA.** No case has been
generated or launched by this change.

## Frozen scientific design

Eight 20-D/U OpenFOAM trajectories use only full40 train restart bins
`b00/b02/b04/b06`. Each phase has a deterministic slew-limited PRBS trajectory
and a tapered multisine/chirp trajectory. The complete 201-point action tables
are written into the predeclaration before any case exists. They obey
`|omega| <= 0.75`, `|Delta omega| <= 0.1` per 0.1 D/U, and start/end at zero.
The formulas and fixed target sequences do not read b01/b05 validation results
or b03/b07 frozen data.

Each solve uses the pinned OpenFOAM image, `dt=0.005`, field interval 0.1,
force interval 0.005, 4,000 steps and 201 field states. Generation copies the
five exact restart files and records their existing full40 SHA-256 values.
The runner re-hashes those files and the actual OpenFOAM omega table, uses an
atomic per-case lock, refuses existing output, limits Spark to four concurrent
`pimpleFoam` processes, requires at least 40 GiB MemAvailable and preserves the
current 381 GiB free-disk budget. The total raw-data contract is capped at
30 GiB.

The independent Curator entry point is fail-closed until all eight raw cases
pass aggregate QC. It re-hashes solver logs, QC, markers, source states, both
baseline force files and the action table. VTK receipts require exactly 201
`TimeValue` frames and a clean pinned `foamToVTK` log. HDF conversion reuses the
official PhysicsNeMo Curator `Source -> NumericalQualityFilter ->
TrajectoryHDF5Sink` pipeline and `run_pipeline`; each HDF is atomic and must
contain `(201,3,128,256)` states and `(201,4)` four-force channels. The output
profile is train-only. A later augmented-train finalizer combines these eight
HDF files with the existing 20 training trajectories while preserving the
immutable full40 train20 normalization. This keeps exact input/output units for
the warm-start e5/free-AR checkpoint; augmented-train descriptive statistics
may be reported separately but must not silently refit the transform.

Normalization amendment: the already frozen CFD-acquisition predeclaration
contains an earlier note proposing a train-only refit. That note does not affect
any action schedule or raw solve. This reviewed execution design supersedes it
before curation/training: the released augment-train28 manifest must identify
the full40 train20 normalization path and SHA and state that the transform was
not refit. A new transform would require analytic checkpoint conversion and is
outside this acquisition.

## Commands submitted for review

The first command is the only one safe before hash review:

```bash
python cfd/tandem_cylinders/make_dynamic_train8_panel.py --write-predeclaration
sha256sum artifacts/tandem_cylinders/dynamic_train8_predeclared_20261003.json
```

After GitLab records that artifact, bind its SHA in all three guarded entry
points, rerun tests, then generate one case per invocation:

```bash
python cfd/tandem_cylinders/make_dynamic_train8_panel.py \
  --case dynamic_train8_b00_prbs \
  --approval-token GENERATE_REVIEWED_DYNAMIC_TRAIN8
bash cfd/tandem_cylinders/run_dynamic_train8_case.sh \
  dynamic_train8_b00_prbs --preflight-only
```

Only after review may an orchestrator set
`DYNAMIC_TRAIN8_APPROVAL_TOKEN=EXECUTE_REVIEWED_DYNAMIC_TRAIN8` and run at most
four independent cases. After all raw QC passes, create and review the curation
authorization, export `VTK_dynamic_train8`, create VTK receipts, curate one case
at a time, and finalize:

```bash
python cfd/tandem_cylinders/curate_dynamic_train8.py authorize --execute \
  --approval-token EXECUTE_REVIEWED_DYNAMIC_TRAIN8_CURATOR
python cfd/tandem_cylinders/curate_dynamic_train8.py curate \
  --case dynamic_train8_b00_prbs --execute \
  --approval-token EXECUTE_REVIEWED_DYNAMIC_TRAIN8_CURATOR
python cfd/tandem_cylinders/curate_dynamic_train8.py finalize --execute \
  --approval-token EXECUTE_REVIEWED_DYNAMIC_TRAIN8_CURATOR
```

This acquisition addresses action-history coverage. It is not a control result,
does not authorize PPO, and cannot change the unchanged dynamic6 validation
thresholds after outcomes are seen.
