# Nine-case matched-start CPU Curator plan (2026-10-03)

## Status and scientific boundary

This is an implemented commissioning pipeline for the nine matched-start
OpenFOAM cases. As of 06:24 on 2026-10-03, the nine-case CFD acquisition had
started and two cases had reached `RAW_TRANSFER_VERIFIED`; VTK export and
Curator execution had not started:

- phases `b00`, `b02`, `b04`;
- branches `m075`, `zero`, `p075` in every phase;
- canonical name `matched_start_acquisition_train_b{00|02|04}_{m075|zero|p075}`.

Spark is the only persistent store. The Curator stage performs no GPU work,
training, validation, or frozen-test access. The output uses a `train` split
only because these labels may later be reviewed as training candidates. The
commissioning manifest says `training_use=FORBIDDEN`, writes no normalization,
and cannot be consumed for training until a separate scientific promotion.

`conf/matched_start_curator_contract.template.json` is intentionally unbound
and fails preflight. It must not be changed to
`BOUND_APPROVED_FOR_CURATION` until all nine raw transfers and the authoritative
phase-manifest SHA have been reviewed.

## Per-case sequence

`scripts/orchestrate_matched_start_curator.py` runs at most two cases by
default and accepts a hard maximum of three. Each worker owns one case through
the full sequence, so no two processes ever share an HDF5 output:

1. Read the atomic raw receipt at
   `artifacts/matched_start_acquisition/transfer_verified/<case>.json`.
   Require status `RAW_TRANSFER_VERIFIED`, raw-only coverage, the exact next
   stage and Curator guard strings, and the canonical worker-manifest path.
2. Verify the worker `.sha256` file itself and then hash every listed raw file
   under `cfd/tandem_cylinders/cases/<case>`. Absolute paths, `..`, duplicate
   rows and paths outside the case prefix are rejected.
3. Verify `case_config.json`: `split=train`, integer `phase_bin` (`0`, `2`, or
   `4`), branch/action target, `expected_field_frames=801`,
   `expected_aligned_force_samples_with_source_t0=16001`, action points, analysis
   window, authoritative phase-manifest SHA, `source_restart_case`, and the
   exact five source-state keys `U,U_0,p,phi,phi_0`.
   Legacy aliases such as `expected_frames` and `force_samples` are rejected.
4. Verify the five pre-action files in `source_restart_provenance` against the
   config hashes and raw manifest. Require identical five-field provenance
   across all three branches of each phase.
5. Parse the OpenFOAM `rearCylinder` omega table in the run-start `U` file and
   require exact agreement with `action_points`.
6. Run the existing `scripts/export_tandem_vtk.sh` on Spark. That script uses
   pinned image
   `opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319`
   and refuses an existing `VTK_curator` directory.
7. Hash exactly 801 nonempty `internal.vtu` files into a per-case VTK manifest,
   then exclusively write an atomic `VTK_READY` receipt under
   `artifacts/matched_start_acquisition/vtk_ready`. A raw receipt alone can
   never start Curator.
8. Launch one official PhysicsNeMo Curator Source → Filter → Sink process for
   that case in
   `data/curated/.staging/matched_start_commissioning_train9_v1/<case>`.
   The source uses official `VTKSource`, PhysicsNeMo `Mesh/BVH`, and
   `run_pipeline`. The sink writes `<case>.h5.tmp`, closes and flushes it, then
   atomically publishes `<case>.h5` with a same-directory, create-if-absent hard
   link before removing the temporary link; existing temporary or final files
   are refused.

Logs, VTK manifests/receipts and case staging targets are all disjoint. A
failed case leaves no falsely complete final HDF5; recovery is manual and must
preserve the failed evidence rather than delete or overwrite it.

## Dedicated final QC and assembly

`scripts/finalize_matched_start_commissioning.py` is the only supported
finalizer. The generic Curator `--finalize-only` path is programmatically
rejected for this profile; a commissioning Curator call likewise requires
exactly one case plus `--defer-finalize --atomic-hdf5`. The dedicated finalizer first
requires exactly nine staging HDF5 files and rejects `.h5.tmp`, validation,
test, missing and extra outputs. It then checks every case:

- HDF5 schema and exact shapes: state `(801,3,128,256)`, mask
  `(801,1,128,256)`, omega `(801,1)`, force `(801,4)`, time `(801,1)`;
- train-only attributes and byte-equivalent embedded case config;
- 801-point time grid and action interpolation;
- finite state/force values, valid-mask coverage and per-frame gauge-pressure
  mean at most `1e-6`;
- VTK receipt, 801-file manifest and every VTK SHA;
- raw-transfer receipt and five-field same-phase restart identity;
- front/rear raw force files merged with duplicate-time conflict rejection.

Each new run has 16000 force samples from `t0+0.005` through `t0+80`. The
finalizer does **not** invent or extrapolate `t0`: it reads the unique exact
`t0` row from
`tandem_backward_dt005/postProcessing/forceFront|forceRear/*/coefficient.dat`,
prepends it, verifies the complete 16001-point `dt=0.005` grid, and only then
aligns forces to the 801 field times.

After all checks pass, the finalizer hard-links verified staging files into a
new temporary profile directory, writes a commissioning manifest and aggregate
QC, and atomically renames the directory to
`data/curated/tandem_cylinders_matched_start_commissioning_train9_v1`. Existing
final/temporary profiles or QC files are refused. It never writes
`normalization.json`.

## Commands after contract binding

Read-only full preflight and command plan:

```bash
python scripts/orchestrate_matched_start_curator.py \
  --contract conf/matched_start_curator_contract.json \
  --max-parallel 2 \
  --plan-output artifacts/matched_start_acquisition/curator_plan.json
```

Execution, only after reviewing that plan:

```bash
python scripts/orchestrate_matched_start_curator.py \
  --contract conf/matched_start_curator_contract.json \
  --max-parallel 2 --execute
```

Dedicated aggregate validation and atomic assembly:

```bash
.venv-curator-py312/bin/python scripts/finalize_matched_start_commissioning.py \
  --contract conf/matched_start_curator_contract.json \
  --qc-output artifacts/matched_start_acquisition/commissioning_qc.json
```

The Curator commands above have not been run. CFD acquisition and raw-transfer
verification continue independently; each case remains ineligible for VTK and
Curator processing until its exclusive `RAW_TRANSFER_VERIFIED` receipt exists,
and the nine-case aggregate remains ineligible until the full contract is bound
and approved.
