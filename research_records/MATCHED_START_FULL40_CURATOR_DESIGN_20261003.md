# Matched-start full40 Curator design (pre-execution)

## Decision

Build a new, versioned `matched_start_full40_v1` profile only after both raw
aggregates pass. Reuse the nine already curated commissioning HDF5 files by
verified hard links, and curate only the 31 remainder cases into an isolated
staging root. Do not modify or write into the running nine-case staging tree.

This is deterministic CFD-to-HDF preparation. It is not training, evidence of
surrogate generalization, or closed-loop control.

## Immutable split

The committed full40 predeclaration is authoritative:

- train: phase bins 0, 2, 4, 6; 20 trajectories;
- validation: phase bins 1, 5; 10 trajectories;
- frozen test: phase bins 3, 7; 10 trajectories.

All five actions from one phase remain in the same split. The 31-case remainder
therefore contributes train 11, validation 10, and frozen test 10; the reused
nine trajectories are train only. No frame- or action-level random split is
permitted.

## Required gates before curation

1. Nine-case raw aggregate remains
   `MATCHED_START_9_CASE_COMMISSIONING_QC_PASS`.
2. All 31 receipts exist and the extension aggregate is
   `MATCHED_START_FULL40_EXTENSION_31_CASE_RAW_QC_PASS`.
3. The nine-case Curator profile is finalized and its per-case HDF5/raw/VTK
   hashes pass finalizer QC.
4. Every case config matches the committed full40 predeclaration, authorization,
   phase split, action table, restart hashes, 801 field frames, and 16,000 raw
   force rows.
5. Spark has at least 64 GiB available memory and 250 GiB free disk before a
   batch. Stop launching below 40 GiB memory or 150 GiB free disk.

## Exact t0 force provenance

For every trajectory, the first force sample is the unique row at the restart
time from the original baseline files:

`tandem_backward_dt005/postProcessing/forceFront|Rear/0/coefficient.dat`.

Before VTK sampling, the full40 Source must compare both files with
`case_config.source_force_sha256` and the extension authorization. The finalizer
must repeat this check before assembling 16,001 force samples. Interpolation may
align the verified 16,001 force series to the 801 saved field times, but it may
not invent the source t0 row.

## Official APIs and isolated outputs

Reuse the installed official APIs already exercised by the nine-case profile:

- PhysicsNeMo Curator `Source`, `Filter`, `Sink`, `VTKSource`, and
  `run_pipeline`;
- PhysicsNeMo `Mesh/BVH` field sampling;
- the existing numerical-quality filter and atomic HDF5 sink contract.

Use one process and one staging directory per case, at most three concurrent
cases. The proposed staging root is
`data/curated/.staging/matched_start_full40_v1/`; the atomic final root is
`data/curated/tandem_cylinders_matched_start_full40_v1/`. A case becomes
eligible only after its raw receipt and separate 801-frame `VTK_READY` receipt
are verified. Existing outputs are never overwritten.

Observed storage is approximately 2.2 GiB VTK and 0.26 GiB HDF5 per 801-frame
trajectory. The 31-case remainder therefore needs roughly 70 GiB VTK plus
9 GiB HDF5, excluding temporary export headroom. Current free space was about
552 GiB during this design audit.

## Leakage controls and finalization

Curating frozen cases is allowed only as a deterministic conversion with
integrity hashes. Before a separately authorized final evaluation, frozen data
must not be used for normalization, model selection, action ranking, drag/reward
summaries, or dashboard outcome plots.

The finalizer must atomically assemble exactly 20 train, 10 validation, and 10
`frozen_test` HDF5 files. Normalization statistics are computed from an explicit
20-file train manifest only. Training and validation DataPipes must consume
explicit file manifests and must never glob the full dataset root. The frozen
manifest stays sealed by SHA until one-time final evaluation is authorized.

The pre-execution planner is
`scripts/plan_matched_start_full40_curator.py`. It is read-only and currently
must report blocked while the 31-case aggregate or finalized nine-case HDF5
profile is absent. It does not start VTK export, Curator, finalization, or
training.
