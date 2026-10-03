# Full40 execution milestones — 2026-10-03

This is an execution and provenance log, not a model-performance claim. All
times below are UTC.

## Immutable development release

- At approximately 09:42, the atomic dev30 release was published at
  `data/curated/tandem_cylinders_matched_start_full40_dev30_v1` after the
  readiness check reported 20 train trajectories, 10 validation trajectories,
  30 verified sources, no blockers, and no frozen-test access.
- The release contains 20 train and 10 validation HDF5 trajectories. The
  frozen-test split is not materialized in this release and remains sealed.
- The release directory is read-only (`0555`). Normalization is computed from
  train20 only and includes the ordered four-force-channel statistics.
- Release manifest SHA-256:
  `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2`
- Train-only normalization SHA-256:
  `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`
- Frozen-test seal SHA-256:
  `7180d122fc7c0b7be2f42c0be0db1ce9780a41e10bf7e65f887e0ed9068f8ed6`

## Reviewed quick-screen run `qs1`

- The persistent quick-screen pipeline started at 09:43:25 as
  `fluid-control-dev30-quickscreen-pipeline-qs1.service`.
- The training container uses GPU 0 and image
  `fluid-control-physicsnemo:2.2.2` with immutable image ID
  `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Dataset, `scripts`, `src`, and `conf` are mounted read-only. The GPU guard
  reserves at least 20 GiB free unified memory; the container is additionally
  limited to 8 CPUs and 64 GiB memory.
- The prescribed sequence is one-step 10 epochs, H20 5 epochs, then validation.
  As of 09:49 the pipeline was still running. No validation metric or research
  gate result is recorded here because none was complete.

## Frozen-test materialization on Worker78

- Curator runs use image
  `fluid-control-physicsnemo-curator:2.2.2-86533e5`, immutable image ID
  `sha256:0f9007d6e62ba54f83ed7139ac378b251c8b5e6b27591f32919221fdc45ea991`,
  with network disabled and per-case isolated output paths.
- At 09:35 the first six frozen-test cases started on Worker78: all five b03
  actions and b07_m0375. At 09:49 all six were still converting and zero frozen
  HDF5 trajectories had been promoted to Spark.
- Frozen trajectories are excluded from dev30 training, validation, model
  selection, and quick-screen input. A Worker HDF5 is not authoritative until
  it is copied atomically to Spark, its source/destination SHA-256 agrees, and
  Spark-side raw, VTK, force provenance, time/grid/channel, and HDF5 QC passes.
- Worker raw, VTK, and HDF5 caches are retained until the final full40 audit;
  this log does not authorize cleanup.

## Interpretation and next gates

The dev30 release makes the reviewed quick-screen executable; it does not show
that the FNO, PPO policy, or closed-loop controller is successful. Gate B/C/D
thresholds remain unchanged. Full40 publication still requires all ten frozen
cases, strict Spark-side QC, normalization and split-manifest verification, and
the reviewed finalizer.
