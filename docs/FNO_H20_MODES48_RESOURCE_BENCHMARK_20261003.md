# Official PhysicsNeMo FNO higher-mode resource probe (2026-10-03)

## Decision

Do not promote the 48×48-mode FNO into the tandem-cylinder control mainline yet. On the same real OpenFOAM training windows and at width 48/batch 4, it has 106.21M parameters versus 47.22M for 32×32 modes, but its short forward/backward throughput is 46.92 rather than 55.63 transitions/s (15.7% lower). Peak CUDA reserved memory is 13.44 rather than 11.75 GiB (14.4% higher). No accuracy experiment was performed, so these numbers do not prove that extra modes hurt or help prediction.

## Controlled measurement

- Host: DGX Spark `SPARK_HOST`, GPU 0 (GB10 unified memory); worker `WORKER_HOST` continued two independent CPU OpenFOAM solves.
- Runtime: pinned project PhysicsNeMo 2.2.2 image, official `physicsnemo.models.fno.FNO` through existing `build_model`; `--network none`.
- Data: 28 real, curated **train-only** OpenFOAM trajectories in `tandem_cylinders_control_gap_v4`; eight sampled H20 windows, float32, same loss and input/output channels for every row.
- Timing: one warm-up plus two timed forward/backward iterations per row; no optimizer, update or checkpoint, no validation/frozen-test read. This measures an in-memory compute step, not DataPipe loading, epoch time or accuracy.
- Memory guard: allocator cap 65% of 121.69 GiB unified device memory; guard preflight required 103.10 GiB `MemAvailable`, observed 114.24 GiB. The guard sampled a minimum 87.25 GiB during the run, above the mandatory 20 GiB floor. No OOM.

| Configuration | Modes | Parameters | Transitions/s | Peak reserved GiB | Minimum `MemAvailable` GiB |
|---|---:|---:|---:|---:|---:|
| width 48, batch 4 | 32×32 | 47.22M | 55.63 | 11.75 | 100.30 |
| width 48, batch 8 | 32×32 | 47.22M | 59.95 | 23.25 | 87.22 |
| width 64, batch 4 | 32×32 | 83.94M | 42.23 | 13.92 | 96.96 |
| width 48, batch 4 | 48×48 | 106.21M | 46.92 | 13.44 | 97.45 |

Batch 8 improves throughput by 7.8% versus batch 4 in this **two-step** probe while almost doubling reserved memory. Width 64 costs 24.1% throughput at batch 4 in this run. Results vary from the preceding five-step benchmark and should not be extrapolated to full training. The new higher-mode row was explicitly opt-in; the default three-row benchmark plan remains unchanged.

The official [PhysicsNeMo FNO API](https://docs.nvidia.com/physicsnemo/26.03/physicsnemo/api/models/fnos.html) exposes `num_fno_modes`, `latent_channels`, and dimensionality. FNO dimension 3 is not a drop-in upgrade for this 2D causal control setup: a temporal third axis would require a separately specified history/action interface and leakage-safe evaluation. A bespoke FNO/local-convolution hybrid is outside the current official-model constraint. Reconsider capacity only after phase-grouped CFD acquisition, Curator QC, and validation show that data coverage and action-effect prediction justify it.

## Provenance

Run command: `bash scripts/run_tandem_fno_h20_resource_benchmark.sh tandem_fno_h20_resource_modes48_20261003 1 2 include-modes48`.

Canonical JSON: `artifacts/benchmarks/tandem_fno_h20_resource_modes48_20261003/benchmark.json`, SHA-256 `f4c246137191bef0dd69efd971600a0781c8f2e1ac8aa89d546914ad68bf1ea2`. Its status is `COMPLETE`, `checkpoint_write_count=0`, `optimizer_created=false`, and `frozen_test_status=NOT_ACCESSED`.
