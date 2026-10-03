# PhysicsNeMo FNO H20 resource benchmark (2026-10-03)

## Result

On DGX Spark GPU0, batch 8 gives only 6.95% more H20 transition throughput
than batch 4 while using 97.7% more peak reserved CUDA memory. Increasing the
FNO latent width from 48 to 64 at batch 4 is 30.6% slower and uses 18.5% more
reserved memory. For the current experiments, width 48 / batch 4 is the safer
resource-efficient default; batch 8 is a possible throughput setting only when
the extra approximately 11.5 GiB allocator reserve is acceptable. This result
does not predict accuracy or justify a training/model change.

| specification | parameters | H20 sequences/s | transitions/s | peak allocated | peak reserved | minimum MemAvailable |
|---|---:|---:|---:|---:|---:|---:|
| width48, batch4 | 47,222,783 | 3.0055 | 60.1096 | 11.411 GiB | 11.752 GiB | 100.140 GiB |
| width48, batch8 | 47,222,783 | 3.2142 | 64.2846 | 22.536 GiB | 23.230 GiB | 88.453 GiB |
| width64, batch4 | 83,935,015 | 2.0869 | 41.7375 | 13.645 GiB | 13.930 GiB | 97.228 GiB |

All specifications completed without OOM. Host `MemAvailable` stayed far above
the required 20 GiB; it was 111.261 GiB after the run. The outer guard sampled
a minimum of 88.462 GiB. No optimizer was created, no parameter update occurred,
and no checkpoint was read, written or overwritten.

## Method

The authoritative run used the official NVIDIA PhysicsNeMo 2.2.2 FNO in pinned
container image
`sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`
on NVIDIA GB10. It reused the project's `build_model`, `rollout`, and
`TandemRolloutDataset` implementations rather than creating a separate model or
data API. Model settings common to all rows were 6 inputs, 7 outputs, five FNO
layers, modes `[32,32]`, two decoder layers of size 128, padding 8 and coordinate
features. Precision was float32 without AMP, matching current H20 training.

Eight deterministic, evenly spaced flattened-dataset indices from eight
distinct real train trajectories were selected from the 28 real
`control_gap_v4/train` OpenFOAM HDF5 trajectories. Every benchmark sample
was a normalized contiguous 21-frame window: one initial field plus 20 target
transitions and the observed action sequence. The JSON report records case and
step metadata and SHA-256 for all 28 source HDF5 files. Validation and frozen
test files were not accessed.

Each specification used two warm-up steps followed by five timed steps. A step
contains the full 20-step autoregressive forward graph, physical-mask field and
force loss construction, and backward propagation. Gradients are cleared but
never applied. The timing begins after CPU HDF5 reads and host-to-device batch
materialization, so it measures model forward/backward throughput, not data
loading or complete epoch throughput. It also excludes optimizer work,
validation, logging and checkpoint I/O.

The short two-iteration pilot independently gave 60.347/64.428/42.101
transitions/s and the same peak memory values, consistent with the five-step
run. Even so, these are short local microbenchmarks rather than long-run power,
thermal or end-to-end training measurements. Different batch rows use different
numbers of real windows, and random initialized weights are sufficient for
resource measurement; loss values must not be compared as model quality.

## Reproduction and artifacts

```bash
scripts/run_tandem_fno_h20_resource_benchmark.sh \
  tandem_fno_h20_resource_20261003_final 2 5
```

The runner rejects an existing output, active FNO training, a non-pinned image,
or a train split other than the expected 28 HDF5 files. The benchmark catches
OOM per specification and stops the wider probe if width-48 safety fails.

Authoritative artifacts:

- `artifacts/benchmarks/tandem_fno_h20_resource_20261003_final/benchmark.json`
  — SHA-256 `630deb2d8acbef743bb187bff79168d82559ed3b4d66eaf994eaf608cf6134f1`
- `artifacts/benchmarks/tandem_fno_h20_resource_20261003_final/benchmark.csv`
  — SHA-256 `aa67a9359bb6eabdb1191f8699f945b9f3f90b684d06395058c6a2b3e899a637`
- `scripts/benchmark_tandem_fno_h20_resource.py`
- `scripts/run_tandem_fno_h20_resource_benchmark.sh`
- `tests/test_tandem_fno_h20_resource_benchmark.py`

The earlier `tandem_fno_h20_resource_20261003` and `_confirm` reports are
retained as pilot/replicate provenance; the `_final` report is authoritative
because it also embeds the exact runtime, image ID and parameter counts.
