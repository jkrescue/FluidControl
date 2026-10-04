# FC-P003C mixed-loss technical probe launch plan

Status: launch-contract draft only. GPU execution is disabled. This document
does not approve training, validation, PPO, CFD, or a scientific conclusion.

## Purpose and boundary

The bounded probe will test one real mixed update using the future reviewed
FC-P003C trainer path. It must combine one real regular training batch with
the fixed train-only dynamic pair `b00:multisine`. This is different from the
completed paired-only backward probe: the regular graph and the paired H100
chunks coexist in one optimizer update, so the old memory result cannot be
used as evidence that the mixed update fits.

The probe starts from the immutable Main-e2 parent and may update only an
in-memory model copy. It must not save a model, checkpoint, optimizer state,
or candidate. It performs one optimizer step only after the regular gradient
has been accumulated, its graph released, and ten true-state pair chunks of
10 endpoints each have been accumulated with scale `10 * chunk_length / 100`.
Gradient clipping occurs once after both contributions. Update equivalence
must be established before launch by reviewed CPU synthetic tests that compare
the mixed helper with its reference update. The real GPU probe performs only
one optimizer step on one scratch model copy: it does not run a second GPU
reference update. It may compare recorded component gradients with their
explicit sum before that unique step, but doing so must not mutate the model.
The exact CPU comparison and tolerances remain owned by the reviewed
Surrogate implementation and must be fixed before GPU output is observed.

## Immutable scientific inputs

- Official image ID:
  `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Main-e2 model:
  `8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240`.
- Main-e2 training state:
  `1e5d4c055812d8f92bc55f58708e839f7cc776071849544f6a26d3d62053cbd0`.
- dev30/train8/train16 manifests respectively:
  `5213c7bb...1ddd2`, `a0bd0e3b...2f35`, `7c62dab9...cf5b`.
- Train-only normalization:
  `f1b4607e...92bc1`, byte-identical across all mounted training sources.
- Dynamic-pair manifest:
  `b756c6d7...2d28c`.
- Fixed action HDF:
  `dynamic_train8_b00_multisine.h5`, SHA `400b3b5f...9e9f`.
- Fixed zero HDF:
  `matched_start_acquisition_train_b00_zero.h5`, SHA `243caa79...a01`.

The launcher must recompute the full hashes, not rely on the abbreviated
forms above. No validation or frozen-test directory may be mounted. The
regular loader receives only the real train leaves from dev30, train8, and
train16. The paired loader receives only the fixed action/zero train leaves,
the pair manifest, and the same train normalization.

## Source and runtime contract

Surrogate owns the trainer, configuration, mixed-loss helper integration,
and tests. Compute will bind their final reviewed commit and exact file SHAs
into a fresh source snapshot. The snapshot manifest must cover every project
module imported by the probe and trainer; the launcher, guard, resolved
configuration, image ID, model/state, manifests, normalization, and HDFs are
verified before output creation or GPU initialization.

The official PhysicsNeMo DataPipe/DataLoader is used for the real regular
batch and dynamic pair. The probe records the actual regular case/start,
pair ID, tensor shapes, force-channel order, RNG/loader metadata, objective
kind, scalar and per-channel losses, gradient norms, clipping, update count,
runtime versions, and parent state hashes before and after. Any nonfinite
loss or gradient fails before the optimizer step. Parent files remain mounted
read-only and must be byte-identical afterward.

A CPU-only official-container preflight using the unchanged mixed-data loader
contract (`seed=20261003`, shuffle enabled, batch 1, H100, base stride 20,
additional stride 2) found 1,368 regular windows and selected
`matched_start_acquisition_train_b04_m075`, start step 180, dataset index 0 as
the first batch. The execution launcher must require this exact metadata and
the observed shapes state `[1,3,128,256]`, target state `[1,100,3,128,256]`,
omega `[1,101,1]`, force `[1,100,4]`, and mask `[1,1,128,256]`. The final
receipt will bind the reviewed loader source and this CPU preflight, so a
loader/RNG change fails rather than silently choosing another regular batch.

Container identity is UID/GID 1000, matching Spark user `USER`; `HOME` and
cache paths use private tmpfs. The container is network-disabled, read-only,
capability-dropped, no-new-privileges, PID-limited, and receives only one
exclusive writable result directory. No host IPC namespace is shared.

## Resource plan and stop rules

- GPU: device 0 only.
- Allocator fraction: initially `0.25`; no automatic increase.
- Outer physical-memory guard: minimum `20 GiB` `MemAvailable`, 2-second
  sampling, plus a 4-GiB margin in launch admission.
- Container limit: `90 GiB`; CPU limit: 8; shared memory: 1 GiB.
- Batch sizes: regular 1 and paired 1; pair horizon H100; pair chunks 10.
- Budget: exactly one reviewed mixed GPU update on one scratch model copy,
  not two reference/candidate updates, an epoch, or a throughput test.

The outer guard must own and terminate the real container/trainer process if
the physical-memory floor is crossed. OOM, nonfinite values, SHA drift,
unexpected loader identity, missing graph-release evidence, more than one
optimizer step, any saved weight, or any validation/frozen visibility is a
hard failure. Failures and partial logs are preserved under their exclusive
generation; they are never overwritten or silently retried.

## Output and review sequence

The future execution root is a new generation below
`artifacts/fcp003c_mixed_loss_technical_probe_20261005/`. Before execution,
the immutable launcher will create an exclusive launch receipt binding all
inputs, the CPU synthetic equivalence receipt, and the reviewed approval
token. A completion receipt is written only after independent validation of
finite results, exactly one GPU optimizer step on the scratch copy, no saved
candidate, unchanged parent files, unchanged parent model outside the
in-memory copy, and the physical-memory floor.

Current `scripts/run_fcp003c_mixed_loss_probe_spark.sh` is deliberately a
preflight-only skeleton. It verifies the already fixed inputs and reports
that the reviewed Surrogate trainer/config binding is outstanding. It has no
Docker or GPU execution path. After Surrogate delivers the final interface,
Compute will replace the placeholder contract with exact hashes, add negative
tests and an immutable launcher copy, and submit it to Evaluation. A separate
Lead GO is required for the single GPU probe.
