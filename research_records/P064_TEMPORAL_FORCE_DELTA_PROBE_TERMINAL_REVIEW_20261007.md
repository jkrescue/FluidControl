# Temporal-force-delta single-window probe — independent terminal review

Engineering execution verified; **not a saved candidate, development evaluation or promotion**. No model forward or optimizer was rerun by this review.

## Actual execution and binding

- Host R3 unit `fluid-control-p064-temporal-force-delta-probe-host-r3-20261007.service`, invocation `7b7e12c6f9374a1da4b8ca7dd44a6030`: PID0, exited, success/exit0.
- Approval `docs/P064_TEMPORAL_FORCE_DELTA_PROBE_HOST_R3_APPROVAL_20261007.json`: `34fddd2c641413f9afea6f0b9d1a7615493cc0d23e2ad178a3c1d59360637aa7`.
- Result `artifacts/p064_temporal_force_delta_probe_host_r3_20261007/payload/result.json`: `200b7763b1fa73bd1380fbcc00c4645f7096f2cd0501a3e6482816b2d071bb17`.
- Supervisor receipt in the same output root: `c4c82f24a8188b8aa94a54c7f438879a31231ef7a8cbdee6e0c8b9067df33d5a`.
- Worker `e5edf3a71e6fbacc69bb5770aa0b05b8ef187f44eb04a7ac0e883ef030bc1d1c`; residual core `cb7343461cd0819a9b5aab5db22ff3a90635bbb75f32a9ffcc48dc87823ef716`; P013 objective `f3cf4b9a745cc0cbee39db9385cbfc398e4834e25bc487d8fb2d6eca07b483d7`. All approval-bound inputs and worker/supervisor were independently rehashed unchanged, including parent checkpoints, normalization and HDF.

Exactly one mechanically fixed b00 train window, start0/H100, same K1 parent in both arms. HDF SHA `45041e79e70043838763e5dd8da0fe9c02dba4cfe6df01356f8dcd1389e32d8f`. Official CPU reload/Reader and precision preflight were separately exercised; production uses high/TF32, unchanged normalized absolute-force loss and two real AdamW steps total, one per arm. Only result and supervisor JSON exist in the output: no checkpoint or candidate saved, no dev/frozen access or CFD.

## Recorded metrics, independently checked arithmetic

| Arm / stage | H1 balanced loss | AR balanced loss | Total |
|---|---:|---:|---:|
| Absolute initial | 0.004305274226 | 0.009931897745 | 0.007118585985 |
| Absolute after one step | 0.004137977026 | 0.009589541703 | 0.006863759831 |
| Residual zero-head persistence | 0.011189917102 | 1.237297534943 | 0.624243736267 |
| Residual after one step | 0.011189972050 | 1.237010240555 | 0.624100089073 |

Initial and backward-pass recorded losses match exactly; `total = .5*(H1+AR)` agrees within float32 summation tolerance. Total decreases3.579730% for absolute and0.0230114% for residual on this same training window. Residual H1 slightly increases; its after-step loss remains far above absolute. The persistence initialization is not an improvement in this window.

Both arms contain28 finite gradient records and30 parameter-update records; the two frozen lifting biases have exactly zero change. Absolute has28 nonzero-gradient/changed tensors, full update L2 `0.0005036214351490268`. Residual has only the final weight and bias with nonzero gradients and measured changes, update L2 `0.0000035492995061524386`; hidden gradient elements are zero, as expected from the zero force head at the first backward. Preclip norms are12.1043272 and55.2733040; clip bound1 and learning rate1.5625e-7 are unchanged. AdamW decay can in principle change zero-gradient parameters; actual measured hidden changes here were zero, not evidence of learned hidden dynamics.

The full-H100 cotangent/recompute method was independently CPU-oracle tested, including multiple batches/chunk sizes. Saved result contains aggregate losses and gradient/update inventories, not raw predictions or before/after model tensors. Therefore this terminal review checks arithmetic, finite values, hashes, runtime and reviewed source wiring; it does **not** independently recompute the actual training losses or reconstruct optimizer state from saved checkpoints. There are deliberately no saved checkpoints.

## Resource and failure record

Actual unit24GiB/noSwap/CPU1/tasks64,600s worker/630s outer/20s stop. Allocator16GiB. Wall22.171348926s;11 supervisor samples give minimum MemAvailable105.288032532GiB. Highest observed allocated7,084,155,904 bytes and reserved7,511,998,464 bytes. The residual phase's reserved peak includes allocator cache carried from the absolute phase; it is not isolated residual model demand. No worker remains.

R1 b40 invocation `aa850705c7c04a2a81e3cce45fea77dd` failed on first model migration, before forward/optimizer. Host R2 `1bd24f27e6444172bdf09e069eb1745d` completed frozen-flow construction, then failed in the absolute arm's initial-loss forward at the explicit6GiB allocator cap; no backward/update completed. These failures are retained, not scientific rejection of residual parameterization.

The bounded host and corrected b40 migration diagnostics both succeeded with identical parent tensor digest/bytes and equal `properties.total_memory == mem_get_info.total == 130663821312`. They do not support the hypothesized mismatch of memory-total denominators. The b40 diagnostic's first attempt instead failed on a readonly cache path, before migration; the later corrected diagnostic is separate. R1's precise cause remains unproven. R3 changed only approved resource limits/identity from R2, not the science.

## Decision support

The measured resource envelope and real one-step gradients permit planning a fixed-budget test, but this probe supplies no evidence that32 updates suffice, no generalization estimate, and no basis to replace B or promote a model. Any subsequent finite budget must be declared in advance and distinguish initialization prior from improvement over each arm's own initialization. Basic canonical CFD delivery and all previous prediction FAIL conclusions remain unchanged.
