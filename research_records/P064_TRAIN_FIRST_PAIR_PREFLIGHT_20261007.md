# Train-only first-pair preflight

This CPU-only check completed successfully; it does not authorize training.

- Unit `fluid-control-train-first-pair-preflight-20261007.service`, invocation `0cbbba94549e46f6b2ff7e850f58f373`, actual PID0/exited/exit0.
- Receipt `artifacts/p064_train_first_pair_preflight_20261007/receipt.json`, SHA `6269562b1c4cc92f32853b9c7260e21f702f7ff8249125071fee27365a2be4ce`.
- Executed source `/tmp/audit_train_first_pair_20261007.py`, SHA `76adf10c4ff2e1bb20a47e20f74b011942a9a6e9909a5db0ab69738f56271501` (identical archival copy beside receipt).
- 2 GiB MemoryMax, swap0, CPU1, CUDA hidden, 120 s limit; actual elapsed 10.527 s, minimum sampled physical Available 122922868736 bytes. No model, optimizer, new CFD, or data modification.

Only the first two frames of each of the 20 manifest-listed train HDFs were decoded. Full-file byte SHA was streamed before/after, without decoding later arrays. Train split, normalization and 20 HDF hashes match the existing manifest. All per-file metadata, first-two-frame hashes, times, actions and four forces are retained in the receipt.

Within each b00/b02/b04/b06 group, all five q0 fields, both masks/times, current omega0 and force0 match the zero case exactly. There are usable matched initial states. Stored t1 equals float32(t0+0.1); stored omega1 exactly equals the recorded action-table interpolation at that stored time.

However, the first rate-limited ramp produces only three distinct endpoint actions per phase: approximately −0.1, 0 and +0.1. The ±0.375 and ±0.75 eventual targets of the same sign are duplicate first-step actions/force responses. Deduplicate to eight nonzero-minus-zero contrasts across four initial states, not sixteen independent contrasts or five distinct first-step amplitudes. Shared phase/zero controls also mean these eight are not statistically independent observations.

The stored time/action quantization differs from nominal endpoints by up to 6.1035e−6. This check establishes saved-HDF pairing, not a new raw-solver action/force clock audit; legacy interpolation limitations remain. It does not justify a universal action-response claim or solve coverage beyond four initial states. An auxiliary-loss proposal must preserve these limitations and avoid double weighting duplicate ramps.
