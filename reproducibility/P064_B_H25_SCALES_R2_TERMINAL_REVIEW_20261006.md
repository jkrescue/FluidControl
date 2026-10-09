# P064-B H25 preparation: independent R2 scale receipt review

The scales pass completed successfully. This is loss-normalization measurement, not H25 training, model improvement, or scientific admission. The first attempt's loader-kind failure remains preserved; no retry was performed by this reviewer.

## Executed identity

- Unit: `fluid-control-p064-b-h25-scales-r2-20261006.service`; invocation `88e5e31e611c45dab28dbe6c3ad11c8c`. Independently observed `MainPID=0`, `active/exited`, `ExecMainStatus=0`.
- Approval: `docs/P064_B_H25_SCALES_R2_APPROVAL_20261006.json`, SHA256 `cb60cf7bcc146a51f085957d6c4c68a7792fe41803fd925fed1aa7c1b3c1d77e`.
- Result: `artifacts/fcp064_b_h25_scales_20261006_r2/payload/result.json`, SHA256 `4b8d28506bad47d76753848094c1b00195f96bb432cb9d96ef9031c8fdad2a8b`.
- Immutable source manifest: `artifacts/fcp064_b_h25_bounded_source_v3_20261006_immutable/source_manifest.json`, SHA256 `e4ca06d0b26d371ebbcdf4323c840e3dd64d8c80a16c8f3f682d6fff7254eb76`.

## Independent saved-record checks

All 1,368 records cover 44 train cases, with dataset-family counts 720/408/240. Sorting the unique `(dataset_index, case, start)` identities to reconstruct global indices and hashing their observed sequence reproduces `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`. All identities match the 1,368 journal-file window events; group events are exactly 1–171. Each measured objective has ten finite, nonnegative per-step field and force values.

Independent `math.fsum(raw_values) / 1368` exactly reproduces both saved scales:

| Component | Fixed scale |
|---|---:|
| Field | 0.001456146538716282 |
| Four-force | 0.003783821943311762 |

The approved source specification exactly matches the result's embedded specification. All 428 source hashes, source-manifest hash, configuration, parent manifest, training-audit receipt, and three dataset manifest/normalization bindings were rehashed successfully. This review did not reread HDF arrays, deserialize models, or rerun inference.

The receipt reports zero optimizer steps, no optimizer creation, no model save, and no validation/frozen-test access. The payload contains only `result.json`. Flow initial/terminal tensor digests are identical (`89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`); frozen aero digest is `3cdac90fe62da9d483a6d38eb2a28b23a52b007028c1dc11c28f049f7f77b851`. These are verified receipt consistency checks, not an independent model reload. Actual precision is historical `high`, CUDA matmul TF32=true, cuDNN TF32=true.

## Resource and cleanup evidence

Recorded worker elapsed time is 541.6890858269762 seconds. Across 274 external resource samples, minimum MemAvailable is 120,598,044,672 bytes; the worker's more frequent samples report minimum 112.2823486328125 GiB. Actual container Memory/MemorySwap are both 48 GiB (no additional swap allowance); the configured allocator ceiling is 32 GiB. Measured CUDA peak allocated/reserved bytes are 521,055,744 / 541,065,216, not the configured ceiling. MemFree is recorded honestly and is not the UMA admission criterion.

Saved container terminal evidence shows exit 0 and OOMKilled=false. Its exact CID was independently absent from the current Docker container list. No model/CFD execution or resource mutation was performed during this review.

Conclusion: the actual R2 scale receipt is internally reproducible and suitable for binding a separately authorized H25 resource probe. Training and downstream prediction/control claims remain untested by this scales pass.
