# FC-E065 — Independent b00 conversion and training-view review

Status: CPU DATA PREPARATION VERIFIED; NOT TRAINING APPROVAL or scientific admission.

The original conversion unit `fluid-control-b00-controlled-train-conversion-20261006.service`, invocation `f5ee31dd92624f0980a509084de9c756`, was independently observed PID0/exited/exit0 before opening its HDF. It ran 10:17:37–10:38:17 UTC (1240 seconds). Approval SHA `4fa7192e13bf7ad3a141bffb483710e2400fd8ee243caa60a6e67ab695927686`; executed source SHA `f96c882a90e7ecaf4a2f8a5fc327764909ab42b4e6bbb11cde8e99205488b075`.

## Actual terminal evidence

- Result: `artifacts/b00_controlled_train_conversion_20261006/result.json`, SHA `f24f2fbc8b283c0781b2a01189d291b43da0203e9ede28208ec575173bbe0bd9`.
- Physical HDF: `b00_projected_ppo_train.h5`, SHA `45041e79e70043838763e5dd8da0fe9c02dba4cfe6df01356f8dcd1389e32d8f`: one train trajectory, 801 frames from148 to228; state shape801×3×128×256, one common binary mask/grid.
- Independently rehashed all1614 original source files against the approved source inventory (1602 U/p plus12 metadata), exact inventory identity, executed dependencies, original progress, and old conversion proof. No original source content or permissions changed.
- Actual pinned official HDF5Reader read all801 frames. Each state/mask/time/omega/four-force array matched HDF values exactly, with finite checks. All801 actions and four-force labels independently matched FP32 conversion of the original real-CFD progress: frame0 initial observation/omega0, frame j>0 applied action and output force of step j−1.
- Times are monotonic; maximum deviation from148+0.1j is `6.103515630684342e-6`, retained VTK float32 representation rather than exact decimal timestamps.
- All48 retained old controlled NPZ packets were independently SHA-checked and compared exactly with corresponding HDF state/mask/time/x/y. The753 newly sampled packets were deleted after streaming writes by the approved converter. They were NOT independently compared afterward, nor resampled. Their field provenance rests on the reviewed per-frame converter checks, original-source hashes and actual official-reader roundtrip; this report does not claim an independent second Curator conversion.
- All16 saved export-container terminal records show exit0, OOMfalse, not running; exact16 IDs were absent from Docker. Root-owned batch scratch remains retained intentionally. No new CFD solver, model, GPU inference, optimizer or normalization fit occurred.

Conversion resources: actual MemoryMax12884901888B, swap0, CPU quota1; MemoryPeak2883301376B. All20571 saved resource observations had MemAvailable≥121431912448B. No physical-reserve failure was observed.

## Exclusive dedicated training view

Created `artifacts/b00_controlled_train_dataset_view_20261006` only after the conversion terminal checks. It contains one hardlink under `train/`, byte-identical original train normalization, manifest and CPU verification receipt. Source inode permissions were NOT changed. Metadata/directories are non-writable; the HDF hardlink retains source permissions. This is a dedicated view required to be used read-only, NOT a claim of an OS-enforced read-only mount. Consumers must prohibit writes and bind/check hashes.

- Manifest SHA `97a82e2a157b87ecf9b9ea626ad079852a8d29ba89628ca5034ef99545929a5f`.
- Normalization SHA `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`, copied original bytes without refitting.
- Verification receipt `verification.json` SHA `41e759289dd993c9a0df0fc7013a3cce7a47cd81ebb5abdc2fc3370a38d1ef9e`.
- Actual official-backed `TandemRolloutDataset(..., train, H100, stride1, all4forces)` exposes exactly701 starts0..700. Both windows0→100 and700→800 were loaded via public `dataset[index]`; normalized current/100target fields, 101 action endpoints,100 future force targets and masks matched independently constructed expectations exactly. No off-by-one omission of frame800.
- HDF SHA was rechecked after DataPipe use, and source mode/mtime remained unchanged. Label provenance explicitly says real closed-loop endpoint coefficients, not legacy interpolated-HDF labels. This is train-only; b01/b03 development data are not in the view.

The independent CPU check used an8GiB/noSwap/CPU1 unit and hidden CUDA. Successful R2 invocation `166133d841f24be2a409caaec40e0b89` exited0; script SHA `6677e5915c0b7dac30a5545cedc07fe8bc34f7713d609395e6e65cf7cb313f5a`, elapsed4.37s, minimumAvailable122889850880B. Its MemoryPeak was unavailable, not zero. Expected hidden-CUDA Warp initialization warning did not prevent official CPU reading.

An earlier audit invocation `281d0c3749d147be82a97efcb685e236` failed before HDF reading/view creation because the audit helper compared an inventory `{sha256,size}` object to a digest string. The exact first file hash matched; only audit parsing was corrected, with the failed unit retained. The successful conversion was not rerun.

## Scope

This closes data-engineering readiness for the separately reviewed candidate A/B preparation. It does not authorize GPU training, establish improved FNO accuracy, admit a new policy, or alter the three measured physical-phase successes, the original10% bias criterion, or K1 H100 FAIL.
