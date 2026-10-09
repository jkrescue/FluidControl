# FC-E090 b02 controlled-train conversion — independent terminal review

Conclusion: **ACCEPT data conversion; ready for a separately authorized train-view construction (可建 view)**. This is not training, model admission, new CFD or approval to start a subsequent experiment.

## Actual execution and bindings

- Unit `fluid-control-p064-b02-controlled-train-conversion-20261007.service`, invocation `f7a7550e035e4ee482205379eefa016a`: independently observed MainPID 0, Result success, ExecMainStatus 0; terminal progress 801/801, last index 800, official_reader_verified true.
- Approval `docs/P064_B02_CONTROLLED_TRAIN_CONVERSION_APPROVAL_20261007.json`: SHA256 `64e910fc20c7f5beeb0805ad159be3e0ff3e8d599f4560b7a0f42149aa53766a`.
- Result `artifacts/p064_b02_controlled_train_conversion_20261007/result.json`: SHA256 `3faf153b2a9857e880634be3347f6596834b3609a5235007f668b141d10b5ade`.
- HDF `artifacts/p064_b02_controlled_train_conversion_20261007/b02_canonical_ppo_train.h5`: SHA256 `96140954487b40f8e7a8dd37cdbb90221fce5656120dedb0301865a046f4377f`; 341,355,532 bytes.
- Independent CPU audit unit `fluid-control-b02-conversion-independent-audit-r2-20261007.service`, invocation `4b3a0f4a54034488b8328a6bdca8ca6c`; CUDA hidden, 8 GiB/no swap/1 CPU/120 s bound. No model, conversion or CFD rerun.

## Independent checks actually executed

Rehashed all 1,614 selected original files (1,602 U/p plus 12 metadata files), including sizes; the terminal inventory equals the approved inventory. Rehashed the bound driver, core, sampler, official-reader adapter/source, original normalization, source result/approval and progress. Selection is exactly controlled branch `mpc`, indices 0–800, with no zero-branch training data.

Read the completed HDF: state `(801,3,128,256)` float32; mask `(801,1,128,256)` uint8; all fields finite, masks binary and identical over time, static x/y finite. All 801 action labels and four-force labels equal the source observations after the specified float32 conversion. Independently reread the 1,600 controlled raw force files (front/rear, 16,000 samples each); all 800 post-step force labels equal the actual interval endpoints exactly in float32. Frame 0 uses the recorded initial observation and zero initial omega; it is not a fabricated post-step label.

Times are strictly increasing from nominal 106 through 186 at 0.1 D/U. Maximum absolute VTK float32 time rounding is `6.103515630684342e-06`; do not claim exact decimal timestamps. The original train normalization is unchanged/not refitted. Force labels are real CFD endpoint coefficients, unlike legacy interpolated-HDF labels; action clock conventions are unchanged.

The pinned producer actually ran official HDF5Reader and compared all 801 frames, every field/dtype/shape and static grid to HDF contents before writing its terminal result. This review checked that executed source, terminal receipt and HDF arrays; it **did not independently resample all VTK fields or rerun the official Reader**. Grid/mask equality across every sampled packet is a producer check in the bound source, not an independently rerun Curator operation.

All 17 exporter receipts show exit 0, no OOM, exact pinned OpenFOAM image and 4 GiB memory/no excess swap; all 17 recorded CIDs are absent in actual Docker state. Root-owned exported scratch remains intentionally retained, not claimed deleted. Host unit has exact 12 GiB MemoryMax and swap 0, peak 2,960,928,768 bytes. All 21,276 recorded resource samples meet the 22 GiB available-memory floor; minimum is 122,282,983,424 bytes. No optimizer/model/CFD solve was executed by conversion.

## Audit history and scope

First independent audit invocation `75385b316cd54740946e460c769870be` stopped before HDF reading because the audit script treated inventory entries as digest strings instead of `{sha256,size}` dictionaries. R2 corrected only that audit parsing and then passed; conversion outputs and sources were not modified or rerun. This is an audit-script engineering error, not a scientific or conversion failure.

This entire 801-frame b02 trajectory is train-only. Earlier b02 short exploratory data already appeared in training; no unseen-test or statistical-independence claim is made. A thin view may reference these arrays with the original train normalization once separately authorized. No view or new training was created by this review. Existing physical-control results and surrogate prediction failures remain unchanged.
