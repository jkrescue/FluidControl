# B04 curated train-data terminal review

Independent data/interface review PASS. This is usable train-only data, not a trained candidate, prediction admission, or additional physical-control result.

## Bound execution and artifacts

- Conversion unit `fluid-control-p064-b04-long-excitation-curator-20261007.service`, invocation `777b61f9027244bbb069db9ee8acda90`, independently observed PID 0 / exited / success / exit 0.
- Approval SHA256 `da6587c6baea40077d7482031c0edadedefe61a3663ca861110532ce22213ea4`. Same-process operational CPU quota sequence was 1 → 4 → 8; the original approval was preserved. Dated amendment: `docs/P064_B04_CURATOR_CPU_QUOTA_AMENDMENT_20261007.md`, SHA256 `566efa8366323a729933e233ac32278d33a0d04e1c13ec85c0ecd63c01d9a824`. Memory 8 GiB, no swap, 50/22 GiB memory guards, 20 GiB disk guard and original deadlines remained unchanged.
- Output root `artifacts/p064_b04_long_excitation_curated_20261007`.
- `result.json`: SHA256 `81eb5d8b2ad985b4949e4d611df41561e75cf32a6bb8cd92ed5b8bf98fe4f487`.
- `manifest.json`: SHA256 `5ec761ecca307d70da3660c7cf8e3cf4e6c75620377992beb41f775599a0ebf7`.
- Actual sole train HDF SHA256 `36da59d57412eee3958d5ded89bd3441453233734f5e76a2ca5c6e066f6fe96c`; its inherited record/file stem is the VTK view name, not the unused config.case label.

## Independent checks

All 801 HDF times equal the actual VTK TimeValue values cast to float32. All action and four-force labels exactly match interpolation at the original VTK times, including true baseline t=120 forces. All saved state/mask/action/force/time/coordinate arrays are finite. Original normalization bytes remain SHA256 `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`, with no refit.

The actual official HDF5Reader and project TandemRolloutDataset passed the existing first/last H100 checks (starts 0 and 700, 202 decoded frames). The reviewer separately sampled only the first and last VTK meshes through official Mesh/BVH, then applied the original mask and valid-domain pressure-mean subtraction. Both sampled states have maximum absolute difference 0 and masks are exact. No full 801-frame re-sampling was performed.

Observed masks are binary and identical across all 801 frames. Coverage is 0.9869384765625 for every frame: finite saved arrays do not imply every grid cell was valid before masking. This observation is not a newly added scientific gate. Minimum recorded available memory was 113.24556350708008 GiB; minimum free disk was 195.09226989746094 GiB.

## Audit lifecycle and limitations

Checker `/tmp/audit_b04_curated.py`, SHA256 `0ae1706f5c38a00840df7257a90405309ca752d9fbd9105d936eadb76d94d12c`, was reviewed by Sota and Lead. The first audit invocation `f58dd53c1d4b41c5b2ab737bca4158cf` failed because its process lacked project PYTHONPATH, before Reader/mesh checks. Its logs remain in `artifacts/p064_b04_curated_independent_audit_20261007`; this is an audit-launch failure, not a conversion/data failure.

Lead authorized an environment-only retry with the same checker and absolute repo src:scripts PYTHONPATH. Audit R2 unit `fluid-control-p064-b04-curated-independent-audit-r2-20261007.service`, invocation `98b7fad7aa1b415ea6d96dce62d9722f`, completed exit 0 with CPU 1 / 8 GiB / no swap / 180 seconds / CUDA hidden. Receipt `artifacts/p064_b04_curated_independent_audit_r2_20261007/receipt.json`, SHA256 `21660346df4739ab109b3500867bf5a83068074317d52d7cd4d4e6c6e731a3ca`, includes per-frame mask coverage. Warp CUDA error 100 was an expected CUDA-hidden import warning; the CPU audit succeeded.

No model forward, optimizer, training, CFD, or source-data write occurred in this audit. A future fixed-schedule training spec may bind these actual artifacts, but this review itself does not authorize training or change the default B controller or historical prediction failures.
