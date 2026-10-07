# Absolute64 terminal engineering review

Independent engineering ACCEPT, not prediction admission. Training unit `fluid-control-p064-absolute64-arm-b-20261007.service`, invocation `3c9d8e1bb01649a3b11c892159a73a2b`, was independently observed PID 0 / exited / success / exit 0 before candidate access.

## Immutable bindings

- Approval: `docs/P064_ABSOLUTE64_ARM_B_TRAINING_APPROVAL_20261007.json`, SHA256 `705c596861cdfaafa9bcd5e10ffeea2f37cbe36cf59c816e019183e92a44d500`.
- Result: `artifacts/p064_absolute64_arm_b_20261007/result.json`, SHA256 `2d715ff7d474e06ab0fcec2bc22056a020e37e0adf29645785b6fe89f44fbad8`.
- Manifest: `artifacts/p064_absolute64_arm_b_20261007/dual_model_manifest.json`, SHA256 `2e3c778b9cec039487602eb38b54f78f42b0a2e0fb8e467790bc585aedc0545c`.
- Update32 comparison: `artifacts/p064_absolute64_arm_b_20261007_update32_comparison.json`, SHA256 `7ff4e3002f2061e3551a406369bb708403567fbb48f3e70289354540f3e59e14`.
- Reviewed trainer SHA256 `d24de464bd85732a89404cdb91eeef13fff5b17c96e2f73789d06e6ef39aecc4`; original 434-file source manifest SHA256 `05c00539ad03cc9059dc101cf9c9cab47d523a8b5bdae29b4af1cea88c66a9c0`.

## Independent checks

One CUDA-hidden CPU audit ran under unit `fluid-control-p064-absolute64-terminal-cpu-review-20261007.service`, invocation `b03dc70e418446a8ba8c5106b8d06b1a`, with 8 GiB memory / no swap / one CPU / 120 seconds, and exited successfully. Its two JSON evidence objects are in `artifacts/p064_absolute64_terminal_cpu_review_20261007/receipt.jsonl`. Checker SHA256 `fae7a44febce922e48b48bd68979fbb7afbe12c21ce4d90c4d2ac11dd6829380` extends pinned original checker `8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587`; seven CPU fixtures and independent source review passed.

The audit verified 64 ordered accumulation groups / 512 actual journal windows, two identical original B 256-window schedules, the original objective scalar aggregation, all 434 source hashes and direct input/source bindings, normalization/view manifests, final-only epoch2 checkpoint paths and hashes, and diagnostic counts `[0,512]`. CPU weights-only inspection found exactly 28 Adam states, each at step64 with the original hyperparameters and finite values. Both frozen aero lift biases equal the parent tensors; frozen flow checkpoint bytes equal the parent. Official save/fresh reload is bound producer evidence, not an additional independent model reload.

The durable update32 comparison and terminal result independently agree with historical B's saved tensor digest: `3cdac90fe62da9d483a6d38eb2a28b23a52b007028c1dc11c28f049f7f77b851`. The same uninterrupted optimizer continued only after this exact match. The historical package snapshot is unavailable; this measured agreement supports first-epoch comparability without inventing historical versions.

Training used the approved 16 GiB allocator / 24 GiB cgroup / zero swap. Recorded minimum MemAvailable was **106.82454681396484 GiB** (producer records `/proc/meminfo` kB divided by 2^20). Observed training MemoryPeak was **6677905408 bytes**. No resource-floor breach was found.

## Limits and handoff

This is engineering completion, not scientific improvement. The audit did not rerun model forwards, independently reconstruct training loss, or rehash the large HDF payload. It checked recorded scalar consistency and the existing data/source identities. Only the final candidate may proceed to the separately conditionally approved fixed 16-origin / 80-endpoint development comparison plus the original fixed-six retention assessment. Preserve all original thresholds and B as the retained baseline; no PPO or CFD is authorized by this report.
