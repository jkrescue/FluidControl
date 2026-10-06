# P064-B H25 bounded training R2 launch

- Approval: `docs/P064_B_H25_TRAINING_R2_APPROVAL_20261006.json`, SHA-256 `852736b03edbd987da0e4f5a0d0cc55f1b1145fa980c989dc5b9cab298346f02`.
- Immutable source manifest: `artifacts/fcp064_b_h25_bounded_source_v4_20261006_immutable/source_manifest.json`, SHA-256 `69b0d2e37c0c01ac1dba8b73d48a2296929ab2e19dd8ac375e2812892273e2b5`.
- Reviewed runner: SHA-256 `032da059f9b62d52ab72148070b544ccdbdb3a5815cf172d8815162ffc0b3384`; independent CPU review accepted 9 tests.
- Unit: `fluid-control-p064-b-h25-training-r2-20261006.service`.
- Invocation ID: `9449ac16be65406eabad0515e88b6513`; initial main PID `2239224`.
- Output: `artifacts/fcp064_b_h25_training_20261006_r2` (exclusive at launch).
- Resources: 48 GiB memory, no swap, 800% CPU, 1024 tasks, 1920 s runtime, 120 s stop grace; inner reviewed contract uses 32 GiB allocator cap and MemAvailable 80/22 GiB startup/runtime guards.
- Numerical protocol remains fixed: P064-B parent, frozen aerodynamic model, H25 flow-only objective, 256 fixed original-order-prefix windows, eight-window accumulation, 32 AdamW updates, and the already verified H10/1368 parent scales. The failed R1 output and unit were preserved.
- R2 produced real `group_complete` events on the same invocation, proving that the truthful H25 records passed the repaired eight-window accumulator and reached optimizer updates. The `window_complete.rollout_steps=100` field records the source HDF window length; the optimized objective horizon remains 25 and is validated inside each accumulation record.
- This is a launch milestone, not terminal success, scientific admission, or permission to extend/select checkpoints.
