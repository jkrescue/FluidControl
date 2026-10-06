# P064-B H25 R2 training: independent terminal review

Engineering ACCEPT, not prediction-accuracy admission. Actual unit `fluid-control-p064-b-h25-training-r2-20261006.service`, invocation `9449ac16be65406eabad0515e88b6513`, was independently observed PID0/exited/exit0. Approval SHA256 is `852736b03edbd987da0e4f5a0d0cc55f1b1145fa980c989dc5b9cab298346f02` (`docs/P064_B_H25_TRAINING_R2_APPROVAL_20261006.json`). R1's first-window H10-versus-H25 accumulation contract failure remains preserved; it completed no optimizer update.

Actual output is `artifacts/fcp064_b_h25_training_20261006_r2/payload`:

- `result.json`: SHA256 `557e0792eee538d8152c4997032309423a1197067c7198089768d0ddb40f5cf7`.
- `dual_model_manifest.json`: SHA256 `decf5f52bc0087fe07f2d3969e39604f19ea273c191ad193660c0af4c02670c0`.
- `training_protocol.json`: SHA256 `8489c0b04aa96bb0ffbf2034886f875830d49970e3321c5d924ff47e7258cd0a`.
- Flow archive/state: `b93d2d11c2f94eb4f5a207a4adca38a11ef97f2f82cdf172c48d4c915eca47ab` / `6a7bd826e1743e963772d5b676349059bb6a095b9010574a1c00311af0bb2810`.

Independent saved-record checks: exactly 32 sequential update records, eight windows each, 256 window-complete events and 32 sequential group-complete events. Every window identity equals the corresponding first-256 entry of the verified scales run. The selected-order SHA matches the fixed protocol. Every actual training record uses a 25-step objective and full 25-step field gradient; metadata `rollout_steps=100` describes the original sampled window, not the optimized horizon. All 32 groups report 30 finite, nonzero parameter gradient norms. All group field/force and normalized-contribution means were independently recomputed exactly.

All 428 executed source hashes and bound configuration/parent/audit/scales/probe receipts were rehashed. All four saved model/state files match the candidate manifest; training-protocol hashes agree across result and manifest. All four original parent files still match their original hashes. Saved aero archive/state are byte-identical to B; frozen aero tensor digest matches scales, whereas flow initial digest matches the parent and terminal digest differs. The executed trainer reports `official_fresh_reload_verified=true` and `model_saved=true`. This review verifies that execution evidence and saved bytes; it does not rerun official model deserialization or inference. Optimizer update counts here are checked against the executed result/log records, not independently re-executed optimization. The accumulation helper's inherited `optimizer_step_performed=false` field describes that helper before the trainer's subsequent update; the trainer's sequential `update`/`consumed_windows` fields describe actual updates.

Elapsed worker time: 299.83931758999825 seconds. Minimum worker MemAvailable: 103.48341369628906 GiB. CUDA peak allocated/reserved: 7,329,976,832 / 8,405,385,216 bytes. Actual container Memory/MemorySwap both equal 48 GiB, terminal exit0, OOMKilled=false; exact CID independently absent. No validation/frozen-test access or scientific admission is claimed.

Separately reviewed quick-evaluation entry `fce5028fcb4b8468fba50e25304d55b8a24dae2ec318c3cd12b374b598f4862e` reuses the unchanged original force-window worker and reviewed H25 loader overlay. Three CPU tests passed independently. It may proceed only under Lead's separate conditional execution approval and final actual metadata bindings; no prediction result exists from this training review. Existing successful physical closed loops used the prior B policy, not this new flow candidate.
