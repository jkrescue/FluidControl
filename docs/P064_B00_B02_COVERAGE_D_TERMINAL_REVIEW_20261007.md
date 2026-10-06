# FC-E091 D25 coverage training — independent terminal engineering review

**ACCEPT engineering completion; suitable for binding the separately approved fixed development evaluation. This is not accuracy admission or authorization for PPO/CFD.**

Actual training unit `fluid-control-p064-controlled-coverage-d-20261007.service`, invocation `171686b7ec154a0194348d26fd736cec`: independently observed MainPID0, Result success, normal ExecMainCode1/ExecMainStatus0. Actual journal contains256 completed training windows and32 accumulation updates, not merely a running process or self-reported counter.

## Final identities

- Approval `docs/P064_B00_B02_COVERAGE_D_TRAINING_APPROVAL_20261007.json`: `fa8a99d22426f7c6e73fc56ff8f0c6062c0b11c0dcae146ee51f40aa7e32a3e8`.
- Output `artifacts/fcp064_controlled_aero_arm_d_b00_b02_20261007`.
- `result.json`: `2c01c3b24a11b30371d40ee9ef799c7fd01090d284778498919080e6323b8733`.
- `dual_model_manifest.json`: `93d579b910d5a03385a31fb44c0050fa0b8d498fb6c621289a8d1038b5c1741a`.
- `training_protocol.json`: `c9734853920a37d1ddb4e3012a1bb96e58be5f05c862ea36d563d734c1700e6b`.
- Independent CPU receipt `artifacts/fcp064_arm_d_terminal_cpu_review_20261007/receipt_r3.json`: `960159c0d99a3017bbd439ef2b107172d6310a94b7761d8cfd50bd13fe2d2fbc`.
- Audit script `/tmp/p064-terminal-review.RY8Rx3/check_p064_d_terminal_r3.py`: `ed7a06f2219f9039a4abf7853c15b88f9572aa6e231f6815abfcbaa488c8fd97`; derived from unchanged checker8b3cd86756f4fc5b36f19d0049969793bbb3e403bbeba8327324eb6b4e9e9587.

## Independent checks actually executed

After the exact terminal-unit gate, rehashed all441 source-closure files and all directly bound inputs, including original train manifests/normalization, K1 parent, schedule/order, b00/b02 manifests and conversion receipts. No live checkpoint was opened. Protocol, result and manifest agree; schedule SHA is `f3f32e70190b8864ca432dc88bacc7efd5c4317ab4a74a67ed6a22613edabfef`.

Independently compiled256 expected journal rows:192 original entries,32 b00 at within-update slot0 and32 b02 at slot4; each controlled source uses starts `i*700//31` for i0..31. All actual events and all saved per-window identities match, including dataset indices3/4 and exact start clocks. Checked32 updates of8 windows, journal interleaving, finite losses, independently recomputed group objective means and clip scale. Both fixed train-panel records are at consumed0/256. Historical original-order/data/precision remain unchanged; D changes coverage and behavior-policy/state-action mix, not just a causally isolated phase label.

Actual CPU `weights_only` inspection verifies28 Adam parameter states, every step exactly32, finite optimizer tensors, lr1.5625e-7/betas(.9,.999)/eps1e-8/weight_decay1e-4. Candidate model tensors are finite. Both frozen lift-network bias tensors exactly equal the K1 parent. Flow model and state checkpoint bytes exactly equal the parent; all four saved role files match manifest hashes. The executed producer reports actual official fresh reload success; the independent reviewer did **not** rerun official model deserialization/forward, training or HDF sampling. This distinction is retained in the receipt.

Training is aerodynamic FNO branch fine-tuning of28 parameter tensors, not only a last-layer readout; flow and two biases remain frozen. Original normalization and data-source identities were verified; this terminal audit did not rehash every original training HDF payload. b02's independently accepted conversion/view provide its data-interface provenance.

## Resources and audit history

Actual training MemoryMax12GiB, swap0, MemoryPeak7,251,238,912 bytes. Recorded minimum MemAvailable106.77628707885742GiB, above the22GiB runtime floor. Independent final CPU audit unit `fluid-control-p064-d-terminal-review-r3-20261007.service`, invocation `dcdb1d4579c04b79bfaf530cb63c4779`, actually exited0 under8GiB/noSwap/CPU1/120s/CUDAhidden. No optimizer update or model forward was performed by audit.

The schedule/identity fixture suite passed9 CPU tests. First audit invocation `de7b3614ff5649bdbe982a8dcc904bd9` stopped on the old checker's flat `source_root` approval schema; second `06e5dace04384be9903471bff06132cd` stopped on the source manifest's new `files` envelope. R3 maps these actual approved metadata schemas into the unchanged checks, retaining441-file hashing and all scientific checks. These were audit-adapter errors, not failed training; their outputs remain preserved. No production source or candidate was changed or rerun.

## Handoff

Fixed b01/b03 development may bind these actual identities for a separately authorized evaluation; no prediction benefit is inferred from completed optimization or save/reload success. Predeclared D selection still requires both pooled H1 force MAEs and fixed-six retention criteria against B. No checkpoint selection, budget extension, PPO/CFD promotion or threshold change is authorized here. Existing B/canonical physical successes, early failures, C50 rejection and complete surrogate prediction FAIL remain intact.
