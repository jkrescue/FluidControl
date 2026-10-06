# P064-B H25: independent one-update resource probe review

Engineering ACCEPT for the actual one-window probe, not training completion or scientific admission. Unit `fluid-control-p064-b-h25-resource-probe-20261006.service`, invocation `74f31b29dfae41e38e509e7761788be5`, independently observed PID0/exited/exit0.

- Approval `docs/P064_B_H25_RESOURCE_PROBE_APPROVAL_20261006.json`: SHA256 `345f2cc3e1189d5a263a017a5e29f7829cabbae823e8d3fba822173541663b26`.
- Result `artifacts/fcp064_b_h25_resource_probe_20261006/payload/result.json`: SHA256 `a0dc4d689d02e98c11df972a555bf7bccee9bc8c2e994fc9aa34ea851b9b90b7`.
- Actual scales receipt bound: `4b8d28506bad47d76753848094c1b00195f96bb432cb9d96ef9031c8fdad2a8b`.

Independent read-only checks confirmed the result's embedded specification equals the saved approval, all 428 source hashes/source manifest/configuration/parent/audit bindings, and the actual scales receipt. The probe used the same first train window as that receipt, 25 rollout steps, complete 25-step field gradient, and 24 force terms with flow-gradient dependence. All 30 recorded parameter gradient norms are finite and nonzero. One Adam update occurred; the flow initial digest matches the scales parent and differs from the terminal digest. Frozen aero digest matches the scales receipt. Model saving is false and the payload contains only `result.json`; validation/test/scientific admission flags remain false.

All four original parent model/state files were independently rehashed against the manifest and remain unchanged. The actual Docker parent mount is read-only. No model deserialization, HDF read, or forward/backward rerun was performed during this review; tensor/gradient checks use the actual saved execution evidence.

Worker elapsed time is 10.006069237017073 seconds. Measured CUDA peak allocated/reserved bytes are 6,745,159,168 / 7,629,438,976. Worker minimum MemAvailable is 104.94927597045898 GiB; the 11 external resource samples have minimum 112,668,164,096 bytes. Actual container Memory/MemorySwap both equal 48 GiB; configured allocator ceiling is 32 GiB. Saved container exit is 0, OOMKilled=false, and the exact CID is absent from the current container list.

The inherited `optimizer_memory_accounting` warning describing a no-update probe is inapplicable here: this actual probe reports optimizer_created=true and optimizer_peak_measured=true, performs one Adam update, and measures its peak. The warning is retained as historical source text, not interpreted as evidence that Adam was excluded. No source mutation or rerun was needed.

Training pending `artifacts/fcp064_b_h25_bounded_training_preparation_20261006/PENDING_TRAINING_SPEC.json`, SHA256 `1dc4dd4dbedbc812abb85fe8d4207a3ee44df4267e930d6b325c81a906950230`, differs from the approved probe only by mode=train, actual probe-receipt binding, and pending authorization/status fields. Numerical protocol/source/resource fields are unchanged. This is preparation acceptance; the planned 256-window/32-update training still requires separate Lead execution authorization. A one-window peak does not guarantee the complete run's peak or model quality.
