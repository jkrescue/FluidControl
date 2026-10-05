# P026 terminal-audit approval JSON compatibility

This is an auditor compatibility correction, not a training, model, data or scientific-gate change.

The first K1 terminal audit, user unit `fluid-control-fcp026-k1-terminal-audit-20261006` invocation `a2f9e95a0aab4161b59ddf873d0be17c`, failed closed at `approved protocol` before writing a candidate-audit receipt. This failure is preserved; it is not a training failure or an integrity PASS.

Read-only recursive comparison found exactly one difference: `effective_protocol.gradient_clip_norm` is integer `1` in the preserved execution approval and floating-point `1.0` in the trainer-produced candidate protocol. The original audit compared serialized JSON, so it distinguished these equivalent numeric spellings. The frozen trainer explicitly uses `1.0`.

Unchanged original evidence:

- Execution approval: `artifacts/fcp026_history_training_k1_20261005/execution_approval.json`, SHA256 `1088285e4e13c7e3553009dd511436e9a5da976e93eed44d6bb0c28ab9515aa3`.
- Candidate protocol: `artifacts/fcp026_history_training_k1_20261005/candidate/training_protocol.json`, SHA256 `daf22b2464744509260f1eb9e0b20d3b80da484c8985f3a22887293bbb40cb30`. This already matches the approval's `effective_protocol_sha256`.

The correction adds `validate_approved_protocol` and changes only the approved-protocol comparison to use it. It requires this one field to have exact Python type int or float (never bool), be finite, and numerically equal the frozen trainer's expected clip norm. It normalizes a shallow copy of only that field to the expected value, then applies the original strict serialized comparison to the entire protocol. Other field types, key sets, values, candidate/trainer bytes and protocol hashes remain strict. Inputs are not mutated.

Validation: 20 portable synthetic tests cover K1/K4, both numeric spellings, no mutation, missing/extra keys, booleans, NaN/infinities, unequal values and other-field type/value differences. One additional test checks the actual preserved approval and candidate JSON byte hashes and reproduces the old comparison failure before verifying the narrow correction. Only this actual-artifact test skips when its evidence files are absent. Existing terminal-tool fixture regressions are also required before integration.

Canonical CPU regression: all 59 tests passed in 0.77 seconds (the 21 compatibility tests plus 38 existing auditor/reload fixtures), with CUDA hidden and bytecode/cache writes disabled. Git whitespace validation passed.

The old 416-file immutable terminal runtime is untouched. A separately reviewed new immutable source revision must bind this corrected auditor before a new actual audit attempt. No real audit, official model reload, HDF scan, GPU execution or formal evaluation was performed for this correction. Successful software tests do not establish candidate integrity or scientific admission.
