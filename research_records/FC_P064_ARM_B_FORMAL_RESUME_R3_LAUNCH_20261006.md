# FC-P064 Arm B formal resume R3 launch record

- Actual unit: `fluid-control-p064-b-formal-r3-20261006.service`
- Invocation: `3bded2dcb4a24f808879987962a9ef8b`
- Initial main PID: `1582344`
- Approval: `docs/FC_P064_ARM_B_FORMAL_RESUME_R3_APPROVAL_20261006.json`
- Approval SHA-256: `19be2d6aad903ffc94b807803bd5fd0902c7ec5b7a0f0b4212423744db89cb56`
- Resume runner SHA-256: `3bb215f93f5d6d468f8b22267b5f1fb485516f865c45d32723fe33b3fe56ec84`
- Resume source manifest SHA-256: `d52a6e76e40189ad2448cde723156a531dcb01976f9e81c45f90dd4e5073b584`
- Numerical source-v2 receipt SHA-256: `5cc0c470675df6e78fc0efc31fa106cc8af0cb34c37d8dd3e6185f3773fbda12`
- Output: `artifacts/fcp064_arm_b_formal_resume_r3_20261006`

The saved approval passed the no-execute preflight before launch. The resume reuses only the exact hashed R2 `precision` and `validation10` products and executes the remaining six original formal stages. Its final receipt must retain the R2 unit, invocation, and nine reused-file hashes separately from the R3 execution identity.

At launch, `MemAvailable` was 120524020 kB, no GPU compute process was listed, no Docker container was running, and the new output path was absent. The unit uses the reviewed 72 GiB/no-swap outer limit, 50/22 GiB `MemAvailable` checks, and the unchanged inherited numerical commands. As documented for R2, `.06` is the outer guard accounting value while nested original GPU evaluators retain their `.15` allocator setting; this record does not claim `.06` is the effective evaluator allocator cap.

The R3 unit subsequently completed the repaired CPU `validation_diagnostic` and `endpoint_gate` stages and entered `dynamic6`. This is an engineering launch milestone, not a formal scientific result or admission decision.
