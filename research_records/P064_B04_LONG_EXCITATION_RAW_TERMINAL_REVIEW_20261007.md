# B04 long excitation raw terminal review

Independent raw metadata/force audit PASS; not curated and not a model-quality or control-performance result.

- Science unit: `fluid-control-p064-b04-long-excitation-cfd-20261007.service`; invocation `e2fa5d4e5fa4465a90a2dfbbdd402fc8`, independently observed PID 0 / exited / success / exit 0.
- Approval: `docs/P064_B04_LONG_EXCITATION_RAW_CFD_APPROVAL_20261007.json`, SHA256 `f2bfcd8f6d1526af266946448ca452d3c271047466d77a9e439d843081338536`.
- Result: `artifacts/p064_b04_long_excitation_raw_20261007/result.json`, SHA256 `ab4d66f9d872fe55245460c319dcd7c58589bab83fddd32825c50dc5a1725b02`.
- Independent audit unit: `fluid-control-p064-b04-raw-independent-audit-20261007.service`; invocation `e1aa37710ec04339bc34386da6489026`, PID 0 / exited / success / exit 0. Actual limits: CPU 1, 2 GiB, no swap, 120 seconds, CUDA hidden. Exactly one terminal audit was executed.
- Receipt: `artifacts/p064_b04_long_excitation_raw_independent_audit_20261007/receipt.json`, SHA256 `f411fb9df76883c4c7d0b8588356fda5bb0c9ffefcd86c722e34a2ab15a2e1f5`.
- Checker `/tmp/audit_p064_b04_raw_terminal.py`: SHA256 `9ecd5aefa5e48ed29813f1f441815f725f56d90181b8f018c4aa6237cf8a2197`; fixture SHA256 `4f64c3d11d93046ef7baa604cf905ab0686f294ef557ab3286f9557207a8b0de`. Three CPU tests passed, including actual pinned component-parser import; Sota independently accepted the final source.

## Independently verified

The solver log contains exactly 16,000 steps at 0.005 intervals from 120.005 through 200 and ends in `End`. Maximum Courant number is 0.245420707; maximum absolute global continuity error per step is 1.35594527e-12. There are 801 U/p frame directories on the exact 120:0.1:200 grid.

Both body coefficient tables have 16,000 rows and all 13 columns finite. The initial t=120 force comes from the bound original coefficient tables, not zero padding. All 801 saved function-object properties were parsed for actual pressure and viscous components. Maximum pressure-plus-viscous minus total discrepancy is 1.0547118733938987e-15; maximum component total versus printed coefficient discrepancy is 4.999580749398547e-10. Coefficient `(f)/(r)` columns were not misidentified as pressure/viscous components.

The 20 bound source entries, old-time/uniform restart payload, constant/mesh files, unchanged U content outside the intended rear boundary, and full 801-point action table were checked. Source hashes remain unchanged. The owned solver container is absent. Resource records match the result: minimum available memory 114.61346817016602 GiB and minimum free disk 197.46954345703125 GiB.

## Scope and handoff

This verifies raw timing, frame presence, forces, source provenance, resource bounds and cleanup. It does **not** independently validate every U/p field value or declare full-field numerical QC. VTK/Curator conversion and field-finiteness checks remain separate, unexecuted work requiring authorization. No model, optimizer, training, or extra CFD was run by this audit. The repeated train-only excitation is not a new independent physical condition or evidence that the existing prediction gate has passed.
