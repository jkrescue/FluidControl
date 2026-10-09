# FC-P026 K1 original formal evaluation terminal review (2026-10-06)

## Decision

FC-P026 K1 completed the unchanged original formal evaluation but **failed the
development-admission decision**. The terminal receipt status is
`FC_P026_ORIGINAL_FORMAL_COMPLETE_NOT_ADMISSION`; `scientific_admission`,
`ppo_auto_launched`, and `frozen_test_accessed` are all false. This candidate
must not enter PPO or real-CFD control.

The matched K4 arm is a separate experiment. It received its own approval and
was observed live only after K1 formal evaluation terminated; its running state
does not alter this K1 decision.

## Terminal execution and integrity

- User unit: `fluid-control-fcp026-k1-formal-20261006.service`
- Invocation: `c039836ab63246ff8772dad66e1b46e5`
- Runtime: 2026-10-05 21:21:31--21:58:17 UTC
- Terminal state: `active/exited`, `Result=success`, `ExecMainStatus=0`, PID 0
- Formal approval SHA256:
  `2d15c3233d8255874fff98aac23a15a5c5a7d0488ce18f2ae166c8a4c6980000`
- Receipt:
  `artifacts/fcp026_history_training_k1_20261005/posteval_fc_p026_k1/receipt.json`
- Receipt SHA256:
  `f2f7a50a26177c65ee048b58fb20df0aee7f4cfa42aed0edd857f08d911ef948`
- All 35 files named by `receipt.json.sha256` were independently rehashed and
  matched. All eight saved container terminal records report exited status,
  exit code zero, and `OOMKilled=false`.
- The receipt's seven-entry candidate hash map exactly equals the formal
  approval map and, after removing the `candidate/` prefix, the previously
  recorded terminal candidate-audit map (`candidate_audit.json` SHA256
  `fa26bf47b6eb448e36046973a479e771b2d37eb605d9630b9022b392ce30d944`).
  This comparison did **not** reread the seven candidate payload bytes, so it is
  evidence of recorded-map continuity rather than a fresh model-byte audit.
- The formal receipt records the 411-file source map and exact original
  protocol: validation10, dynamic6, six force windows, and the unchanged
  development gate.

The 1,109 external memory samples remained above the 20 GiB floor. Minimum
physical `MemFree` was 28.046733856 GiB and minimum `MemAvailable` was
110.021461487 GiB.

## Endpoint and rollout results

Validation10 endpoint readiness passed. At H100, pooled total-Cd NRMSE was
0.00609718, total-Cd MAE was 0.0109280, rear-Cl MAE was 0.0401863, and velocity
relative L2 was 0.0435911. The strict start-0 action comparison passed with
delta-Cd MAE 0.0191297 (limit 0.023), sign 8/8, and cross-action ordering 20/20.

Dynamic6 also passed its predeclared diagnostic. Its pooled H100 total-Cd NRMSE
was 0.0178932 (limit 0.1) and strict start-0 delta-Cd MAE was 0.0103930 (limit
0.023). The dynamic6 H100 aggregate rear-Cl MAE was 0.0849813 and velocity
relative L2 was 0.0872107. These endpoint/rollout components do not override
the force-window failure.

## Six-window decision

Only `full40_dynamic_validation_b01_zero` passed all three window metrics. The
unchanged gate therefore passed 1/6 branches, with total-Cd passing 5/6,
rear-Cl fluctuation RMS passing 2/6, and rear-Cl mean passing 4/6.

| case | total-Cd error | rear-Cl RMS error | rear-Cl mean error | joint |
|---|---:|---:|---:|---|
| b01 minus | 0.00452229 | 0.0683388 | 0.00871125 | FAIL |
| b01 zero | 0.00157248 | 0.00505306 | 0.00509351 | PASS |
| b01 plus | 0.0123465 | 0.122262 | 0.0159393 | FAIL |
| b05 minus | 0.0161149 | 0.0682180 | 0.0485904 | FAIL |
| b05 zero | 0.00547017 | 0.00247624 | 0.0310912 | FAIL |
| b05 plus | 0.0241096 | 0.0813537 | 0.0152861 | FAIL |

The four rotating rear-Cl RMS errors average 0.0850431 and all exceed their
approximately 0.0294 fixed limits. Relative to FC-P018, their changes are only
about 1e-5 to 4e-4; the matched K1 arm did not materially repair the rotating
window-amplitude error. The scientific outcome is therefore
`DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`, with no threshold change and no PPO
authorization.

## Matched K4 start observation

After K1 formal termination, Root issued the independent K4 execution approval
at commit `ef2ddd3`, SHA256
`fca9c5a1106c85fb55a54590784453bd5246d55c21bbaa8ded1e3c72562b3e86`.
The approved dry run passed. A fresh observation then confirmed:

- unit `fluid-control-fcp026-history-k4-20261006.service`, invocation
  `eee5a6fbad40411cac2f05e00520b079`, MainPID 941365, `activating/start`;
- container `79a39768f1398bc5ff9d1085f027a4b1802978120107ca5f9008ffdd300a75e3`,
  name `fcp026-history-training-k4-20261005`, using the pinned official image
  `b40d5888...a22e` and `--history-k 4`;
- actual GPU preflight: CUDA free 32.9703 GiB, host MemAvailable 115.0164 GiB,
  allocator cap 7.3014 GiB, required available 31.3014 GiB;
- the live GPU was subsequently observed at 96% utilization.

This is evidence that K4 actually started, not a completion or accuracy result.
Its planned 1,368 windows / 171 updates, original optimizer/objective, 20 GiB
resource floors, terminal audit, official reload, and separate full formal
evaluation remain required. No K4 admission or PPO conclusion is recorded here.
