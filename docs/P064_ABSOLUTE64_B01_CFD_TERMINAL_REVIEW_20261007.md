# FC-E111 absolute64 canonical b01 CFD independent terminal review

All six predeclared physical windows PASS the unchanged drag reduction ≥2%, rear-Cl fluctuation RMS ratio ≤1.05 and absolute mean rear-Cl / zero RMS ≤10%. Prediction selection FAIL remains; this single already-exposed b01 run neither replaces default B nor establishes independent generalization or statistical superiority.

## Actual provenance and audit

- Science unit `fluid-control-p064-absolute64-symmetry-canonical-b01-cfd-20261007.service`, invocation `6fa0baeb90034371b3b49f3d8d51192a`, independently observed PID0/exited/success/exit0 before reading terminal result.
- Approval `docs/P064_ABSOLUTE64_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json`, SHA `d6303082b0f97087961a9171505a2b4289623eb5f634f2502ac4f7d65639143b`.
- Result `artifacts/p064_absolute64_symmetry_canonical_b01_cfd_20261007/result.json`, SHA `5fa8f47bb3a4939b6b8377f5ccd8c178dec560220929e36e48b8875e90a7504e`; progress SHA `559c782a10c036fa9c120c84f99cdd35fcb48dcbfed2b8676c00ff8a88df96b3`.
- Driver `037f3c959d6fa2f313c4a964d00d263a4587d37b78b42cfa7300b10f286101c9`; fixed final PPO policy `bacbdaf3fc43716331e11285955e0f558c295cc7801b0192c0c81b5e2e98c0b0`, VecNormalize `6d40e429f385817328919f4c5d1e0400e41551f60a27e4c8102641bc99926c7f` from independently accepted E110.
- Independent raw audit unit `fluid-control-p064-absolute64-b01-raw-audit-20261007.service`, invocation `b4b0efda17754ce092d7bf7c01d393a1`, PID0/exited/exit0, CUDA hidden,2GiB/noSwap/CPU1/120s. Checker SHA `ae422846d6ff18ae93e98a7599120cef89cf8d32d6e3046ef37dfddb1dd7ba64` was narrowly reviewed with3 CPU fixtures before execution.
- Receipt `artifacts/p064_absolute64_b01_raw_audit_20261007/stdout.jsonl`, SHA `d114a326919c1f2abf4b1a35ab51f3de09e33b43eac5de821d776f8fd126dad3`; stderr empty. A separate producer check was also reported under invocation `6a6b40f61ec44760a0c3c795750d2a5b`; this duplicated only read-only CPU verification, not model, training or CFD execution. This review binds the independent b4b0 execution.

## All original windows

| Window | Samples | Drag reduction % | Rear-Cl RMS ratio | Mean bias % |
|---|---:|---:|---:|---:|
| First6.2 |1240|5.8908007302|0.9760435312|9.0800186317|
| Early trailing6.2 |1240|5.3045635537|0.8489483447|1.3145044207|
| Early12.4 |2480|5.5976965536|0.9155254541|5.1971322572|
| Primary(150,210] |12000|4.1362382900|0.8101959756|1.6817003164|
| Historical inclusive[150,210] |12001|4.1361456839|0.8101966404|1.6730855905|
| Full80 |16000|4.3736771226|0.8283446452|1.2044249789|

All3200 raw force SHA bindings and1600 solver logs verified; recomputed metric maximum difference4.440892098500626e-16. All800 canonical observation/action mappings and single rate filters passed;799 exact feedback links; raw force endpoints equal saved float32 observations with maximum difference0. All20 restart hashes unchanged, paired-zero force columns exactly equal retained B's same-b01 zero arrays. Owned solver cleanup and original resource checks passed. Maximum |omega|0.6652474999427795; max step delta0.10000000000000003 is floating-point representation of0.1, not a relaxed limit.

Science wall1109.2920159689966 seconds, average1.38661501996 wall seconds per paired feedback cycle including overhead. Minimum recorded Available121253580800 bytes over4002 samples. This is online feedback with actual CFD, not proof of physical real-time control speed; D/U is not automatically seconds.

## Matched b01 descriptive comparison

| Frozen policy | Main drag reduction % | RMS ratio | Bias % |
|---|---:|---:|---:|
| Retained B / E085 |4.0090689485|0.8171901322|3.6366076181|
| Exploratory G |3.9513457770|0.8163775204|3.0716935621|
| Absolute64 |4.1362382900|0.8101959756|1.6817003164|

Absolute64's main drag reduction exceeds B by0.12716934 percentage points and G by0.18489251 points in this deterministic comparison. No significance or broad superiority is inferred. Some early/full bias comparisons are worse than B despite passing10%; do not cherry-pick primary metrics. Controlled trajectories differ as expected; they were not required to be bitwise identical. G reference result is pinned to `f6319771a93541275f4fe7183d49fba9d607100fcdb7d9bf31b9fc92946594a4`; B reference to `c56a5cbccdf218953a0e1ea040da1c1ee7e2f1b574e7b509939a622f65b7495f`.

Action-cost proxies from all800 applied actions (first previous action0): mean omega²0.2574809670, omega RMS0.5074258242, endpoint-rectangle sum omega²×0.1=20.5984773626 D/U, delta RMS0.06306896593. Retained B mean omega²0.2404474707 and integral19.2357976572 are smaller. These are squared-action proxies only; torque and power conversion have not been verified, so no physical energy or net-power savings claim is made.

E109's248→328 future-time B run is a different interval and is reported separately, not pooled into this matched b01 comparison. No new physical threshold, prediction gate, model default, training authorization or automatic retry follows from this report.
