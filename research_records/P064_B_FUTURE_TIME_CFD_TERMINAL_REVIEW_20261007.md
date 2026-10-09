# Fixed B future-time CFD terminal raw verification — E109

The fixed B policy passed all six original physical windows on this newly generated time segment. This is a new future-time validation at the same Re100/L/D5 deterministic setup, not an independent physical condition or proof of statistical independence. It does not change the surrogate prediction FAIL or establish a result for the separately running Absolute64 policy.

## Identity and execution

- Science unit: `fluid-control-p064-b-future-time-cfd-20261007.service`, invocation `4b3eb2a2fcab4585aafd5740f8d0cea3`; actual MainPID 0, exited, Result success, ExecMainStatus 0 before audit.
- Approval: `docs/P064_B_FUTURE_TIME_CFD_APPROVAL_20261007.json`, SHA256 `dbcedcce5d680c98e9e611ca092fcd4953879ffef9e8ef9fbf613cd11c93da25`.
- Result: `artifacts/p064_b_future_time_248_328_cfd_20261007/result.json`, SHA256 `d53cb2ea32f66af6c4067d0c7eb90e7aecc8b815634588b6fe448d291acc5b98`.
- Fixed E082 B policy SHA256 `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`; same canonical adapter and single physical rate limiter, no online FNO/MPC.
- The source author executed the read-only checker after independent source review by Recovery (3 CPU fixtures passed). This is an independently reviewed checker and raw recomputation, not a claim of an independent third-party execution.
- Audit source SHA256 `f6d530084a2cf3fc4be72a230260ea482f4b2202a44de7dea8b6e3c096174804`; audit unit `fluid-control-p064-b-future-time-raw-audit-20261007.service`, invocation `df1165c3c5bd4e708765d9133b1c7b15`, actual exit 0, 2026-10-07 01:39:59–01:40:01 UTC. CPU1/2GiB/noSwap/120s, CUDA hidden, no model or solver calls.
- Audit receipt/stdout: `artifacts/p064_b_future_time_raw_audit_20261007/stdout.jsonl`, SHA256 `f4be832f6d844de4e25d2cccdf470db4a30e9fc794a27c1ffb8ed34cb1b47a89`; stderr empty.

## Recomputed physical results

Every row passes unchanged drag reduction ≥2%, rear-Cl fluctuation RMS ratio ≤1.05, and absolute mean rear-Cl / paired-zero RMS ≤10%. Percent values below are ratios multiplied by 100, not new criteria.

| Fixed window | Samples | Drag reduction % | Rear-Cl RMS ratio | Mean-Cl bias % |
|---|---:|---:|---:|---:|
| First 6.2 | 1240 | 5.5239005402 | 0.9656664890 | 8.5938139442 |
| Early trailing 6.2 | 1240 | 4.9955221962 | 0.8558199963 | 0.1748124959 |
| Early 12.4 | 2480 | 5.2597115264 | 0.9134502404 | 4.2094857578 |
| Primary (268,328] | 12000 | 3.9948739165 | 0.8165882753 | 2.8533042395 |
| Historical inclusive [268,328] | 12001 | 3.9948449918 | 0.8165925823 | 2.8446463223 |
| Full 80 | 16000 | 4.2280906363 | 0.8341788028 | 0.4040285390 |

All 3,200 raw force-file SHA256 values and 1,600 paired solver logs were checked. Raw recomputation differs from producer metrics by at most `4.440892098500626e-16`. All 800 canonical action mappings and single-filter transitions were checked; maximum |omega| `0.6557590961456299`, maximum step change `0.10000000000000003` (floating-point representation of the fixed 0.1 limit).

## Baseline, feedback, provenance, and resources

The fixed, unselected zero-action continuation 228→248 completed 200 segments / 4,000 solver steps. All baseline clocks, zero-action boundary tables, saved source inventory, parent and generated restart hashes were checked. The actual t248 probes and force outputs reconstruct the stored 69-component initial observation exactly; it equals the paired first input. Both paired branches start from that generated restart (only the applied rear-cylinder boundary can differ). All 799 subsequent feedback links and force-to-observation endpoint mappings passed. No comparison to an old b01 zero array was required: this is a different time segment with a genuinely newly solved zero branch.

Science wall time `1454.3036421969882` seconds; minimum MemAvailable `119106936832` bytes across 5,482 samples. Original resource/solver health bounds and all three owned solver-container cleanup checks passed. Historical source data were not modified. This run did not authorize conversion into training data.

## Interpretation

The predeclared future interval supports retained B control performance without selecting a favorable phase after observing the baseline. Same mesh, Reynolds number, geometry, deterministic dynamics and frozen policy limit the inference; it is not a new independent physical regime. Earlier failures, predictive accuracy limits, and the default B/G policy identities remain unchanged. No net actuation-power claim is made, and no policy or threshold was selected from these results.
