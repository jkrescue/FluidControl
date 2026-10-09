# E114 retained-B continuation: independent terminal review

The full new tail (328,408], all four predeclared20-D/U blocks, and both joined summaries PASS the unchanged physical criteria: drag reduction≥2%, rear-Cl fluctuation RMS ratio≤1.05, absolute mean rear-Cl divided by paired-zero fluctuation RMS≤10%. No new warm-up interval was discarded. E109's original six windows are preserved unchanged.

This supplies160D/U/1600feedback intervals of the same paired trajectory across an independently verified restart, not one uninterrupted process, a new Reynolds number/geometry, independent statistical replication, surrogate prediction admission, or model improvement. Default B remains the same frozen policy. Historical prediction failures remain.

## Actual identity

- Science unit `fluid-control-p064-b-continuation-328-408-cfd-20261007.service`, invocation `73a2568857074c70b494639269a602bf`: independently observed PID0/exited/Resultsuccess/ExecMainStatus0 before reading terminal results.
- Approval `docs/P064_B_E109_CONTINUATION_328_408_APPROVAL_20261007.json`, SHA256 `6e08555e336e271fe0baf6790feaf3d92c7c6e0771fc9aa93d7b0fd608f3f812`.
- Result `artifacts/p064_b_continuation_328_408_cfd_20261007/result.json`, SHA256 `752b92d1063e51a8fb6a45ea539b173c3c5ffbd24c6a83255e0fa649f392063e`.
- Frozen B policy SHA256 `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`; VecNormalize `1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145`.
- Prior recovery review: [one known interval](P064_B_E109_RECOVERY_REPLAY_INDEPENDENT_REVIEW_20261007.md), SHA256 `94ab146d4dab66cea17880c44fc789311fb2dd0fd83e61d087f6a72332f6fa40`.
- Unique independent read-only audit unit `fluid-control-p064-b-continuation-independent-audit-20261007.service`, invocation `7b6eceeef4d74792ac885d962b394df6`, PID0/exited/success0. Actual CPU1/2GiB/noSwap/120s, CUDA hidden; no policy, model or solver rerun.
- Audit receipt `artifacts/p064_b_continuation_328_408_independent_audit_20261007/receipt.json`, SHA256 `a26368b60007b53a422351f1d086fcda095cd823a98ded331e513c324de07af1`.
- Independently reviewed checker `/tmp/audit_p064_b_continuation_terminal.py`, SHA256 `1af8e223289b0def57c89e1abc19ebf80452acf7b2ea4de4fc04ec65161baa66`; four CPU fixtures passed. Actual one-shot audit stderr is empty.

## Independently recomputed metrics

All intervals below are open-left/closed-right. Percent columns multiply the original ratios by100; criteria were not relaxed.

| Window | Raw samples/branch | Drag reduction % | Rear-Cl RMS ratio | Mean-Cl bias % | Original gate |
|---|---:|---:|---:|---:|---|
| (328,348] |4000|4.2414348032|0.8225297589|4.8866253123|PASS|
| (348,368] |4000|4.3017729344|0.8275557884|6.8816026588|PASS|
| (368,388] |4000|4.0692257762|0.8168277993|0.9174709077|PASS|
| (388,408] |4000|3.9175867383|0.8173977454|5.8391490701|PASS|
| **New primary (328,408]** |16000|**4.1325891506**|**0.8207712887**|**1.2352025888**|**PASS**|
| Joined (248,408] |32000|4.1803349609|0.8275000018|0.8197802269|PASS|
| Joined (268,408] |28000|4.0735445085|0.8190894448|0.5085330591|PASS|

## Raw and recovery checks

All6400 old+new force-file SHA256 values were verified, with32000 unique finite .005-spaced samples per cylinder/branch. The junction belongs to old time328; new samples begin328.005. No silent duplicate removal was used. All branch means/RMS/peaks and seven paired-window metrics were independently recomputed; maximum difference from saved metrics0.

The first new input69 exactly equals E109's final output69. The limiter separately retains double action0.19905773401260382, not the rounded float32 observation action. All799 subsequent saved feedback links, independently recomputed canonical orientations/sign-restored requests and single physical filters pass. No policy inference was repeated; the audit does not claim to independently regenerate the policy's saved requests.

All1600 new wake-probe endpoint files were directly parsed: probe coordinates and column order0..31, unique endpoint time, and all64 float32 velocity components match the saved output. Four force components match the raw coefficient endpoints and action clocks match, covering the complete69-component observations for both branches. Probe-inventory SHA256 is `c7a225f2f719dd07f7343518d37351832bdea7a1ae6b50bc80dacf739306a962`.

All1600 new solver logs contain the correct20-step time grid, clean End and no FOAM FATAL. All56 bound parent restart/configuration files remain unchanged; copied t328 old-time fields match, with only the expected controlled rear-boundary action table excluded from the U comparison. Both owned solver containers are absent, stopped without OOM and retain8GiB/no-extra-swap limits in terminal receipts. Approval/input hashes and the ten inherited E109 source-file hashes match.

Minimum recorded MemAvailable122477780992bytes. Recorded continuation wall time1098.8131197360344seconds gives1.373516399670043wall seconds per feedback interval including loop overhead; it does not establish physical real-time control. D/U is not automatically seconds. Mean applied omega squared0.23908553125217036; maximum step change0.10000000000000003 is floating-point representation of0.1. Squared action is only a proxy: torque/power conversion is unverified, so no net-energy-saving claim is made.
