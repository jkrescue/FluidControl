# FC-E053 independent terminal review — diverse-reset final PPO / real CFD

The actual final diverse-reset PPO policy completed the unchanged paired124-cycle real-CFD protocol. Unit `fluid-control-exploratory-diverse-ppo-cfd-20261006.service`, invocation `464de68ee1114eea8e8ae214d18dc045`, ran2026-10-06 07:33:58–07:36:26UTC; PID0/Resultsuccess/exit0. Worker wall145.11208140599774s. This is operational completion and a descriptive improvement overFC-E051, not original80D/U admission or completed constrained control.

## Identity

- Approval `docs/EXPLORATORY_DIVERSE_PPO_CFD_APPROVAL_20261006.json`: `67fda1a404f844d89b986442a4a9000561b02757417d5a8d366fdf9f9e6033db`.
- Executed driver `89e0d8bea92440babd3d647eed31758db9cfc2a43e31ed6e9d9bdf5047b77b6e`; reviewed source/test delta changes producer identity only. Seven independent tinyCPUtests PASS0.04s; actual trainingJSON consumed successfully without policy loading during review.
- Result `artifacts/exploratory_diverse_ppo_real_cfd_20261006/result.json`: `8c909aa4bd0b73e3cf570dd55cb2a1abd7346a9c424695a5e0056b4e5e833bdc`.
- Training result `cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640`; finalpolicy `8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b`; identityVecNormalize `6988d4d161bc69c8bbd89d477e9320ad9ef264d35c9dee0bbf63954d4cdfce70`.

The exact nine import-source files, actual trainingapproval/result/policy/normalizer hashes were independently rechecked. The full paired solver loop and numerical observation/action/window helpers are AST-identical toFC-E051. Policy inference is deterministicCPU, actual69 physicalCFD observations, one canonical slew/amplitude filter and unchanged linear boundary ramp. No onlineFNO, MPC action substitution, new training or CFD rerun by this reviewer.

## Raw independent arithmetic and matched zero

Rehashed all496 generated coefficient files against result identities. Both bodies/branches have2480 finite, strictly increasing samples. Exact predeclared open-left windows contain2480/1240/1240 points on the.005 time grid. Direct NumPy formulas independently reproduce all saved branch means, centered/total RMS, peak and paired metrics within absolute1e-14.

Crucially, **all columns of both complete zero-branch raw coefficient arrays are exactly equal toFC-E051**, not merely close aggregate scores. The same start148, actioninterval.1, end160.4, original reward/action protocol and zero trajectory make the comparison matched. No favorable-window selection.

| Window | Drag reduction | Rear-Cl fluctuation RMS ratio | Absolute mean rear-Cl / pairedzero RMS | Original10% / sensitivity20% |
|---|---:|---:|---:|---|
| Full12.4D/U (148,160.4] | +0.5057596986% | 0.9737074308 | 0.1237209858 | fail / pass |
| First6.2D/U (148,154.2] | +0.6872659156% | 0.9605219543 | 0.1071973373 | fail / pass |
| Trailing6.2D/U (154.2,160.4] | +0.3242480613% | 0.9864394989 | 0.1402443436 | fail / pass |

Using the unchanged fixed train-b00 long-term zero reference RMS1.1826535012844825 instead, the bias ratios are0.1232281589 /0.1067693910 /0.1396869267, also fail10%/pass20%. The latter is explicitly a sensitivity calculation, not a threshold change. Every observed drag reduction remains below the original2% target, and12.4D/U is shorter than original80D/U. A relaxed bias limit alone does not complete the objective.

Full PPO/zero totalCd:2.288205522883871 /2.29983717243621. Full PPOrearCl mean0.1457362135211723, centeredRMS1.1469714142888385, peakabs1.788904166; zero centeredRMS1.1779425502882805, peak1.64726217. First/trailing PPOmean0.12627119410790502 /0.16520123293443956, centeredRMS1.1314297273487257 /1.1619792805818645. Full rear-Cl fluctuation reduction is2.6293%, while the peak remains abovezero.

## Comparison with original four-zero-start policy

FC-E051 full/first/trailing drag reductions were+0.4117553% /−1.6358561% /+2.4594279%; RMS ratios1.1058626 /1.1998321 /.9909864; bias ratios.5273107 /.4172662 /.6373532. Diverse resets substantially reduced mean bias and full/first fluctuation, removed the adverse first-half drag sign, and modestly improved full-window drag, but trailing drag reduction is smaller. Report all outcomes; do not claim dominance in every metric.

All124 requested actions are distinct, between−0.20271998643875122 and−0.11491861194372177, with no saturated endpoints and one rate-limited endpoint. Maximum applied magnitude.2027199864 and delta.1. FC-E051 requested+.75 at every cycle and saturated117 endpoints. This is evidence of changed closed-loop behavior under the reset-distribution intervention, not proof that it alone explains all remaining errors or guarantees robust state-responsive control across other starts.

The69-dimensional observation still omits the full62-force rewardhistory. Short-return bootstrap, diluted reward, model action-response bias and training-grid versus CFD-probe interpolation remain. No reward/weight/architecture changes, gate relaxation, or policy selection follows from this result. Any increased PPO budget is a separate prospective intervention, not part of these measurements.

## Solver, resources and cleanup

All248 branch segments completed20 steps cleanly. Maximum Courant PPO.245365912 /zero.245224870; maximum absolute global continuity perstep6.74681567e-13 /6.96047689e-13. Actual input observations match the preceding output observation, with finite69 channels and appliedaction feedback.

620 resource observations: minimumMemAvailable122083807232bytes, above22GiB. TraceSHA `8d3d2bc6d4759daced1d7e5fa7a6c6a4f197880d877732547466634eaf3e8d06`. CPU8GiB controller, two8GiB/noSwap solver containers; no onlineGPU.

Saved terminal evidence hashes `244bf69f21a6b959f90c7e16a6381aa5bdcfcd9561cbd71ef0fd2c6dc68c1a36` and `07769dc014a427593befb624b02c0f81682581dd112cc00e1a808783cb7ea700` show both not-running/noOOM. Exact IDs `7f4c1db51ff4b811094199ef0e418c7bfd74e4432fdb11e3959f08a424b9e79e` and `0e076aa207159a3c1c5180abaf5ff2040837341b35de35e2d5e40b8d1af06f09` are independently absent from Docker. Exit137 records intentional termination of sleeping container supervisors, not solver failure. Executed driver reports originalrestart unchanged; reviewer did not reread original field tree. No separate cleanup.json is claimed.

Conclusion: genuine surrogate-trained PPO→real CFD has produced small matched short-window drag benefit and improved behavior, but original constrained physical success remains incomplete. Original K1 formalFAIL and the separate historical CFD-only80D/U success retain their identities.
