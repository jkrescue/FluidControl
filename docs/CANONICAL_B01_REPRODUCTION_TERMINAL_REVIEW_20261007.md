# Canonical b01 basic closed-loop engineering reproduction — independent terminal review

Verdict: **800 real feedback cycles reproduced successfully; all six windows pass the unchanged physical criteria.** This confirms the new safe entrypoint reproduces the already-opened E085 case, not a new holdout, improved model or completion of the overall high-accuracy prediction goal.

## Actual identity and execution

- Unit `fluid-control-canonical-reproduce-b01-20261007.service`, invocation `f6c3fc3464074493b19ea5418ddfada5`: MainPID0, SubStateexited, Resultsuccess, ExecMainStatus0. No restart or retraining.
- Final approval `docs/CANONICAL_B01_REPRODUCTION_APPROVAL_20261007.json`, SHA `f10547240d5a89430f2967bfb9859868fc00ee625447fcc56ce68b2b93b4152d`.
- Unmodified E085 driver SHA `ebc6f2663e5eeae5d22a6e4ebfd06debd4ec2672f31accfc1db30f67131e25ba`; frozen E082 policy and VecNormalize remain approval-bound. Source/input/policy-overlay/restart hashes were checked before launch and again after completion.
- Result `artifacts/canonical_b01_reproduction_20261007/result.json`, SHA `f22047b62c8bdb50547122e61d3e523d0cb101d90b1ed98037e4524cb98c342a`; progress SHA `04998f89fda4feda281f18140aaa5541fcc068ddb8eb477605b981d955af30a0`.
- Runtime1104.363636s. This is CPU policy + real OpenFOAM feedback, no online FNO, MPC, GPU training or optimizer updates.

## Independent raw evidence

The previously used read-only canonical audit (`/tmp/audit_b02_base_readonly.py`, SHA `cef05af9e430072300020a819ec320b328abf4754197b00701537fda018df2d8`) was reused without numerical changes, with a thin new-output/unit/approval/reference wrapper at `/tmp/canonical-delivery-review/audit_reproduction_terminal.py`. An additional explicit check covered feedback continuity and raw force endpoints; these are not attributed to the older audit alone.

Actual audit unit `fluid-control-canonical-reproduction-terminal-audit-20261007.service` ran with2GiB/noSwap/CPU1/120s, CUDA hidden, exit0 in1.482s, memory peak5.2MiB. No policy/model load or solver rerun.

- All3200 raw force files rehashed; four force series each have16000 unique finite .005-time samples. All branch means, centered/total RMS and six-window ratios recomputed; maximum difference from saved metrics4.440892098500626e-16.
- All1600 solver logs contain20 steps and clean End without FOAM FATAL.
- All800 physical→canonical observations/orientations/pivots/margins/sign-restored requests and the single slew/amplitude filter recomputed. Maximum|omega|.6657415628433228, maximum|delta|.10000000000000003 (rounding).
- All799 consecutive input observations equal the previous output observation exactly. All800 cycle-end raw front/rear Cd/Cl match output observation indices64:68 after FP32 conversion, maximum difference0.
- New/old E085 input observations, output observations, requested actions and applied actions are exactly equal across800 cycles. Paired-zero full raw arrays, all columns, are exactly equal. Historical E085 result SHA `c56a5cbccdf218953a0e1ea040da1c1ee7e2f1b574e7b509939a622f65b7495f`; minute metric differences are reduction-order rounding, not trajectory changes.
- Terminal progress contains800 cycles and exactly the result rows. All20 source-restart files remain unchanged. Both exact owned solver containers exited without OOM, retained8GiB/no excess swap limits, and are absent after cleanup.
- All4000 resource samples maintain MemAvailable≥122040713216bytes, above22GiB runtime guard and20GiB reserve.

## Original physical criteria, unchanged

Primary window is(150,210],12000 samples per branch, after20D/U settling from130. Historical inclusive companion[150,210] has12001 and is separate. Criteria remain drag reduction≥2%, rear-lift centered RMS ratio≤1.05 and absolute mean rear lift / paired-zero centered RMS≤10%.

| Window | Samples/branch | Drag reduction | Rear Cl RMS ratio | Mean-bias ratio | Original result |
|---|---:|---:|---:|---:|---|
| First12.4D/U |2480|5.4339718765%|.9181581792|3.9147005324%|PASS|
| First6.2D/U |1240|5.8700750911%|.9816194239|8.0613985373%|PASS|
| Trailing6.2 of first12.4 |1240|4.9978257764%|.8479563067|.2317213707%|PASS|
| Primary final60 |12000|4.0090689485%|.8171901322|3.6366076181%|PASS|
| Historical inclusive final60 |12001|4.0089803915%|.8171935391|3.6276577148%|PASS|
| Full80 |16000|4.2407884776%|.8347686669|.2679444480%|PASS|

Primary rear-lift fluctuation reduction is18.2809867843%. Action-square descriptions reproduce E085 exactly: mean(omega²)=.24044747071480532, omegaRMS=.4903544337668472, endpoint rectangle sum(omega²)×.1=19.235797657184428D/U, deltaRMS=.06272814087822465. These are action-cost proxies, not measured power or net energy savings; torque/power conversion has not been verified here.

## Delivery interpretation and retained limitations

The safe default-preflight/explicit-execute launcher and frozen canonical chain now have an actual fresh-output reproduction. The basic online-feedback demonstration is real: observations come from the just-advanced CFD, not a saved observation replay. Official FNO is used in the established HydroGym/SB3 training chain, while deployment runs CPU policy inference against CFD. This review validates engineering reproduction and this fixed physical protocol only.

E083/E086 early bias failures, E080 negative noncanonical second seed, C50/D25/H25 rejected candidates and the full B surrogate precision FAIL remain unchanged. This duplicate b01 trajectory is not an independent phase/seed sample and does not resolve prediction or broad-generalization shortcomings. No next training or CFD run is authorized by this report.
