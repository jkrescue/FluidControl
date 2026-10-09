# P064 B-policy b01 paired CFD — independent terminal review

**Primary original physical criteria pass:** `(150,210]`,12,000 samples/branch, independently recomputed total-drag reduction **3.9275159299%**, rear-Cl centered RMS ratio **0.815727792287**, absolute mean rear Cl / paired-zero centered RMS **0.027294976565**. Criteria remain >=2%, <=1.05 and <=0.10. Early first6.2 D/U still fails the original10% mean-bias criterion.

This uses exactly the same new B-trained policy as the preceding b00 trial, at predeclared restart130. b01 is already opened development and historically exercised, **not an untouched test or statistically independent generalization proof**. This supports physical performance at a second observed initial phase, not significant superiority or formal H100 model acceptance. Old successful policies remain distinct and preserved.

## Actual identity and independent evidence

- Unit `fluid-control-p064-b-projected-ppo-b01-long-cfd-20261006.service`, invocation `432c12de32b0444d9a6f6626e12616d1`: the existing45-second monitor observed actual MainPID0, exited, ExecMainStatus0 before audit; no retry or duplicate scientific run.
- Approval `docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_APPROVAL_20261006.json`, SHA `3a19e326ebfd37df24060b8b5717b5af08b405ed63165e4034974aeb7b0abcb6`; immutable driver `4b8fa43f8ac020521ae8cde1047b6ec2f80d35ec0512bc7606d1050835010619`.
- Result `artifacts/p064_b_projected_ppo_b01_long_cfd_20261006/result.json`, SHA `0be19e0dfdf8df4d60e2f5040673f2a3ec25cbadb133548671ce31f061e75c88`.
- Policy `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`, VecNormalize `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`, candidate B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`. Approval input/source hashes, immutable driver and restart130/constant/system tree bytes were independently rehashed unchanged.

All3,200 raw coefficient file hashes match. Both cylinders and both branches contain16,000 finite samples on the exact.005 grid from130.005 to210. NumPy recomputation directly from these saved files reproduces all means, centered/total RMS, peaks and paired ratios in all six windows with maximum discrepancy4.44e-16. No evaluator/model/CFD was rerun.

| Predeclared window | N/branch | Drag reduction % | Rear centered RMS ratio | Mean-bias ratio | Original joint criteria |
|---|---:|---:|---:|---:|---|
| (130,142.4] early12.4 | 2480 | 3.6176475790 | .8780513137 | .0659482202 | pass |
| (130,136.2] first6.2 | 1240 | 4.2256062032 | .9240772588 | .1278252613 | **fail10% bias** |
| (136.2,142.4] trailing6.2 | 1240 | 3.0096291696 | .8248542359 | .0040752992 | pass |
| (150,210] primary60 | 12000 | 3.9275159299 | .8157277923 | .0272949766 | pass |
| [150,210] historical companion | 12001 | 3.9274182681 | .8157293384 | .0272072974 | pass |
| (130,210] full80 | 16000 | 3.8191684931 | .8249791070 | .0072018431 | pass |

Primary policy/zero total Cd means2.209964204065/2.300309214919; rear centered RMS.956438292176/1.172496880968; policy mean rear Cl-.032003274889. Primary absolute rear Cl peak1.350205188 is below zero1.647302403, but full-window policy peak1.704317136 exceeds that zero peak. RMS benefit is not uniform transient peak suppression. The20% bias sensitivity would accept the first6.2 mean metric, but does not replace the original10% criterion.

All800 projected requests exactly equal `.5*(recorded pi(o)-recorded pi(R(o)))`. Independent sequential single amplitude/rate filtering reproduces every applied action within1e-7;69-component observations and paired endpoint clocks match. Maximum|omega|=.6549400985, maximum|delta|=.0959317237; no saturated or rate-limited endpoints. This checks recorded arithmetic and the reviewed reflection source, not a policy rerun. Actual feedback is CPU PPO to OpenFOAM, not online FNO inference or MPC action replacement.

All1,600 solver logs contain20 timesteps, cleanEnd and noFOAM FATAL. Both exact owned container IDs `8057e7e026c0b157358e86ae4220a2c3a25bb977bc1ec241487749785af55b46` and `06f374d364883caed1122306c7504c1725ac332acb4d9c76502a1a48d67ad48b` have saved terminal inspect stopped/OOMfalse,8GiB memory and8GiB memory+swap, and are absent from actual Docker inventory. Exit137 belongs to cleanup of sleeping service containers, not failed solver segments. All4,892 saved resource observations are above22GiB; minimum MemAvailable116,194,476,032 bytes. Wall1347.922996s corresponds to approximately1.685s per.1D/U feedback cycle; online simulation feedback is not a hard-real-time claim.

## Matched historical comparison and limits

New zero front/rear arrays (time,Cd,Cl;16,000 each) are **exactly equal** to old K1-based projected b01 arrays, result SHA `961e1bc3ccb7a9f9dae4b54e9f8233c906507c794cff9a497d806391e0fc5c37`. New primary drag reduction exceeds old3.9236372469% by only **0.00387868305 percentage points**. This negligible numerical difference does not establish statistically significant control improvement; no replication variance was estimated. The policy differs, while the paired zero/restart/measurement protocol is matched.

Together with the newpolicy b00 result, this is two observed physical initial-phase confirmations using the same frozen policy and projection. b00 entered B training; b01 entered development evaluation. Neither observation erases model-prediction failures, old-train retention costs, or unchanged frozen-flow/H100 limitations. The full formal surrogate evaluation remains separate. This report changes no gates and authorizes no additional execution.
