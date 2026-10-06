# FC-E077 — independent same-six H100 review: H25 candidate not adopted

The bounded evaluation completed; the H25 flow candidate regressed overall and is not promoted to PPO. This does not invalidate the existing B-policy's observed real-CFD benefits, and does not relax any prediction or physical criterion.

## Actual execution and independent checks

Unit `fluid-control-p064-h25-quick-ar-20261006.service`, invocation `b34a1af84199467bad07b61758b92b49`. The transient unit was observed after completion with PID0/dead/exit0; its later InvocationID field was empty, not a running handle. Invocation remains bound in the saved supervisor receipt and launch observation. The exact named Docker container is absent.

- Approval `docs/P064_H25_QUICK_AR_APPROVAL_20261006.json`: `27e2c37525b16330894048b064bd706dfa2419e7b4836614a5a838751683a394`.
- New result `artifacts/p064_b_h25_quick_ar_20261006/result.json`: `1b7bd2a2e99f9d02398df4cbcefc2d7dc5a486a02866d9a64856d0db9e9dafe0`.
- Supervisor receipt: `f70b9ee80e9e1761ce1fe5f2a7f13f7b6cb9691fd4132fc0263caf7ecccbb9da`.
- Reused parent B result: `artifacts/fcp064_arm_b_formal_resume_r3_20261006/force_window/result.json`, `6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173`.
- Candidate manifest `decf5f52bc0087fe07f2d3969e39604f19ea273c191ad193660c0af4c02670c0`; training engineering review is [R2 terminal](P064_B_H25_TRAINING_R2_TERMINAL_REVIEW_20261006.md).

Independently rehashed all 411 original numerical sources, the worker/supervisor and 22 bound inputs. All six case names, HDF hashes, 101 timestamps, realized action endpoints and true four-force arrays match the parent exactly. Official dual loader changes only the explicit reviewed H25 identity contract; numerical worker is unchanged `eee1f59…a8576`, batch1/high/TF32, six start0 H100 rollouts. No parent rerun, model/field reread or inference was performed by this reviewer.

All 600 predicted force endpoints are finite. Independent full-series force MAEs, tail indices39–100 (62 samples) Cd/Cl means and centered Cl RMS, and action-minus-zero mean-Cd errors were recomputed; maximum saved-window-statistic discrepancy is 4.44e-16. Field per-step scalar diagnostics and their means/final values were checked; no raw-field-array independent recomputation is claimed because this worker saves diagnostics, not all predicted grids.

## Matched numerical outcome

Every case has 100 predicted endpoints; pooled force MAEs below exclude the seed. Cd error is the absolute error of the sum of front/rear Cd, not a sum of component MAEs.

| Metric | Parent B | H25 child |
|---|---:|---:|
| Rear-Cl MAE, 600 endpoints | 0.0623860029175 | 0.0830122845010 |
| Total-Cd MAE, 600 endpoints | 0.0207225337625 | 0.0467238451044 |
| Mean absolute action-minus-zero tail-Cd error, four pairs | 0.0156519934535 | 0.0201665626899 |

| Case | Cl MAE B→child | Cd MAE B→child | Mean velocity relative L2 B→child | Mean raw-pressure relative L2 B→child |
|---|---|---|---|---|
| b01 minus | .053949→.032707 | .023693→.036119 | .056661→.063819 | .192589→.217690 |
| b01 plus | .113139→.110941 | .035939→.066129 | .048647→.053114 | .151022→.159492 |
| b01 zero | .034894→.070838 | .004339→.037808 | .017664→.029264 | .056127→.092818 |
| b05 minus | .085212→.105114 | .020686→.058131 | .053046→.060448 | .163073→.182197 |
| b05 plus | .057995→.084807 | .029876→.041140 | .054433→.066415 | .183171→.218561 |
| b05 zero | .029128→.093667 | .009802→.041016 | .019942→.034677 | .067328→.109111 |

All six cases worsen in total-Cd MAE and both mean and final-H100 velocity/raw-pressure relative L2. Rear-Cl has local improvements in b01 minus/plus, but pooled error worsens. These field means are arithmetic means of step-relative errors, not pooled field-SSE ratios. Pressure demeaning is diagnostic only and does not replace raw pressure. All cases and regressions remain retained; this is an already-opened development panel, not fresh heldout confirmation or a complete formal rerun.

Elapsed 24.011134641012177 seconds; minimum sampled MemAvailable112.83304977416992 GiB. Bound resource contract: 48 GiB no-swap container/outer limit, allocator fraction .15, startup Available80/runtime22 GiB, 600-second worker deadline. Receipt/observations and actual cleanup confirm terminal completion; no training or CFD was performed here.

Lead decision, supported by these matched results: retain the successful original B-policy; do not train new PPO on this H25 candidate or claim model improvement. The 32-update training completed mechanically, but its proposed numerical benefit was not demonstrated. Next work is reproducibility/delivery of the existing successful closed-loop chain, not automatic budget growth, threshold relaxation or a new architecture. Existing B full-prediction FAIL and original physical 2%/1.05/10% criteria remain unchanged.
