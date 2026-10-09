# P064 B-bound projected PPO: independent paired CFD terminal review

Primary outcome: the predeclared `(168,228]` window passes the unchanged physical criteria. Independent raw recomputation gives **3.8952838833% total-drag reduction**, rear-lift centered RMS ratio **0.815623043405**, and absolute mean rear-lift / paired-zero centered RMS **0.011378146878** (criteria >=2%, <=1.05, <=0.10). No threshold was relaxed.

This is an **in-sample b00 physical trial of a new B-trained PPO policy**, not independent generalization or formal surrogate admission. The prior successful K1-based policy remains a separate artifact. B's aerodynamic training included this b00 controlled trajectory; H100 field failure is not repaired by this trial.

## Actual execution and identities

- Unit `fluid-control-p064-b-projected-ppo-long-cfd-20261006.service`, invocation `3a078c62ed9e4f7b8876f0f166bdb510`: independently observed MainPID0, exited, Result=success, ExecMainStatus0.
- Approval SHA `5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99`; executed driver `83e08d66aa6c52b4f0164b9ab41eb3a161a52d7b50fa67db078d6b65cf2cbb26`.
- Result `artifacts/p064_b_projected_ppo_long_cfd_20261006/result.json`, SHA `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`.
- B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`; final policy `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`; VecNormalize `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`; training result `3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3`.

## Independent raw checks

All 3,200 recorded raw-force file hashes match actual files. Both cylinders and both branches contain exactly 16,000 samples on the .005 grid from148.005 through228. All six windows' sample counts, means, centered and total RMS, absolute peaks, and paired ratios were independently recomputed with NumPy directly from the raw coefficient files; maximum statistic difference from saved results is 8.88e-16.

| Fixed window | Samples/branch | Drag reduction % | Rear centered RMS ratio | Mean-bias ratio | Original joint criteria |
|---|---:|---:|---:|---:|---|
| (148,160.4] early12.4 | 2480 | 3.2349352673 | .8767102902 | .0707129739 | pass |
| (148,154.2] first6.2 | 1240 | 3.3827511284 | .9219299419 | .1354645134 | **fail mean-bias10%** |
| (154.2,160.4] trailing6.2 | 1240 | 3.0871149920 | .8239568207 | .0059625735 | pass |
| (168,228] primary60 | 12000 | 3.8952838833 | .8156230434 | .0113781469 | pass |
| [168,228] historical companion | 12001 | 3.8951185721 | .8156221708 | .0112830523 | pass |
| (148,228] full80 | 16000 | 3.7550441050 | .8243067111 | .0137088909 | pass |

Primary total Cd means are2.208043962303 (policy) vs2.297539654165 (zero). Primary rear Cl centered RMS is.964560681839 vs1.182605971764; mean rear Cl is-.013455864446. Primary absolute rear Cl peak is1.350631516 vs1.647306236, but full-window peak is1.679784159 vs1.647306236: reduced late RMS/peak is not uniform peak suppression.

All800 recorded projected requests equal exactly `.5*(pi(o)-pi(R(o)))`. Independently applying the single amplitude/rate filter from previous applied action reproduces all800 actions within1e-7; maximum magnitude.6547635496, maximum step.1, no saturated endpoints and two rate-limited endpoints. Both branches' endpoint times agree. Projection arithmetic audit uses recorded requests; it does not rerun the policy. Source review establishes the69-component reflection order. This is direct CPU policy feedback to OpenFOAM, without online FNO inference or MPC action substitution.

All1,600 solver logs independently contain20 timesteps, clean `End`, and no FOAM FATAL. Approval-bound source files, restart148/constant/system tree hashes, input artifacts and immutable driver were rehashed unchanged. Four thousand resource observations have minimum MemAvailable121,152,520,192 bytes, all above22GiB. Wall time1092.70294s for80D/U and800 feedback cycles is approximately1.366s/cycle; this is online simulation feedback, not a demonstrated hard-real-time .1s controller.

Saved terminal inspect records for exact owned containers `9f5554b61485245ed251a6c267a3547f1dd0de174c4e960b342e46d89f24a34e` and `a9af107121e86d15f9fd3dfd513feb3000c109cc97b89976d16f15f8a806f947` show stopped/OOMfalse, each8GiB memory and8GiB memory+swap (no additional swap). Both exact IDs are absent from `docker ps -aq`. Their137 exit codes describe stopped sleeping service containers, not failed solver segments; actual solver success is independently supported above.

## Comparison and limits

Both zero-branch raw front/rear arrays (time,Cd,Cl;16,000 samples each) are exactly equal to the old K1-based projected b00 trial, result SHA `199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6`. Thus paired references and restart chronology are matched. New policy primary drag reduction exceeds old3.8919799397% by only **0.0033039437 percentage points**. This does not establish a meaningful or statistically significant control improvement; no replicate uncertainty estimate was made. It confirms retained physical performance in this single in-sample phase, despite the documented development-error and training-retention tradeoffs. The early first-half original10% bias failure remains visible;20% sensitivity is not substituted for the original rule.

No model, GPU, training, or CFD was rerun for this independent review. Formal candidate acceptance and additional-phase robustness remain separate decisions; the successful old policy is not replaced by this report.
