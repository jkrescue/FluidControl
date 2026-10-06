# FC-E072 — same B-policy b07: independent real-CFD terminal review

The primary `(130,190]` window independently meets the unchanged physical criteria: **3.9026772898% drag reduction**, rear-Cl centered RMS ratio **.814994542203** (18.5005% lower), and absolute mean rear Cl / paired-zero RMS **.012920799096**. All six predeclared windows meet2%/1.05/10%; even the early first6.2 bias is .083038892388. No15% sensitivity or relaxed criterion is needed.

This is actual CPU-policy/OpenFOAM feedback, not merely surrogate simulation or training progress. It extends the same B-trained projected policy to b07 without retraining. b07's fixed-action H5 data were already opened, so it is not a universally unseen phase or untouched holdout. Three observed initial phases are not statistically independent robustness evidence. Prediction-model admission remains FAIL.

## Identity and independent checks

Unit `fluid-control-p064-b-projected-ppo-b07-long-cfd-20261006.service`, invocation `c45add13aeff41fe9526e835e384a52d`, independently queried MainPID0/exited/ExecMainStatus0. Approval `docs/P064_B_PROJECTED_PPO_B07_LONG_CFD_APPROVAL_20261006.json` SHA `df4d7881226f8ebf53da3aa47ac29f32ba2f0c94d59e431dae3a84dced4b7f56`; immutable driver `c3d63d9d9114a2ec32b8a5d6a4e7a6dee5aa8e31143167fb63229951656777d5`.

Result `artifacts/p064_b_projected_ppo_b07_long_cfd_20261006/result.json` SHA `dd579e7443c6693daef4173ed53ea2cb6836878fafff365bc12c1db8fe4ab7fc`. Same policy `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`, VecNormalize `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`, B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891` as the preceding b00/b01 trials.

Independently rehashed all approval source/input bindings, immutable driver and original restart110/constant/system inventory. All3200 raw coefficient file SHAs match. Four branch/cylinder streams each have16000 finite unique samples on the exact .005 grid110.005..190. Independently recomputed every branch mean, centered/total RMS, peak and paired metric from these files across all six windows; maximum difference4.44e-16. No CFD, policy or FNO was rerun.

| Fixed window | N/branch | Drag reduction % | Rear centered RMS ratio | Mean-bias ratio | Original criteria |
|---|---:|---:|---:|---:|---|
| (110,122.4] early12.4 | 2480 | 2.6829948324 | .853265057086 | .043955807585 | PASS |
| (110,116.2] first6.2 | 1240 | 2.2132841663 | .885599555235 | .083038892388 | PASS |
| (116.2,122.4] trailing6.2 | 1240 | 3.1528696755 | .817807558197 | .004881769588 | PASS |
| (130,190] primary60 | 12000 | 3.9026772898 | .814994542203 | .012920799096 | PASS |
| [130,190] historical companion | 12001 | 3.9026890436 | .814989211744 | .012976254157 | PASS |
| (110,190] full80 | 16000 | 3.7086847354 | .820637113105 | .014706399557 | PASS |

Primary policy/zero total Cd means2.209037575886/2.298750385115; rear centered RMS .964527170572/1.183476846318; policy rear mean Cl +.015291466566. Full80 policy/zero rear absolute peaks1.561030697/1.647295989. These coefficients do not establish net energy savings: actuator power has not been included.

All800 projected requests exactly match `.5*(pi(o)-pi(R(o)))` from recorded requests. Independent sequential single slew/amplitude filtering reproduces applied actions within1e-7; all69-observation lengths, previous/current applied action slots and110→190 endpoint clocks agree. Maximum absolute omega .653975993395, maximum delta .10000000000000003 (floating-point representation of.1), zero saturated endpoints and five recorded rate-limited endpoints. This is recorded arithmetic verification, not a fresh policy evaluation. Deployed feedback uses CPU PPO plus project reflection/filter, with no online FNO or MPC substitution.

All1600 solver logs contain20 time steps, cleanEnd and noFOAM FATAL. Two saved owned container terminal records show stopped/noOOM and8GiB memory with8GiB memory+swap; both exact CIDs are absent from current Docker inventory. All4009 resource samples exceed22GiB, minimum MemAvailable120378871808 bytes. Wall1097.565439s for800 cycles is approximately1.372s per.1D/U control interval, not a physical hard-real-time claim. No restart occurred.

## Interpretation

The same B-trained projected policy now has successful primary physical windows at b00/b01/b07. Preserve the earlier b00/b01 transient10% bias failures; b07 success does not retroactively change them. No matched old-policy b07 result is asserted or invented. This is finite fixed-Re100/L/D5 evidence, not independent-phase statistics, generalization to other geometries, surrogate precision acceptance or whole-goal completion. The full B force-window prediction failure and signed H1 readout limitations remain recorded. No new training or experiment is authorized by this report.
