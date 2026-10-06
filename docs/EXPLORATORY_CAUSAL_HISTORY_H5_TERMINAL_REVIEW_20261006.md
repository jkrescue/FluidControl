# Actual causal-history H5 paired CFD: independent terminal review

## Execution and provenance

Unit `fluid-control-exploratory-causal-h5-real-cfd-20261006.service`, invocation `6adc59fae65344d2b49b57cbe5b30f70`, completed 2026-10-06 06:08:12–06:12:13 UTC. Independent query found MainPID0, active/exited, Result=success, ExecMainStatus0. This is engineering completion, not physical-goal admission. No retry or new scientific execution was performed by this reviewer.

Approval SHA256: `f927f6b956f847766d899745ebc0e679a7638db63f27cfb1f9a489eb69fb6c0e`. Executed immutable driver SHA256: `0917cd5c62e43fc3f7b2cdc23900aa9d0932dff9ef4842a155bcf4288524b14d`. Result: `artifacts/exploratory_causal_history_h5_real_cfd_20261006/result.json`, SHA256 `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`. All 20 approval source-file hashes were independently matched. The result records the original restart unchanged; the reviewer did not repeat a full restart payload audit here.

K1 manifest `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`, flow `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`, and aero `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5` remain the matched frozen models. This run retained CPU inference; the separate GPU precision experiment did not change it.

## Actual measured outcome

Ten applied actions were `[0.1,0.2,0.25,0.25,0.2,0.1,0,-0.1,-0.2,-0.30000000000000004]`. Unlike H2's ten HOLD actions, H5 induced nonzero feedback control. Action magnitudes stay within 0.75 and changes within 0.1 (floating representation notwithstanding).

Independently read all raw front/rear coefficient records in each new case, restricted to `(148,149]`. Both branches contain exactly 200 samples with the exact expected `148 + .005*[1..200]` time grid. Recomputed means and fluctuation RMS match saved values exactly.

| Actual raw-force statistic | H5 MPC | Matched zero |
| --- | ---: | ---: |
| Mean total Cd | 2.4137825099145 | 2.413592168615 |
| Mean rear Cl | 0.8665601175300001 | 0.8871138481919999 |
| Rear Cl fluctuation RMS | 0.3889004655283842 | 0.3953805314509264 |
| Peak absolute rear Cl | 1.459758176 | 1.468057611 |

Paired drag reduction `1-Cd_MPC/Cd_zero` is **−0.0000788622460641264**, i.e. drag is **0.0078862% worse**. Rear-Cl fluctuation RMS ratio is **0.9836105589246837** (1.63894% lower). H2 canonical-history feedback had exactly zero benefit; the earlier instantaneous-cost H2 trial had 0.232927% worse drag and RMS ratio0.943109. These are descriptive comparisons over one matched start and only 1 D/U, not evidence for the original long-window physical criteria.

## Prediction, cost and causality checks

Selected one-step physical force MAEs across ten cycles, ordered front Cd/front Cl/rear Cd/rear Cl, are `[0.00011434555053710938,0.0013359144330024719,0.005998331308364868,0.009711787104606628]`. Saved selected-prediction-versus-zero direction indicators agree with actual paired endpoint effects on respectively **7/10, 4/10, 6/10, 9/10** cycles. These indicators are not a pure same-state action-effect validation: after the first action, the controlled and zero trajectories differ, and an unexecuted alternative at the controlled state lacks CFD ground truth. They nevertheless warn against assuming all immediate drag-response signs are correct. The measured one-step errors coexist with a near-zero integrated drag difference; neither alone proves that short-window transience is the sole cause.

Reconstructed initial actual62 history from its bound raw CFD sources, then advanced persistent history only with each measured MPC endpoint. Under the actual Python3.12/NumPy runtime, independently recomputed all **50 candidate costs and 250 per-stage ledgers/components**, with exact equality to saved records. Final history ends at149.0; predicted endpoints never enter persistent history. Maximum predicted canonical mean-bias ratio is `0.006794879498193016`; all250 mean-bias penalty terms are zero. Therefore **the physical 10% mean-lift-bias term did not block this trial's action selection**. Relaxing this inactive term cannot explain or repair the observed result. The positive one-D/U physical mean in the table is a different statistic from the causal62-history ledger.

## Resources and cleanup

CPU inference latencies span `9.529155479976907`–`10.623129790998064` seconds per cycle; total unit wall interval is approximately241 seconds. This is not real-time performance at a0.1D/U control interval. Sixty saved resource observations give minimum MemAvailable `121418903552` bytes and MemFree `1399365632` bytes. This approved CPU workflow uses MemAvailable startup50/runtime22GiB, not a MemFree floor; low file-cache-adjusted MemFree is not reported as a GPU experiment.

Both saved owned-container records report exit137 and OOMKilled=false, consistent with deliberate termination of the persistent idle containers after solver completion. Both full CIDs were independently absent from `docker container ls -aq --no-trunc`:

- `a0dd70055437ef4ce4ab7943a5b2521554ad1b3036fd8c15a4a8bc32bd657e25`; terminal JSON SHA256 `5377b368f137c5c0acba26b47ba6423b74c5cc90fd2d08f59c0ff34313f7c652`.
- `56dd52d7742db47878c2ee8a40f772fc110a666162ef575831e1fb1ce0229091`; terminal JSON SHA256 `cca7a29f4193905c8ae9c58e6185e2729c986cf24423de59180a7f31dfc23495`.

## Next bounded hypothesis

H5 resolves the H2 HOLD behavior but has **not demonstrated drag reduction**. The 1D/U near-zero drag result is insufficient grounds either to declare success or to reject this fixed controller solely on long-window performance. A separately approved, fixed H5 matched-zero trial over approximately two shedding periods (e.g.12.3D/U,123 cycles) is a reasonable next test, retaining the same objective, weights, candidates and constraints. Predeclare the full-window paired statistics and inspect temporal drag accumulation plus prediction errors; do not select a favorable subwindow or sweep horizons/weights. It remains shorter than the original80D/U physical assessment.

Resolve execution throughput using the separately reviewed precision and Curator engineering before such a run, with any GPU inference override explicitly declared and checked on multiple actual states. No new model is a mandatory prerequisite for this exploratory test, and no further execution is authorized by this report. Existing model-admission failures and original physical thresholds remain unchanged.
