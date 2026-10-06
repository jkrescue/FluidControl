# FC-E051 independent terminal review — operational loop complete, physical goal incomplete

The unique final surrogate-trained SB3 PPO policy actually drove paired real OpenFOAM for 124 cycles, without online FNO or MPC action substitution. Unit `fluid-control-exploratory-final-ppo-cfd-20261006.service`, invocation `fd6d92f7ea9946b49c91c07e21f1d74b`, ran 2026-10-06 07:07:51–07:10:18 UTC and is PID0/exit0/Result=success. Reported work duration: 144.2376935010543 seconds. This is exploratory 12.4D/U, not original 80D/U physical admission or repair of K1's full formal FAIL.

## Identity and independent checks

- Approval `docs/EXPLORATORY_FINAL_PPO_CFD_APPROVAL_20261006.json`: `7ace192519a08795fe9217473fae33941fc5edbb1075daeeb3701e672c521cb3`.
- Executed immutable driver: `44b488a97a2882e1325da8871d3ac4905cdae2a6f2cbb17202ced91afc58b91a`; later source capture commit `f473abe` is not claimed as launch HEAD.
- Result `artifacts/exploratory_final_ppo_real_cfd_20261006/result.json`: `4007493f22de5855cbd0574e0ec006ca715941b8396f4e48af6527dc11e03d47`.
- Final policy: `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1`; identity VecNormalize: `54a08a438501aac0663e50da931f41aabb63cdeb8255aa80051e7af1b4eaaba2`; actual training result: `138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4`.

Independently read the complete driver/protocol and nine-source closure, checked source hashes, and ran ten tiny CPU tests (0.07s). At terminal, rehashed all 496 generated coefficient files against recorded hashes and independently recomputed both branches from raw physical forces. Exactly 2480/1240/1240 strictly increasing, finite samples on the 0.005 grid belong to the predeclared open-left full/first/trailing windows. Means, centered RMS, total RMS and peaks agree with saved results within absolute 1e-14. All 248 branch solver segments report 20 steps and clean completion. No model loading, HDF rereading or CFD rerun was performed by this review.

## All predeclared windows

Drag reduction is `1 - mean(Cd_total_PPO)/mean(Cd_total_zero)`. Mean-bias ratio in this table is `abs(mean(Cl_rear_PPO))/RMS_fluct(Cl_rear_zero)` from the same paired window.

| Window | Samples/branch | Drag reduction | Rear-Cl fluctuation RMS ratio | Mean-bias ratio | 10% / 20% sensitivity |
|---|---:|---:|---:|---:|---|
| (148,160.4], full 12.4D/U | 2480 | +0.4117553020% | 1.1058626226 | 0.5273107117 | fail / fail |
| (148,154.2], first 6.2D/U | 1240 | −1.6358561333% | 1.1998320941 | 0.4172662479 | fail / fail |
| (154.2,160.4], trailing 6.2D/U | 1240 | +2.4594278852% | 0.9909864238 | 0.6373532396 | fail / fail |

Full PPO/zero total Cd means: 2.2903674709412902 / 2.29983717243621. Full rear-Cl means: −0.6211417245494072 / −0.00012979507508894752; centered RMS: 1.3026426379258063 / 1.1779425502882805. PPO absolute rear-Cl peak is 2.367989822 versus zero 1.64726217. First/trailing PPO means are −0.49151134437321126 / −0.7507721047256032; their centered RMS values are 1.4133208439467626 / 1.167335344029713.

For the unchanged existing train-b00 long-term zero reference RMS 1.1826535012844825 (`train20_physics_summary.json`, SHA `b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7`), the corresponding bias ratios are 0.5252102360 / 0.4156004644 / 0.6348200076: also all above both 10% and 20%. This separately labelled sensitivity calculation changes no criterion. Relaxation cannot repair the full-window RMS worsening or establish the missing long-window result.

## Actual policy behavior and limits

All 124 deterministic requested actions equal +0.75. Applied actions ramp .1, .2, .3, .4, .5, .6, .7, .75 and then remain saturated: 117/124 saturated endpoints, seven rate-limited endpoints. One canonical amplitude/slew filter and linear CFD boundary ramp are used. Actual 69-channel observations propagate from the preceding CFD segment, with the prescribed probe positions/order/physical units. Training grid interpolation and direct CFD probes are not asserted numerically identical.

Thus the learned policy executed, but on this trajectory behaved as a constant saturated controller, not demonstrated useful state-responsive control. H5 training from four zero starts only reached applied magnitude .5, whereas deployment reaches .75; this supports testing reset-distribution mismatch, not proving it is the sole cause. The 69 observations also omit the full 62-force reward history, making the reward history-dependent beyond the visible observation. Short-return bootstrap, reward-history dilution and surrogate bias remain alternative explanations. No architecture, reward weight or threshold change follows from this review.

The next approved scope is preparation/CPU verification of a separate fixed 24-real-start reset adapter (four original zero starts plus twenty base-train frame62 starts), preserving K1/H5/PPO4096 and the existing reward. It is not GPU training approval or a claim of causality. Existing CFD-only PPO success over its original 80D/U protocol remains separate.

## Resources and cleanup

620 observations: minimum MemAvailable 122930147328 bytes, above runtime22GiB. Resource trace SHA `2259a5ef8669419c4cf3045dbb27d92a6ec781940847a1ee191ca2c36b02a41e`; progress SHA `9e53233c299c7f1103937dce7c954f283c473149f67e4353d4e0add30ed9d727`. CPU controller and both solvers are each bounded at8GiB with no swap; no online GPU/FNO.

Saved terminal Docker evidence hashes `a31c327b7033872afa61b3b62285f2beb5c69c88881cc7bdfe1db0025ee3ecbd` and `d4dae01ebda55e0a98cbfc85a9da2ace543bf47ca042b382cfa4c94bf0c48dc0` show not-running/no-OOM. Both exact container IDs are independently absent from Docker. Exit137 belongs to the deliberately stopped sleeping container supervisor, not the clean solver executions. No separate cleanup.json exists. The executed driver reports original restart bytes unchanged; this review did not separately rehash the original field tree.

Conclusion: genuine FNO-trained PPO→real-CFD integration is operationally complete; constrained physical control is not. No admission, automatic rerun, or threshold selection.
