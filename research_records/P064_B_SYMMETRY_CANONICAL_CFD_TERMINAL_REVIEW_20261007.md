# FC-E083 — canonical-coordinate PPO real-CFD terminal review

Independent read-only review, 2026-10-07. Actual unit `fluid-control-p064-b-symmetry-canonical-ppo-long-cfd-20261007.service`, invocation `545ba2aebfb6412aaf41ee3281ccc802`, finished with MainPID=0, SubState=exited, ExecMainStatus=0. No solver or model was rerun for this review.

## Outcome

The primary `(168,228]` window passes the unchanged physical criteria: total drag reduction **3.9567229236%**, rear lift fluctuation RMS ratio **0.8165429747** (18.3457% lower), and absolute mean rear lift / paired-zero fluctuation RMS **1.0659980225%**. Criteria remain drag reduction ≥2%, RMS ratio ≤1.05, mean-bias ratio ≤10%.

The first 6.2 D/U window fails the original mean-bias criterion (17.5561%); this is not an all-window pass. No surrogate scientific admission or whole-project completion is inferred. The frozen B surrogate's full prediction-accuracy failure remains unchanged.

| Window | Samples/branch | Drag reduction % | Rear Cl fluctuation RMS ratio | Mean-bias ratio % | Original physical criteria |
|---|---:|---:|---:|---:|---|
| early `(148,160.4]` |2480|5.9068297176|0.9435097641|8.3477957333|PASS|
| first `(148,154.2]` |1240|6.1886429051|1.0147467507|17.5561055128|FAIL — mean bias|
| trailing `(154.2,160.4]` |1240|5.6250081144|0.8565964999|0.8603520397|PASS|
| primary `(168,228]` |12000|3.9567229236|0.8165429747|1.0659980225|PASS|
| historical companion `[168,228]` |12001|3.9566098775|0.8165450161|1.0562736876|PASS|
| full `(148,228]` |16000|4.3389753899|0.8391702849|1.6975219801|PASS|

## Independent checks actually executed

All 3,200 recorded raw force-file SHA256 values were recomputed. Four front/rear, controlled/zero streams contain exactly 16,000 finite, unique samples on the 0.005 grid. All six windows were independently selected with their declared endpoint conventions; means, centered/total lift RMS, peak lift and paired metrics agree with the result to at most 4.440892098500626e-16.

All 800 saved physical 69-component input observations were independently reflected in FP32 (reverse probe order, negate transverse velocities, both lift coefficients and omega). The first maximum odd-component pivot, orientation, margin, reflection-fixed flag and full canonical observation match exactly. Physical requests equal FP32 orientation × canonical policy request; all next orientations recompute from saved physical output observations. The original single slew/amplitude filter and current/next applied-action observation clocks pass. This controller uses one canonical policy call and sign restoration, not the old average of two policy calls.

All 1,600 solver logs contain 20 steps and clean End, without FOAM FATAL. Both terminal container receipts show no OOM and the approved 8 GiB/no-excess-swap limits; both owned container IDs are absent. Source-restart preservation is recorded in the executed result; unchanged prelaunch source closure was not redundantly rehashed. There was no online FNO or MPC selection.

All columns of both complete zero force streams are exactly equal to the previously reviewed B b00 run, not merely the first interval. The old zero file hashes were rechecked. Wall time: 1109.2949996789685 s. Across 4,000 resource samples, minimum actual MemAvailable was 122,369,617,920 bytes, above the 22 GiB runtime floor.

## Matched interpretation and action cost

The same second seed 20261007 without canonical-coordinate training/control (FC-E080) had primary drag reduction **−0.6174007162%**, RMS ratio 1.0062202287 and bias 1.6945780148%. With the same B model, seed, PPO budget, reset panel and reward, the predeclared coordinate-wrapper intervention recovered a physically effective b00 controller. It changes both training coordinates and deployed action mapping; this single experiment does not isolate those two contributions or establish cross-seed/general-phase robustness.

The earlier successful first seed had 3.8952838833% primary drag reduction, RMS ratio 0.8156230434 and bias 1.1378146878%. The small drag difference is not evidence of meaningful superiority. Prior failed trials remain part of the evidence.

From all 800 actual applied endpoint actions, mean omega² = **0.24215872756405044**, omega RMS = **0.4920962584332972**, rectangular sum omega² × 0.1 D/U = **19.37269820512404**, and delta-omega RMS (first previous action zero) = **0.06253740728244239**. Maximum |omega|=0.7 and maximum |delta|=0.1 (floating representation 0.10000000000000003). Old successful B values were respectively 0.21966350539302873, 0.46868273425957213, 17.5730804314423 and 0.0522090388340161. These are action-square proxies, not physical energy. Torque and power conversion have not been validated here; no net energy-saving claim is made.

## Exact provenance

- Result: `artifacts/p064_b_symmetry_canonical_ppo_long_cfd_20261007/result.json`, SHA256 `165b78194f84676b5ea0091b1d160ccf25f913a9f49c74a9424a7d0f03cde7dc`.
- Approval: `docs/P064_B_SYMMETRY_CANONICAL_CFD_APPROVAL_20261007.json`, SHA256 `afb03b9e86931d1031d8e0ab1ce76dc180816ac76446d46823cf4aa350ee9f1e`.
- Executed immutable driver: `artifacts/p064_symmetry_canonical_cfd_source_20261007_immutable/scripts/run_p064_symmetry_canonical_b_ppo_long_cfd.py`, SHA256 `11096d99c684de170138f1bdd4eedf409211512af5cd4665e0523228f167ae2a`.
- Shared canonical wrapper SHA256 `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`.
- Training review: `docs/P064_B_SYMMETRY_CANONICAL_PPO_TERMINAL_REVIEW_20261007.md`, SHA256 `72a03435e0ed954743694ca785e8de1616c6860db457b2ad20ef6f40d330e562`.
- E080 result SHA256 `6221a7d2f8868eba8622f2e6b76d110d206dd4d9304627d890e01d71b6a7d893`; old successful B result SHA256 `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`.

This is actual CPU-policy/OpenFOAM feedback after FNO-based PPO training. It is neither online FNO inference nor a new surrogate-accuracy pass. No next experiment was executed or automatically authorized by this audit.
