# FC-E085 — fixed canonical policy b01 real-CFD terminal review

Independent read-only raw audit, 2026-10-07. Actual unit `fluid-control-p064-b-symmetry-canonical-ppo-b01-long-cfd-20261007.service`, invocation `3a678d0c1d604f0eb821255c2848fdd4`, MainPID=0, SubState=exited, ExecMainStatus=0. No solver, model or policy inference was rerun for this review.

## Physical outcome

The fixed E082 canonical policy passes the original primary physical criteria at b01: **4.0090689485% drag reduction**, rear lift fluctuation RMS ratio **0.8171901322** (18.2810% lower), absolute mean rear lift / paired-zero fluctuation RMS **3.6366076181%**. All six preregistered windows pass unchanged drag>=2%, RMS ratio<=1.05, bias<=10%. In this run there is **no early-window failure**: first6.2 bias is8.0614%. This does not erase E083 b00 first6.2 bias17.5561% FAIL, or earlier policies' early failures.

| Window | Samples/branch | Drag reduction % | Rear Cl fluctuation RMS ratio | Mean-bias ratio % |
|---|---:|---:|---:|---:|
|early `(130,142.4]`|2480|5.4339718765|0.9181581792|3.9147005324|
|first `(130,136.2]`|1240|5.8700750911|0.9816194239|8.0613985373|
|trailing `(136.2,142.4]`|1240|4.9978257764|0.8479563067|0.2317213707|
|primary `(150,210]`|12000|4.0090689485|0.8171901322|3.6366076181|
|historical companion `[150,210]`|12001|4.0089803915|0.8171935391|3.6276577148|
|full `(130,210]`|16000|4.2407884776|0.8347686669|0.2679444480|

## Checks executed

All3200 raw force-file SHA256 values were recomputed. Four streams each contain16000 finite unique samples on the exact0.005 grid from130.005 to210. All branch means, centered/total lift RMS, rear absolute lift peak and paired metrics were independently recalculated with explicit open-left/inclusive companion selections; maximum difference from saved metrics4.440892098500626e-16.

All800 physical69 observations were independently canonicalized in FP32: reverse32 probe positions, negate transverse velocities, front/rear Cl and omega; first maximum odd-component pivot, direction, margin and fixed flag match the recorded canonical observations. Every sign-restored request and next orientation matches, followed by the original one physical slew/amplitude filter and current/next action clock. No old two-policy half-difference formula is used.

All1600 solver logs have20 time steps and clean End without FOAM FATAL. Two owned terminal container receipts show no OOM,8GiB/no-excess-swap limits and no remaining owned IDs. Result records source restart unchanged and no online FNO/MPC/scientific admission. Previously reviewed unchanged source closure was not needlessly rehashed. Wall1234.018868929008s;4431 resource samples, minimum actual MemAvailable119373869056 bytes, above22GiB.

Both complete zero streams match the old B b01 reference exactly across all16000 rows and **all columns**; old zero hashes also rechecked. Old B b01 primary drag reduction3.9275159299%, RMS ratio0.8157277923, bias2.7294976565%. These small mixed differences do not establish meaningful superiority. b01 is already opened development data, not an independent untouched holdout; two phase successes do not establish statistical independence or universal robustness.

## Action-square proxy

From all800 applied endpoints: mean omega² **0.24044747071480532**, omega RMS **0.4903544337668472**, rectangular sum omega²×0.1D/U **19.235797657184428**, delta-omega RMS with first previous action0 **0.06272814087822465**. Maximum |omega|0.6657415628433228; maximum |delta|0.1 within floating precision. Old B b01 values respectively0.22048854397911996/0.46956207681106443/17.639083518329596/0.051886154615702236. These are action-cost proxies, not physical power or net energy. Torque/power conversion has not been validated here.

## Provenance and limits

- Result `artifacts/p064_b_symmetry_canonical_ppo_b01_long_cfd_20261007/result.json`: `c56a5cbccdf218953a0e1ea040da1c1ee7e2f1b574e7b509939a622f65b7495f`.
- Approval `docs/P064_B_SYMMETRY_CANONICAL_B01_CFD_APPROVAL_20261007.json`: `ee010bbe1932e0b48b2f2e90a1b9dd77d463f86dad6367e0c0dd49a46f0b45f1`.
- Immutable driver `artifacts/p064_symmetry_canonical_b01_cfd_source_20261007_immutable/run_p064_symmetry_canonical_b_ppo_b01_long_cfd.py`: `ebc6f2663e5eeae5d22a6e4ebfd06debd4ec2672f31accfc1db30f67131e25ba`.
- Fixed E082 policy `5c05699e0851787d85d40c407647f80c19d3aebeb7dff82e019336cde77c6c6e`, Vec `1d25005144b6436c3e2641ee89d1585e3c9f8b9fdb1f26b9cd39c7d83610c145`, canonical adapter `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`.
- Old B b01 result `artifacts/p064_b_projected_ppo_b01_long_cfd_20261006/result.json`: `0be19e0dfdf8df4d60e2f5040673f2a3ec25cbadb133548671ce31f061e75c88`.

This is genuine CPU-policy/OpenFOAM feedback after FNO-based PPO training, not online FNO/MPC. Full B surrogate prediction admission remains FAIL; no whole-project completion or new execution permission follows. Concurrent FC-E086 uses a different seed's final policy and has separate output/approval; its outcome is not inferred from this success.
