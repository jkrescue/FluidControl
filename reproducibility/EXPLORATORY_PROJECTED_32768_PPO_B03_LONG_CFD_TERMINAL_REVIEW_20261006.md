# FC-E061 — independent b03 paired-CFD terminal review

The fixed projected-policy b03 trial completed 800 paired intervals. Actual unit `fluid-control-exploratory-projected-32768-ppo-b03-long-cfd-20261006.service`, invocation `47612677a9f64dfc968917fada5e9ba8`, is active/exited with MainPID0 and ExecMainStatus0. Runtime reported 1099.821862204 seconds. No restart or additional scientific execution was performed by this review.

Result: `artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006/result.json`, SHA256 `d4d755faf913d393ca1466ee74614c0fb11f33b8f662a76a1cb7e4de3dd17a2f`. Approval SHA256 `3ca5531815c48cff59fd1ca0d96e40e2305402435cb2e28d2adb26eeaf9328a6`; executed driver SHA256 `6516456f07d765728055f58036bed97a5e37036f205c4d1ee2bf2456be8cc3b0`.

## Independent raw-data calculation

All 3200 recorded coefficient-file hashes were recomputed. The four concatenated front/rear × controlled/zero streams each contain exactly 16000 samples on the 144.005:0.005:224 grid, without duplicate removal, interpolation or missing samples. Header columns were verified as time, Cd at column1 and Cl at column4. Every saved branch metric and paired ratio for all six preregistered windows was recomputed independently; maximum absolute difference was 8.881784197001252e-16.

|Window|Samples per branch|Drag reduction|Rear-Cl centered RMS ratio|Absolute rear-Cl mean / paired-zero centered RMS|
|---|---:|---:|---:|---:|
|(144,156.4]|2480|2.61682122%|0.8496429150|0.0347139447|
|(144,150.2]|1240|2.15053838%|0.8804771060|0.0645066180|
|(150.2,156.4]|1240|3.08312249%|0.8165615766|0.0049217445|
|Primary (164,224]|12000|3.89714021%|0.8148587790|0.0168885808|
|Historical companion [164,224]|12001|3.89718369%|0.8148537854|0.0169324719|
|(144,224]|16000|3.69070682%|0.8200777565|0.0138235002|

The primary satisfies all original physical references: drag reduction ≥2%, centered lift RMS ratio ≤1.05, mean-lift bias ratio ≤0.10. Its rear-lift fluctuation RMS is 18.51412210% lower and normalized mean bias is 1.68885808%. All six reported windows also satisfy those three references; the inclusive historical companion is not substituted for the primary. The 20% bias reference remains sensitivity only, not a relaxed criterion used to obtain this result.

## Identity, execution and resources

The approval, nine project-source bindings, immutable driver and original restart144/constant/system tree hashes were independently rechecked. All 800 saved projection decisions match `0.5*(raw_policy_request-reflected_policy_request)` and the single unchanged amplitude/slew filter. Maximum absolute applied rotation was 0.6525258422, maximum increment approximately0.1, no saturated endpoints and five rate-limited endpoints. This review did not deserialize or rerun the policy; its frozen identity remains the approval-bound final32768 PPO, with no online FNO or MPC substitution.

All 1600 solver logs contain a clean End and no FOAM FATAL; all recorded solver segments contain 20 steps. Independently scanned maximum Courant number was 0.245351545. The 4000 saved resource observations have minimum MemAvailable 121825087488 bytes, above the 22GiB operating guard and 20GiB reserve.

Both owned container IDs were checked absent from the current Docker container list. Their retained terminal receipts show no OOM. ExitCode137 belongs to the deliberately terminated idle `while :; do sleep 3600; done` container processes, not failed CFD solver segments; the solver logs and actual driver unit independently establish successful completion. Original source trees remain unchanged.

## Scope

This is successful fixed-policy physical confirmation at b03, complementing the earlier b00/b01 results for the same Re100 and L/D5 configuration. It does not establish independent statistical samples or universal phase/geometry robustness. b03 fixed-action H1–H5 data had already been opened under FC-E060; this is a new physical-policy trajectory, not a universally unseen-phase claim. Short-time prediction evidence and the preserved K1 H100 development FAIL remain separate. No threshold tuning, checkpoint selection or new training followed from this review.
