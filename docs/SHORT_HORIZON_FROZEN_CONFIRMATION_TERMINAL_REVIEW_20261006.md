# FC-E060 — prospective sealed H1–H5 confirmation completed

Actual unit `fluid-control-short-horizon-frozen-confirmation-20261006.service`, invocation `7a887ed792ec4d60a429f4a7a3660b3a`, completed PID0/active-exited/exit0. Lead-authorized first registered confirmation opening followed candidate freeze; original seal and preparation drafts remain unchanged. No optimization, new CFD, checkpoint selection or retry occurred. The concurrent b01 CFD was untouched.

Approval SHA `1ae960dcee7c9aa81e438a990dbf88397f5593decd1cdbb3ba1336ea4888c162`; R3 immutable372-source closure; actual worker `f64b0f5aa714c087f1b0cac3e6b70773fbd353ace3b89212da8326cd4a8b9eec`, derived evaluator `b1bd8e28765439ce93ae05070f26b9cabce3730917b26fe1bf6ee8e4da1ab162`. Source capture in Git is retrospective, not launchHEAD.

Artifacts under `artifacts/short_horizon_frozen_confirmation_20261006/`:

- `payload/result.json`: `77ab4fb85f238d1e76b6e5a20c18f8992182d24a82b38d7b4b5a52483f9fce20`.
- `payload/evaluation.json`: `cab65822d1c8ce5588cbcd6e1ce105d245c62d9214067a1befa4c14993fa95ec`.
- `payload/segments.json`: `fb39a1efe690424c92267c7f01f6bf4c2ab54df0d9615e95c35a783ae6a8250d`.
- `payload/coverage.json`: `0e8ffcc2d0b33101ff6d82d75c853d70185d9513b655f8d265b7f278148a641b`.
- `supervisor_result.json`: `1135443598a5ed59a25f8cc7e9f5ae4635682743fcfc091778e95f50d5898f3a`.

## Actual scope and metrics

All ten frozen b03/b07 fixed-action cases, starts0:25:775, H1–H5:1600 endpoints total,320 perhorizon, failed0/nonfinite0. Official installed PhysicsNeMo HDF5Reader read dynamic arrays; h5py read static metadata. K1 manifest7adca21e…ecc7 and train normalizationf1b460…2bc1 remained fixed. Historical checkpoint high/TF32 identity was validated before explicit highest/noTF32 inference. Actual output records highest/false/false and6442450944-byte allocator cap. Default evaluator and H100 protocol were not modified.

|Endpoint horizon|Pooled velocity relativeL2|Pooled pressure relativeL2|TotalCd MAE|RearCl MAE|
|---|---:|---:|---:|---:|
|H1|0.0022426952|0.0064234466|0.0070819531|0.0191755319|
|H2|0.0043465692|0.0121630376|0.0077086698|0.0195609978|
|H3|0.0062701259|0.0175132911|0.0087761424|0.0204336208|
|H4|0.0080011590|0.0224016454|0.0097627319|0.0213490572|
|H5|0.0095414810|0.0268274147|0.0106288686|0.0221162728|

H1/H5 all-force MAEs respectively: frontCd0.0002451718/0.0002241347; frontCl0.0015307362/0.0018775447; rearCd0.0072165665/0.0107073760; rearCl0.0191755319/0.0221162728. Relative field metrics pool physical SSE/reference sums before sqrt; they are not means of case-relative ratios. Pressure is ROI-centered kinematic solver pressure, not absolute pressure. Force MAEs are coefficients. H1 is true-current-state conditioning; later endpoints are free autoregression, not teacher forcing.

Independent reviewer recomputed savedJSON coverage, four-force errors, pooled field sufficient statistics and eight start0 action-minus-zero pairs. Maximum tiny discrepancy from FP32 versus Python-double subtraction was≤1.3e-11. Start0 delta-totalCd MAEs H1..H5:0.001690387726,0.002159953117,0.005255877972,0.008219659328,0.012037873268. Delta-rearCl:0.002800524235,0.005820468068,0.011684849858,0.013544742018,0.011757295579. These are same-phase shared-start responses; later distinct-state trajectories are not presented as causal action interventions. Reviewer did not reread HDFs or reload models.

## Resources and interpretation

Actual supervisor12GiB/no-swap,1CPU; errornull/return0; minimum MemAvailable117072576512bytes, above22GiB guard and20GiB reserve. Process-group supervisor completed without retry. CPU preparation checks:2 actual-reader synthetic tests,3 source transformation/reversibility tests and4 worker authorization/guard tests passed; these are distinct from actual GPU evidence.

This is quantitative short-time prediction evidence on predeclared previously sealed fixed-action phases: H5 velocity error0.954% and centered-pressure error2.683%, with explicit force errors above. No arbitrary post-hoc categorical accuracy gate is invented. It does not establish arbitrary policy-action-distribution accuracy, long-rollout accuracy, all-phase physical robustness, or replace preserved K1 H100 development FAIL. Prediction accuracy is separate from original physical mean-lift10% and20%sensitivity; changing either physical threshold cannot change these prediction errors. No automatic retraining/tuning follows this final-test opening.
