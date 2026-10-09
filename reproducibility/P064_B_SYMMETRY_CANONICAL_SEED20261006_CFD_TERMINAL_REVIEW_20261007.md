# FC-E086 — canonical seed20261006 b00 CFD terminal review

Independent raw-data audit, 2026-10-07. Actual unit `fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-long-cfd-20261007.service`, invocation `be48ee07057c42879d222548045233ef`, MainPID=0, SubState=exited, ExecMainStatus=0. No solver/model/policy inference was rerun for review.

## Result and unchanged criteria

Primary `(168,228]` passes original drag reduction>=2%, rear lift fluctuation RMS ratio<=1.05, and absolute mean rear lift/paired-zero fluctuation RMS<=10%: **3.7440147745% drag reduction**, **0.7737896930 RMS ratio** (22.6210% lower), **3.6740531816% mean-bias ratio**.

Early12.4 and first6.2 windows fail mean bias at **11.2623%** and **21.1797%** respectively. The first6.2 also exceeds20% sensitivity; neither10% nor20% is silently changed. The other four windows pass all original criteria. Physical success does not change the full B surrogate-prediction FAIL.

| Window | Samples/branch | Drag reduction % | Rear Cl fluctuation RMS ratio | Mean-bias ratio % | Original criteria |
|---|---:|---:|---:|---:|---|
|early `(148,160.4]`|2480|6.0172583030|0.9200776306|11.2622520531|FAIL bias|
|first `(148,154.2]`|1240|6.4693222678|0.9938978884|21.1797126892|FAIL bias|
|trailing `(154.2,160.4]`|1240|5.5651808382|0.8280021326|1.3449659002|PASS|
|primary `(168,228]`|12000|3.7440147745|0.7737896930|3.6740531816|PASS|
|historical companion `[168,228]`|12001|3.7439190435|0.7737967937|3.6643747487|PASS|
|full `(148,228]`|16000|4.1924958456|0.8014660264|0.2568376236|PASS|

## Independent checks executed

All3200 recorded raw force-file SHA256 values were recomputed. The four front/rear and controlled/zero streams each contain16000 finite unique samples on the0.005 grid. All six explicit open-left/inclusive-companion windows, branch means/centered and total lift RMS/rear peak and paired metrics recompute within4.440892098500626e-16 of the saved result.

All800 physical69 input observations were independently canonicalized in FP32 (reverse32 probes; negate transverse velocities, front/rear Cl and omega). First maximum odd-component pivot, direction, margin, fixed flag and full canonical observations match. Physical requests equal FP32 orientation×canonical policy request; output observations reproduce every next orientation. Original single slew/amplitude filter and actual action clocks pass. No old two-request half-difference projection is used.

All1600 solver logs contain20 steps and clean End, without FOAM FATAL. Two owned terminal receipts show no OOM, approved8GiB/no-excess-swap constraints; both container IDs are absent. Source restart preservation is recorded by the executed result, with no online FNO/MPC/scientific admission. Previously validated unchanged source closure was not redundantly rehashed. Wall1108.8715211490053s;4000 memory samples; minimum actual MemAvailable121698250752 bytes, above22GiB.

Both complete zero force streams match the old trained B b00 reference exactly across16000 rows and all columns, with old zero-file hashes rechecked. E083 zero was independently verified against that same reference, so the two canonical-seed comparisons share the same zero series.

## Fixed-seed interpretation

The predeclared seed20261007 E083 primary was3.9567229236% drag reduction/RMS ratio0.8165429747/bias1.0659980225%; this seed20261006 gives3.7440147745%/0.7737896930/3.6740531816%. Both specified seeds pass the original primary criteria with frozen B, same canonical interface, PPO budget, reset panel, reward and b00 evaluation. This supports reproducibility across **these two fixed seeds only**, not arbitrary-seed robustness, independent unseen-state generalization or a statistically significant superiority claim. Early bias differs and remains failing. The earlier noncanonical seed20261007 negative result remains archived, as does H25 rejection. No additional seed, tuning or execution is automatically authorized.

## Action-square proxy

All800 actual endpoint actions give mean omega² **0.3043258170340099**, omega RMS **0.5516573366085236**, rectangular sum omega²×0.1D/U **24.346065362720793**, delta-omega RMS (first previous action0) **0.06592147642424317**. Maximum |omega|0.686103880405426, maximum |delta|0.1 within floating precision. E083 proxies were0.24215872756405044/0.4920962584332972/19.37269820512404/0.06253740728244239. These are descriptive action costs, not physical energy; torque/power conversion was not validated and no net energy saving is claimed.

## Bindings

- Result `artifacts/p064_b_symmetry_canonical_seed20261006_ppo_long_cfd_20261007/result.json`: `9a7db1cff37e614dbefa908ec132cd8376cdfddd1f3149e2ba3436cebad57df4`.
- Approval `docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_APPROVAL_20261007.json`: `da5f4f7681174f24fc5315097de723cbe47e71679b7c202aee11a6277e6bbe52`.
- Immutable driver `artifacts/p064_symmetry_canonical_seed20261006_cfd_source_20261007_immutable/run_p064_symmetry_canonical_seed20261006_b_ppo_long_cfd.py`: `fd51fa43c16800f3549a2a63baef80a7d3948254f38f3353fa46727db201f1ce`.
- Policy `edd616c507798e355776c420974f53c256d78af04170dab77fe63beb1af0baa5`, Vec `3035aad7e6ff5dd0a789a3a56bc6029b137221ffc0297cc223d6325825cf9018`; shared canonical adapter `a55b569986b6e62fd23d1c46dbe4f117659795aef3953cac81359506d1ac38ae`.
- E084 training review `docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_TERMINAL_REVIEW_20261007.md`: `d2bf82ce004e6d07213d50d930d0d58b4bafa009c44d87972ca0bdb8081aa707`.
- E083 result `165b78194f84676b5ea0091b1d160ccf25f913a9f49c74a9424a7d0f03cde7dc`; old B b00 result `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`.

FNO was used for PPO training; deployment is CPU policy plus actual OpenFOAM feedback, not online FNO/MPC. The full high-accuracy-surrogate plus constrained-control goal is not declared complete.
