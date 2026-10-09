# P064-B complete prediction evaluation: independent terminal review

2026-10-06. Engineering execution completed; **DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL** remains the scientific conclusion. The four rotating branches fail the original retained lift-fluctuation prediction requirement. This does not erase the separately verified real-CFD closed-loop benefit, and that benefit does not establish surrogate admission.

## Executed identity and recovery

Unit `fluid-control-p064-b-formal-r3-20261006.service`, invocation `3bded2dcb4a24f808879987962a9ef8b`, independently queried MainPID0 / active-exited / ExecMainStatus0. Output `artifacts/fcp064_arm_b_formal_resume_r3_20261006`; receipt SHA `30d3d0746580b8423a9f626a0ebe1b76c129acab7800158f74d7d8df3d8a1799`. Approval SHA `19be2d6aad903ffc94b807803bd5fd0902c7ec5b7a0f0b4212423744db89cb56`.

R2 invocation `01806bdc150841f7b9efd04360a441f2` failed because validation_diagnostic argparse omitted the P064 kind. Its completed precision and validation10 evidence were not recomputed. All nine approved prior-file hashes were independently verified: eight files copied byte-identically, and the old memory log retained as the exact prefix of the combined log. R3 executed the six remaining stages. The receipt explicitly records both invocations. R1/R2 failures and outputs remain preserved; the CLI repair is not a scientific change.

Independently rehashed all 35 receipt output files, all 411 numerical-source files, and all seven candidate files without model deserialization. Source-chain receipt SHA `1be4606582f1d96fa9ab7aeb1b13e45b6ef09f702c9278c5de1580b5a451aaa4`; resume runner `3bb215f93f5d6d468f8b22267b5f1fb485516f865c45d32723fe33b3fe56ec84`. Candidate manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`; aerodynamic archive `57d4634df22ce96c1c4467a2ed52412be452375129af05b89f10a690e363356e`; frozen flow archive `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`. This review did not independently reload tensors; existing approved CPU reload evidence remains the tensor-loading proof. No model/HDF/CFD rerun was performed.

## Matched endpoint results, not mixed aggregations

Comparator: `artifacts/fcp026_history_training_k1_20261005/posteval_fc_p026_k1`. Only identical panels/horizons are compared below. Summary NRMSE is a mean across cases; the diagnostic pools squared errors/reference sums before taking the square root. These are different quantities.

| H100 panel / metric | K1 | B |
|---|---:|---:|
| validation10 rear-Cl MAE | .04018628365 | .04311831568 |
| validation10 total-Cd MAE | .01092804753 | .01126262278 |
| validation10 pooled Cd NRMSE | .00609718127 | .00626446610 |
| validation10 mean-case Cd NRMSE | .00580968247 | .00603305750 |
| validation10 velocity relative L2 | .04359112110 | .04359112110 |
| dynamic6 rear-Cl MAE | .08498129532 | .08639715487 |
| dynamic6 total-Cd MAE | .02844553694 | .02853910227 |
| dynamic6 pooled Cd NRMSE | .01789316332 | .01783577002 |
| dynamic6 mean-case Cd NRMSE | .01571786983 | .01579845614 |
| dynamic6 velocity relative L2 | .08721070016 | .08721070016 |

Recomputed H100 pooled metrics from 290 validation and 606 dynamic segments. Saved diagnostic values differ from recomputation at approximately 1e-11 from recorded FP32-derived quantities, not a material discrepancy. Validation10's earlier full 1240-endpoint review also found B rear-Cl MAE worse at H1/10/50/100; see `P064_B_FORMAL_R2_PARTIAL_TERMINAL_REVIEW_20261006.md`. Frozen flow fields do not improve.

Strict start0 action-minus-zero Cd MAE: validation K1 .01912969351 → B .01933880150; dynamic K1 .01039302349 → B .00987213850. Independently recomputed all 8/4 delta pairs and 20/6 ordering pairs; signs and ordering are correct. These endpoint gates pass, but do not substitute for a force time-window gate. The separate force-window producer's six final endpoints give pooled NRMSE .00438642619 and delta MAE .00980696082; its different sampling/producer aggregation must not be conflated with the rolling dynamic6 summary.

## Original retained H100 force-window failure

Raw saved force series: `force_window/result.json`, SHA `6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173`. Independently recomputed all six branch statistics from indices39..100: 62 samples spaced nominally .1 D/U, spanning about6.1 D/U as the discrete representation of the requested trailing6.15. Actual FP32 timestamps are retained.

| Branch | Cd mean absolute error | Cl fluctuation RMS absolute error | Cl mean absolute error | Joint |
|---|---:|---:|---:|---|
| b01 minus | .0066673131 | .0483455120 | .0062311939 | FAIL |
| b01 zero | .0008770708 | .0269937708 | .0016041096 | PASS |
| b01 plus | .0130710756 | .1439358617 | .0202747596 | FAIL |
| b05 minus | .0167993413 | .0912204925 | .0496610026 | FAIL |
| b05 zero | .0060720069 | .0247019132 | .0277044286 | PASS |
| b05 plus | .0260702439 | .0620918664 | .0092308562 | FAIL |

Thresholds remain 1% of same-window zero Cd for Cd, and2.5% of zero Cl fluctuation RMS for both lift statistics. Lift error limits are .0294155092 (b01) and .0293871658 (b05). B passes Cd5/6, fluctuation2/6, mean5/6, jointly2/6; K1 jointly1/6. All four rotating branches still fail fluctuation fidelity. Mixed changes—including worse zero-branch RMS errors—do not establish broad improvement. This is a force-statistics failure, not an invented H100 field-L2 threshold failure. The development standard's historical provenance remains distinct from final real-CFD physical2%/1.05/10% criteria; neither was relaxed.

## Resources, cleanup and interpretation

All eight combined container terminal records show exit0, not running and no OOM; independent current `docker ps` was empty. Combined R2/R3 memory log has1127 samples, minimum MemAvailable116345077760 bytes, above the20GiB reserve. The actual PyTorch allocator fraction was .15; outer .06 and inner guard fractions are startup accounting, not a .06 enforced allocation cap. Docker72GiB/no excess swap and runtime MemAvailable22GiB protection were retained under the recorded Lead continuation authorization.

The final receipt explicitly sets scientific_admission, ppo_auto_launched and frozen_test_accessed false. Existing B-trained projected PPO has already completed real online CFD feedback at b00/b01 with primary physical criteria met; no FNO forward is called by that deployed policy. Those observed/development phases are not fresh independent tests. Current high-accuracy surrogate completion remains unmet. The separately authorized saved-JSON same-six-case true-field H1 versus free-AR comparison has already completed: `docs/P064_B_SAME6_CACHED_H1_AR_REVIEW_20261006.md`, SHA `c7433e7b7bc364ded006d9ae17d1f1afea9465ce461385b57b926e60dc8ba1b7`, source/report commit `ed638be`. Its batch8-versus1 and precision limitations remain explicit; AR minus H1 is not a causal decomposition. The next preparation is a signed H1 600-endpoint batch1 diagnostic, not yet executed. No automatic retraining, threshold relaxation or larger architecture is authorized by this review.
