# FC-E093 — fixed-lag saved-force diagnostic

One Lead-authorized CPU-only execution completed at 2026-10-06 20:20:52 UTC. Unit `fluid-control-p064-fixed-lag-json-20261007.service`, invocation `a730e633fec4417b9386a532fdb1e46d`, PID0/exited/exit0. CPU usage57,086,000ns; startup MemAvailable123,796,348,928B. MemoryMax1GiB, swap0, CPU1, CUDA hidden,60s limit. MemoryPeak is unset, not zero. No model, HDF, training or CFD execution.

Source `5be17944684e1899f4562aa99d94ca3cd41863c76dc0fe16465686d993c2aa68`; tests `0443a3372dfb194afec55383c512e67dd2e5220aca1ab11ac82de8e9faf8c9c4`; prospective protocol `73edd122938718114da2b7393fb1a94b84f6792bd7ba53e2b9dd57c9f778b5c4`, under `/tmp/e093-review`. Author and independent reviewer each ran7 CPU fixtures successfully; independent source review accepted before execution.

Inputs are the signed batch1 H1 result `artifacts/p064_teacher_forced_h1_signed_force_20261006/result.json` SHA `1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef` and AR `artifacts/fcp064_arm_b_formal_resume_r3_20261006/force_window/result.json` SHA `6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173`. Exact six cases,600 H1 truth/action/time matches, HDF/model/norm identities passed. Existing [E073 review](P064_TEACHER_FORCED_H1_TERMINAL_REVIEW_20261006.md) establishes matching batch1/high-TF32 execution and initial fields; this CPU job did not revalidate runtime tensors.

Output `artifacts/p064_b_fixed_lag_saved_json_diagnostic_20261007/result.json` SHA `7738b963c474b6d62aa42037095a11100e71c9154e0b9cc5274fff731421ead3`. All three lags are retained, with no best-lag selection or shifted admission score. Prediction target t=2..99 is compared with truth[t+lag],98 points/case and588 pooled. Four-force and signed-sum totalCd MAE/RMSE/bias/centered RMS, each case, four action-minus-zero contrasts and first-step signed contrasts are saved.

| Stream / metric | lag−1 | lag0 | lag+1 |
|---|---:|---:|---:|
| H1 rearCl MAE | .09499144 | .04510976 | .12678350 |
| AR rearCl MAE | .11964918 | .06246926 | .11816191 |
| H1 totalCd MAE | .03665116 | .02233390 | .03166106 |
| AR totalCd MAE | .04081158 | .02103200 | .02890986 |

All six cases, in both H1 and AR, have smaller rearCl MAE at lag0 than at either adjacent lag. Thus these saved trajectories do **not** support a simple whole-one-step displacement explanation. Fractional phase error and state-dependent error remain possible; this is not proof that all timing is exact.

At lag0 the H1 action-minus-zero centered-RMS differences are negative for all four contrasts: rearCl −.049756/−.008996/−.010719/−.032036 and totalCd −.014952/−.025637/−.024983/−.000380 (b01minus/plus,b05minus/plus). These are response-trajectory amplitude discrepancies, not a causal isolation of action sensitivity because later states diverge.

The first-step comparison has producer-verified equal initial fields. For b01minus/plus, predicted totalCd action-minus-zero is +.0009704/−.0010092 versus true −.0004296/+.0004433 (opposite signs). For b05minus/plus, predicted rearCl action-minus-zero is −.0020398/+.0025975 versus true +.0024121/−.0024664 (opposite signs). Other channels/contrasts remain in the output; these small four-contrast observations support a local response concern, not a global response-sign verdict. H1 and AR first predictions are the same, so these discrepancies do not require accumulated AR field error.

Scope: fixed B dynamic6/high-TF32 data, not controlled b01/b03 highest-noTF32 development replay. No causal decomposition, new threshold, model calibration, admission, automatic retraining or further experiment is authorized by these results. D's predeclared failure and existing physical-control successes remain unchanged. The useful next hypothesis, if separately authorized, should target demonstrated local force/action response rather than assuming a one-step label delay or merely increasing data dose.
