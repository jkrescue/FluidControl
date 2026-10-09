# FC-E093 saved fixed-lag diagnostic — independent numerical review

**ACCEPT descriptive JSON arithmetic; not model admission or a new physical-control result.** No model, HDF, inference, training or CFD was rerun.

Actual unit `fluid-control-p064-fixed-lag-json-20261007.service`, invocation `a730e633fec4417b9386a532fdb1e46d`, independently queried MainPID0/Result success/ExecMainStatus0, MemoryMax1GiB/swap0/RuntimeMax60s. Author report `docs/P064_FIXED_LAG_SAVED_JSON_DIAGNOSTIC_20261007.md` SHA608f02b951edaeb613cd639693cabfee91275073e5ce89d7a0a2fc00f5f24c3c and result `artifacts/p064_b_fixed_lag_saved_json_diagnostic_20261007/result.json` SHA7738b963c474b6d62aa42037095a11100e71c9154e0b9cc5274fff731421ead3 independently rehashed. Source5be17944684e1899f4562aa99d94ca3cd41863c76dc0fe16465686d993c2aa68/test0443a3372dfb194afec55383c512e67dd2e5220aca1ab11ac82de8e9faf8c9c4/protocol73edd122938718114da2b7393fb1a94b84f6792bd7ba53e2b9dd57c9f778b5c4 match the executed provenance.

Independent NumPy implementation (not an invocation of the producer's `analyze`) reopened only the existing H1 and AR JSONs. Input SHAs1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef and6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173 match. All six case identities/HDF identities,600 targets, times and applied-action endpoint pairs align. Recomputed all2220 MAE/RMSE/signed-bias/centered-RMS scalars across two streams, three lags, each case/pooled, four phase-matched action-minus-zero contrasts, and the first-step contrasts. Maximum absolute discrepancy6.661338147750939e-16. All stored contrast arrays match exactly.

Fixed prediction indices2..99 use98 points/case,588 pooled for every lag; compare prediction[t] with truth[t+lag]. No best-lag selection or corrected admission score is introduced.

| Stream/MAE | lag−1 | lag0 | lag+1 |
|---|---:|---:|---:|
| H1 rearCl | .09499144088 | .04510976163 | .12678349998 |
| AR rearCl | .11964918099 | .06246925838 | .11816191041 |
| H1 totalCd | .03665116247 | .02233389740 | .03166106313 |
| AR totalCd | .04081157782 | .02103200018 | .02890985667 |

Every one of six cases has lower rearCl MAE at lag0 than either adjacent lag, in both streams. These records do not support a simple whole-one-step displacement explanation. They do not exclude fractional/state-dependent phase errors.

First-step action-minus-zero b01 totalCd predicted minus/plus are +.00097042322/−.00100916624, whereas truth is −.00042957067/+.00044327974. b05 rearCl predicted −.00203979015/+.00259745121 versus truth +.00241208076/−.00246644020. The reported opposite signs are correct. H1 and AR first predictions are exactly equal, so this discrepancy does not require accumulated autoregressive field error. Equal initial fields and batch1/high-TF32 runtime are established by the prior E073 producer/review; this small-JSON audit does not independently reload fields or validate runtime tensors.

All four lag0 H1 action-minus-zero centered-RMS differences are negative for rearCl and totalCd, as reported. Later contrasts combine action and divergent states, so they cannot isolate action sensitivity causally. Small first-step sign contrasts motivate a local-response hypothesis, not a universal sign-error conclusion. This dynamic6/high-TF32 evidence is not the controlled b01/b03 highest/noTF32 development distribution and cannot be directly extrapolated to it.

D's predeclared development failure, B/canonical actual control benefits, early failures and complete surrogate prediction FAIL remain unchanged. E094's proposed paired-precision check is preparation only, not executed by this review; there is no new training authorization or automatic remedy.
