# FC-E073 — independent signed H1 terminal review

2026-10-06. Diagnostic execution and saved-array integrity verified; no scientific admission, model update, CFD action or training is inferred.

Actual unit `fluid-control-p064-b-teacher-forced-h1-20261006.service`, invocation `76d21e62134b44c0a97d65b6ad991669`, independently queried PID0/exit0. Approval `docs/P064_TEACHER_FORCED_H1_APPROVAL_20261006.json` SHA `482628b32be9fcd3879404356e3deeb186c38a96bf59635b76893e8d10d3e0c8`. Result `artifacts/p064_teacher_forced_h1_signed_force_20261006/result.json` SHA `1eacc9219f2f608c54e6ef48d4856624af64b8ad00eb477bbd5772eff8ad61ef`; supervisor receipt SHA `951967332df858729e310d9342c26f7b63d37c6d904629760a3dce78a4964b9d`.

Independent read-only review rehashed worker507f6d90, supervisor57f65114, all nine bound inputs and411 numerical sources. No model deserialization, HDF reopening or inference rerun. Candidate B manifest92766915/aero57d4634d, frozen flowdc41fc91, train normalizationf1b4607e and high/TF32 precision match the original force-window reference6195b21e. This diagnostic deliberately does not use the separate highest/noTF32 short-replay protocol.

All six cases contain exactly100 steps. Independently matched every input/target time, current/next physical action, target four-force vector and HDF identity against original AR indices0..100. Each H1 input is the true current state; K1 history is reset each step. Independently recomputed every case and pooled count, signed bias, MAE, RMSE and centered RMS for all four force channels from600 rows. The600 dual /1200 submodel forward counts follow the reviewed fixed loop and are theoretical counters, not separate hardware trace evidence. Optimizer steps0/model_updated=false are recorded; no fresh independent tensor reload was performed.

## Matched numerical findings

Batch1 H1 and the AR reference's first prediction agree exactly across all six cases and all four forces (maximum absolute difference0). This removes the earlier cached batch8-versus1 first-step mismatch for this comparison.

Pooled rear-Cl truth mean −.009185383876, prediction mean −.008304740190, signed bias +.000880643686; MAE .045200950125, RMSE .056784922888. Pooled centered RMS truth1.198123401320 versus prediction1.173602006843. Pooled bias can cancel across phases; it is not a per-branch physical lift-bias acceptance measurement.

Below are signed prediction-minus-truth errors for the identical trailing62 samples (targets39..100). All entries were independently recomputed from raw signed arrays, not inferred from absolute MAE.

| Case | H1 mean-Cl error | AR mean-Cl error | H1 Cl RMS error | AR Cl RMS error |
|---|---:|---:|---:|---:|
| b01 minus | +.0276558912 | +.0062311939 | +.0252557129 | +.0483455120 |
| b01 plus | +.0178130131 | −.0202747596 | −.0614688096 | −.1439358617 |
| b01 zero | +.0000717388 | +.0016041096 | −.0211002744 | −.0269937708 |
| b05 minus | −.0121466943 | −.0496610026 | −.0370160980 | −.0912204925 |
| b05 plus | −.0087346598 | +.0092308562 | +.0068724010 | +.0620918664 |
| b05 zero | −.0100158095 | +.0277044286 | −.0289341429 | −.0247019132 |

All four rotating branches have smaller RMS-error magnitude with true-state H1 conditioning than with free AR, but b01 plus and b05 minus retain true-input amplitude errors exceeding the original diagnostic zero-reference2.5% scales (.0294155/.0293872). This supports both a residual true-state force-readout problem and additional state-trajectory dependence; it does not prove a unique causal decomposition or justify an architecture/threshold change. Mean bias changes are not uniformly improved. The original complete B force-window admission remains FAIL.

## Runtime and limits

Receipt elapsed24.0060s, minimum sampled MemAvailable112.174331665GiB. Actual unit properties independently show12GiB memory/noSwap/CPU1/tasks64/120s stop; exact owned Docker name is absent after terminal. GPU allocator target6GiB is recorded with fraction .04930554517; outer600s monitored budget/630s runtime remained bounded. The supervisor completion receipt is written before cleanup; independent unit exit0 and Docker absence are therefore necessary complementary evidence, not assumed from the receipt alone.

This is opened dynamic6 development data, not new heldout confirmation. It explains a failure of the current model under a fixed diagnostic; it is not optimization, policy selection, new physical validation or overall goal completion. Existing B-policy b00/b01 physical results and the separately running b07 CFD trial are unaffected. Further work requires a bounded hypothesis and separate authorization; no automatic retraining follows.
