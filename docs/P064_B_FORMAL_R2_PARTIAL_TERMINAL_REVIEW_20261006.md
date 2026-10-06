# P064-B formal R2: partial terminal review

R2 is **failed, not running**. Unit `fluid-control-p064-b-formal-r2-20261006.service`, invocation `01806bdc150841f7b9efd04360a441f2`, independently observed MainPID0, ActiveState/SubState failed, Result exit-code, ExecMainStatus1. Root observed termination at12:53:10 UTC. This is an engineering CLI/source-routing failure after completed validation10, **not a completed scientific full-gate rejection or acceptance**. No full `receipt.json` exists.

## Failure and retained evidence

The next `validation_diagnostic` CLI rejected candidate kind `FC_P064_ARM_B_CONTROLLED_AERO_FORCE_FNO`; its actual choices stopped at P029 and omitted P064. Log: `artifacts/fcp064_arm_b_formal_20261006/validation_diagnostic.log`, SHA `4ccb9bb93685286b99c591e577a8468e5ed04955da671450806cef7de856db2b`. Preserve the failed unit/output. A reviewed identity-routing repair may reuse completed evidence; this report authorizes no rerun, retry or source mutation.

Approval `docs/FC_P064_ARM_B_FORMAL_APPROVAL_20261006.json` SHA `cd58fd47e991ec6dac200bd82d414347f72b778ea78415377427e430dfd47478`. Earlier R1 wrong-cwd failure is separate and remains preserved. Original scientific thresholds have not changed.

## Independently verified completed validation10

New B files under `artifacts/fcp064_arm_b_formal_20261006/validation10/`:

- `evaluation.json`: `15c149760171890798a4e1c3aaeb7e9f31d080cbae6bc6244f71967dd009f9a3`.
- `segments.json`: `8d68e2212f9e29a8ced2f0add63d0fbf09f7fb5d009a6dae0ac1df9d85bfc364`.

Matched historical K1 files under `artifacts/fcp026_history_training_k1_20261005/posteval_fc_p026_k1/validation10/`:

- `evaluation.json`: `a4f84771731f04bc75435e053692b8d000998ee2e6dfe5ae970bd46364a91caf`.
- `segments.json`: `1419d2379eaf855dc03f2b127e149cee67421129f6be5e0694f506f91c5cadb7`.

Read-only JSON recomputation verified ten cases, exact case/horizon/start/target-total-Cd correspondence, stride25 and horizons1/10/50/100. Counts320/320/310/290 total1240 endpoints; every case/horizon stable with failed_segments0. Four-force per-case MAEs and total-Cd MAE/RMSE were independently recomputed from saved segment errors/predictions/targets; report summaries match equal-case averaging. Field sufficient-statistic sums and relative norms were recomputed from case records, with floating reduction-order tolerance1e-12. No HDF, model or new inference was opened.

| Horizon | K1 rearCl MAE | B rearCl MAE | K1 totalCd MAE | B totalCd MAE | Common velocity relative L2 |
|---|---:|---:|---:|---:|---:|
| H1 | .02021761495 | .03778500591 | .00766464137 | .00878357142 | .00227667248 |
| H10 | .02780779332 | .04583977001 | .01175935194 | .01173352003 | .01509229490 |
| H50 | .02948903963 | .04183507283 | .01198412141 | .01156699273 | .02861461532 |
| H100 | .04018628365 | .04311831568 | .01092804753 | .01126262278 | .04359112110 |

All saved per-case field-error/reference sums, channel MAEs and velocity metrics are exactly equal across K1/B. This verifies unchanged saved field statistics, not a fresh comparison of model tensors or full prediction arrays. H100 common pressure relative L2 is .14288417123. B rearCl MAE is worse at all four reported horizons; Cd is slightly better at H10/H50 and worse at H1/H100. Do not compare this fixed-action validation10 panel with controlled b01/b03 development as a matched improvement test.

## Important aggregation distinction

`scripts/evaluate_tandem_fno.py:839` averages case-level RMSE/NRMSE in `summary`; `scripts/audit_full40_validation_gate.py:181` instead pools squared errors and reference squares across segments. At H100:

| Quantity | K1 | B |
|---|---:|---:|
| Mean-case Cd NRMSE (`evaluation.summary`) | .005809682472 | .006033057498 |
| Genuinely pooled Cd NRMSE (segment recomputation) | .006097181273 | .006264466104 |

Comparing B mean-case .006033 with historical K1 pooled .006097 would falsely suggest improvement. Each matched definition worsens slightly. The pooled values alone remain below the original .10 endpoint limit, but do not establish the other endpoint checks or full admission.

## Resources and interpretation

Validation10 container terminal record SHA `f41acaecc2d11d44c9ac1560b4c785eba8b043029dc38875999bae99569fc80f` reports exited/ExitCode0/OOMKilledfalse, from12:41:52.308 to12:53:07.953 UTC. Both guard logs end exit0; outer minimum MemAvailable108.243225GiB. CUDA-free observations near1GiB are not unified physical available-memory measurements. Actual evaluator allocator fraction is .15; outer .06 is startup accounting, not an enforced allocator cap. Lead separately approved same-run continuation under72GiB/noSwap and22GiB Available runtime reserve, without changing science.

Historical K1 "H100 FAIL" specifically includes failed trailing62-sample force-window fidelity, not a velocity/pressure threshold failure: K1 endpoint and dynamic Cd components passed, but force windows passed only1/6. Those full B components remain unevaluated in this partial run. Frozen flow does not prevent the changed aerodynamic model from improving or worsening force gates. Nothing here revokes already verified real-CFD physical results, establishes broad surrogate accuracy, relaxes physical10% mean-bias limits, or creates a new scientific admission.
