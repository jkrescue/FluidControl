# P024 response-scale diagnostic — FC-E034

Actual unit fluid-control-fcp024-response-scale-20261005.service, invocation
b85dcf9d70e443fe92ae4ef5d72d3c76, endedexited/success/PID0/status0.
Result artifacts/fcp024_response_scale_20261005/result.json SHA
6eecbd75c5b18cd821d6cf814c6d9319f755bf8dc70f2eb0c0e7e9326bf8ba0b.
Source81a45c6, approval53926e4d, protocol93c01f44. Runtime101.595seconds.
Actual container/image/mount observation: FC_P024_RUNNING_EXECUTION_20261005.json.
Root and independent reviewer verified terminal state/source identities and
resource samples. Independent audit rederived all10panel aggregates, exact0/1raw
reproduction, all60journal windows in prescribed order, identical repeated rows
and all96scaled FP32 values. No operational blocker or scientific admission.

## Protocol and observation

Fixed0/1/-1/8/64multipliers of the measured P023HIGH96-vector, same six train
windows, officialFNO, normalization, original objective and H1/fullH100 recurrence.
No optimizer, backward, saved candidate, heldout or new policy.600flowcachecalls,
6000paired modelcalls,60window evaluations. Each scale repeats twice.
Scale0/1 exact raw-prediction/metric/aggregate reproduction completed before any
later scale; this verifies reconstruction rather than assuming it from toy tests.

All numbers below are percentage changes relative to scale0; negative means
smaller error. These are training-panel sensitivity results, not acceptance.

| Scale | H1 mean bias squared | H1 RMS error squared | H1 centered MSE | AR mean bias squared | AR RMS error squared | AR centered MSE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | +0.070729 | -0.417274 | -0.139289 | -0.199089 | -0.083577 | -0.249289 |
| -1 | +0.056750 | +0.383509 | -0.073366 | -0.044162 | +0.011545 | +0.240124 |
| 8 | -0.407964 | -2.912865 | +0.022267 | +0.155518 | -0.298982 | -1.679812 |
| 64 | -4.386146 | -20.182609 | +19.096009 | +2.092909 | -1.615007 | -9.976034 |

Scale64 also worsens H1 original objective0.00352291→0.00392288, while AR objective
improves0.00885168→0.00837178. AR absolute RMS error only improves about1.002%.
No fixed scale simultaneously improves all listed quantities. Stronger input
effect alone does not remove the observed tradeoffs along this learned direction.
This does not establish an unavoidable tradeoff, global capacity limitation or
an optimal coefficient/learning rate. No best-scale selection is authorized.

## Resources and next action

53external samples: minMemFree30.915092GiB/minMemAvailable110.739491GiB.
6727internalchecks:30.936146/110.761051GiB.53GPUguard samples,minCUDAfree30.951664GiB,
exit0. Both20GiB floors held. Old-parent restoration is code-enforced before
writing result; do not call it an independently reloaded final checkpoint.

Do not multiply saved model weights for deployment or launch PPO. Scientific
next-action review now focuses on reoptimizing the input direction with explicit
control-relevant statistics, rather than more scaling or repeated unmodified
training. Root selected one bounded actual-learning intervention instead of a
further gradient-only diagnostic: the already specified symmetric mean/RMS loss
with all old weights frozen. This tests a different direction without claiming
the cause is proven. CPU-only plan FC_P025_ISOLATED_STATISTICS_PLAN_20261005.md;
GPU execution requires separate final review. Complete criteria remain unchanged.
