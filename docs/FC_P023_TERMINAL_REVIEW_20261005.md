# P023 terminal review — FC-E033

Actual unit `fluid-control-fcp023-input-block-20261005.service`, invocation
`39aec740a9914226bb1f74c2d29e7917`, ended active/exited,success,PID0,code1/status0.
Result `artifacts/fcp023_input_block_20261005/result.json` SHA
`adfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb`.
Runtime869.977seconds. Sourcef13a6a0; approval714db1f9 and protocol4591677d.
Actual container/mount evidence is FC_P023_RUNNING_EXECUTION_20261005.json.

## What was tested

Same six real training windows and P018 parent. Freeze old official PhysicsNeMo
force-FNO weights; optimize only96new current-force input coefficients. Both
arms causal, LOW1.5625e-7 andHIGH1e-5, each16updates, originalJ0/fullH100.
No new model architecture, physical criterion change, heldout or candidate save.

## Independent verification

Exact source/dependency/protocol identities rehashed.192windowbackwards,32updates,
48ordinary endpoint plus24zero-input-ablation window evaluations. Twelve panels
and all comparison/repeat dictionaries independently recomputed. Repeated raw
rows exactly agree; this is observed repeatability, not a rigorous error bound.
Actual96values reproduce update and cumulative norms; sole24x4x1x1 optimizer state
has768moment bytes, old-base hashes unchanged. Final restoration is enforced by
the executed code before result writing, not independently reloaded on a GPU.

## Results relative to common initial state

| HIGH metric | H1 one-step | AR100-step |
|---|---:|---:|
| Original prediction objective | -0.070508% | -0.213732% |
| Mean-lift prediction bias squared | +0.070729% | -0.199089% |
| Lift RMS-amplitude error squared | -0.417274% | -0.083577% |
| Centered waveform MSE | -0.139289% | -0.249289% |

HIGH beats LOW on the recorded conditions, but H1 mean-bias squared increases
by2.4742730913131575e-7 relative to initial. Hence original local_support=false.
This is a small improvement in several training-panel quantities, not repair of
the previously failed complete development assessment or real-CFD control.
AR absolute RMS error only changes0.07451033→0.07447973 (about-0.0411%).
H1 bias regression is dominated by window816 (+3.19705e-6), while923/975/1077
improve. Aggregate agreement must not hide this operating-phase dependence.

HIGH final blockL2=.001553586037,maximumabsolute=.000165001038;
LOW L2=.000024479938,maximumabsolute=.000002502642. Both terminal zero-force-input
ablations exactly reproduce initial raw predictions. HIGH maximum rearCl output
change from ablation is.0013356862(H1)/.0013809714(AR), LOW.0003790543/.0004270079.
Block magnitude and output effect are not simply proportional; this does not
prove an unavoidable bias/waveform tradeoff or a unique numerical cause.

## Resources and disposition

435external samples: minMemFree28.611729GiB, minMemAvailable108.414654GiB.
46831internal checks: minima28.377514/108.179558GiB.437GPUguard samples:
minCUDAfree28.437504GiB,exit0. Both20GiB floors held. No candidate or newPPO.

Do not promote or automatically extend this experiment. Next proposed mechanism
test: reconstruct the measured96-vector, reproduce scale0/1 exactly, then inspect
a few preregistered signed/scaled directions without optimizer or model saving.
Its question is response sensitivity and curvature, not best-scale selection or
lowered acceptance. Requires separate plan, implementation review and execution.
