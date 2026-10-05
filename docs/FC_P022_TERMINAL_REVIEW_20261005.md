# FC-P022 causal force-input comparison — complete, local support not met

Root observation and independent terminal review agree. Result
`artifacts/fcp022_causal_conditioning_20261005/result.json` SHA256
`69e5d4a8a93b8036187a18a5c40cb270aec53462a49ddab94417fa0b908096d8`.
Exact invocation `f1f3f7b31e70440693da2661a12cfc04` retained active/exited,
Resultsuccess, MainPID0, ExecMainCode1/ExecMainStatus0; guard exit0.
Reviewed implementation074d979, approval273fe63f097049fe28f9d3f6f241308a95eefb6fab23ca2dbe0d23559043cf78.

## Verified execution

Six fixed real training windows; identical expanded P018 initial state and fresh
AdamW in each arm; original objective and full100-step recurrence. Each arm16
updates, each averaging6 raw window gradients before clipping. Independent review
verified192 distinct journal backward events,32 updates,600 frozen-flow calls,
19200 training forward and19200 checkpoint-recompute calls,4800 endpoint calls
(48 windows). All eight endpoint-panel aggregates, full comparisons and repeat
resolution dictionaries independently recomputed from stored per-window metrics.
A/B initial rows and all repeated endpoint rows exactly agree. Zero observed
spread is not a rigorous numerical uncertainty bound.

## Results on the fixed training panel

H1 means one-step prediction from real flow/current force; AR is free-running
100-step prediction. Statistics below average the five nonzero-action windows;
original objectives separately average all six windows. These are training-panel
metrics, not validation or real-CFD control benefits.

| Metric | Initial | A: zero new input | B: current force |
|---|---:|---:|---:|
| H1 original objective | .003522910 | .003519588 | .003507791 |
| AR original objective | .008851685 | .008362288 | .008371235 |
| H1 mean-bias squared error | .000349824 | .000359300 | .000351674 |
| AR mean-bias squared error | .000767301 | .000783459 | .000772637 |
| H1 RMS-amplitude squared error | .000956622 | .000918318 | .000917538 |
| AR RMS-amplitude squared error | .006898063 | .006825872 | .006825220 |
| H1 centered waveform MSE | .002245066 | .002514264 | .002499056 |
| AR centered waveform MSE | .013250078 | .012033968 | .012055629 |

B improves four mean/amplitude squared errors relative to A, but fails the
predeclared joint local conditions: versus initial, H1 mean-bias error+0.52877%,
H1 centered waveform error+11.3133%, AR mean-bias error+0.69550%; AR objective
also remains worse than A. Therefore local_support=false. Improvements in other
entries cannot override these failures. No automatic further epochs or full
training are justified by this result alone.

## Resources and evidence limits

Whole harness979.19seconds. External490 host samples: minfree26.807617GiB,
minavailable106.579044GiB. Harness44383 checks minfree26.801697GiB; innerguard492
samples minCUDAfree26.804150GiB. Both20GiB floors held. ActualAdam moments
377782456bytes per update; peak reserved A4534042624bytes/B4452253696bytes.

Original and expanded restoration is code-enforced before successful result
writing, but terminal model weights were deliberately not saved. No independent
reload/re-inference or realized added-column norm claim is possible from this
artifact. New-column post-clip gradient norms are measured, not weight norms.
No validation/frozen data, deployable candidate, new PPO or closed-loop result.

## Next action

Analyze input-column learning scale versus updates of existing weights before a
new experiment. A frozen-parent/new-input-only comparison is a hypothesis, not an
approved training run. Do not infer that extra information is useless, that the
learning rate is optimal, or that a different model is necessary. Preserve the
existing physical10% mean-load constraint pending17:32UTC review; it is not the
surrogate's mean prediction-error metric. This result changes no criterion.
