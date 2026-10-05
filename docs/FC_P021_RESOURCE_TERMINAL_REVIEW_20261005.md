# P021 real-size causal-force resource check

Root and independent evaluation verified the r2 engineering result. This is not
model training, prediction improvement, scientific admission or closed-loop success.

Result: `artifacts/fcp021_causal_resource_probe_r2_20261005/result.json`
SHA256: `975fc40bbb88d5d0ab3d239ee0ce994bc0635aabcad9b7d74568bf73f1a8ce7a`.
Unit `fluid-control-fcp021-causal-resource-r2-20261005.service`, invocation
`abb778c24a494eaa881eaa2669c5b57f`, terminal active/exited, success, PID0,
ExecMainCode1/ExecMainStatus0. Guard exited0. Actual container was
`69ccfb0be99f24872432a6cf069712d008f51cc89089aee8c3c22b3b2833a63d`,
official image b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e.

## What was measured

Same P018 parent, fixed real train window816 (dynamic_train8_b00_prbs/start90),
full100-step chain with checkpoint10 and pairedbatch2. Frozen flow generated100
states; original six-input force model supplied100 paired reference calls.
Each expanded official FNO arm performed100 forward and100 backward-recompute
calls. No optimizer, parameter updates, candidate saves or heldout access.

| Measurement | Zero-input A | Current-force B |
|---|---:|---:|
| Forward/backward seconds | 4.860219 | 4.597026 |
| CUDA peak allocated GiB | 3.379 | 3.378 |
| CUDA peak reserved GiB | 3.705 | 3.779 |
| New input-column gradient norm | 0 | 0.02195512825 |
| Initial output max difference from legacy | 0 | 0 |

All28 trainable tensor gradients were finite, frozen biases had no gradients,
and original/expanded initial tensor identities were preserved after cleanup.
A/B predictions exactly matched at initialization, as expected for zero new
columns; their H1/AR normalized objectives were0.0050170836/0.0136952251.

17 external host samples: min MemFree28.595875GiB, MemAvailable108.324940GiB.
Inner guard min CUDAfree28.598488GiB. Both20GiB requirements held. Whole harness
29.048545seconds. 192-window backward estimate907.8955seconds excludes optimizer,
endpoint panels and setup. Adam moment377782456bytes plus188891228bytes temporary
allowance is a projection, not measured training memory.

## Preserved startup failure and recovery

r1 failed before model creation when imports reduced free memory below extra
30GiB startup floor; no20GiB violation. Failure SHA
`30c1d17c7692a85ff60f62e11eb5700d1fdda1a49163e543253f44cf828317b5` remains.
Reviewed r2 advised clean-cache release only for44 verified regular training HDF
files, rehashed through the same read-only descriptors (6834205387bytes). No
data writes or global cache purge. Measured MemFree changed32340988 to38853188KiB;
reclamation is advisory, not guaranteed. All thresholds and model code unchanged.

## Next action and limitations

Root approves CPU preparation only for a matched sixteen-update-per-arm test
on the existing six training windows, original objective and identical optimizer.
Full-size trained feedback, optimizer resources and efficacy remain unmeasured.
The new force-input columns were zero here, so learned nonzero feedback stability
cannot be inferred. Six-window causal data coverage is not full-training coverage.
No scientific training launch is approved by this report; freeze and independently
review the implementation first. Physical and surrogate criteria are unchanged.
