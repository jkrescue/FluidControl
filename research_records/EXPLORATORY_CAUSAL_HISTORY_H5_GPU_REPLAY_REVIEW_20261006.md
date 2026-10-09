# Ten-state H5 GPU replay — independent terminal engineering review

Actual unit `fluid-control-h5-gpu-replay-20261006.service`, invocation `c4d66fdf436d40a0b91455680a121cc3`, completed 2026-10-06 06:22:37–06:22:48 UTC. Independent query: MainPID0, active/exited, Result=success, ExecMainStatus0. No replay/model execution was performed by the reviewer.

Approval: `docs/EXPLORATORY_CAUSAL_HISTORY_H5_GPU_REPLAY_APPROVAL_20261006.json`, SHA256 `b6350c342d0a1700639305a2806b5ee2bc1b4a288a0706d3d7dec66a35a4159b`. Reviewed executed source SHA256 `e2e6c6fada4e41eb48493d2ced79118173677ae8b5a2822cab41d2173a71dd0e`. All11 source and16 input file hashes independently matched the approval, including the ten actual stored CFD current-frame packets, common normalization/baseline and K1 identities.

Output root: `artifacts/exploratory_causal_history_h5_gpu_replay_20261006`.

- `result.json`: `3568870390df28c4f63145317ad89f941ff7d1c6073f5a756b06ab2a46273a69`.
- `supervisor_result.json`: `4401baa3d17299269be4ca44b76756a24eed5370b722a99d1c31d686a3d0916e`.
- `memory.jsonl`: `b0bd6b23e19ef61d2917bc696156f925f2f15ee2dd7c610dfaca0e086bc70bcc`.
- `run.log`: `8057e222a15e0f38a5cb445e1425b066bd334754e6a1d3bcc83941c46d481e9b`.

The fixed comparison source is actual CPU H5 result SHA `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`, with K1 manifest `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`. Each of the ten current states reruns all five held-action candidates for H5; causal62 history advances using only the original measured CFD endpoints, never replay-selected actions or predictions. This is replay on a recorded trajectory, not a new closed-loop trajectory.

Historical `high/True/True` checkpoint precision identity is verified before loading. The result records the explicit post-load override `highest/False/False` (float32 matmul/CUDA matmul TF32/cuDNN TF32). This is not the original precision protocol. The worker records unchanged model tensors, no optimizer, CFD or saved model, and scientific_admission=false; the reviewer did not reload tensors.

## Recomputed numerical evidence

All **10/10 selected indices and exact actions agree**, and all10 full candidate-cost rankings agree. Actions are `[.1,.2,.25,.25,.2,.1,0,-.1,-.2,-.30000000000000004]`. The reviewer independently reconstructed GPU costs from saved CPU costs plus recorded differences and recomputed rankings/selections and difference reductions. No model was rerun. Saved force differences cover10states×5candidates×5leads×4channels.

| Force channel | Maximum absolute GPU−CPU difference | RMSE |
| --- | ---: | ---: |
| Front Cd | 1.1920928955078125e-7 | 2.384185791015625e-8 |
| Front Cl | 2.086162567138672e-7 | 6.29862759352824e-8 |
| Rear Cd | 6.556510925292969e-7 | 2.770686506473102e-7 |
| Rear Cl | 2.2649765014648438e-6 | 7.197442876792683e-7 |

Maximum absolute candidate-cost difference is `8.67788286562643e-7`; maximum per-stage component difference is `1.2810193341650233e-6`. The predeclared exact selection-consistency criterion is met for these ten recorded states. This is not exact numerical identity, a global equivalence claim, surrogate-accuracy admission or evidence of control benefit. No additional numerical tolerance was invented.

First synchronized H5 replay took `0.7313071800163016` seconds; the remaining nine took `0.3508230559527874`–`0.35513469902798533` seconds. Worker replay elapsed is `4.440570732986089` seconds, excluding model/input initialization. These are inference timings, not whole-cycle CFD/control throughput.

## Resources and use of evidence

Queried actual systemd limits:12GiB memory, swap0, oneCPU equivalent,330s maximum. Supervisor uses300s deadline and0.5s sampling. Its23 observations have minimum MemAvailable `121039630336` bytes and MemFree `1429213184` bytes; final elapsed11.01193603s. Explicit UMA guards are Available50GiB startup/22GiB runtime; CUDA free is observational, not an old20GiB CUDA-free gate. No scientific criterion is changed by this resource policy.

No remaining source or observed execution blocker to **preparing a separately approved accelerated fixed-H5124-cycle trial** using this exact highest/no-TF32 override. Bind this report and result in that trial's approval; retain original model-failure and exploratory labels. The ten-state engineering evidence does not guarantee unchanged decisions at unseen states or establish12.4D/U or80D/U physical performance. No new execution is authorized by this report.
