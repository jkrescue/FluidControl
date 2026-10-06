# K1 post-load no-TF32 GPU probe: independent terminal review

## Actual execution

The retained unit `fluid-control-k1-uma-no-tf32-20261006.service` has invocation `6dcaaafb208c444c8d154bcebbdd27f3`, MainPID0, active/exited, Result=success and ExecMainStatus0. Queried execution timestamps are 2026-10-06 06:05:46–06:05:54 UTC. ExecStart identifies the approved staged script and exact approval SHA. No restart or model execution was performed by this reviewer.

- Source: `/tmp/k1-uma-no-tf32-review/probe_k1_uma_no_tf32.py`, SHA256 `7f8567b0f41474294a45a864d4024cf0d64508b7e8351e19feb8e8aee9219b73`.
- Approval: `docs/K1_UMA_GPU_NO_TF32_APPROVAL_20261006.json`, SHA256 `d4593bb72688b7227b55c9cb6c34aac7af8c1876abb52f8322714c63407b2bb5`.
- Output root: `artifacts/k1_uma_gpu_no_tf32_probe_20261006`.
- `worker_result.json`: `a45ff74160df30448963357f8d660d5a26f812c3fbc0de24d6b573db134bb6bc`.
- `supervisor_result.json`: `b7b644a63f02cedd1dcabb83710371b555a09a1498356d5ddbe2870e1bc2520e`.
- `inference_precision.json`: `79da46d914ee6112a5c5575bc574df7c0aa3e560f73b127344ffe7bee842da35`.
- `memory.jsonl`: `927d3dbf3e5e3a7c67920f82071b0b74cd00f78c8635d20187a13dc4bd1d960d`.
- `run.log`: `8057e222a15e0f38a5cb445e1425b066bd334754e6a1d3bcc83941c46d481e9b`.

The official loader used the historical identity settings first. The explicit post-load inference override records `high/True/True` before and `highest/False/False` after (float32 matmul precision / CUDA matmul TF32 / cuDNN TF32). This is NOT the original precision protocol. Worker evidence reports unchanged model tensor digest, no gradients, optimizer or CFD, and no scientific admission. The reviewer inspected saved evidence rather than independently reloading tensors.

## Saved numerical comparison

Compared all five held-action H2 candidates against the same t148 input and K1 manifest in the earlier CPU trial and old GPU probe. Shared identities and the original CPU comparison are recorded in `docs/K1_GPU_CPU_INFERENCE_COMPARISON_20261006.md`. CPU result SHA is `45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb`; old GPU worker result SHA is `7e3c373d895ae839ad360369988c49aa52a1ecc14d0f6d94b4793b83efb0559c`. K1 manifest remains `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`; sample is `b9bdf87db1b4744633fb3c175050449952a6661ec95676a055d7f62111f80bc9`.

Each channel below uses ten predictions (five candidates × two leads). All three new GPU decisions are exactly identical to each other.

| Channel | New GPU vs CPU max absolute | New GPU vs CPU RMSE | New GPU vs old GPU max absolute |
| --- | ---: | ---: | ---: |
| Front Cd | 0 | 0 | 0.00001537799835205078 |
| Front Cl | 0.00000011920928955078125 | 0.00000007052516678690452 | 0.000276714563369751 |
| Rear Cd | 0.0000005960464477539062 | 0.0000003674277068954096 | 0.0008113384246826172 |
| Rear Cl | 0.0000018775463104248047 | 0.0000009882065434211048 | 0.013054251670837402 |

New GPU H2 costs in action order `[-.1,-.05,0,.05,.1]` are `[1.1982166006908372,1.1921240145237144,1.188576628475099,1.1875697026455352,1.1891018069107153]`. Their differences from CPU are `[-6.90300450578718e-7,-4.5171934082155474e-7,5.513832879167069e-8,-4.6494386407580635e-7,-3.857583892319383e-7]`. All three implementations rank indices `[3,2,4,1,0]` and select `+.05` at this input.

The observed rear-Cl discrepancy falls from approximately 0.013 to below 0.00000188 after this combined precision override. This supports reduced-precision backend settings as a contributor at this tested input; it does not isolate matmul versus convolution effects, establish universal CPU/GPU equivalence, or validate longer trajectories. No numerical equivalence threshold or PASS is invented. H5 remains CPU-only unless separately authorized otherwise.

## Resource and scope

Queried systemd limits: MemoryMax=12884901888 bytes, MemorySwapMax=0, CPUQuotaPerSecUSec=1s (one CPU equivalent), RuntimeMax=200s. The supervisor independently records the matching actual cgroup cap and swap0. Its 16 saved observations have minimum MemAvailable `120607776768` bytes and MemFree `1474117632` bytes; final observed elapsed time is `7.510825539997313` seconds. This explicitly approved UMA probe uses available-memory startup50/runtime22GiB guards; MemFree and CUDA free are observational, not the old GPU-floor protocol.

CUDA allocated peak is `755589120` bytes; reserved peak `796917760` bytes. Observed CUDA free is `1502056448` bytes for the first two calls and `1502380032` for the third. Synchronized inference times are `0.5205077980062924`, `0.14072805701289326`, and `0.13990906300023198` seconds. These timings are this bounded probe only, not closed-loop real-time performance. The queried retained systemd MemoryPeak value is not used as a reliable whole-worker peak claim; CUDA peaks and sampled host values are reported separately.

Recommendation: before a future GPU control trial, compare highest/no-TF32 inference against CPU at multiple actual CFD states (for example the independently generated H5 frames), then explicitly bind this inference-precision override in that trial's approval. The present single-state result does not authorize a device change or imply global equivalence; current H5 stays CPU.

Conclusion: operationally complete, reproducible three-call engineering evidence for the explicit inference-precision override. No CFD/control experiment, retraining, model admission, physical-goal completion, or alteration of historical scientific gates occurred.
