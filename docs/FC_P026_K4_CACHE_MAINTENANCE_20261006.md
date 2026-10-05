# P026 K4 bounded cache-maintenance preparation and approval

Root reviewed the isolated change and independent review: helper now accepts
explicit `--history-k 4` with only the fixed K4 r1 receipt. Default K1 and its
existing r1-r4 choices are preserved. No training code, data or model changes.

Helper SHA: `a4fce7fb311d5293b2594fad4313c46d146a74a06e813f317f649bb025ebc00a`.
Tests SHA: `675e7e979ebeca18d8d3e630d8d443b6eb48cc414e4770cba809dfe3bee5ebcc`.
Implementation/independent/Root canonical fixture tests: 24 passes in
0.04/0.03/0.05 seconds, respectively. Tests do not read real HDF or advise caches.

The pinned 44 training-file allowlist, same-descriptor SHA/stat checks,
no-symlink path checks, exclusive receipt creation, 300-second deadline and
20/20.5/20.75 GiB memory checks remain unchanged. No global cache clearing or
data writes are permitted. This is OS clean-file-cache advice, not numerical
training or a source of scientific improvement.

Root authorizes **one** K4 r1 pass after fresh memory checks for the actual
running K4 invocation `eee5a6fbad40411cac2f05e00520b079`. The exact output is
`artifacts/fcp026_history_training_k4_20261005/cache_advice_20261006_r1.jsonl`.
Physical free memory was around22 GiB during preparation, above the20 GiB
floor but with limited headroom. The pass must use the reviewed helper hash
and cannot overwrite any prior receipt. Execution success is not yet recorded
by this approval; verify the actual receipt separately. No r2, automatic loop,
training restart, scientific admission or PPO execution is authorized here.
