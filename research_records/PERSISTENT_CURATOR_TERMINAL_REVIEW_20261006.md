# Persistent Curator: actual CPU equivalence and latency review

User unit `fluid-control-persistent-curator-replay-20261006.service`, invocation
`9b119400a2744651b4da45b9ed9ff5e3`, completed06:11:55–06:14:35UTC, active/exited,
MainPID0, Resultsuccess, ExecMainCode1/Status0. Approval SHA
`ad2050de4ed4631c80e6b107090dee0ba26530c1776a87487c4a323e336dd51c`.
No restart or additional scientific computation was performed by this review.

The reviewer independently reopened all21 old/new pairs and their corresponding
saved historical samples:20 distinct existing frames across MPC/zero branches at
148.0–148.9, plus a final repeat of MPC148.0. All105 field comparisons
(`state`, `mask`, `time`, `x`, `y`) match exactly in keys, dtype, shape and array
bytes among old CLI, persistent function and historical sample. The original
pressure gauge, invalid-point mask and float32 stored TimeValue are preserved.
NPZ ZIP-file SHA identity is deliberately not the equivalence criterion.

Measured warm-cache, single-CPU replay timings:

| Measurement | Original CLI | Persistent callable |
|---|---:|---:|
| First call |6.132717s|1.948667s|
| Later20-call mean |5.761954s|1.571468s|
| Total21 calls |121.371792s|33.378027s|

Persistent one-time import took2.777943s. Warm-call ratio is approximately3.67x;
including import,21-call totals are121.37s versus36.16s. This is sampling-call
throughput on existing files, not total closed-loop speedup or real-time proof.
Old CLI always precedes the paired new call, so cache/order effects are not
separately isolated. No claim that all saved time is Python import overhead.

319 external memory observations span159.168597s; minimum MemAvailable
121,422,364,672bytes. Actual cgroup/unit limits were4GiB MemoryMax, no swap,
1CPU,330s outer duration, KillModecontrol-group. Reported self cumulative
ru_maxrss was1,400,156KiB; it is not per-call RSS or whole-cgroup peak.
There was no model/GPU/CFD execution, data modification or scientific admission.

Executed staged sources remain preserved:

- sampler6c1ae12c0b7382347547b2049a3f82d9f18cc9b5c5c4ff71a5b9d5895d067416
- replay521b26b52b987e7a38c9b2faa3020b6871557337bd52791e990211564eee8a86
- supervisor5c7dde9d354b12988e3d54f84a2641dc363b814356eb2cdcbdb507b20dd621b1

Evidence:

- equivalence/result.json:d8786fde2de19f609df320c0d5df21a8cffa91f91cca6a02d55d83ce53bbcddc
- equivalence/records.jsonl:b2281a8185a9e74fed68d8be0e3da30a4cc999fbc900c072788946e34abbddbf
- supervision/result.json:054ad8fcfac0a9fa8bf72ffdf8b0bc9a4fa606c7cc2e3977b62da0dda026ec5f
- supervision/memory.jsonl:94e3e1e061c1da85c4a5fb8553d6eb566f11fe219d3f7cf71d123546f7113f28
- supervision/run.log:a4a410c05bbd5c8a9f279da186f7b9dc61bb502b66a7648f3275da4bd686aab2

Prefixes denote artifacts/persistent_curator_equivalence_20261006 and
artifacts/persistent_curator_supervision_20261006. Result rows were independently
checked equal to JSONL, and supervisor result binds the actual result SHA.
Seven synthetic CPU tests also passed; real-array equality above is separate
actual evidence. This enables reviewed integration of the callable into a future
separately approved trial; neither old H5 source nor its outputs were changed.
