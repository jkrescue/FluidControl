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

## Actual r1 execution and independent receipt review

Root reports the approved command exited 0 under its 310-second outer timeout
(session 51920). The saved receipt independently records completion in
10.184359 seconds, from 2026-10-05 22:05:52.096 UTC to 22:06:02.280 UTC
(2026-10-06 06:05:52–06:06:02 Asia/Shanghai).

Receipt SHA-256:
`115ab435a28d0b23d8e1a8fea8823467b15cad00ec99956231ca5c38a1a374a6`.
Its 90 records comprise one begin, 44 file-begin/advised pairs, and one
complete, with no aborted record. The begin record binds the approved helper,
K4, exact output and pinned P018 audit SHA. All 44 unique sorted paths and
reported SHA values match that audit's train-only allowlist. All three
same-descriptor stat snapshots match each file-begin stat, covering
6,834,205,387 bytes. This review checked saved JSON evidence only; it did not
reread any HDF file or issue cache advice.

Across 178 receipt memory observations, minima were MemFree **21.424427 GiB**
and MemAvailable **105.820568 GiB**. Initial free memory was 21.424427 GiB;
completion free memory was 24.310375 GiB. Recorded samples remain above the
applicable 20/20.5/20.75 GiB checks. The increase is an observed operational
change, not proof that all advised pages were evicted or that training improved.

The concurrently growing `resource_watch.jsonl` snapshot through Unix timestamp
1791238023 (22:07:03 UTC) contained 187 samples, with minima MemFree
**21.550716 GiB** and MemAvailable **105.905724 GiB**; snapshot SHA-256 was
`6100cc274b6ace0ffac05c2fba4b0686ddfda5a75847ca9845b4399ca4baf7d0`.
This identifies that read snapshot, not a final immutable training log.

The actual service `fluid-control-fcp026-history-k4-20261006.service` remained
live: invocation `eee5a6fbad40411cac2f05e00520b079`, MainPID 941365,
ActiveState `activating`, SubState `start`. Its interim ExecMainStatus 0 is
not terminal success. The observed training log had reached 81/1368 windows
and 10 updates. Cache maintenance completed; training and scientific review
did not. No restart, second pass, model acceptance or PPO authorization follows.

## Separately approved and completed r2

Root subsequently reviewed and committed the fixed K4 r1–r4 name extension
(`37d9b64`), with 30 canonical CPU tests passing in 0.04 seconds. The extension
does not authorize repeated execution automatically. Root reports one separately
approved r2 execution, session 20657, exit 0, using helper
`3ca107dd1ddb2ac70a457f752088b423bf96f438f0b4f963d482929625d77ed9`.

Independent saved-receipt review found 90 records: begin, 44 matching
file-begin/advised pairs, and complete. All 44 paths and hashes match the pinned
audit; every file's before/hash/advice stat records agree. The receipt binds K4
and the exact r2 output. No HDF reread or additional advice was performed by
the reviewer. Receipt SHA-256:
`ea9ac02c38c8441c1afaeee4f610d44aeed13fe10cb72b6ac67343dc47134cf7`.

Execution ran from 2026-10-05 22:14:33.939901 UTC to 22:14:44.171206 UTC
(2026-10-06 06:14:33–06:14:44 Asia/Shanghai), elapsed 10.231633539 seconds.
Recorded minima were MemFree **21.483437 GiB** and MemAvailable
**106.142365 GiB**, above all applicable floors; completion free memory was
24.287720 GiB. The growing host-watch snapshot through timestamp 1791238512
(22:15:12 UTC), 423 samples, had minima MemFree **21.342857 GiB** and
MemAvailable **105.867161 GiB**. Its snapshot SHA-256 was
`5c2210fee8a9646a83ae62a0a2566e9536692d10cfa777958eb2bc35827a0ebc`.

The original r1 receipt retains SHA `115ab435a28d0b23d8e1a8fea8823467b15cad00ec99956231ca5c38a1a374a6`;
r3 and r4 receipts were absent at review. The same training invocation
`eee5a6fbad40411cac2f05e00520b079`, PID 941365, remained `activating/start`.
This establishes completed r2 maintenance only, not terminal training success,
scientific admission, or permission for r3/r4.

## Separate r2 preparation and authorization

Subsequent live K4 observations showed MemFree declining again to about21.7GiB
while the same invocation continued training. Root separately authorizes one
r2 pass after fresh checks; this is not an automatic consequence of r1.
The reviewed helper now permits fixed r1-r4 names for both arms, retaining
exclusive creation; this merely removes the two-line K4 r1-only parser check.
All exact44/path/hash/stat/advice/memory/deadline behavior is unchanged.

Helper SHA: `3ca107dd1ddb2ac70a457f752088b423bf96f438f0b4f963d482929625d77ed9`.
Tests SHA: `50a3cefd48c596daaf7c4860ea7f8e38e0186e1c78ab92313851ba0365126db0`.
Implementation30CPU tests0.02s; independent30tests0.03s. These are mocked
software tests, not cache operations. Original r1 receipt remains unchanged.
Exact new target:
`artifacts/fcp026_history_training_k4_20261005/cache_advice_20261006_r2.jsonl`.
No r3/r4 execution is approved. Record actual r2 completion separately;
do not restart training or alter its protocol.
