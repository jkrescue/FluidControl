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

## Separate r3 authorization

At 335 consumed training windows, the same K4 invocation remained live and
the latest host-watch sample recorded MemFree22,736,464KiB (about21.68GiB).
Root authorizes one additional bounded r3 pass with the unchanged reviewed
helper `3ca107dd1ddb2ac70a457f752088b423bf96f438f0b4f963d482929625d77ed9`.
Target is exclusively the same K4 directory's
`cache_advice_20261006_r3.jsonl`; r1/r2 must remain unchanged.
The helper rechecks current memory and all44 pinned training files; no global
cache operation, data/model change, training restart or automatic repeat.
This paragraph records approval, not execution success. r4 remains unapproved.

## Actual r3 receipt review

Root reports session 82030 exited 0 after the separately approved single pass.
Independent JSON-only review verified 90 records: one begin, 44 file-begin/
advised pairs, and one complete. All sorted unique paths and reported hashes
match the pinned audit; all file-begin and before/hash/advice stat snapshots
agree. The receipt binds the unchanged helper, K4 and exact r3 output.
Receipt SHA-256:
`55d34429f523b2d081dc3d7e7cfba83ef66fcb2cd8a816ebd0960e5e628de533`.

Recorded start/end Unix timestamps were 1791239010.4860294 and
1791239021.1663043; elapsed time was **10.680523248 seconds**. Recorded memory
minima were MemFree **21.533192 GiB** and MemAvailable **106.133457 GiB**,
above every applicable floor. Completion MemFree was **24.242355 GiB**.
The concurrent, still-growing host-watch snapshot through timestamp1791239050
contained 682 samples, with minima MemFree **21.171146 GiB** and MemAvailable
**105.867161 GiB**; snapshot SHA-256:
`ef091049c40e26f4c6d593ed183fdf824149040385974f5e963c0b2453f426c6`.

Rehashed r1/r2 receipts retain their recorded SHA values (`115ab435…374a6`
and `ea9ac02c…134cf7`). r4 was absent and remains unapproved. The same actual
service invocation `eee5a6fbad40411cac2f05e00520b079`, PID941365, remained
`activating/start`, not terminal. This review performed no HDF rereads or
cache advice and establishes maintenance completion only, not training
completion or scientific acceptance.

## Actual r11 receipt review

Root reports separately approved session93348 exit0. Independent saved-JSON
review verified90 records: begin,44 ordered file-begin/advised pairs,complete.
All44 paths and reported hashes match the pinned audit, with identical per-file
start/before/hash/advice stat snapshots. Begin binds reviewed helperd116b55b…b654e25,
K4 and exactr11 output. Receipt SHA:
`ac969392e69677efee3cb5d6577edc4bdcdcae0bf6514139e4a6bdea3c28f37f`.

Start/end Unix timestamps were1791242364.592915/1791242375.2992246;
elapsed **10.706622560 seconds**. Receipt minima: MemFree **21.920593 GiB**,
MemAvailable **106.166386 GiB**, above applicable floors. Growing host-watch
snapshot through1791242403,2299samples, minima: MemFree **21.006046 GiB**,
MemAvailable **105.671360 GiB**; snapshot SHA:
`a54a6e2e200a07280cd2e8c206d3cf39f8775334043e63150f0bb18952d4f5fb`.

Allr1–r10 receipt hashes were rechecked unchanged. Actual service remained live
under invocation `eee5a6fbad40411cac2f05e00520b079`, PID941365,
`activating/start`. No HDF reread or cache advice was performed by review.
No further pass, training completion or scientific acceptance follows.

## Separate r11 authorization

At timestamp 1791242339 the same live K4 invocation had reached 1186
windows/148 updates; MemFree was 22,635,404 KiB (about 21.59 GiB).
Root authorizes one bounded r11 pass with reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Exclusively create K4 `cache_advice_20261006_r11.jsonl`, preserving r1-r10.
All exact 44-file checks and memory/deadline guards remain unchanged.
No global cache action, model/data modification, restart or later pass is
authorized. Actual receipt verification remains required.

## Separate r4 authorization

At468 consumed windows, the same live K4 invocation's latest host sample
recorded MemFree22,610,148KiB (about21.56GiB). Root separately authorizes
one r4 pass using unchanged helper
`3ca107dd1ddb2ac70a457f752088b423bf96f438f0b4f963d482929625d77ed9`.
The only new receipt is K4's `cache_advice_20261006_r4.jsonl`, exclusively
created; preserve r1/r2/r3. All44 pinned train files and current memory are
rechecked by the same reviewed code. No model/data changes, global cache
clearing, training restart or further maintenance loop is authorized.
This is approval only; completion requires actual receipt review.

## Actual r4 receipt review

Root reports session29710 exited0 after this separately approved single pass.
Independent saved-JSON review verified exactly90 records: begin,44 ordered
file-begin/advised pairs,complete. All44 paths and reported hashes match the
pinned audit; all per-file before/hash/advice stat snapshots equal file-begin
stat. Begin binds the same reviewed helper, K4 and exact r4 output. Receipt SHA:
`aff19e4d6003e9884ac0448c46339acc557c8d613aeb762886f7807104e2d0c2`.

Start/end Unix timestamps were1791239538.2520118/1791239548.96068,
elapsed **10.708794733 seconds**. Receipt memory minima were MemFree
**21.356167 GiB**, MemAvailable **106.240269 GiB**, above applicable floors.
The still-growing host-watch snapshot through1791239580,938 samples, had
minima MemFree **21.006046 GiB**, MemAvailable **105.867161 GiB**; snapshot SHA:
`af4605102f9e953a65242ae3c046f5688a5e15c647f18766e52727a9d967308a`.

All three prior receipt hashes were rechecked unchanged: r1 `115ab435…374a6`,
r2 `ea9ac02c…134cf7`, r3 `55d34429…28de533`. Actual training remained live
under invocation `eee5a6fbad40411cac2f05e00520b079`, PID941365,
`activating/start`; this is not training completion. No HDF files were reread
and no advice was issued by this review. No further maintenance pass is
approved, and no model admission or PPO conclusion follows.

## Separate r5 authorization

At592 consumed windows, same live K4 invocation, host MemFree was
22,526,976KiB (about21.48GiB). Root authorizes one bounded r5 pass using
reviewed helper `d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Only K4's new `cache_advice_20261006_r5.jsonl` may be created; r1-r4 remain
unchanged. The exact44/path/hash/stat/memory/deadline checks still apply.
No further pass, global cache action, model/data modification or restart is
authorized by this entry. This is approval, not evidence of completion.

## Actual r5 receipt review

Root reports the separately approved single execution, session53568, exited0.
Independent JSON-only review verified90 records: begin,44 ordered file-begin/
advised pairs,complete. All44 paths and recorded hashes match the pinned audit;
all file-begin and before/hash/advice stat snapshots agree. The receipt binds
helper `d116b55b…b654e25`, K4 and exact r5 output. Receipt SHA-256:
`bbf63b35439627e2649ed5b50ddfa1ee2770378f9579c15d3b54517f57d9b4f3`.

Start/end Unix timestamps were1791240032.4967554/1791240042.649896,
elapsed **10.153431135 seconds**. Recorded minima were MemFree
**21.325340 GiB**, MemAvailable **106.216824 GiB**, above applicable floors.
The growing host-watch snapshot through1791240078,1178 samples, had minima
MemFree **21.006046 GiB**, MemAvailable **105.731537 GiB**; snapshot SHA:
`46063b89ab460ec3fc1681912e94448a7d93269b45efb76909bc1a7d2236ef61`.

All prior r1–r4 receipt hashes remain unchanged (`115ab435…374a6`,
`ea9ac02c…134cf7`, `55d34429…28de533`, `aff19e4d…4e2d0c2`). The actual
training service remained live under invocation
`eee5a6fbad40411cac2f05e00520b079`, PID941365, `activating/start`.
This review reread no HDF files and issued no advice. Further passes remain
unapproved; maintenance completion is not training or scientific completion.

## Separate r6 authorization

At 22:47 UTC the same live K4 invocation had consumed 707 windows and
completed 88 updates. Its latest MemFree sample was 22,479,944 KiB.
Root authorizes one bounded r6 pass with reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Only the new K4 `cache_advice_20261006_r6.jsonl` may be created; preserve
r1-r5. Recheck all 44 pinned paths, hashes, stats and memory floors.
No global cache action, model/data changes, restart, or further pass is
authorized. This is approval only, not evidence of execution completion.

## Actual r6 receipt review

Root observed session 52028 exit 0. Independent saved-JSON verification found
90 records: begin, 44 ordered file-begin/advised pairs, and complete. All 44
paths and reported hashes match the pinned audit, and every file's start,
before/hash/advice stat snapshots agree. The begin record binds reviewed
helper `d116b55b…b654e25`, K4, and the exact r6 output. Receipt SHA-256:
`ef6388ec90218fed1192480f09118c59fada983bea19af0b6a69350b0ba447f0`.

Start/end Unix timestamps were 1791240482.125588/1791240493.006071;
elapsed **10.880671498 seconds**. Recorded minimum MemFree was
**21.300934 GiB**, MemAvailable **105.828136 GiB**, above applicable floors.
The growing host-watch snapshot through timestamp 1791240518 contained 1390
samples, with minima MemFree **21.006046 GiB**, MemAvailable **105.731537 GiB**;
snapshot SHA-256:
`1af723fac16e4a4fc69391736aad1e4f19acf6b57777b92779fc5061b78c95a6`.

All r1–r5 receipt hashes were rechecked unchanged. The actual service still
reported invocation `eee5a6fbad40411cac2f05e00520b079`, PID 941365,
`activating/start`; this is live training, not terminal success. No HDF reread
or additional cache advice was performed during review. No further pass or
scientific acceptance is authorized by this maintenance result.

## Separate r7 authorization

At host-watch timestamp 1791240833, the same live K4 invocation had reached
805 windows and MemFree 22,880,008 KiB (about 21.82 GiB). Root authorizes
one bounded r7 pass with unchanged reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Exclusively create K4 `cache_advice_20261006_r7.jsonl`; preserve r1-r6.
All 44 pinned path/hash/stat and memory/deadline checks remain mandatory.
No global cache operation, data/model edits, restart or subsequent pass is
authorized. Execution success remains to be verified from the actual receipt.

## Actual r7 receipt review

Root observed session 98182 exit 0. Independent saved-JSON review verified
90 records: begin, 44 ordered file-begin/advised pairs, complete. Every path
and reported hash matches the pinned audit, and all per-file start/hash/advice
stat snapshots agree. The receipt binds the approved d116b55b…b654e25 helper,
K4 and exact r7 output. Receipt SHA-256:
`2ff7d82b2fab2bd2b0563ca6e9eba6214a02b3427b61c6a49839ed5aa4b20cec`.

Start/end Unix timestamps were 1791240855.8846467/1791240867.343941,
elapsed **11.459630691 seconds**. Receipt minima were MemFree
**21.911907 GiB**, MemAvailable **106.242134 GiB**, above all applicable floors.
The growing host-watch snapshot through 1791240881, 1565 samples, had minima
MemFree **21.006046 GiB**, MemAvailable **105.671360 GiB**; snapshot SHA:
`c4ca5e34d54645d89e8cba3edfafee1a45365a83bd95f16a7b20ee25ebfe10db`.

All six earlier receipt hashes were rechecked unchanged. The actual training
invocation remained `eee5a6fbad40411cac2f05e00520b079`, PID 941365,
`activating/start`. No HDF files were reread and no cache advice was issued
by this review. This establishes r7 maintenance completion only; no later pass,
training completion or scientific acceptance is implied.

## Separate r8 authorization

At host-watch timestamp 1791241196, the live K4 process had reached 897
windows/112 updates, with MemFree 22,864,488 KiB (about 21.81 GiB).
Root authorizes one bounded r8 pass using unchanged reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Create only K4 `cache_advice_20261006_r8.jsonl`, preserving r1-r7.
All 44 pinned path/hash/stat checks and memory/deadline limits apply.
No global cache action, model/data edits, training restart or later pass is
authorized. This entry is approval only; actual completion needs review.

## Actual r8 receipt review

Root observed session 16425 exit 0. Independent JSON-only review verified
90 records: begin, 44 ordered file-begin/advised pairs, complete. All 44
paths and reported hashes match the pinned audit, with identical per-file
start/before/hash/advice stat snapshots. The begin record binds the reviewed
d116b55b…b654e25 helper, K4 and exact r8 output. Receipt SHA-256:
`362e86fe2f7d3e2ca9cc0cd58767845bfcef043638d631087492e2cbec0b581c`.

Start/end Unix timestamps were 1791241220.1606305/1791241232.236007;
elapsed **12.075637369 seconds**. Receipt minima were MemFree
**21.714046 GiB**, MemAvailable **106.073368 GiB**, above applicable floors.
The growing host-watch snapshot through 1791241233, 1735 samples, had minima
MemFree **21.006046 GiB**, MemAvailable **105.671360 GiB**; snapshot SHA:
`5abf78c30ce195d0faf3981c32185669d8730e434d57ee6e6a52446c51b896cb`.

All r1–r7 receipt hashes were rechecked unchanged. Training remained live:
invocation `eee5a6fbad40411cac2f05e00520b079`, PID 941365,
`activating/start`. This review performed no HDF reread or cache advice.
No later maintenance pass, training completion or scientific acceptance is
authorized or established by this result.

## Separate r9 authorization

At timestamp 1791241619 the same live K4 invocation had reached 1002
windows/125 updates; MemFree was 23,104,516 KiB (about 22.03 GiB).
Root authorizes one bounded r9 pass using unchanged reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Create only K4 `cache_advice_20261006_r9.jsonl` and preserve r1-r8.
The exact 44-file identity checks and all memory/deadline limits apply.
No global cache action, data/model modification, restart or later pass is
authorized. This approval does not establish execution completion.

## Actual r9 receipt review

Root confirmed session 83760 exit 0. Independent saved-JSON review verified
90 records: begin, 44 ordered file-begin/advised pairs, complete. All 44
paths and reported hashes match the pinned audit; every file's start,
before/hash/advice stat snapshots agree. Begin binds the approved
d116b55b…b654e25 helper, K4 and exact r9 output. Receipt SHA-256:
`c65a530003086f85b620be20b4456d0028da0b1cc708b7610372a5a4c3a4dc7b`.

Start/end Unix timestamps were 1791241642.3366215/1791241653.9030135;
elapsed **11.566590663 seconds**. Receipt minima were MemFree
**21.807205 GiB**, MemAvailable **106.010109 GiB**, above applicable floors.
The growing host-watch snapshot through 1791241655, 1938 samples, had minima
MemFree **21.006046 GiB**, MemAvailable **105.671360 GiB**; snapshot SHA:
`e62eb4e9c3f870ca2bb6e5542f9c2c019be8d53ec7e86a61b91963af76589719`.

All r1–r8 hashes were rechecked unchanged. Actual training remained live under
invocation `eee5a6fbad40411cac2f05e00520b079`, PID 941365,
`activating/start`. No HDF reread or cache advice was performed by this
review. No subsequent pass, training completion or scientific acceptance is
authorized or established by the maintenance result.

## Separate r10 authorization

At timestamp 1791241959, the same live K4 invocation had reached 1089
windows/136 updates. MemFree was 23,152,348 KiB (about 22.08 GiB).
Root authorizes one bounded r10 pass with unchanged reviewed helper
`d116b55b67c50d2e618ec4188fe4fc8a4bc8b732deda8ca89f5b942d4b654e25`.
Exclusively create K4 `cache_advice_20261006_r10.jsonl`; preserve r1-r9.
All 44 pinned file identities and existing memory/deadline guards apply.
No global cache action, data/model edits, restart or further pass is approved.
Execution completion must be established from the actual receipt.

## Actual r10 receipt review

Root confirmed session 38880 exit 0. Independent JSON-only review verified
90 records: begin, 44 ordered file-begin/advised pairs, complete. All reported
paths/hashes match the pinned audit and all per-file start/before/hash/advice
stat snapshots agree. Begin binds reviewed helper d116b55b…b654e25, K4 and
the exact r10 output. Receipt SHA-256:
`f3b0a45fde0b66050b927e3303fe6478ff40aa5bef32e1a6aa0aa855b506b67c`.

Start/end Unix timestamps were 1791241984.3123283/1791241994.7507544;
elapsed **10.438632036 seconds**. Receipt minima were MemFree
**21.978786 GiB**, MemAvailable **106.055191 GiB**, above applicable floors.
The growing host-watch snapshot through 1791241999, 2104 samples, had minima
MemFree **21.006046 GiB**, MemAvailable **105.671360 GiB**; snapshot SHA:
`ac0cccfe7ff660708a2b5e64f0b9aeb163cf71bcfc3aa194d0444abca3cdf87c`.

All r1–r9 receipt hashes were rechecked unchanged. The actual training service
remained live under invocation `eee5a6fbad40411cac2f05e00520b079`, PID 941365,
`activating/start`. No HDF reread or advice occurred during this review.
This result neither authorizes a subsequent pass nor establishes training
completion or scientific acceptance.
