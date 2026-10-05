# P026 K1: one bounded clean-cache intervention

Operational resource maintenance only; no new scientific experiment or admission.

## Authorization and exact scope

Lead approved one pass over exactly the 44 existing training HDF files listed in
`artifacts/fcp018_reduced_rate_training_20261005/candidate_audit.json`, SHA
`03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`.
The existing training source, parents, data, optimizer, objective, sampling order,
and continuous MemFree/MemAvailable >=20 GiB guard were not changed.

The source helper `scripts/advise_p026_verified_train_cache_once.py`, SHA
`5ba8916f0f3fbbfbfd9ac538e3990b8068ffd526701b74278e849280de3ab8d5`,
was reviewed before Lead executed it once. Startup operational headroom was
20.75 GiB free, with 20.5 GiB before each file hash and the unchanged dual20 floor
during/after work. These are resource checks, not scientific thresholds.

Each file was opened read-only through confined directory descriptors with
`O_NOFOLLOW`; only regular files from train20/train8/train16 were accepted.
SHA256 was streamed in 1 MiB buffers on that descriptor, then clean-cache advice
`POSIX_FADV_DONTNEED` was applied immediately to the same verified descriptor.
Device/inode/size/mtime/ctime were compared before hashing, after hashing, and
after advice. No global cache clearing, heldout path, model file, CFD simulation,
or second pass was used. The fixed output is exclusive and prevents rerunning
this helper silently.

## Actual execution and independent receipt review

Lead's SSH execution exited0. Receipt:
`artifacts/fcp026_history_training_k1_20261005/cache_advice_20261006_r1.jsonl`
SHA `792d0402237346b6f22b114f895f68e0bae9e3210c99576d2657b55247399537`.

Execution began 2026-10-05 19:56:51.023 UTC and finished after 7.451348842 s.
Independent read-only JSONL review verified 90 rows: begin, 44 ordered begin/advice
pairs, and complete; all 44 paths and digests equal the pinned audit map, all
three per-file stat identities agree, timestamps are ordered, and no abort is
present. The helper uses no writable data descriptor; the receipt records
`data_writes=false`. This review did not re-read or rehash HDF contents.

Logged pass minimum MemFree was 21.252487 GiB and MemAvailable 106.689064 GiB.
MemFree changed from 21.256115 to 26.729771 GiB during the pass. This observed
5.473656 GiB increase is consistent with releasing clean file cache, but concurrent
training/system activity prevents attributing every byte exclusively to advice.

At the independent follow-up, the existing watcher had 869 samples, with overall
minimum MemFree20.936733 GiB and MemAvailable105.834663 GiB. The same training unit
`fluid-control-fcp026-history-k1-20261005.service` remained activating/start,
MainPID598666, invocation `b3759e7e1acc4de7a1aa9f6e8d38de9a`; exact official
container `ed4f0ad7a42621712b6689d3f694ed90067f154ed7bf78c72e0993bbad930f65`
remained running with host PID599092. No restart or training completion is claimed.

## Interpretation

Read-only measurements preceding intervention showed approximately3.11 GiB
container anonymous memory and5.28 GiB file cache (5.19 GiB inactive, zero dirty
or writeback). Recent free-memory decline exceeded available-memory decline,
supporting cache accumulation rather than an observed anonymous-memory leak.
The train-file size-only inventory was44 files/6.36485 GiB, maximum0.25902 GiB.
This does not prove the identity of every cached page.

Continue the existing training and unchanged resource guard. This receipt does
not authorize recurring advice, another pass, candidate admission, PPO, or CFD.

## Separately authorized r2, 2026-10-05 20:15:46 UTC

After explicit receipt-name amendment review and separate Lead authorization,
Lead executed exactly one r2 pass with helper SHA
`94d3c43b5b4341f4fa541dd9c630b5dad83785720f65ca092a1728b2fa4c7cb8`.
The original r1 file and its digest above remain unchanged. The implementer did
not execute either pass. Source/guard semantics remain those documented above;
the amendment only permits explicit fixed, exclusively created receipt names.

Actual r2 receipt `cache_advice_20261006_r2.jsonl` in the same K1 output directory
has SHA `c8f3292ce93231e2c7ef26b000a2cb51f579047f0fbb4dbfe4df8fc3871f313a`.
Lead's process exited0 after8.668314628s. Independent read-only review again
verified90 ordered rows,44 unique exact approved paths/digests, all before/hash/
after-advice stat identities, no abort and `data_writes=false`. Minimum logged
MemFree21.717365GiB and MemAvailable106.596531GiB respected every original floor
and operational headroom check. Observed free changed from21.722134GiB to26.353031GiB;
as with r1, concurrent activity prevents exclusive causal attribution.

At follow-up the same unit/invocation b3759e7e, MainPID598666 and container
ed4f0ad7a426 (host PID599092) remained running. The external watcher had1412
samples; minima remained free20.936733GiB/available105.834663GiB. No restart,
model/protocol change, candidate completion or scientific conclusion follows.
The r3 receipt was independently checked absent; no further pass is authorized.
