# Scoped training-file cache maintenance during K4 formal evaluation

The K4 training invocation has terminated successfully. Its formal evaluation
is live under `fluid-control-fcp026-k4-formal-20261006.service`, invocation
`d5d2201c8e2c4bf2ab40201cca0dcb1e`. During validation10, timestamp1791243589,
MemFree was22,940,844,032 bytes (about21.36GiB), MemAvailable118,868,238,336.

Root authorizes one r13 pass, conditional on independent code review, using
staged helper SHA `98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c`.
Its only change from reviewed d116b55b is allowing the exact next K4 receipt
name. K1 remains limited to r1-r4. The existing 44 train-file path, hash,
same-descriptor stat, memory and deadline checks remain unchanged. The output
is exclusively `artifacts/fcp026_history_training_k4_20261005/cache_advice_20261006_r13.jsonl`.
Earlier receipts must not change. These are training files no longer used by
the formal evaluation; no validation files, models or datasets are modified.

No global cache operation, training/formal restart, threshold change, or later
pass is authorized. This is prospective approval, not execution completion.

## Actual r13 review

Root confirmed session17425 exit0. Independent saved-JSON review verified
90 records: begin,44 ordered file-begin/advised pairs,complete. All44 paths
and reported hashes match the pinned audit; every file's before/hash/advice
stat snapshots equal its file-begin stat. Begin binds exact helper98efe4a3…43ae2c,
K4 and r13 output. Receipt SHA:
`73bd9f5b7308eeb4c231669a6652d4947189b3c576f2c9b123471f228c0c72e5`.

Start/end timestamps1791243637.877229/1791243644.2484798;
elapsed **6.371486484 seconds**. Recorded minima were MemFree
**21.268818 GiB**, MemAvailable **110.594776 GiB**, above applicable floors.
Completion MemFree was29,689,204,736 bytes. Allr1–r12 receipt hashes were
rechecked unchanged. The completed training resource-watch remains unchanged
at SHAd8d29ad690469358657161d6a4e424f6ae4c5991151209eb23cce248e6bece1c;
it is historical training evidence, not a live formal-memory log.

Formal invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e`, PID1156613,
remained activating/start. No HDF reread or advice occurred during review.
This maintenance completion implies neither formal success nor authorization
for additional cache passes.
