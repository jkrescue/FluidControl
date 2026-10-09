# P029 formal training-cache maintenance — actual execution review

Lead separately approved exactly one pass of the reviewed wrapper SHA `b6b7914964319ee366ccbaac1affac66c8f915d0d5b87c3bf7f7f059ed089414`. The reviewer executed that pass once, remotely, with a 180-second outer timeout; process exit was 0. No further pass is authorized by this report.

Immediately beforehand, the exact 44 training paths from the actual scales approval had no readers or writers reported by `fuser` (return 1; empty stdout/stderr). All were under their original `train/` directories, with no validation paths. The unique output directory did not exist; the executed wrapper hash matched. Preflight physical MemFree was 23,597,535,232 bytes and MemAvailable 119,406,137,344 bytes, both above the unchanged 20 GiB floor.

The wrapper imports the reviewed P029 wrapper `c0069d06428f5a834919d2e132f66b2ef62c858fe1353497def9cc085f8f0481`, changing only its receipt destination. The underlying exact44 read-only same-descriptor SHA/stat/advice implementation remains `98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c`.

Actual receipt: `artifacts/fcp029_formal_train_cache_advice_20261006/cache_advice_20261006_r1.jsonl`.

SHA: `60710f6b448ed6f325cb15b4d259e66ebcedb9df6eea3f82d2fe537dfc2a8581`.

Independent saved-JSON checks found 90 records, 44 file pairs, completion in 6.411125292 seconds, and exact agreement for every expected/observed payload SHA and all before/after-hash/after-advice stat identities. Receipt-start MemFree was 23,592,988,672 bytes; final MemFree was 30,455,431,168 bytes (approximately 28.36 GiB). Across all recorded memory fields, minimum MemFree was 23,589,253,120 bytes and minimum MemAvailable 119,388,811,264 bytes. Both remained above 20 GiB.

The executed source is retained byte-identically at `artifacts/fcp029_formal_train_cache_advice_20261006/advise_p029_formal_train_cache_once.py`, SHA `b6b7914964319ee366ccbaac1affac66c8f915d0d5b87c3bf7f7f059ed089414`.

Only clean cache advice for the completed training inputs was issued. No validation cache advice, dataset write/delete, global cache clearing, model update, or intervention in the active formal process occurred. Root independently observed the same formal invocation still live after the pass. This is operational resource evidence, not a scientific result or surrogate admission.
