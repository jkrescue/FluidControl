# Historical monitor payload hash cache repair

Operational repair only; no scientific result, candidate admission, GPU/CFD execution or cache-advice operation is authorized by this report.

The minute watcher repeatedly streamed completed historical checkpoint payloads through `verify_receipt` / `file_sha256`. Four examined completion receipts alone referenced 2,833,558,274 bytes of model/checkpoint payload. This confirmed avoidable recurrent reads, but did not establish that this watcher alone caused all observed system page-cache growth. Dashboard sampling showed repeated PNG/JSON reads; its large-checkpoint involvement was not established.

Root approved isolated implementation, independent review, then deployment. The initial reviewed source `643e8a093dca873a7b284ac949830eedcb9e137edc4caf9a5fb988b2100d71d8` and test `fd2ec01cfec8abe330ee920f1ad338d41c7e2cc21a7b247daad8d1c96b9c1511` passed 13 synthetic CPU tests independently and after canonical deployment. No overlapping tracked edits existed in the owned files. Only the previously stopped `fluid-control-training-evaluation-watchdog.timer` was restored; other services were not stopped or edited.

Actual invocation `00b1abc5f4be4b22aa3754de1b266c48` exited0 but exposed a runtime compatibility defect: opening ancestor `/` with `O_RDONLY|O_DIRECTORY` was denied in the service execution context. The monitor correctly reported unavailable payload verification rather than falsely verifying it; its cache remained empty. Root authorized the narrow fix: directory descriptors use `O_PATH|O_DIRECTORY|O_NOFOLLOW`, while the payload descriptor remains read-only. An execute-only-directory regression was added. No service configuration or scientific code changed.

Final canonical identities:

- `scripts/watch_training_evaluation_state.py`: `d182a126ad3f92f5cc364b30356a748ee42a09f11f3bafeba46db60b03ffaa2f`.
- `tests/test_watcher_historical_sha_cache.py`: `71f7e7925f8b93e9d096e5c5428d2114937858d10d3beebebf9f91fd954392ad`.
- Canonical CUDA-hidden CPU command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /home/USER/env_isaaclab/bin/python -m pytest -q -p no:cacheprovider tests/test_watcher_historical_sha_cache.py`: **14 passed in0.04s**.

Actual final service verification:

- Invocation `27ea683416fd4fbcaa912424a8d4c02a`: exit0/MainPID0; 16 entries labelled `new_same_fd_sha256`.
- Invocation `ce93b7271f414d2ca1f6d8a675ecf1d5`: exit0/MainPID0; the same16 entries labelled `cached_prior_sha_stat_unchanged`; no nonempty historical receipt issues in latest.json.

The persistent cache is `artifacts/monitor/historical_payload_sha_cache.json`, explicitly `monitor_only_not_scientific_audit`. Only completed-receipt `.pt`/`.mdlus` payloads use it. Canonical path plus device/inode/size/mtime_ns/ctime_ns must match; a new hash checks same-FD and final path identities. Missing, changed or symlink paths fail verification; malformed/oversized caches fall back to a full hash. Entries are bounded at1024 and written atomically. Cache hits are prior verified byte hashes with unchanged stat identity, **not fresh byte audits**. Current authority, receipts, source checks and live logs remain uncached. First cache population necessarily reads historical model bytes once.

Resource interpretation: AGENTS.md requires physical **MemAvailable >=20GiB**. Observed low MemFree with roughly115GiB available is predominantly reclaimable-cache state, not proof of exhausted physical memory. This repair changes no resource admission or execution threshold; the separate resource owner handles that policy correction. No global cache clearing, file deletion, HDF modification or scientific threshold change was performed here.
