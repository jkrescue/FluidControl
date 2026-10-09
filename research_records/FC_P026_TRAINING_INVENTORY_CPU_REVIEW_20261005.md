# P026 training inventory and launcher — independent review

## Verdict

The actual CPU inventory execution completed successfully. Receipt identities, original sampler order, family counts and both effective protocol hashes independently recompute correctly. This is inventory/preparation evidence, not training approval, an all-HDF-byte audit or scientific admission. No rerun or GPU work was performed by the reviewer.

## Actual retained execution

- Container: `fluid-control-p026-training-inventory-cpu-20261005`.
- ID: `2fa6d157760a8ee65befa76442aa745bc89c06493a2c690edaf68b28e642698c`.
- Actual image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Start/finish: `2026-10-05T19:10:27.534876487Z` / `2026-10-05T19:10:37.074489499Z`.
- Terminal state: exited, PID 0, exit code 0, OOMKilled false.
- runc, no device requests, `CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`; receipt reports CUDA unavailable.
- Two CPUs, 4 GiB memory and memory-plus-swap limit, network none, readonly rootfs. Stage, immutable source, configuration, metadata views and three train directories are readonly; only the dedicated inventory output mount is writable.

Receipt: `artifacts/fcp026_training_inventory_cpu_20261005/inventory.json`.

SHA256: `76c84cae2e08595d5326159926396ed8b1bc31a6c1fdd09a92bbae5b9ef2a526`.

## Independent receipt checks

The ordered list contains exactly 1,368 unique `(family, case, start)` identities, all train split. Sorting by original family/file/start reconstructs the global dataset indices; hashing their recorded consumption order independently yields the original sampler SHA256 `177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`.

| Family | Windows | Padded starts |
|---|---:|---:|
| base | 720 | 20 |
| train8 | 408 | 16 |
| train16 | 240 | 32 |
| Total | 1368 | 68 |

All warm flags equal `start >= 3`: 1,300 warm and 68 padded windows. This is the same K4-availability classification for both matched arms; no K1 windows are removed.

Recorded protocols equal the reviewed trainer's effective dictionaries, and their canonical hashes recompute:

- K1: `daf22b2464744509260f1eb9e0b20d3b80da484c8985f3a22887293bbb40cb30`.
- K4: `72b3638c4f0fbad687bcc4b216365e8c68935611778f7be562386abaa1db7a3d`.
- Trainer: `562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e`.
- Existing loader: `c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae`.

The reviewed preflight pins configuration, loader, manifests and normalization; it builds real-HDF metadata indices and iterates only the official sampler, not field batches. It explicitly reports `full_hdf_hash_performed=false` and `training_performed=false`. Full data-byte verification remains the training launcher's responsibility.

## Launcher preparation review

Reviewed preflight SHA256 `dce38fc338eaf100483c0d7631307f23fb2787620296dc8b4a53c73a54394913`; training launcher `b14a48e89bdee441c365bb5c729573a7696b40d24bca8c0a588d5c2aa7304f32`; preparation tests `db97fb0787d3d2764faa015b39a0b7892df119792064c883fdf9a3fa1c298602`.

Independent CUDA-hidden preparation plus trainer tests: **18 passed in 0.99 seconds**; launcher `bash -n` passed. Full static review confirmed arm-specific outputs/approvals, shared K1/K4 lock, exact protocol/inventory/trainer-hash bindings, separate parents, readonly inputs, exact-44 same-descriptor hash/cache advice, checked GPU-idle query, startup 30 GiB free/50 GiB available and continuous both-20 GiB floors. Timeouts are 14,400 inner / 14,500 watcher / 14,540 outer seconds. Watcher death records failure and repeatedly stops only the owned container until the run pipeline exits; final success checks require watcher liveness, no violation markers and runtime/exit receipts.

Static tests do not execute the four-hour trainer or establish all-window memory stability. Scientific training still requires Lead approval and the plan's formal-history-caller prerequisites. No source or canonical document edits were made by this review.

## Retained raw-evidence hashes

- Docker inspect stdout: `5246a6113f4766d7e7a4b5e6144d17fe3bbe5962b3d323cbccd544f369e47ff6`.
- Docker logs stdout: `02529be5bd4b8337af0c9326a465eb78b59580e5ecf1dac5c9ab4f93070f6911`.
- Docker logs stderr: `6ee92fb3fc14591838763f9feeb4e35e5251287557333fa7dd486b2af0199724`.

Raw output remains available from the retained container. These are observed evidence hashes, not a claim that separate raw archives were created.
