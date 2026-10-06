# FC-P029 training attempt 1 — resource-guard termination

Independent read-only review. This is an operational failure, not a completed training experiment or scientific accuracy result. No restart, cache advice, model/data reads, or artifact move was performed by this review.

Approved user unit `fluid-control-fcp029-flow-train-20261006.service`, invocation `b343d13dffb6402fa91bec055cafad32`, failed with exit1. Approval SHA `4412292ee5695ee86e6dca1facdc1436587e161199da6e5394cea582003e3566`. Container `8eeb7df7b7b6e1add383ebd97034263f4f4cf5604095e807d770737061a26171` exited1/noOOM at `2026-10-06T02:47:02.023262424Z`.

The actual log records ten completed windows and one completed eight-window optimizer group. The exception occurs in the runner's `guard()` before processing the eleventh window, raising `RuntimeError: physical/CUDA memory floor`. No payload directory, result, terminal candidate, or checkpoint was created. The original immutable423 numerical source remains unchanged.

Evidence under `artifacts/fcp029_control_aware_flow_training_20261006` at review time:

- `run.log`: `393f90879c932e0bb325fea3f3530f0e38f87744db892c036e5db5174b173afa`.
- `resource_watch.jsonl`: `ad942b0cf5844fcea6faa9a1a773e8c4658af69edbd43428083d978d970df546`.
- `evidence/container_terminal.json`: `6304865409a5f69eb3c4feb732a8ffaf90e33dbca17f4553ba3aef3ed14385cc`.

Seventeen host watcher samples have minimum MemFree21,529,120,768 bytes (~20.05GiB), MemAvailable116,449,464,320 bytes. The outer CUDA guard sampled minimum20.134373GiB and exited1 with its child. These sparse samples **do not prove that the20GiB floor was always maintained**: the more frequent inner check detected a floor violation. Its failing sample was only retained in process memory, not serialized after the exception, so the precise low-water value and which physical/CUDA-free measurement triggered it are unavailable.

The no-update probe had only ~1.07GiB above the floor, and training introduces at least .351837GiB persistent Adam moments plus temporary allocations. Those effects and changing clean-file residency plausibly explain inadequate headroom; the saved evidence does not uniquely attribute the failure to one component. There is no reported CUDA OOM or numerical-nonfinite failure. Do not lower the guard or silently retry.

## Minimal recovery boundary, not execution authorization

Root proposes retaining the complete stopped directory at the new exclusive path `artifacts/fcp029_control_aware_flow_training_20261006_failed_attempt1`, then a separately approved fresh attempt at the original candidate path. This avoids changing the frozen official-CPU-reload profile's exact candidate/unit paths. Before any move, verify the full evidence file map; after the move, verify unchanged bytes and record old/new paths, old invocation, approval and container ID. Do not rewrite embedded historical paths.

A retry must have a new observed invocation and start from the original parent with fresh Adam/full171 updates, not resume the ten consumed windows. Require separately verified additional resource headroom and explicit Root approval for one attempt. Preserve20GiB continuous guards, objective/data/order/learning rate and all scientific gates. The prior successful scales/probe are not a guarantee of full-training capacity. No archive or restart is authorized by this report itself.
