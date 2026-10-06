# P029 first training failure and bounded clean restart

The first actual training invocation `b343d13dffb6402fa91bec055cafad32` exited1 at2026-10-06 02:47:03UTC after one optimizer update and ten completed training windows. The inner physical/CUDA memory guard raised before the next window. There is no payload or checkpoint. The sampled outer CUDA minimum20.134GiB does not prove the floor was never crossed between samples; the failing inner sample was not persisted. This is an engineering failure, not a scientific result or completed training.

Independent report: `docs/FC_P029_TRAINING_ATTEMPT1_FAILURE_REVIEW_20261006.md`, SHA `f38bccbcaee7f007297fb789a8d105aecc61457f0f2afef35e561235deaa4eed`.

Root authorizes preservation of the exact stopped output directory to the new exclusive path `artifacts/fcp029_control_aware_flow_training_20261006_failed_attempt1`. Before and after the move, verify the following five file hashes. Do not rewrite the old logs or their historical paths. Do not remove any CFD data or model.

| Relative file | SHA256 |
|---|---|
| evidence/container_created.json | f583dad81cc1213216f5d0df292982b66faa8a57f8d23b35607e651c12bc2744 |
| evidence/container.cid | af57ea8820062715047d5e6d77c5b5f9f42bf4840ca6cfbdb3882eb53636262a |
| evidence/container_terminal.json | 6304865409a5f69eb3c4feb732a8ffaf90e33dbca17f4553ba3aef3ed14385cc |
| run.log | 393f90879c932e0bb325fea3f3530f0e38f87744db892c036e5db5174b173afa |
| resource_watch.jsonl | ad942b0cf5844fcea6faa9a1a773e8c4658af69edbd43428083d978d970df546 |

One clean retry of the same user unit and original command is conditionally authorized only after a separately reviewed exact-file cache intervention provides additional physical headroom. The previous44-file train-cache pass alone was insufficient. Verify the extra completed-validation file scope, no active readers, same-descriptor SHA/stat, then observe at least35.5GiB physical free and50GiB available before restart. Continuous20GiB host/CUDA guards remain unchanged. The extra headroom reduces risk but is not a guarantee of full training capacity.

The retry must create a new actual InvocationID, begin from the unchanged original parent and fresh Adam state, consume all1368 original windows in the fixed order, and perform171 updates. It must not resume the failed in-memory update. Numerical source423, data, protocol, fixed scales, receipt bindings and original execution approval SHA `4412292ee5695ee86e6dca1facdc1436587e161199da6e5394cea582003e3566` remain unchanged. The candidate output path is reused only after preserving the failed directory; downstream evidence must bind the new actual invocation, not the old failure. No automatic further retries, scientific admission, PPO or CFD launch are authorized by this record.
