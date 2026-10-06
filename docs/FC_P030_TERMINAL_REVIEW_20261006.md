# FC-P030 actual terminal review: operational failure

Status: FAILED — aggregation interface error, not a scientific result. No admission, policy training or CFD action is supported. Review performed from existing source and small execution artifacts only; no HDF/model reread, GPU work or restart.

## Actual execution identity

- Unit: `fluid-control-fcp030-train-horizon-20261006.service`; invocation `d7739a9bf4fe42f68a584e8ec5edc684`. Independently observed terminal MainPID=0, ActiveState=failed, SubState=failed, Result=exit-code, ExecMainStatus=1.
- Approval SHA: `2f80dab06ca1916495272dac0adf155745969534ce3d5b1626e7a4295eadc98f`.
- Frozen v2 source manifest: `a14bd2c00ff5c905ccd6193843736a6ccc2806cd2c2d7a51827635cb09fdb6e3`; its 17 source members were independently verified during preparation. Neither source nor numerical configuration was changed by this reviewer.
- Actual container: `1d4b1ed558ab69f71c1f3d19d54f0ec15f69e3d71f504d89a2d26a0425c79199`, official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`. Started 04:15:49.124089298 UTC; finished 04:18:28.404143151 UTC; exit=1, OOMKilled=false, Running=false. Actual limits: 12 GiB memory and swap, network none, read-only root, GPU0.

## Failure and limits of evidence

The log records all 44 distinct origins completing, with counts 1..44. Aggregation then fails at driver line 320 / core line 387: `TypeError: tuple indices must be integers or slices, not str`.

The driver passes a dictionary keyed by `(case,start,dataset_index)` into `grouped_and_paired`; the accepted core iterates a sequence of metadata dictionaries and accesses `row['case']`. Iterating the driver's dictionary supplies tuple keys instead. This is a real producer/consumer interface defect. The synthetic CPU suites and independent source review did not exercise the complete driver-to-core aggregation call and therefore missed it; their PASS was not end-to-end execution proof.

No `result.json` or durable raw prediction records exist. Consequently neither pooled/paired at-lead metrics nor family/phase comparisons can be independently recomputed from this failed attempt. Completion logs establish progress through the rollouts, not scientific accuracy. No claim about K1 versus P029 improvement or regression is warranted.

`selection.json` was persisted before rollout and binds the actual approval SHA. It contains exactly 44 start0/H100 rows: base20/train8/train16, phase counts 15/15/7/7, and family-phase counts base5 each, train8 2 each, train16 b00/b02 8 each. Selection SHA: `5daf596740d7e6d552897c4161cf75c431d64c72fb8f0b1f63a3614daedf0912`.

## Resources and retained evidence

The inner GPU guard completed with exit1, 79 samples, minimum CUDA free 21.538467407226562 GiB and MemAvailable 111.16032028198242 GiB. The outer 79-sample watcher has minimum MemFree 23,101,534,208 bytes and MemAvailable 119,332,265,984 bytes. These sampled minima and explicit traceback support an interface failure, not a recorded memory-floor or OOM failure; samples do not assert unsampled continuous values.

Artifacts remain under `artifacts/fcp030_train_horizon_diagnostic_20261006`:

| Artifact | SHA256 |
|---|---|
| run.log | 92e46af0718ba5dd57a8320cc9e1727db8e3ea3cad990a47b16ab9aeb77b98ee |
| resource_watch.jsonl | 936cd0031695704384c6e821ebdb76d349ff31c43d29c3cca356b471223fb1af |
| evidence/container_created.json | f5ba4cf5c21ee2b55765cf9f9ae3db78b87d722e0be64bb5f9de0b1af9db955f |
| evidence/container_terminal.json | 8ba98386f66b64f4bfe686663623c2ff9c1c852ad86da524a87183c73854c552 |

Bounded correction: pass the already validated metadata rows to the core, add an actual-core 44-row driver aggregation regression, and persist raw records before aggregation. Preserve this attempt and v2 source. A new immutable version and separately authorized rerun are necessary; this review does not authorize either execution or altered scientific thresholds.
