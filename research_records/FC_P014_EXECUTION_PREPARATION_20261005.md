# P014 preparation — not execution approval

The fixed-six train objective diagnostic is committed as `59215be`. Root and independent reviewer each ran the 18 CPU tests successfully. The numerical script, test and plan have immutable copies under `artifacts/fcp014_train_objective_source_20261005_immutable`; their digests match the committed files.

The proposed launcher preserves the pinned P013 training source, image, precision and train-only mounts. It verifies the pinned candidate audit, all 44 actual training HDF files and exact directory sets once before execution. The GPU script does not repeat this bulk hash scan. The launch log records the canonical mapping digest. Original validation/frozen datasets are not mounted.

Execution is not approved yet. Required before launch:

1. The exact P013 formal invocation must finish with retained exited state and exit code zero. Scientific FAIL still counts as a completed evaluation, not as candidate admission.
2. Independently verify its full receipt and referenced outputs. Bind the actual receipt SHA in a separate Lead approval, together with launcher, diagnostic/test/plan and the actual frozen GPU guard SHA.
3. Verify no GPU compute process is active, acquire the execution lock, and meet launch memory headroom. Retain at least 20 GiB MemAvailable and MemFree throughout.
4. Execute only the immutable launcher copy and preserve output, failures and resource logs. No numerical changes, optimizer or model saving are authorized.

Launcher protection review identified and corrected lost watcher exit status, an inadequate TERM-only timeout, GPU-query failure being interpreted as idle, and a watcher/container startup race. Parent supervision now checks watcher liveness during the bounded container invocation; any failure leaves a marker and repeatedly attempts bounded stop/kill of only the named diagnostic container. Independent isolated shell mocks verified normal watcher termination and an injected awk failure without invoking real Docker.

Current formal evaluation remains live. No P014 GPU invocation, result, new PPO or FNO-assisted CFD success exists at this preparation milestone.
