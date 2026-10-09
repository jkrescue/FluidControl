# FC-P013 fixed-six read-only diagnostic approval — 2026-10-05

Lead: Root. Approved only for the already specified P009-parent versus P013-r2 terminal comparison. This is not scientific admission, parameter selection or PPO approval.

Training completed1368 updates at08:38 UTC. The actual captured Docker container emitted die exitCode0 followed by destroy; exact invocation journal records the GPU guard exit0. Original transient systemd unit was garbage-collected, so the original finalizer refused the missing-unit state. Preserve that operational failure. Do not execute this diagnostic until an independently reviewed collected-unit recovery has produced a completion receipt from authoritative terminal records and the full unchanged candidate audit.

## Exact candidate and execution identities

- Candidate: `artifacts/fcp013_independent_force_fno_training_r2_20261005`.
- Candidate audit SHA: `1c280b291ae7ded1f8e63ccc46ba40a7e085e7d7b6fa69f0dddff18ffac704a7`.
- Aerodynamic model SHA: `2eb4dde99c5f1a8ba8d8f2821c3871e906a6f13dd3d01f1ed1e8eba5379f1a03`.
- Aerodynamic checkpoint state SHA: `b0d859e6576c679bc7c09812bff6d55db6341253061236a45e1c52b9e17993ca`.
- Dual manifest SHA: `23d2917ef038196f066934f51d0be2770c23b26099eb7ab97439ce58488e13bb`.
- Training result SHA: `239f6567d662157e8f0bec8b1277cc219f7580ac3dde3238657dea8d6b4821d0`.
- Flow model/state remain exact P009 parent identities recorded in the manifest and candidate audit.
- Reviewed launcher: `scripts/run_fcp013_fixed_diagnostics_spark.sh`, SHA `cf0bc8d57cb02ff23d0c006fca26e7fd63330a7b52a478a1ef73785561c62bef`; run a byte-identical immutable copy.
- Original training source1634c05, diagnostic code in immutable23a4ec020ed6 chain and exact pinned2.2.2 image remain as checked by the launcher; no working-tree numerical code enters this diagnostic.

## Protocol, resources, decisions

Run the existing six exact train-only H100 windows on both P009 and terminal P013. Compute the original physical H1/free-AR four-force errors, u/v/p errors, tail62 rear-Cl mean and centered-RMS errors. No optimizer, save, model selection, validation or frozen data, policy or CFD control. Hash model tensors before/after. Recheck all44 train HDF byte hashes and exact file sets against the completed audit.

Use Main GPU0 alone, readonly mounts, network-disabled isolated container, timeout1200 seconds, allocator fraction0.15, container memory40GiB. Start only with MemAvailable>=50GiB and MemFree>=30GiB. Existing runtime guard preserves MemAvailable>=20GiB. No simultaneous GPU experiment or bulk transfer. Root monitors resource/log progress and diagnoses any failure rather than blindly restarting.

Output is exclusively `artifacts/fcp013_independent_force_fno_training_r2_20261005/fixed_six_diagnostics`. Never overwrite earlier output.

Completion verifies panel identity, finite values and unchanged tensors; it is not a new scientific PASS threshold. Compare every window and report regressions. Proceed to separately approved original formal evaluation after integrity/diagnostic review; only unchanged full admission can permit a newly trained compatible PPO and original real-CFD paired control validation.
