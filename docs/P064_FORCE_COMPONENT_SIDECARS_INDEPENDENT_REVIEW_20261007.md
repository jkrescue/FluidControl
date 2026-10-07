# Train-only component sidecars: independent review

ACCEPT for data preparation, not predictive improvement or training authorization.

Actual conversion: `fluid-control-p064-force-component-sidecars-20261007.service`, invocation `19ccdb8f8c3742c49957e9b01f8c707a`. Approval `docs/P064_FORCE_COMPONENT_SIDECARS_APPROVAL_20261007.json` SHA256 `06ae2e41c8e22153408bc23e3c0e1f2f1dbbbb5b244ceb40638f3f5de3494ff0`; output manifest `artifacts/p064_force_component_sidecars_20261007/manifest.json` SHA256 `f8b2f6564e761d360c6a0101c6d6d036f49f7ce9643ecb7755b86cd43e6f9c3a`.

Independent read-only audit verified all45 compact sidecar hashes, all20493 timestamps and pressure+viscous=raw-total arithmetic, and exact equality of sidecar time/old-total arrays to the original HDF time/force arrays. Counts are base20:20/16020; train8:8/1608; train16:16/2064; controlled b00:1/801. The b00 source is the existing conversion result's HDF SHA `45041e79e70043838763e5dd8da0fe9c02dba4cfe6df01356f8dcd1389e32d8f`.

For each sidecar, first/last raw dictionaries were rehashed and independently parsed (90 samples); official PhysicsNeMo HDF5Reader also actually read first/last rows (90 samples), with every stored field exact. All family and all-train physical pressure/viscous means and population standard deviations were independently recomputed. These are new label statistics, not a refit of the old normalization.

Maximum pressure+viscous minus raw total: `1.3322676295501878e-15`; time mismatch: `6.103515630684342e-06`; existing HDF total minus raw total: `1.243862659752043e-05`. This last discrepancy is retained, not hidden by redefining viscous force or replacing old total labels. Raw pressure and viscous labels are genuine separate dictionary entries. Source-path inspection supports possible time/interpolation/output-rounding mechanisms but does not uniquely attribute the discrepancy.

Root observed running→PID0/exit0. Reviewer read the original invocation's successful worker and supervisor journal; the supervisor recorded actual8GiB/noSwap/CPU1/120s limits. The unit was subsequently garbage-collected, so current missing properties are not reported as retained execution properties. Warp CUDA100 is the CUDA-hidden import warning. No conversion or scientific model was rerun; the audit did not read flow-state datasets or rehash large HDF payloads.

Machine-readable review receipt: `docs/P064_FORCE_COMPONENT_SIDECARS_INDEPENDENT_RECEIPT_20261007.json`. Audit implementation was `/tmp/audit_p064_force_component_sidecars.py`; its single read-only execution exited0. Scope is readiness for a separately specified component-supervision experiment; no auxiliary weight, centering, initialization, or accuracy improvement is selected by this review.
