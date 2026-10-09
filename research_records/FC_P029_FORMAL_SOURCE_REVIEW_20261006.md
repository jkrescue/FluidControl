# FC-P029 formal source-only freeze review

Independent read-only verification of `artifacts/fcp029_formal_source_20261006_immutable` recomputed every declared SHA: 411 numerical files, 12 CPU-reload source files, and four external orchestration files. The complete actual file set equals these maps plus the declared root receipt, numerical source-chain receipt, training configuration, and CPU source manifest. All files are mode 0444, all directories 0555; no symlinks or extra files were found.

The numerical map equals the recorded 408-file base plus exactly seven reviewed overlays. Its source-chain map and the CPU manifest agree with the root receipt. The original external numerical runner remains SHA `03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3`; this review does not change numerical commands or gates.

- Root receipt: `e4327333f92108d87558d780dde47b4f1741f9065ef54c982068129ec3d763d6`.
- Numerical source chain: `3e86ce83523eaeb6335efe6943817a069a1be83a0e1ca7fd6b58a0fdf12dad43`.
- CPU source manifest: `ec8d39553250c1c38001ddcc7ead06c6a57192c00af135188f4bb6cca747ea8e`.

Canonical operations tests were rerun after correcting the staging-versus-canonical test import path: 19 passed in 0.07 s. Lifecycle AST assertions remain intact, with only explicit identity-profile argument routing normalized in the recovery-function comparison.

Scope: source bytes and metadata only; no model, checkpoint, HDF, GPU, or evaluation execution. This establishes preparation integrity, not a candidate audit, execution approval, scientific admission, or PPO readiness. Actual future training invocation and candidate/audit/reload hashes must come from completed executions. At the review observation, P028 formal remained live under invocation `eb4e12507302498bb8944373e0717a25`, PID 1436370, `activating/start`; its status-zero field was not treated as terminal success.
