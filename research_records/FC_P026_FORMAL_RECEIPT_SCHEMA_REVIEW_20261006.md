# P026 formal receipt-schema review

Engineering preparation; no actual candidate or formal evaluation executed.

Reviewed runner SHA: `03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3` (`scripts/run_fcp026_posteval.py`). New adversarial test SHA: `b3a96028cdd36d211e8565246f18145309f6bbea86cc26d80c2b2186b6d1148d` (`tests/test_p026_formal_terminal_proofs.py`).

The runner previously checked the two externally reviewed receipt file hashes and `reviewed_by_lead` flags. It now also validates their actual canonical P026 content before formal execution: exact K1/K4 statuses and identities; protocol/result/manifest hashes; all seven candidate file hashes; audit 171 updates, 1,368 windows and eight-window accumulation; exact training unit/invocation and retained terminal state; reload binding to that audit's exact SHA; saved/audited/reloaded tensor identity; CPU-only/no-forward/no-optimizer/no-save flags; no-admission/no-PPO flags; and model configuration and role-loader identity against the approved formal runtime.

Receipt file-map normalization removes only the literal `candidate/` prefix after requiring the exact seven prefixed keys. Missing, repeated or unexpected prefixes/files are rejected. Lead review and original byte checks remain mandatory; schema checks do not replace independent provenance review or actual official CPU reload.

The implementation adds one validator and replaces only the former receipt loop with its call. All original numerical commands, source-chain functions and constants are unchanged. An AST regression against the pre-integration canonical runner verified every existing function except this preflight call site, and all top-level constants, were identical. The numerical source snapshot is not refrozen or modified; the orchestrating runner is approved separately.

Validation: 53 new adversarial receipt tests plus 29 existing runner tests passed (82 total, 0.12 seconds). Root read the full diff and independently reran all 82 successfully. The tests use synthetic JSON and temporary candidate metadata only, not actual trained artifacts, model loading, HDF data or GPU execution. Test-path setup mistakes in initial runs were corrected without changing production logic: the staged tree must retain scripts/tests/src layout and canonical test-helper imports must be available. The AST baseline comparison is a staging-only regression and skips once the patched file is canonical.

After exact-byte integration, canonical regression from a fresh empty temporary cwd passed 81 tests in 0.17 seconds; the single staging-only AST baseline test skipped as designed. No other test skipped.

This integration neither reports training completion nor authorizes formal evaluation, scientific admission, PPO or real-CFD execution. The original numerical criteria remain unchanged.
