# P026 diagnostic aggregation roundoff compatibility

Preparation only. No actual audit rerun or candidate acceptance.

Both actual failed audit attempts remain preserved:

- `fluid-control-fcp026-k1-terminal-audit-20261006`, invocation `a2f9e95a0aab4161b59ddf873d0be17c`: approval clip norm JSON `1` versus `1.0`; separately corrected in commit 701d5b9.
- `fluid-control-fcp026-k1-terminal-audit-compat-20261006`, invocation `f0ce7dc0cf5a47f3930abbf432121d5d`: failed at panel aggregate recomputation; no candidate-audit receipt written.

Read-only recomputation used only the exact pinned `aggregate` and `grouped_panel` function ASTs and the saved result JSON SHA256 `ec5476c389112ea4f27a2adf2f82ac3fad93ba6dcf9bb5625b60cfd80fdf5071`. No trainer imports, model/archive/HDF access or new container execution was needed. All four panels (0/456/912/1368 windows), both H1/AR domains and warm/padded summaries were checked: 136 leaves, 123 exact and 13 float differences. No key, structure, type or integer-count differences. Maximum discrepancy was two binary64 ULPs, absolute maximum `1.3877787807814457e-17`. Replacing builtin sum with math.fsum in these explanatory pure-function recomputations reproduced every saved leaf exactly.

Host auditor Python is 3.11.15, NumPy 1.26.4. The pinned training image metadata identifies Python 3.12. The functions use builtin `sum`, not NumPy. Python documents that float summation changed in 3.12: https://docs.python.org/3/library/functions.html#sum . This explains why byte-exact float serialization is not a portable aggregate reconstruction check; it is not evidence of a training/force-statistic discrepancy.

The proposed helper requires identical Python types, dictionary keys, list lengths, exact integer counts and None. It rejects bool, nonfinite values and unexpected leaf types. Only finite floating leaves allow `abs(a-b) <= 4*max(math.ulp(a),math.ulp(b))`. Only the two diagnostic comparisons (panel aggregate and history subgroups) use it. Existing strict equality, raw source/candidate hashes, optimizer/objective checks and all scientific admission gates remain unchanged. The comparison checks reporting arithmetic, not whether a candidate meets a numerical criterion.

Tests cover the inclusive four-ULP boundary, rejection above four ULPs, structural/type/boolean/nonfinite differences, unchanged original strict comparison and all 136 actual leaves. The actual-artifact test is optional away from Main; portable tests need no real artifacts. Canonical regression passed all 81 CPU tests in 0.98 seconds (38 existing terminal tools, 21 protocol compatibility, 22 aggregate compatibility), with CUDA hidden and bytecode/cache writes disabled; whitespace validation passed. A new approved immutable audit revision is required before any real retry. Old immutable sources and original approval, result, model and dataset bytes must remain untouched.
