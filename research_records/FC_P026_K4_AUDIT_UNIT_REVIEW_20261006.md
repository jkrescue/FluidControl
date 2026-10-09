# Exact K4 training-unit audit compatibility

The live K4 unit is `fluid-control-fcp026-history-k4-20261006.service`.
The prior auditor incorrectly constructed a 20261005 unit for both arms.
The reviewed correction uses an explicit map: K1 keeps its original 20261005
unit; K4 requires its actual 20261006 unit. No pattern matching or relaxed
invocation, approval, container, source, resource or scientific checks were added.
Training code, data and the running process are unchanged.

Auditor SHA256: `fc15ad1d691e857d1c70a2c00e56e32a516529db47d26e66c2f57fc54fe937ab`.
New test SHA256: `534d5039902a5b484303d6d63cab8311afdf259521f9ffa899d9318dc3575f62`.
Implementation, independent review and Root each passed seven CPU fixtures;
Root measured 0.01 seconds. Fixtures stop at a mocked terminal query; they
do not load candidate models, HDF data or execute an actual terminal audit.
Valid exact names reach the query; wrong dates, cross-arm names, suffixes and
short invocation identifiers are rejected. Existing downstream production
checks use evidence equality rather than the erroneous date construction.

The actual running observation is recorded in
`FC_P026_K4_RUNNING_EXECUTION_20261006.json`, SHA256
`e5614bad493bf4bf0e494cd6c113d3b3b2b54528651e6a0340772e08a69cf0bd`.
Independent inspection matched the actual unit, invocation, container, image,
start time and approval bytes. The stated 23:14:39 UTC observation is during
training, not an invented start-time observation or evidence of completion.

Before actual terminal auditing, freeze and verify a new runtime source tree
containing this auditor. Preserve the prior K1 runtime and results unchanged.
This software correction is not scientific acceptance or PPO authorization.

## Actual source-only freeze independently verified

Root executed reviewed freeze script ee3cb87b…2359fb4 with normal Python and
PYTHONOPTIMIZE unset. Independent read-only verification checked all 416
source hashes in `artifacts/fcp026_terminal_runtime_source_20261006_k4_unit_compat_immutable`:
415 match the previous runtime exactly; only `scripts/audit_fcp026_candidate.py`
changes to reviewed SHA `fc15ad1d691e857d1c70a2c00e56e32a516529db47d26e66c2f57fc54fe937ab`.
The file set matches the manifest exactly, plus its manifest/receipt. Source
files are mode0444 and directories0555. New manifest SHA:
`2417a44d8b30c59b94ba743d8f0f579e845765824a4f43e95c75b0b052fbb889`;
receipt SHA:
`c604bd148d4240530194f1bd837045a313b1e625b7693df084b9ac4bf1168152`.

All 416 old-base source hashes remain unchanged; its manifest remains
`277ec97a99e7c21b280832bb24c3da57f3ac8228a8682ae44d62cea527ad1f70`
and receipt `d583147b5213beec37fef45a889715e43ef329968a89d230cd4af7c7ba8a3e75`.
The new receipt binds reviewed commit5054cbed6b363eb48ac0a49927edcc3562e68565
and this single unit-name correction. This is source preparation only: no
candidate audit, model/HDF loading or scientific admission was performed.
