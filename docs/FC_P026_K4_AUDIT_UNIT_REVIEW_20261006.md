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
