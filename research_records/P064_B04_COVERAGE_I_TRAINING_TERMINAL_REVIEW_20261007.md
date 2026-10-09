# I training terminal independent review

Engineering review PASS. Fixed-six retention passes both original tests; full development assessment has not yet run, so this is not prediction admission or controller promotion.

Training unit `fluid-control-p064-b04-coverage-i-training-20261007.service`, invocation `7c1d1d634af94321b638ba6da2febe45`, was independently verified PID 0 / normal exit 0 / success before any candidate read. Approval `docs/P064_B04_COVERAGE_I_TRAINING_APPROVAL_20261007.json`: SHA256 `f380a076a92e1ffa98a497c1fe7933cfb6cc880b4b5e9c3e1708d4cd93a66cf6`.

Output root: `artifacts/fcp064_controlled_aero_arm_i_b00_b04_20261007`.

- Result SHA256: `728bb93b0003db8992a456dfbf0b415dde635a74566deb8ba37a6a5169214d07`.
- Dual manifest SHA256: `b9b28da3d8dbe3e4b71bd814ac4c69d6ca75c88080cc1fe5002eab13541c76ad`.
- Independent receipt: `artifacts/p064_b04_coverage_i_terminal_audit_20261007/receipt.json`, SHA256 `b7cc2bbf345848ce1188ad242b84c6d47d3a2738bf51ab686dc62b9d2cf4dfc0`.

## Checks and scope

The reviewed thin checker `/tmp/check_p064_i_terminal.py` (SHA256 `a8406a022a912473e6c5bdf47c2656537a898eb3fcd49bc021da13bdb827cda4`) reuses the SHA-pinned original B/D audit. Three CPU fixtures passed. Exactly one terminal audit ran: unit `fluid-control-p064-b04-coverage-i-terminal-audit-20261007.service`, invocation `f1e69aacf14a4fbc9792d8342a430f78`, exit 0, CPU 1 / 8 GiB / no swap / 120 seconds / CUDA hidden / explicit absolute project PYTHONPATH.

It checked all 443 source hashes, actual journal order and 256 windows / 32 updates, 192 original + 32 b00 + 32 b04 entries, within-update replacement slots 0 and 4, and fixed schedule SHA `010b806342f4115f051fbd3e51bd691558e5e31c4a4375f0d447530331e369e2`. CPU weights-only checkpoint inspection confirmed 28 finite Adam states at step 32 with original optimizer settings. Flow checkpoint bytes and the two frozen lift biases remain unchanged. Saved official model/state hashes and producer fresh official reload evidence were checked; this audit did not construct or forward a model, independently rerun official reload, or rehash large training data payloads.

Actual training memory cap was 12 GiB with no swap; recorded peak 3,066,359,808 bytes and minimum available memory 106.68467330932617 GiB. Saved objective arithmetic was checked, not independently reconstructed from model predictions.

## Original fixed-six retention

The six identities/history/flow-history hashes match B. The reviewer recomputed each original balanced objective from the saved four-channel MSEs and averaged all six windows; no auxiliary objective or new gate was introduced.

| Objective | B | I | Relative change | Original non-degradation rule |
|---|---:|---:|---:|---|
| H1 | 0.003976855239898214 | 0.003812214266152599 | −4.139979% | PASS |
| Continuous AR | 0.008946200532894485 | 0.008763004034790356 | −2.047758% | PASS |

These are fixed train-panel retention measurements, not significance or generalization claims. The planned fixed b01/b03 16-origin H1–H5 development evaluation still requires its separate approval. See the pre-terminal scope clarification `docs/P064_I_ASSESSMENT_SCOPE_CLARIFICATION_20261007.md` (SHA256 `5decd5c549cbd8074806ea502e5f1abec27a487ff5107c8d0d05e7dbee457c96`). The original AND selection rule remains intact; B remains the default, and no PPO/CFD or further training is authorized by this review.
