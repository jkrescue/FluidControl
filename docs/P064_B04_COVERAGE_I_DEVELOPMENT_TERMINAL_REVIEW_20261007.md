# I fixed-development independent terminal review

Engineering/array verification PASS; original prediction selection **FAIL**. Keep B as default. No PPO or CFD is authorized by this report.

## Bound evidence

- Development unit: `fluid-control-p064-b04-coverage-i-development-h1-h5-20261007.service`, invocation `51303b44822849a08c602e559a09739d`.
- Approval: `docs/P064_B04_COVERAGE_I_DEVELOPMENT_APPROVAL_20261007.json`, SHA256 `9e736b357540384eaff5577310b5a521cf4d8d8eec8a2e2043e173f6d8644b29`.
- Result: `artifacts/p064_b04_coverage_i_development_h1_h5_20261007/result.json`, SHA256 `924ef17c729235ea351137848b6b9f5439b57e1e89d80f85ae522cdcdcdfe18d`.
- Independent receipt: `artifacts/p064_b04_coverage_i_dev_independent_audit_20261007/receipt.json`, SHA256 `b4589d462c29ba9cb82d25ecb87693de4a6c8401f3e5b24a12ac99a1cb61bead`.
- Independent audit unit: `fluid-control-p064-b04-coverage-i-dev-independent-audit-20261007.service`, invocation `c251f2babed54e0281ae4824b04fb005`, PID0/exited/success/exit0; CPU1, 2GiB, swap0, 120s, CUDA hidden. Checker SHA256 `64305d3150b17aa064c0576040bd4b5a7610d1b726caa9de8d7fdea763c57171`; original arithmetic base SHA256 `2d24af20d625ce2d04abbec7049e6ab32048f271d3ee04c99127ac54e0d3357f`. Three CPU fixtures passed and Sota independently accepted the lifecycle-only adaptation before execution.

The producer transient unit was collected before this audit. Success is established from one exact-USER_INVOCATION_ID manager Started/Consumed pair (timestamps 1791355012499797/1791355024604243), full approval SHA in Started, no failure/OOM evidence, pinned successful supervisor receipt, bound parent/token receipt and absent parent/evaluator processes. This is **not** a captured current-systemctl ExecMainStatus or retained current limits. Supervisor receipt SHA256 `bd20ea7a7e33a3441c0c1a08a034a1029277eeef27a873192afea277f2390aae` records returncode0/errornull and actual12GiB/swap0; manager peak493568000 bytes/swap0. Evaluation wall12.0227s and sampled minimum Available112.6489GiB. Audit peak4714496 bytes/swap0; no model forward or inference rerun.

## Independent arithmetic and decision

All 16 NPZ / 80 endpoints were verified: complete H1–H5 per-origin, phase and pooled metrics, persistence, finite arrays, saved source/action/clock alignment, and exact matched B truth/actions/initial state/frozen predicted fields. All 440 source, 192 runtime and 10 input bindings passed. Maximum reported arithmetic roundoff was 2.614e-8. B predictions are pinned saved outputs, not independently rerun.

| Pooled metric | B | I | Relative change |
|---|---:|---:|---:|
| H1 rear Cl MAE | 0.138998316601 | 0.145544018596 | +4.71% |
| H1 total Cd MAE | 0.038065373898 | 0.039605192840 | +4.05% |
| H5 rear Cl MAE | 0.165744980914 | 0.167668650276 | +1.16% |
| H5 total Cd MAE | 0.030762493610 | 0.030916076154 | +0.50% |

Both pooled metrics worsen at every H1–H5 lead. Both phases' H1 metrics worsen. The exception among phase/lead force comparisons is b01 H5 rear Cl (about0.99% improvement); it does not reverse the predeclared pooled-H1 rule. H1 persistence MAEs are0.090333518572 rear Cl and0.020318217576 total Cd, better than I; at H5 persistence is0.439982240088/0.101057592779, worse than I. Full phase/lead/origin results remain in the machine-readable receipt rather than selectively reporting favorable endpoints.

The separately audited original fixed-six training retention improves: H1 B0.003976855239898214 → I0.003812214266152599 (−4.14%); continuous-AR B0.008946200532894485 → I0.008763004034790356 (−2.05%). See `P064_B04_COVERAGE_I_TRAINING_TERMINAL_REVIEW_20261007.md` (SHA256 `2afc99ec9eb4a8027d546a360f77507c9258f9dcf05308291c67b89780609ac5`). These improvements do not offset failure of both required development H1 metrics under the original AND selection rule. No significance claim, threshold change, or new H5 hard gate is introduced.

I changes training coverage through b04 fixed late PRBS excitation as well as phase; this is not an isolated causal test of phase alone. Frozen-flow equality is expected by construction, not evidence of improved flow prediction. Existing successful real-CFD closed-loop delivery and historical negative results remain unchanged.
