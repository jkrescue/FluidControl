# Representative256 fixed-six independent saved-array review

Decision: **arithmetic PASS; original joint retention FAIL; retain B.** No development/PPO/CFD authorization is implied.

## Bound evidence

- Evaluator unit `fluid-control-p064-fit256-fixed-six-20261007.service`, invocation `04e6ea0eb268448387a3476b71e1cadd`; Lead/producer observed MainPID0, success, exit0. Saved launch and supervisor receipts independently agree on approval `370ddda32e75710f65e49375feb4da22e143d9b7be398f4a8d3a1de3d5bdcdae`, supervisor returncode0/errornull.
- Result `artifacts/p064_fit256_fixed_six_20261007/result.json`, SHA `57add3a45f4cedcde94fbc242337e64cd6d54d51156e4d2be27c29cb8c309289`.
- Independent audit source `/tmp/audit_fit256_six_saved.py`, SHA `892488ea8fa93bcee359c36eb4c1932ab324752b46d9d2507ffeae4fdbbcb488`; two independent analytic CPU fixtures PASS (weighted four-channel loss, total-Cd cancellation, tail62 centering). Test `/tmp/test_audit_fit256_six_saved.py` SHA `264936bb84405a28ec76e234343b3ddf22558983bc6047017212d43fb163f21c`.
- Actual audit unit `fluid-control-fit256-six-saved-audit-20261007.service`, invocation `40c23cc9e0404f2eaa03aeccdf192abf`, MainPID0/Resultsuccess/ExecMainStatus0. CPU100%, MemoryMax2GiB/noSwap, Tasks64, TimeoutStart120s, CUDA hidden; no model imports/forward/optimizer. Its stdout was captured through `--pipe`, not the journal. An identical second CPU arithmetic invocation persisted `/tmp/p064_fit256_six_saved_audit_receipt.json`, SHA `9e9a88b606684276dcd7de35d6b622ae42ad253de7e8d791c787e01a77b9ea86`; no model or scientific evaluation was repeated.

## Recomputed scope

All 12 rows (two models × fixed six windows), 2,400 four-force vectors (=2 domains ×100 endpoints ×6 windows ×2 models; 9,600 scalar force predictions, not2,400 scalars) were independently recomputed from saved normalized arrays using NumPy float64. Exact window global indices are `[160,816,923,975,1077,1233]`, starts `[320,90,100,0,0,0]`; case identities/order/train splits were checked. B/candidate target arrays, force standard deviations, history metadata and producer-recorded frozen-flow-history SHA are pairwise identical; both use highest/noTF32.

The original normalized objective is `0.125*(MSE_frontCd+MSE_frontCl+MSE_rearCd)+0.625*MSE_rearCl`, separately for teacher-forced H1 and continuous AR100; total is half of each. All saved channel MSEs, balanced objectives, total, ten-chunk metadata, P020 physical four-force MAE/MSE, tail62 signed bias, squared bias, centered residual MSE and predicted/truth RMS were checked. Six-window aggregates, five-nonzero statistics and original warm/padded subgroups match. Additional total-Cd and rear-Cl physical residual MAE/RMSE are descriptive only, not new gates.

P020 float64 statistics use rtol1e-10/atol1e-12. Original objective scalars are accumulated chunkwise in float32: comparison tolerance rtol2e-6/atol2e-8 accommodates reduction rounding, not an admission tolerance. Independent aggregate recomputation differs by less than6e-10; the strict original `candidate <= B` decision agrees exactly with that from saved aggregates.

| Same-precision original objective | B | Representative256 | Relative change |
|---|---:|---:|---:|
| H1 | .004181029585500558 | .004842338986539592 | +15.81690317% |
| Continuous AR100 | .009002923189351955 | .010287666596317043 | +14.27029177% |

Both objectives worsen, so the unchanged AND nonregression rule is false. These are normalized composite prediction errors, not physical drag/RMS percentages. Historical B high/TF32 objectives must not replace this freshly matched highest/noTF32 B control.

This audit verifies saved-array arithmetic and source-defined metric semantics, not an independent model re-forward, actual tensor reconstruction, generic capacity failure, or physical control effect. The source guard's nonmutation/flow-freeze checks and engineering resource evidence remain producer/Lead evidence. The selected256-point fit improvement does not imply fixed-six retention or development generalization. Lead accepted this audit and rejected replacement of B: archive the unadopted candidate; do not run this candidate's development/PPO/CFD. Its development metrics remain unexecuted/unknown.
