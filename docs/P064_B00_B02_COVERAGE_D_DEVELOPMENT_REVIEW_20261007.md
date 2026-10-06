# FC-E092 D25 fixed development evaluation — independent review

**Engineering execution/array arithmetic ACCEPT; predeclared joint improvement rule NOT MET. Retain B; D is not supported for PPO/CFD promotion by this experiment.** H1 total-Cd improves slightly but rear-Cl worsens. This is a mixed result, not a claim that all D predictions degraded.

## Actual execution and identities

Unit `fluid-control-p064-b00-b02-coverage-d-development-h1-h5-20261007.service`, invocation `6538db4425e14b009bacdb93baf03916`, independently queried PID0/Result success/ExecMainStatus0. Output `artifacts/p064_b00_b02_coverage_d_development_h1_h5_20261007`.

- Approval `docs/P064_B00_B02_COVERAGE_D_DEVELOPMENT_APPROVAL_20261007.json`: SHA `a9a0ecf0fd0a6f44a8900257c2abde4916995616c040de06ba8c09020b5f2158`.
- `result.json`: `a9c9d2088217b599b6491ed5950e3baab024c8126033548b62d3e27843f0e51b`.
- Candidate manifest: `93d579b910d5a03385a31fb44c0050fa0b8d498fb6c621289a8d1038b5c1741a`; training result `2c01c3b24a11b30371d40ee9ef799c7fd01090d284778498919080e6323b8733` and [independent training review](P064_B00_B02_COVERAGE_D_TERMINAL_REVIEW_20261007.md) SHA `82a5c50ba114c3aaa96bc33bdb756f950935bdac6d49542a7d0bb5d7a3d8c235`.
- B comparison result `artifacts/p064_arm_b_development_h1_h5_20261006/result.json`: `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665` (read existing outputs; no B rerun).
- Independent array audit `artifacts/p064_d_development_independent_audit_20261007/receipt.json`: `495dffe6ea5fa58ca983ee2947c0701163bafdef31a89b019e782fcaa6ed0c70`; CPU unit invocation `1cdff2c910ad454f93ec5a7bd6133488`, actualexit0,2GiB/noSwap/CPU1/120s/CUDAhidden.

## Independent checks

Rehashed440 bound source files,192 runtime files,9 inputs,16 saved physical NPZs and B comparator NPZs. Fixed b01/b03 starts0,100,…700 yield16 origins × H1–H5 =80 endpoints; no omitted origin or error-based selection. Independently recomputed per-origin, per-phase and pooled four-force MAE, absolute **sum of two Cd errors** (not sum of MAEs), persistence and masked physical field SSE/reference sums. All reported metrics match within rtol1e-12/atol1e-10; largest absolute reduction-order discrepancy is2.6135239750146866e-8 in large field sums, not force MAE.

B/D initial/truth fields, truth forces, masks, grids, times, realized actions and **all predicted fields** are exactly equal. Source-selection force/action clocks and VTK-rounded time tolerance also pass. Field equality is expected from frozen flow FNO, not field-accuracy improvement. No model/HDF conversion or inference was rerun by this review. Postload highest/noTF32, optimizer0, unchanged model tensors and no scientific admission are confirmed in the bound result. Supervisor reports return0/errornull;25 resource samples, minimumAvailable121380089856bytes, elapsed12.011715425s. This is opened development, not an untouched test.

## Matched pooled force MAE (16 origins per horizon)

| Horizon | B rear Cl | D rear Cl | B total Cd | D total Cd |
|---|---:|---:|---:|---:|
| H1 | 0.1389983166 | 0.1394301741 | 0.03806537390 | 0.03782491013 |
| H2 | 0.1310303132 | 0.1312674661 | 0.02827411890 | 0.02814401314 |
| H3 | 0.1323736757 | 0.1325417687 | 0.02531950921 | 0.02517134696 |
| H4 | 0.1487967335 | 0.1489039185 | 0.02797585353 | 0.02785810828 |
| H5 | 0.1657449809 | 0.1653391613 | 0.03076249361 | 0.03059633449 |

Both phases' H1 rear-Cl regress: b01 `.1552804578→.1558962036`, b03 `.1227161754→.1229641447`. H1 total-Cd improves in both: b01 `.04417800158→.04392603785`, b03 `.03195274621→.03172378242`. At H5 both metrics improve slightly in both phases: rear-Cl b01 `.1804115644→.1797434941`, b03 `.1510783974→.1509348284`; total-Cd b01 `.03376064450→.03355977684`, b03 `.02776434273→.02763289213`. Full H2–H4 phase values and all16-origin rear-Cl errors remain in the independent receipt.

Hold-current-force persistence, matched same16 starts, gives H1 rear-Cl `.09033351857`, total-Cd `.02031821758`, better than D; H5 `.4399822401` / `.1010575928`, worse than D. These horizons have different physical targets; do not infer that persistence is universally better or that AR errors alone explain H1 behavior. Unchanged field metrics: pooled velocity relative L2 H1 `.01031610442`, H5 `.04304008059`; pressure `.03204168499` / `.1370057299`. No physically homogeneous mixed u/v/p MAE is claimed.

## Predeclared retention and decision

Read existing actual training panels at fixed original indices `[160,816,923,975,1077,1233]`; all B/D case/start/history identities match. Independent equal-six arithmetic gives H1 balanced objective B `.003976855262105043` → D `.003941269397425155`; AR B `.008946200483478606` → D `.008911167562473565`. Thus the stated retention condition (both no higher than B) passes. These are stored training-panel metrics, not a newly executed full formal evaluation.

The separate **development** rule requires both pooled H1 rear-Cl and total-Cd MAEs strictly lower than B. Rear-Cl is higher, so the joint rule fails despite retention and H5 improvements. No post hoc weighting, threshold relaxation, ratio sweep, extra epochs or candidate selection follows automatically. D's coverage intervention also changes behavior-policy/state-action coverage and mechanically re-spaced starts; this result does not prove phase coverage alone caused either change. Existing B/canonical physical control benefits, early-window failures and complete surrogate prediction FAIL remain unchanged. D full formal evaluation, PPO training and real CFD deployment were not executed.
