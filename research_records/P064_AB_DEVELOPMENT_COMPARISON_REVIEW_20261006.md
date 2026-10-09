# P064 K1/A/B development comparison — independent saved-array review

Status: DEVELOPMENT_COMPARISON_VERIFIED_NOT_SCIENTIFIC_ADMISSION.

## Executed identity and independent scope

A evaluation invocation `68ef4125cc5b494dad9b52a320c143c2` and B `6613b11d351f48fba27ba33fdff54534` both retained MainPID0, normal exit0 and Result=success. Approval SHAs A `e42e28f1b4651ccb13d30c2508e9ef65e6a76c3a24d0b0c7517bc2ed2c36eeb6`, B `5e05d12d415959523bcdfd473990dacf61cef902aa1a1874ded8c79e983db590`. Results:

- K1 `9ea3e0e781e76265bbc65ea52d6fec92ebe5b5cb7a93d91c3c9addb454f7c4de`.
- A `c8b0242658a101120603514e6d2e5076c827c518965c92470810fe9693840fe6`.
- B `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`.

For each A/B evaluation, independently rehashed16NPZ,440 source entries,192 runtime entries and8 inputs; recomputed all80 endpoints, four-force MAE, total-Cd MAE from signed front+rear error before absolute value, masked physical field SSE/reference, persistence and every origin/phase/pooled aggregation. Maximum absolute discrepancy2.613523975e-8 in large field sums from reduction order; all comparisons satisfy rtol1e-12/atol1e-10. No model reload, forward, training, HDF reread or CFD was performed by this review. NumPy-only checker `/tmp/audit_p064_candidate_dev_arrays.py` derives directly from the independently used K1 checker, changing only arm/approval/output binding.

Each panel is fixed b01/b03 controlled development, starts0,100,…700, five future endpoints each. These are already-opened development trajectories, not untouched tests or statistically independent phases. Actions are recorded realized future commands; this is conditional retrospective replay, not online prediction/MPC.

A/B sampled minimumMemAvailable120453775360/121369776128 bytes,22 observations each, supervision spans10.5128/10.5095s. Supervisor errornull/return0, unchanged-model receipt, zero optimizer, highest/noTF32 effective flags verified; source-bound official load retains historical high/TF32 validation before override. These unchanged-model/precision statements combine recorded execution with source checks, not an independent model construction.

## Predeclared comparison and limits

B versus A pooledH1 strictly decreases both designated force MAEs: rearCl by10.9652%, totalCd by4.72263%. Relative to K1, reductions are12.5088%/3.94921%. This satisfies the predeclared descriptive development comparison; it is not formal surrogate admission or proof of better future physical policy control.

| Lead | K1 rearCl MAE | A rearCl MAE | B rearCl MAE | K1 totalCd MAE | A totalCd MAE | B totalCd MAE |
|---|---:|---:|---:|---:|---:|---:|
| H1 | .158871165 | .156116880 | .138998317 | .039630465 | .039952166 | .038065374 |
| H2 | .152044217 | .150462177 | .131030313 | .029367138 | .029684521 | .028274119 |
| H3 | .153251002 | .152501113 | .132373676 | .026036341 | .026595727 | .025319509 |
| H4 | .167256397 | .165742605 | .148796733 | .028167088 | .028651252 | .027975854 |
| H5 | .179850096 | .177951261 | .165744981 | .030521851 | .030791938 | .030762494 |

All values are absolute coefficient errors, not percentages. B H5 rearCl improves6.85934% versus A, but totalCd improves only0.095625%; versus K1, B H5 totalCd worsens0.788427%. Per-origin B beats A at H1 rearCl12/16 and Cd11/16; H5 rearCl12/16 and Cd7/16. Neither aggregate improvement nor lower mean guarantees improvement at every origin.

## Every phase and lead

Entries show K1 / A / B. All16 endpoints per lead remain in the denominator; no case exclusion.

| Phase | Lead | rearCl MAE K1 / A / B | totalCd MAE K1 / A / B |
|---|---|---|---|
| b01 | H1 | .167390110 / .165051956 / .155280458 | .046172589 / .046496876 / .044178002 |
| b01 | H2 | .150909421 / .149929818 / .136143653 | .032059260 / .032313354 / .030229948 |
| b01 | H3 | .148084225 / .148063352 / .133854428 | .023113608 / .023927644 / .021978602 |
| b01 | H4 | .172386875 / .172895653 / .157143988 | .027062334 / .027800627 / .026937693 |
| b01 | H5 | .189924812 / .190671136 / .180411564 | .033152774 / .033531040 / .033760644 |
| b03 | H1 | .150352221 / .147181805 / .122716175 | .033088341 / .033407457 / .031952746 |
| b03 | H2 | .153179013 / .150994536 / .125916974 | .026675016 / .027055688 / .026318289 |
| b03 | H3 | .158417778 / .156938874 / .130892923 | .028959073 / .029263809 / .028660417 |
| b03 | H4 | .162125919 / .158589557 / .140449479 | .029271841 / .029501878 / .029014014 |
| b03 | H5 | .169775380 / .165231386 / .151078397 | .027890928 / .028052837 / .027764343 |

b01 H5 totalCd worsens0.684752% versus A and1.833543% versus K1. At b01 origin0, B rearCl H1/H5=.056370/.042884 versus A .022862/.008492; selected local improvements must not hide this regression.

Persistence holds each origin's initial true force. Pooled H1 persistence rearCl/Cd=.090333519/.020318218, still better than B .138998317/.038065374. At H5 persistence=.439982240/.101057593; B improves over the stale baseline but this does not establish broad absolute accuracy. B H1 rearCl beats persistence at b01 3/8 and b03 4/8; H1 Cd1/8 and3/8. H5 rearCl7/8 eachphase and Cd8/8 eachphase.

## Frozen flow and training-retention tradeoff

Independently compared every saved array across K1/A/B: all16 predicted flow sequences, true states, initial fields, masks, true forces, action endpoints and timestamps are exactly equal. Hence velocity/pressure are unchanged, not repaired. Pooled velocity relativeL2 H1/H5=.010316104/.043040081; pressure=.032041685/.137005730. The only trained role is aerodynamic readout.

Recomputed saved six-original-train-window summaries from individual rows, no rerun and no new test population. Global indices160/816/923/975/1077/1233 are unchanged; start and terminal panels have consumed0/256. “Five nonzero” excludes historical index160 as the existing diagnostic convention, not a new admission statistic.

| Diagnostic | Common parent | A terminal | B terminal |
|---|---:|---:|---:|
| Six-window H1 objective | .003488336884 | .003437512894 | .003976855262 |
| Six-window AR objective | .008805384403 | .008827898137 | .008946200483 |
| Five-nonzero H1 absolute tail-RMS error | .0259902190 | .0255834177 | .0181662327 |
| Five-nonzero AR absolute tail-RMS error | .0743077463 | .0741958490 | .0776229178 |

B improves the H1 tail-RMS statistic but worsens both original objectives and AR tail-RMS error. A also has a small AR-objective regression. This is a retention tradeoff, not comprehensive improvement; no post-hoc gate is added. The small fixed panel cannot establish retention across all44 trajectories or all1368 training windows.

Original H100 failure remains. No new PPO/CFD outcome follows from this comparison, no10% physical-bias relaxation is needed, and prior three observed-phase physical successes are neither revoked nor expanded into general robustness claims.
