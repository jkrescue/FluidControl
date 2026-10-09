# FC-P020 terminal independent review — FC-E030

## Verdict

Operationally complete; predeclared local-support conditions **not met**. This is a six-window train-only finite-update comparison, not a candidate, full-training result, heldout evaluation or scientific admission. No threshold changed and no PPO/CFD experiment was run. P018 original formal FAIL remains in force.

Both arms used the same P018 tensors and fresh AdamW, 16 updates each, all SIX fixed training-window raw gradients averaged before clipping. Five nonzero windows supply the statistic summaries; zero remains separately reported. Arm B adds the fixed5/16 sum of normalized H1/AR tail62 mean-error-square and RMS-amplitude-error-square to the original objective. No sweep or checkpoint selection occurred.

## Verified execution and provenance

- Service `fluid-control-fcp020-symmetric-statistics-20261005.service`, exact invocation `4567f6d393414bba8baf2239d16960a7`: retained active/exited, Result=success, ExecMainCode=1, ExecMainStatus=0, MainPID=0. Not a running default exit status.
- Output `artifacts/fcp020_symmetric_statistics_20261005/result.json`; SHA `a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042`.
- Approval SHA `ab69dfc9559a7e5458aba08c85b57d8d4aa41529ac7980e28f739ef667fab30a`; source/approval commit `c315cbc`; external startup evidence `docs/FC_P020_RUNNING_EXECUTION_20261005.json`.
- Probe SHA `139dee2c3dcd4ee97de78a7a0e343ca9ab9d9708a1cafab447f0d9e4988b5d8c`; launcher `f2d2dba8fc7668beeb346ab89e935fae402e42406b5b50898fbbd32dc2067e11`; plan `727fe550182748817461f7a2a5912acfddab59ba0c2df3a9c8e7c81bf08cd6f3`.
- All approval source/dependency SHA entries and actual candidate files rehashed; fixed P018 audit identity matches. Exact-invocation journal contains the successful44-HDF byte-verification event with the pinned audit mapping SHA, and one guard-complete exit0. This terminal review checked that startup evidence rather than rereading all44 HDF files again.
- 32 sequential update records and192 unique `(arm, update, window)` backward events; fixed order160/816/923/975/1077/1233. All output numbers finite. All eight endpoint-panel aggregates independently recomputed (floating summation precision accounted for), raw repeated rows and A/B initial rows exactly identical. Full comparison checks and repeat-assessment dict independently reproduce saved output. Statistic replay-output maximum difference0.
- Reviewed implementation checks28 finite trainable gradients/Adam states, frozen flow/two force biases and final restoration before exclusive result write. Restored tensor hashes: flow `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb`; aerodynamic `6f58aea89ecdde46bfaafac0f181d2603bbc96e3bba1faa7d8a818dcf6f7984d`. No candidate checkpoint saved. Tiny CPU fullgraph and regression tests17 passed independently before execution; not substitutes for this actual-run review.

## Matched endpoint evidence

Objectives are SIX-window macros; physical statistics below are FIVE-nonzero-window macros, tail indices38:100. RMS amplitude error is not centered residual waveform error.

| Metric | Shared initial | A original terminal | B symmetric terminal |
|---|---:|---:|---:|
| H1 original objective | 0.003522910670 | 0.003519619912 | 0.003493990303 |
| AR original objective | 0.008851685920 | 0.008362007356 | 0.008383282916 |
| H1 bias MSE | 0.000349823628 | 0.000359521527 | 0.000354625179 |
| H1 RMS-amplitude error MSE | 0.000956622879 | 0.000918437429 | 0.000888191251 |
| AR bias MSE | 0.000767300389 | 0.000783213504 | 0.000764951632 |
| AR RMS-amplitude error MSE | 0.006898064713 | 0.006825313776 | 0.006804428760 |
| H1 centered residual MSE | 0.002245067607 | 0.002514654211 | 0.002481441425 |
| AR centered residual MSE | 0.013250079934 | 0.012034743377 | 0.012017440140 |

B improves all four statistical terms versus A, but H1 bias MSE rises `4.801551509539558e-6` and H1 centered residual MSE rises `0.00023637381820897953` relative to the shared initial model. Both domain original objectives decrease, yet that cannot override the two failed local conditions. Saved and recomputed conclusion: `LOCAL_CONDITIONS_NOT_MET`, `strict_local_conditions=false`, `local_support=false`. Both-repeat observed spreads are zero; this is not a rigorous numerical-error bound or a new tolerance.

## Resources and limits

382 external host samples: minimum MemAvailable106.7127571106GiB and MemFree24.6204109192GiB. Inner guard384 samples: minimum available106.7146263123GiB, CUDAfree24.6222801208GiB, exit0. Probe247 host checks: minima available106.7560272217/free24.6630592346GiB. No resource/deadline violation markers; both20GiB floors satisfied. The result directory contains only approval/launcher copies, log, resource samples and diagnostic JSON.

No terminal model or full gradient vectors were saved: this review does not claim independent model re-inference or vector re-dot. Finite endpoint evidence supports a limited relative benefit over the original-loss arm, not simultaneous repair relative to the starting model, generalization, capacity adequacy or net physical benefit. Root authorizes PREPARATION ONLY for current-force-conditioning data/causality/persistence-baseline investigation; no GPU, architecture implementation or training is approved. Existing physical mean-lift0.10 and surrogate error criteria remain distinct and unchanged.
