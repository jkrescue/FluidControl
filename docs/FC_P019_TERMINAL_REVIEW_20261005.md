# FC-E029 — P019 local gradient alignment diagnostic

Independent terminal review, 2026-10-05. Diagnostic completion is not surrogate admission or a training intervention.

## Execution and provenance

`fluid-control-fcp019-gradient-alignment-20261005.service`, invocation `275365360254440aba18ed96aac58630`, is retained loaded/active/exited, success, ExecMainCode1/Status0, MainPID0. Exactly60 unique gradient passes (six windows × five gradient kinds × two repeats) match the exact-invocation journal. One GPU-guard completion event reports exit0 and matches `run.log`.

Result: `artifacts/fcp019_gradient_alignment_20261005/result.json`, SHA `1bd66e3cbf7c1200ff0af96d23bd62803d59422129eec5c5dddf7615fbcb183f`.

- Source commit `6de42a9`; probe SHA `03e0ba375bf131bcc30bbe0d96c8b44bcc5c18ef1ed99de5d9e563e1b6525a64`.
- Approval SHA `f4ac0117df82fad065ee774757abc062cb5183421ca21ae555764a62c850049d`.
- Preregistered plan SHA `88426236f904994ab2d2f76e8d2b2b6d3a5604df137abd0c5221d77fd3d26060`.
- Launcher SHA `75545e9adf916dee823977c4c4ff41e2eae858ed3d10439ba0b8864aacfe61bc`.
- External running evidence: `docs/FC_P019_RUNNING_EXECUTION_20261005.json`, SHA `5719e608b59b61de03206714128079b08e0bf52efd1a85392c385ac25626adab`; this observation is not itself terminal proof.

Approval/source/dependency/candidate hashes match. The startup journal's44-HDF verification map matches pinned P018 audit SHA `03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`; map SHA `037d9d1958ed7673b7483dec9b3f9e22909afe0490c16e66b4169eb265dcd154`. P018 terminal model/state, protocol, configuration and train-only normalization are unchanged. Each original per-window objective exactly reproduces the P018 terminal panel in both repeats.

The108 host resource samples have minimum MemAvailable **102.286327 GiB** and MemFree **20.551254 GiB**. The108 internal guard samples report minimum available102.279716GiB / CUDAfree20.544643GiB. Both20GiB requirements held, but headroom was narrow; this is not evidence of spare capacity for concurrent work.

## Local directional evidence

The direction below is the **negative raw original objective gradient**, not an AdamW step. Statistics are squared physical tail62 mean error and squared physical tail62 fluctuation-RMS amplitude error; they are not centered-residual MSE or real-CFD control acceptance metrics.

| Five-nonzero statistic | Along five-nonzero original gradient | Along six-window original gradient |
|---|---:|---:|
| H1 mean error squared | -0.166742317 | -0.237139969 |
| H1 RMS error squared | -1.567300210 | -0.993538667 |
| AR mean error squared | -0.020016946 | -0.237960353 |
| AR RMS error squared | +0.859184292 | +0.468013701 |

Positive derivative indicates local increase along the specified negative gradient. The AR-RMS five-window values in repeats1/2 are +0.859184292277/+0.859184299210; six-original values are +0.468013700760/+0.468013706242. Their signs agree. Other three aggregate derivatives are negative in both repeats.

Per-window derivatives use each window's **own** original gradient; these are not the aggregate-gradient derivatives:

| Global index | H1 mean | H1 RMS | AR mean | AR RMS |
|---|---:|---:|---:|---:|
| 160 zero | -4.700330 | +0.235079 | -2.024694 | -0.754884 |
| 816 | +67.452940 | +3.458267 | -84.246349 | -18.976924 |
| 923 | +16.968148 | -14.347062 | +39.683066 | -66.832365 |
| 975 | +36.215608 | -10.054589 | -49.860735 | -35.568714 |
| 1077 | -1.031992 | +0.562112 | +2.958346 | -0.726270 |
| 1233 | +12.074582 | -24.448323 | -2.647059 | -27.742748 |

All six individual AR-RMS derivatives are negative, despite the positive aggregate derivative. This supports **local cross-window gradient interference under aggregation**, not universal within-window objective/statistic conflict. All24 per-window signs agree across repeats; maximum observed derivative spread is3.7532e-8. Original-output and VJP-replay output differences are zero; maximum measured cotangent cast difference is2.32146e-10. Repeat spread is observational, not a rigorous error bound or a new significance gate.

## Verification limits and decision

All reported values are finite. Independent checks verified64 gradient-summary algebra identities and recomputed the five/six objective and physical-statistic macros. Reviewed implementation and14 CPU tests include actual pinned mixed-batch20 objective/VJP comparisons against full autograd, tail62 analytic derivatives, complex real inner products, and mutation/gradient-cleanup invariants. Full gradient vectors were not persisted: independent vector re-dotting was not possible without another run, and none was launched.

No optimizer, parameter update, saved candidate, held-out evaluation, PPO or CFD control run occurred. Model tensor identities were checked unchanged by the probe. This diagnostic neither establishes the cause of all P018 failures nor demonstrates that a modified loss will improve held-out results. It does not license changing admission thresholds; P018 remains complete formal FAIL.

Next: **P020 preparation only**, a proposed two-arm16-update fixed-six-window comparison of original versus symmetric tail-statistic loss. All six training-window gradients are averaged; the statistical summary separately focuses on five nonzero windows. No GPU approval or execution yet. Freeze the exact protocol and obtain independent review/Lead approval first; retain original criteria and require later full evaluation for any candidate.
