# FC-E094 — paired first-step precision diagnostic

One approved execution completed at 2026-10-06 20:30 UTC: unit `fluid-control-p064-first-step-precision-20261007.service`, invocation `22bc9a45344a4969bae30c33f4d85c04`, PID0/exited/exit0. Approval SHA `73f79f4f8c717d5849898cd6625f5f17dca961430fbefbe531cf30b242579f59`. No optimizer, training, CFD, model save or threshold change.

Frozen worker `e706d4b2e8f0ef76684e9e68fd686a1cb95e9b0e81029d319b6d6c998d5e80cf` and supervisor `2a1a5402a1714415d52ad644eed3026393cf7c21c1430a0665e7699a62688460` reside in `artifacts/p064_first_step_precision_source_20261007_immutable`. Exact b40 image and411-file E073 source closure, B manifest927669/aero57d463/flowdc41, dynamic6 manifestbfa490, train normf1b460, E073 reference1eacc9 and AR6195 were checked before execution. Source preparation independently accepted; author/reviewer each passed5 CPU tests. Canonical test only changes its import path, not executed science.

Output `artifacts/p064_first_step_precision_20261007/result.json` SHA `7de34f709d3993ed3e1fc2cd28d4dbd3b3a72a1f3ac1e867b18bda03a5a00eab`; supervisor receipt SHA `788f435bbfa83c400871bcd834b923dca251059d1fcac0f76b1e677768447dfa`. Sota independently accepted all6×2 rows, actual flags, paired inputs, all8 action-minus-zero rows and exact6×4 high-to-E073 first-step reproduction. Both role tensor digests match before/after. Counts12 dual/12 flow/12 aero/24 submodel forwards are theoretical loop counts, not hardware instrumentation.

Each case uses the same true q0, recorded omega0→omega1, normalization and batch1; q1 is never a model input. Historical high/TF32 official loading is preserved. Each first high call must reproduce E073 with atol=rtol=0 before its highest/noTF32 counterpart. All six pass exactly. The full vectors and observed flags are retained; no binary small-signal admission rule is applied.

| First-step response | CFD truth | high/TF32 | highest/noTF32 |
|---|---:|---:|---:|
| b01 minus totalCd−zero | −.000429571 | +.000970423 | +.001063466 |
| b01 plus totalCd−zero | +.000443280 | −.001009166 | −.000929832 |
| b05 minus rearCl−zero | +.002412081 | −.002039790 | −.002060771 |
| b05 plus rearCl−zero | −.002466440 | +.002597451 | +.002480268 |

Individual rearCl predictions shift by approximately−.0063 to−.01154 between precisions, so arithmetic sensitivity is real. Much of that shift cancels in same-phase subtraction, and the displayed local response reversals remain under highest/noTF32. Thus switching off TF32 does not remove these specific discrepancies. This is not a universal sign failure, a proof of their unique cause, or validation of either precision as physical truth. Only six small first-step responses were evaluated; full four-channel results, including contrasts not reversed, remain in the artifact. Dynamic6 is opened development and differs from the controlled b01/b03 panel.

Runtime18.0045s, minimum sampled MemAvailable112.7111GiB, well above22GiB. Host and container each12GiB/noSwap/CPU1, allocator6GiB,600s inner/630s outer; caps are separate, not an aggregate12GiB promise. Owned container is absent after terminal cleanup. No further precision rerun is recommended by this evidence. Preserve D's failed joint-H1 choice rule, complete FNO prediction FAIL and successful canonical real-CFD results. Next priority is safe reproduction of the existing canonical closed loop, while control-relevant force response remains an unresolved precision objective; no automatic new scientific experiment follows.
