# FC-E088 — C50 fixed development comparison, independent terminal review

Engineering execution and saved-array arithmetic verified. **C50 did not improve over B on the fixed development comparison.** No prediction admission or new PPO/CFD result follows.

Actual unit `fluid-control-p064-arm-c50-development-h1-h5-20261007.service`, invocation `d7a29fc521d44952a85827a8a4cf911c`, PID0/exited/exit0. Approval SHA `b21a102f85139b8e0af7a280f63eb82f2e8504ed0cc039aeff58737de8573dc0`. Output `artifacts/p064_arm_c50_development_h1_h5_20261007`; result SHA `9fa7c88759bf83fdca87d79ff305c0f3d4654368c7ec3059d7f9ba3f96c48c1a`.

## Actual independent checks

Rehashed 440 source entries, 192 runtime entries, nine inputs, all 16 source HDF files and all 16 result NPZ files. NumPy-only recomputation covered all 80 endpoints, each origin, each phase and pooled summaries: four individual force MAEs, total-Cd MAE (absolute front-plus-rear signed error, not sum of individual MAEs), masked physical field SSE/reference, and hold-current-field/force persistence. Largest absolute summation-order difference was 2.6135239750146866e-8 in large field sums; every comparison passed rtol1e-12/atol1e-10. No model reconstruction/forward, HDF array deserialization, optimization or CFD rerun occurred in this review.

Fixed panel: b01/b03 controlled development, starts0,100,…700, H1–H5, 16 endpoints per lead. Every saved action and force target matches the selected record, and stored times match nominal times within1e-5 (VTK float32 rounding is retained). Compared with B result `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`, all true fields, initial fields, masks, x/y grids, times, actions, true forces **and predicted flow fields are exactly equal** across all16 records. Thus the force comparison is matched and the frozen flow has not improved.

Execution records show highest/noTF32 inference after historical checkpoint-load identity validation, frozen model tensors, optimizer0 and scientific_admission=false. Supervisor error=null/returncode0; actual cgroup12GiB/noSwap, 23 memory samples, minimum MemAvailable121320198144 bytes, supervision span11.01157455s. These precision/model-unchanged statements are bound execution evidence, not a second independent model run.

## Pooled force errors

Absolute coefficient MAE, not percentage force errors. All16 origins remain in every denominator.

| Lead | B rearCl | C rearCl | B totalCd | C totalCd | Persistence rearCl | Persistence totalCd |
|---|---:|---:|---:|---:|---:|---:|
| H1 | .138998317 | .139148227 | .038065374 | .042953398 | .090333519 | .020318218 |
| H2 | .131030313 | .131252904 | .028274119 | .032148622 | .179995085 | .041096397 |
| H3 | .132373676 | .132429112 | .025319509 | .028874982 | .267659909 | .062011182 |
| H4 | .148796733 | .149145835 | .027975854 | .030530229 | .353433461 | .082366511 |
| H5 | .165744981 | .166302826 | .030762494 | .032763399 | .439982240 | .101057593 |

C is worse than B on both designated pooledH1 metrics and both metrics at every later lead. At H1, C is also worse than A (.156116880 rearCl, .039952166 totalCd) for totalCd, though its rearCl remains lower than A; this does not rescue the failed B-to-C comparison. C H1 remains worse than hold-current-force persistence. At H5 C beats the stale persistence baseline, but that is not broad accuracy admission.

## Each phase, all leads

| Phase | Lead | B rearCl → C rearCl | B totalCd → C totalCd |
|---|---|---:|---:|
| b01 | H1 | .155280458 → .155424902 | .044178002 → .049686693 |
| b01 | H2 | .136143653 → .136318039 | .030229948 → .035725057 |
| b01 | H3 | .133854428 → .133883418 | .021978602 → .027668819 |
| b01 | H4 | .157143988 → .157262716 | .026937693 → .030113742 |
| b01 | H5 | .180411564 → .180656866 | .033760644 → .035708293 |
| b03 | H1 | .122716175 → .122871553 | .031952746 → .036220104 |
| b03 | H2 | .125916974 → .126187770 | .026318289 → .028572187 |
| b03 | H3 | .130892923 → .130974807 | .028660417 → .030081145 |
| b03 | H4 | .140449479 → .141028953 | .029014014 → .030946717 |
| b03 | H5 | .151078397 → .151948785 | .027764343 → .029818505 |

All10 phase/lead mean pairs regress in both metrics; this is not a claim that every individual origin regresses. Against persistence, C H1 rearCl wins3/8 b01 and4/8 b03, totalCd1/8 and3/8. At H5 rearCl wins7/8 each phase and totalCd8/8 each phase.

Frozen-flow pooled velocity relativeL2 H1/H5=.010316104419337488/.04304008059131634; pressure=.032041684989035424/.13700572985397347, exactly B. These are distinct field metrics, not a mixed physically homogeneous u/v/p MAE.

## Interpretation and limits

The already-produced six-original-train-window retention panel was also recomputed from its six saved rows (indices160/816/923/975/1077/1233), without inference. Common initial H1/AR objective=.003488336884/.008805384403. B terminal=.003976855262/.008946200483; C terminal=.004539801487/.009506991735: both C objectives worsen versus B and the parent. Using the existing five-nonzero convention (exclude index160), H1/AR absolute tail-RMS errors are parent .025990219014/.074307746335, B .018166232749/.077622917792, C .018917421210/.078267947878. C keeps some H1 improvement versus parent but is worse than B, and AR worsens. This small saved training panel is not a complete44-trajectory retention evaluation. No new full formal C evaluation was run.

The controlled-data fraction increase25%→50%, with the same parent/budget/objective, did not improve this fixed development panel. C128 starts use the same spacing rule but are not a strict superset of B64; this is the prespecified dose-profile comparison, not a causal proof that controlled data generally harms learning. These b01/b03 trajectories are already-opened development, not unseen tests or statistically independent phases. Future realized actions condition this retrospective replay; it is not online FNO/MPC.

Do not promote C to PPO based on these results or change gates retrospectively. Existing B/canonical policy physical benefits, their early failures, the original complete prediction FAIL and the rejected H25 candidate remain intact. No new training, CFD or broader test is authorized by this report.
