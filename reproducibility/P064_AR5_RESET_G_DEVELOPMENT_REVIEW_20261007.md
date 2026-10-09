# G reset-AR5 fixed development evaluation — independent review

**Valid negative result under the original selection rule: do not replace B.** G improves both pooled H1 force MAEs, but fails the predeclared fixed-six continuous-AR100 retention requirement. H5 rear-Cl regression is disclosed below, not introduced as a new post-hoc gate. No significance claim is made.

## Execution and independent arithmetic

Actual unit `fluid-control-p064-ar5-reset-g-development-h1-h5-20261007.service`, invocation `ef7cd3cfb6f34975935bab28a7f5b3c7`, was independently observed PID0/exited/Result=success/ExecMainStatus=0. Approval `docs/P064_AR5_RESET_G_DEVELOPMENT_APPROVAL_20261007.json` SHA `ec5fe47cb9ff50dd9710157c2da3776d86d375c77545f804a10e30cca134660e`.

Result `artifacts/p064_ar5_reset_g_development_h1_h5_20261007/result.json` SHA `1da616b91019b3084c58e45a6e0e85010a6baecd48baa55dd91b03aa03ed180a`; candidate manifest `6123b587065ad46104c328f276949ba07d3d7b61266b6c781cb5e9a538cfd681`. B reference result remains `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`, without rerunning B.

Independent CPU unit `fluid-control-p064-ar5-reset-dev-array-audit-20261007.service`, invocation `3aa92651520144db97fbe241fea6004e`, exited0 under 2GiB/noSwap/CPU1/120s. All 16 NPZ SHA bindings, 80 endpoints, every H1–H5 per-record/phase/pooled field SSE/reference sums, four force MAEs, total-Cd MAE and persistence metrics were recomputed. Total-Cd error is the absolute sum of front/rear Cd errors before averaging, not the sum of component MAEs. Largest absolute floating-point difference was 2.6135239750146866e-8 in large field sums, within relative tolerance. 449 source, 192 runtime and 10 input bindings rehashed successfully.

B/G initial state, all truth fields/forces, actions, rounded VTK times, mask/grid and predicted flow fields are exactly equal. Selection remains b01/b03 starts0,100,...700, with five realized actions each. Physical force labels are float32 CFD endpoints. Inference is highest/noTF32, with zero optimizer steps and unchanged model tensors. Field equality is expected because the independent flow branch is frozen, not evidence of improved flow prediction. These are already-open development trajectories, not an independent holdout or online FNO control test.

Actual evaluation lasted10.5096s; 22 memory samples had minimum MemAvailable121339428864 bytes. Supervisor return0/errornull, with the approved12GiB/noSwap host and6GiB allocator contract. No new CFD or training was run by this review.

## Matched force MAEs — all leads retained

| Panel | Lead | B rearCl | G rearCl | B totalCd | G totalCd |
|---|---:|---:|---:|---:|---:|
| pooled | 1 | .138998316601 | .137072701938 | .038065373898 | .035423174500 |
| pooled | 2 | .131030313205 | .131141861435 | .028274118900 | .026426687837 |
| pooled | 3 | .132373675704 | .133615727071 | .025319509208 | .023450106382 |
| pooled | 4 | .148796733469 | .149185585789 | .027975853533 | .027317453176 |
| pooled | 5 | .165744980914 | .167154481169 | .030762493610 | .030589848757 |
| b01 | 1 | .155280457810 | .150756228715 | .044178001583 | .041801899672 |
| b01 | 2 | .136143652722 | .133884835057 | .030229948461 | .026568427682 |
| b01 | 3 | .133854428306 | .133717603981 | .021978601813 | .018185377121 |
| b01 | 4 | .157143987715 | .157678080723 | .026937693357 | .025352694094 |
| b01 | 5 | .180411564419 | .182705239393 | .033760644495 | .032393187284 |
| b03 | 1 | .122716175392 | .123389175162 | .031952746212 | .029044449329 |
| b03 | 2 | .125916973688 | .128398887813 | .026318289340 | .026284947991 |
| b03 | 3 | .130892923102 | .133513850160 | .028660416603 | .028714835644 |
| b03 | 4 | .140449479222 | .140693090856 | .029014013708 | .029282212257 |
| b03 | 5 | .151078397408 | .151603722945 | .027764342725 | .028786510229 |

Pooled H1 hold-current-force persistence is rearCl .090333518572 and totalCd .020318217576, better than G on both; H5 persistence is .439982240088/.101057592779, worse than G. Persistence uses each segment's fixed initial force, not rolling future truth. Pooled velocity relative-L2 is .010316104419 atH1 and .043040080591 atH5; pressure relative-L2 .032041684989/.137005729854, all exactly unchanged from B.

## Original fixed-six retention, independently averaged

Training-result identities are B `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980` and G `d6b406038f452806b4852c36cf818d43600684fa0f7a034dfc7460bf722e5581`. Fixed indices160,816,923,975,1077,1233 have matching identity/history/flow-history SHA. G panels explicitly use diagnostic continuousAR100, not training reset-AR5. Means below are independently recomputed from the six saved per-window objective values, not independently rerun model losses.

| Normalized objective | B | G | Retention |
|---|---:|---:|---|
| H1 balanced | .003976855262105043 | .003752365300897509 | improves |
| continuous AR100 balanced | .008946200483478606 | .009264696273021400 | fails |

These normalized high/TF32 training diagnostics are not physical coefficient MAEs and are not numerically interchangeable with highest/noTF32 development metrics. Original selection required both pooled H1 MAEs to improve **and** both fixed-six H1/AR means not to regress. The latter fails. Thus G is not promoted; B and the already verified controllers remain. The basic800-step real-CFD reproduction remains valid, while complete surrogate prediction quality is still unresolved. This result does not automatically authorize continuation, PPO or CFD. A separately and explicitly approved exploratory control experiment remains possible without changing this prediction-selection failure or replacing the retained B model; no such execution is claimed here.
