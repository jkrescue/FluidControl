# Absolute64 fixed development: independent review

**Valid negative selection result: do not replace B.** Both pooled H1 force MAEs improve, and pooled H5 also improves slightly, but both predeclared fixed-six H1/continuous-AR100 retention means regress. No statistical-significance or generalization claim is made; the original selection rule is unchanged.

## Execution and independent arithmetic

R2 unit `fluid-control-p064-absolute64-development-h1-h5-r2-20261007.service`, invocation `dd78b8303e164f869b3afad7cc33e098`, independently observed PID0/exited/success/exit0. Approval `docs/P064_ABSOLUTE64_DEVELOPMENT_H1_H5_R2_APPROVAL_20261007.json`, SHA256 `fc75c61fbda8a1256d60e4a3a9571123685e9c79748d0ec97a2c00550c7921c9`.

Result `artifacts/p064_absolute64_development_h1_h5_20261007_r2/result.json`, SHA256 `2b7558530c03808f3e5fac4cd00bec78c79c0c3e6e552f17ba90efdbd1f2a805`. Candidate manifest SHA256 `2e3c778b9cec039487602eb38b54f78f42b0a2e0fb8e467790bc585aedc0545c`. Historical B development result remains `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`; B was not rerun.

One independent CUDA-hidden CPU arithmetic audit used unit `fluid-control-p064-absolute64-dev-array-audit-20261007.service`, invocation `efee7137ab6f4314a6cf53a42aeeafec`, and exited0 under 2GiB/noSwap/CPU1/120s. Evidence: `artifacts/p064_absolute64_dev_array_audit_20261007/receipt.json`. The audit reuses the previously checked saved-array arithmetic with candidate labels only changed. All 16 NPZ hashes, 80 endpoints, every per-origin/phase/pooled H1–H5 field SSE/reference sum, four force MAEs, total-Cd MAE and persistence metric passed independent recomputation. Largest absolute roundoff was 2.6135239750146866e-8 in large field sums, within relative tolerance. All 450 source, 192 runtime and 10 input bindings were rehashed.

Every initial state, truth field/force, prescribed action, time, mask/grid and predicted flow field equals B exactly. Field equality is expected from the frozen independent flow branch, not improved flow prediction. The panel is the already-open b01/b03 development set, starts0,100,...700 with five realized transitions each, not a new holdout. Inference is highest/noTF32; zero optimizer updates and unchanged model tensors are bound producer evidence.

## Matched physical-force MAEs

| Phase/lead | B rear-Cl | Absolute64 rear-Cl | B total-Cd | Absolute64 total-Cd |
|---|---:|---:|---:|---:|
| pooled H1 | .138998316601 | .126895332709 | .038065373898 | .036246638745 |
| b01 H1 | .155280457810 | .146851062775 | .044178001583 | .042098015547 |
| b03 H1 | .122716175392 | .106939602643 | .031952746212 | .030395261943 |
| pooled H5 | .165744980914 | .162616755813 | .030762493610 | .030755061656 |
| b01 H5 | .180411564419 | .178757516667 | .033760644495 | .033826172352 |
| b03 H5 | .151078397408 | .146475994959 | .027764342725 | .027683950961 |

Total-Cd MAE is the absolute sum of front/rear Cd errors before averaging, not the sum of their MAEs. Pooled H1 hold-current-force persistence is .090333518572 rear-Cl / .020318217576 total-Cd, better than the candidate on both; H5 persistence is .439982240088 / .101057592779, worse. All H2–H4 results, full phase/lead comparisons and per-origin data remain in the bound result and independent receipt, not selectively discarded. The small b01 H5 Cd regression is explicitly retained.

## Fixed-six retention and decision

Training results are original B `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980` and absolute64 `2d715ff7d474e06ab0fcec2bc22056a020e37e0adf29645785b6fe89f44fbad8`. Fixed indices160,816,923,975,1077,1233 have matching identity/history/flow-history SHA. The following means were independently averaged from six saved per-window objectives, not independently rerun losses:

| Normalized high/TF32 objective | B | Absolute64 |
|---|---:|---:|
| H1 balanced | .003976855262105043 | .004376321196711312 |
| continuous AR100 balanced | .008946200483478606 | .009106266195885837 |

Both retention metrics regress. The predeclared rule requires both pooled H1 MAEs to improve **and** both fixed-six means not to regress; therefore selection FAIL remains despite the H1 improvement. These normalized training diagnostics are not physical coefficient MAEs and are not interchangeable with highest/noTF32 development values. No new H5 hard gate was introduced.

R1 failed before model evaluation because an inherited helper enforced a 12GiB cgroup upper bound against the explicitly approved 24GiB unit. R2 changed only that helper limit and its binding; numerical worker code was unchanged. R1 is an engineering failure, not a scientific negative result. R2 used 16GiB allocator / 24GiB noSwap, startup50/runtime22 guards; observed minimum Available was 121019887616 bytes over22 samples. Final memory-sample elapsed time was10.508914373 seconds (not a full end-to-end wall-time claim).

Retain B and the already verified controllers. The basic800-step real-CFD reproduction remains valid; overall surrogate prediction quality remains unresolved. This report does not authorize further training, PPO or CFD or erase previous failures. Separately approved exploration remains possible without relabeling this selection failure.
