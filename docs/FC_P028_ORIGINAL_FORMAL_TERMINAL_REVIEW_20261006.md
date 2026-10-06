# FC-P028 original formal terminal — independent review

Status: **scientific admission FAIL; execution completed normally**. Read-only review on 2026-10-06. No model/HDF loading, GPU execution, threshold changes, PPO, or CFD launch was performed by this review.

## Execution and integrity

The retained user unit `fluid-control-fcp028-original-formal-20261006.service` is `active/exited`, exit status 0, with the unchanged invocation `eb4e12507302498bb8944373e0717a25`. The final development-gate container finished at `2026-10-06T02:27:41.389894496Z`.

Output: `artifacts/fcp028_original_formal_20261006`. Receipt SHA256: `63fd75d4e90176dd94998f2844f2f70cb5a7e357bd59d5362591019ed8655154`. Approval SHA256: `c061e50dc1d868e80d1c858ea79b03e8fe5560a204bbb30d2ef9236373838d55`.

Independently rehashed all 35 receipt-listed output files and all 411 frozen numerical-source files: all match. All eight initial/terminal Docker evidence pairs retain the same respective container ID, official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`, terminal exit 0, not running, and no OOM. These include precision and all seven numerical stages. No inference of scientific success is made from exit status.

All 1,119 host-memory samples remain above the unchanged 20 GiB floors: minimum MemFree **30,096,756,736 bytes (28.029 GiB)**; minimum MemAvailable **118,128,828,416 bytes (110.016 GiB)**. The receipt records no automatic PPO launch or frozen-test access.

The immutable original development auditor was rerun only on the existing force-window JSON. All discrete decisions and non-roundoff results agree; host/container floating-point RMS differences are at most approximately `6.67e-16`, not a changed threshold or scientific discrepancy. No candidate archives were re-read.

## Matched validation10 comparison: K1 → P028

The same split, cases, observed actions, horizons, stride 25, batch 4, seed and K1 history profile are retained. Segment counts are 320/320/310/290 for H1/H10/H50/H100. The frozen aerodynamic model yields identical H1 force metrics. Values below are reported metrics, not a newly defined gate.

| Horizon | Physical field MAE K1 → P028 | Total-Cd NRMSE K1 → P028 | Rear-Cl MAE K1 → P028 |
|---|---|---|---|
| H1 | .000961692 → .000734393 | .00486921 → .00486921 | .0202176 → .0202176 |
| H10 | .00632490 → .00471157 | .00717206 → .00744139 | .0278078 → .0279850 |
| H50 | .0121853 → .0111219 | .00679524 → .00762851 | .0294890 → .0449691 |
| H100 | .0191041 → .0197010 | .00609718 → .00997537 | .0401863 → .0693762 |

Short field prediction improves, but this does not transfer to improved force prediction. H100 field and force metrics worsen. The validation10 action-difference Cd MAE changes from **.01912969 (PASS)** to **.02739353 (FAIL against .023)**. Signs remain 8/8 and cross-action ordering 20/20; persistence and pooled-Cd subchecks remain positive. Endpoint gate SHA: `da3f8cf9932e87a21e2028c530b6387c6e64ba40c84621f464328b2adf9079fc`.

## Dynamic6 and complete trailing-window evidence

Dynamic6 diagnostic still passes its original checks: pooled H100 total-Cd NRMSE **.02551845** (K1 approximately .017893), strict start-0 delta-Cd MAE **.01575005** (K1 .01039302), under .1/.023 respectively. This endpoint diagnostic is not the complete admission gate. The force-window-derived action subset also passes: endpoint Cd NRMSE .01109932, delta-Cd MAE .01576424, signs 4/4, ordering 6/6. Its endpoint population is distinct from the pooled dynamic6 diagnostic.

Original tail62 window thresholds remain 1% of same-window zero Cd, and 2.5% of same-window zero rear-Cl fluctuation RMS for both RMS and mean errors. Absolute errors and original pass flags:

| Branch | Cd error | Rear-Cl RMS error | Rear-Cl mean error | Cd/RMS/mean pass | Joint |
|---|---:|---:|---:|---|---|
| b01 minus | .02567270 | .07544660 | .01329049 | F/F/T | F |
| b01 zero | .00058898 | .00989260 | .02428507 | T/T/T | T |
| b01 plus | .04367907 | .09457386 | .06924763 | F/F/F | F |
| b05 minus | .04812960 | .04890888 | .07144683 | F/F/F | F |
| b05 zero | .00841976 | .00992184 | .04271768 | T/T/F | F |
| b05 plus | .04415018 | .08963440 | .06817936 | F/F/F | F |

Counts K1 → P028: joint **1/6 → 1/6**, Cd **5/6 → 2/6**, RMS **2/6 → 2/6**, mean **4/6 → 2/6**. RMS errors improve on b01-plus (.122262 → .094574) and b05-minus (.068218 → .048909), but neither reaches the unchanged threshold; all other RMS errors worsen. Cd error improves only on b01-zero; mean error worsens on every branch. Thus the report retains both improvements and failures rather than only counting the unchanged joint result.

Development gate SHA: `770f004f3a9ea1072abd66719fe31aa34d2e2643206de6839fb9515121f439ff`; force-window result SHA: `80a23c9e5c74e3a71802f00a3b162d1058a01bd45f35f9e908d4a9fd707f1e48`; dynamic6 diagnostic SHA: `08ab92b8fb8162919b8386d4099ec55b298b9f477bfd44e2d8c48a2e67066396`.

## Interpretation and boundary

The fixed H10 field-only intervention improves short field error yet does not repair the original long-horizon force admission criteria. This supports testing the already separately designed control-aware flow objective; it does not establish its effectiveness, authorize its execution, or prove a unique cause. P028 is not accepted for surrogate-policy training or real-CFD control, and cannot inherit prior PPO readiness. All original scientific and physical acceptance gates remain unchanged.
