# Fixed 27-frame force diagnostic — independent review

Engineering and saved-row arithmetic: ACCEPT. Diagnostic only; no prediction admission, model promotion, or new execution authority.

## Bound execution

- Unit: `fluid-control-p064-coarse-roi-force-recoverability-20261007.service`; invocation `656922d7fdf64f6c85470f4404393644`.
- Approval: `docs/P064_COARSE_ROI_FORCE_RECOVERABILITY_CPU_APPROVAL_20261007.json`, SHA256 `71c278b8b5334437bc19ffa71c048d245461ee53d81bf76ed57965fbfeb4050a`.
- Result: `artifacts/p064_coarse_roi_force_recoverability_20261007/result.json`, SHA256 `45a52fa20c8371fd0c0ad1c4d7bb15336556b1c10ac6f700635ec148ce108507`.
- Supervisor receipt: same directory, `supervisor_receipt.json`, SHA256 `b09c101bb2e4536ab17b9d9ada5e172c269ab0623affef331522771a6e458a29`.
- Worker `ccb4a13885c6ecc11c5378c137cd514dc1702340ea7fb61f6aed35bdada76b3e`; helper `7d0485bc07aea692f1ca5d5d4897af6f0d70d218db9c7eefc54e8f2e6454c196`; supervisor `60af78e39b383976a488c244862563f934a1b796cd05a833d40036eb8f0a1d97`.

Independent preflight ran six CPU fixtures (6 PASS, 0.01 s) and the actual curator-runtime supervisor dryrun (exit 0). Geometry and 27 property-file path/time/hash bindings passed. Dryrun did read geometry/property metadata, but not the selected HDF frames. Limits were 2 GiB/no swap/CPU1/120 s, with a 100 s child timeout, startup Available ≥50 GiB and runtime ≥22 GiB. The supervisor enforces actual systemd limits, and the worker additionally checks memory before each selected frame.

The transient unit was already garbage-collected when independently inspected. The retained invocation journal contains the successful result SHA, and the bound supervisor completion receipt matches the exact unit/invocation/spec/result. This review does not invent retained systemd limits or treat not-found as a live failure. The CUDA error 100 import warning is expected with CUDA hidden; this CPU diagnostic completed successfully.

## Independent saved-row checks

All 27 unique rows cover the three fixed b00 train cases (−0.75, zero, +0.75) at indices 0,100,…,800; all 18 action-minus-zero pairs were retained. Independently checked fixed times, all property-file SHAs, pressure+viscous totals, component fractions, cancellation ratios, pressure-proxy errors, and every saved MAE/RMSE/bias entry. Maximum component-sum discrepancy: `8.881784197001252e-16`; maximum reported HDF-total alignment discrepancy: `7.291611403559273e-08`. The latter is the producer's recorded comparison, not a second HDF read.

| Quantity | Front Cd | Front Cl | Rear Cd | Rear Cl |
|---|---:|---:|---:|---:|
| Mean pointwise viscous absolute-component fraction | 25.1062% | 16.7978% | 31.0584% | 23.1488% |
| Pressure-proxy MAE | 0.04628398 | 0.02216861 | 0.06517566 | 0.10064850 |
| Pressure-proxy RMSE | 0.04629106 | 0.02461019 | 0.08908511 | 0.12369449 |
| Action-minus-zero pressure-proxy MAE | 0.00017160 | 0.00075323 | 0.08540617 | 0.13335935 |

The fraction is `abs(viscous)/(abs(pressure)+abs(viscous))`, averaged over points, not a signed percentage of total force. Pressure contributes the complementary fraction. Rear-Cl viscous fraction reaches 79.4208%; pressure/viscous cancellation makes total-force percentages misleading. Viscous effects therefore cannot simply be omitted.

The proxy samples pressure at R+2h but integrates with R dθ and the bound physical normalization. It approximates pressure traction only, not total wall traction. Its rear-body and action-response errors do not establish that coarse fields cannot represent force, nor isolate a model failure cause. No fit, model forward, optimizer, or CFD execution occurred. No scientific admission threshold was changed.

Receipt wall time: 4.503382199 s; minimum sampled Available: 114.630855560 GiB. Independent review re-read the small result/receipt/property files and recomputed arithmetic; it did not repeat the 27-frame producer or full-HDF hashing.
