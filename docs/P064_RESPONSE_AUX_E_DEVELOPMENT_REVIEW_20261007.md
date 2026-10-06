# E response-auxiliary fixed development — independent numerical review

**Numerical authenticity ACCEPT; predeclared candidate-choice conditions NOT MET.** Both pooled H1 force metrics and both fixed-six retention metrics worsen slightly versus B. Small differences do not become improvement. Retain B; no automatic PPO/CFD promotion, extra updates or coefficient search.

## Actual bindings and independent checks

Actual unit `fluid-control-p064-response-aux-development-h1-h5-20261007.service`, invocation `78e385e2abe6486a83d9e53471063cd2`, MainPID0/exited/ExecMainStatus0. Approval `docs/P064_RESPONSE_AUX_E_DEVELOPMENT_APPROVAL_20261007.json`, SHA `481e0e906ce739ee6122f7187f6bb6e3ad91798fbb7a83b2c3935c90881654fa`. Result `artifacts/p064_response_aux_development_h1_h5_20261007/result.json`, SHA `3c4970d7d60e83567157137b28064f805c2ec28a3c990d031819ef5d68a53f80`; manifest `50a87f0129f3372b1d6d3c2a48d38f6507e8e7822c61e39ad027ecdba76bc92a`. Supervisor SHA `90221bb36a99bb4bd05eb172eb36d51e9f0969c1e87d7b6ec6e83f38738d704c` reports return0/errornull,12GiB/noSwap. Minimum observedAvailable121170219008bytes over23samples; wall11.010079s.

Independent NumPy-only audit `/tmp/audit_e_dev_arrays.py` reuses the D evaluator-array audit, changing only candidate-label/report labels. Actual CPU unit `fluid-control-p064-response-aux-dev-audit-20261007.service`, invocation `1aefdd128a904ca793b62be1982ad753`,2GiB/noSwap/CPU1/120s/CUDAhidden, exit0 in1.276s. No model inference or B rerun.

- All449 bound source entries,192 runtime files and10 inputs rehashed.16 saved NPZ files/80 endpoints verified, all per-record, per-phase and pooled four-force MAE, totalCd MAE, hold-current-force persistence, masked field SSE/reference sums independently recomputed. Largest absolute roundoff2.6135239750146866e-8 is a large field-sum accumulation difference, not a force-metric discrepancy.
- Both b01/b03 have fixed starts0,100,…,700 and H1–H5. E/B truth, initial fields, mask/grid/time, realized actions and predicted flow fields match exactly. B reference result SHA `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665` is reused read-only. Frozen flow is unchanged by design, not improved field prediction.
- Same highest/noTF32 protocol, zero optimizer updates and unchanged inference tensors. These are already-opened controlled development trajectories, not new holdout evidence or prospective unknown-action prediction.

## Predeclared pooled force comparison

Values are coefficient MAE, not percent force errors.

| Horizon | B rearCl | E rearCl | B totalCd | E totalCd |
|---|---:|---:|---:|---:|
|H1|.138998316601|.139014150482|.038065373898|.038086727262|
|H2|.131030313205|.131033398211|.028274118900|.028287049383|
|H3|.132373675704|.132365694270|.025319509208|.025327499956|
|H4|.148796733469|.148800834082|.027975853533|.027977392077|
|H5|.165744980914|.165722916485|.030762493610|.030765276402|

Both H1 metrics also worsen on each phase: b01 Cl.155280457810→.155301975086 and Cd.044178001583→.044208027422; b03 Cl.122716175392→.122726325877 and Cd.031952746212→.031965427101. H5 is mixed: pooled Cl slightly lower but Cd slightly higher; b01 Cl lower/Cd higher, b03 Cl higher/Cd lower. This is not a consistent improvement.

Pooled H1 persistence Cl.090333518572/Cd.020318217576 is better than E; pooled H5 persistence Cl.439982240088/Cd.101057592779 is worse than E. Thus H5 outperforming hold-current-force does not satisfy the stricter predeclared B comparison.

Velocity relativeL2 remains exactly B: pooled H1.0103161044193 and H5.0430400805913. Pressure/field component sums and all phase/horizon field metrics were also verified unchanged; mixed u/v/p relative metrics are not treated as physically homogeneous error quantities.

## Fixed-six retention, from existing terminal saved records

Independent arithmetic over all six per-window objective records confirms indices160,816,923,975,1077,1233 with exactly matched identity/history/flow-history hashes. No extra forward pass:

- H1 mean: B.003976855262105043 → E.003981937033434709: **retention not met**.
- AR mean: B.008946200483478606 → E.008949131064582616: **retention not met**.

These are normalized training-objective values under the original training precision, not the physical coefficient MAEs or heldout metrics above. Actual auxiliary losses were saved as scalar records only and cannot be independently recomputed from prediction arrays; this limitation remains in the training review and is not filled by this development audit.

## Scope and disposition

Engineering execution success and the already-reproduced canonical real-CFD basic loop remain valid. E fails its prospective jointH1 and fixed-six non-regression conditions; it does not replace B. Full surrogate precision FAIL, earlier rejected candidates and seed/early-window physical failures remain recorded. No threshold relaxation, global causal claim or overall-goal completion follows.
