# B CPU H5: independent ten-cycle terminal review

Engineering/raw-and-cost review PASS, not physical or prediction admission. The only intended scientific change from the original CPU H5 trial was K1 aerodynamic checkpoint → retained B. CPU high/TF32 configuration, t148 start, ten feedback cycles, five held H5 candidates, mean canonical cost and fail-stop behavior remained unchanged; no PPO fallback ran.

Actual CFD unit `fluid-control-p064-b-causal-history-h5-real-cfd-20261007.service`, invocation `3dd6413fd91d4ba9bb72c61e49920792`, independently observed PID0/exited/Result success/ExecMainStatus0. Wall timestamps: 07:20:54–07:25:34 UTC (approximately 280 s); ten feedback intervals cover only 1 D/U, not physical real-time control. Approval SHA `70c2e91fdab53731d43cf0c561262c385dad2396a2291ca4f6fcaea6465aeef9`; immutable driver SHA `761d462d3c1309274bc04816b33d3c687dab574e375b0e64b4cedd3b0479ec9d`.

Result `artifacts/p064_b_causal_history_h5_real_cfd_20261007/result.json`, SHA `f189508e962e17c5e98fa8fa6381c18664a4af9a58bd23da5da6977a50324939`. Independent receipt `artifacts/p064_b_h5_independent_audit_20261007/receipt.json`, SHA `fa1b21586896ecf26951a2a663b41ced813e3a018c3f01670978fe27ff840eb7`.

## Actual outcome

Independently read both branches' 200 force samples on `(148,149]`, dt .005. Means and RMS match the producer. Ten actions are `[.1,.2,.25,.25,.2,.1,0,-.1,-.2,-.30000000000000004]`: exactly the original K1 CPU H5 actions. Both controlled and zero saved force statistics consequently match that fixed reference (`d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`) value-for-value.

| Metric | B-controlled | Paired zero |
| --- | ---: | ---: |
| Mean total Cd | 2.4137825099145 | 2.413592168615 |
| Mean rear Cl | .8665601175300001 | .8871138481919999 |
| Rear Cl fluctuation RMS | .3889004655283842 | .3953805314509264 |

Drag reduction is **−.0078862246%** (slightly worse), and rear-Cl fluctuation RMS ratio **.9836105589**. Replacing K1 with B did not improve this trial's executed control. This short transient does not assess the original long-window physical criteria; neither prediction failures nor thresholds are changed.

Selected-next-step force MAEs, ordered front Cd/front Cl/rear Cd/rear Cl, are `[.000226855278,.002164547145,.007243460417,.046649804711]`, versus original K1 `[.000114345551,.001335914433,.005998331308,.009711787105]`. All four are larger here, despite identical selected actions. Saved selected-prediction-minus-paired-zero signs agree with actual controlled-minus-zero signs on `5/10,1/10,6/10,10/10` cycles. These are descriptive trajectory comparisons: after the first step the two states differ, and unexecuted alternatives have no matched-state CFD truth.

## Independent checks and limitations

Reconstructed the initial 62-sample history from pinned raw sources, advanced it only using each real controlled endpoint, and recomputed all 50 candidate costs and 250 stage ledgers/components using the unchanged canonical cost. Checked candidate actions/rate, state feasibility and tie ordering, chosen action and selected prediction error. All 250 mean-bias penalties are zero: relaxing the original 10% limit cannot change these recorded scores. Recalculation uses saved predictions, not independent FNO inference.

All 20 solver logs have 20 clean steps and satisfy original Courant/continuity bounds. Verified current-packet SHA bindings, all source/input pins, original restart/config tree unchanged, and both owned containers absent with no OOM. This is not an independent resampling of all current CFD fields or proof of unexecuted candidate accuracy.

Controller actual limits: 8 GiB, no swap, CPU4. Both solver records bind 8 GiB memory/swap-total caps. Sixty resource samples give minimum MemAvailable `121233772544` bytes (above 22 GiB). CPU inference took 11.1801–11.5370 s per decision; controller peak was `2426826752` bytes. Resource samples ended around 263.936 s before final cleanup.

Independent audit R2 `fluid-control-p064-b-h5-independent-audit-r2-20261007.service`, invocation `2a01407fe4f1492cb3eb9708534667b4`, exited0 with CPU1/2 GiB/no swap/120 s and CUDA hidden. Auditor source SHA `2a3b4b4b800d05210ec968a806eb6fdb125314aa880d76d62ff43fd722d6af97`. R1 `fe45918ebb0a4ec8bcfba706541f6d29` failed on a reviewer list-versus-array `.shape` interface error; preserved source SHA `da331c23bbe68cdd622b44ebdab6786248bd4755f80f273cf7ec2c999763e82f`. Root approved only explicit `np.asarray` normalization for R2. No model, optimizer or CFD was repeated by the reviewer.

Default B PPO delivery remains separate. This result neither replaces it nor licenses longer/GPU MPC without separate approval and declared engineering-equivalence limits.
