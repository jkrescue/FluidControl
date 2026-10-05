# FC-P013 terminal dashboard update — 2026-10-05

The existing private dashboard now separates completed training, fixed-six train-window diagnostics, and the live formal evaluation. It preserves existing historical flow visualizations and explicitly labels them as historical, not new P013 field results.

The completion, candidate audit and fixed-six result are SHA-bound. The formal-running label additionally requires the exact approved service invocation and immutable launch command. Training completion is not admission. The six-window table shows physical rear-Cl MAE for P009 and P013; H1 regresses in 6/6 and autoregressive prediction in 5/6. Field metrics remain identical.

Validation: 11 CPU tests passed; embedded JavaScript syntax check passed; dashboard service restarted successfully. The existing local `/api/state` returned verified terminal evidence and the actual formal-running state. Existing Chrome tabs matching the private localhost dashboard were reloaded, without opening additional tabs or performing visual browser QA.

At this observation, formal validation10 endpoint checks have completed and dynamic6 evaluation is active. GPU utilization was 96%, host MemAvailable approximately 111 GiB and MemFree approximately 31 GiB. These are point-in-time measurements, not fixed dashboard constants. No new PPO or FNO-assisted CFD closed loop was started.

Next: finish the unchanged formal suite; independently review the proposed read-only training-objective diagnostic before execution. Preserve scientific failures and do not change admission thresholds.
