# FC-P027 independent terminal review — 2026-10-06

Operationally complete; offline diagnostic only, not surrogate or physical closed-loop admission. This review read saved outputs, evidence and source bytes; it did not launch a job, load a model, reread HDF data, change thresholds or modify training.

## Actual execution and provenance

Output root: `artifacts/fcp027_short_horizon_diagnostic_20261006`.

- `result.json` SHA256: `7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366`.
- Actual approval SHA256: `fe218527b6f85daf08999673b9525a2b93144e1722235f5769dc2ea055e567a5`; its parsed content exactly equals the result's source_spec.
- Actual container: `bdb33419469473e7670923c16aa6300af47bdb9040b850e030ecbc0b48403987`.
- Actual image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Saved Docker terminal evidence: exited, Running=false, PID0, ExitCode0, OOMKilled=false; started `2026-10-06T00:36:40.819596971Z`, finished `2026-10-06T00:37:32.25585331Z`.
- Created and terminal evidence agree on container/image/command. The command binds the exact approval SHA, immutable diagnostic source, reviewed GPU guard, allocator .06 and timeout900. Only the diagnostic output bind is writable.
- All **414** source_files byte hashes in the actual approved result spec were independently recomputed and matched. This is source verification, not another candidate archive/HDF audit.
- `evidence/container_terminal.json` SHA256: `ead276231658e80ea5cb5079952ca106f57cd0290b525723d79a8fa632799456`.
- `run.log` SHA256: `a9093bb1de025a503fdfa0bc0ccaeefe04bf345fc0d94c4f4b45243dc832ffe1`.

There are exactly44 distinct family/case pairs: base20, train8 and train16. The log has consecutive origin_complete counts1–44 with exactly the same case set. Every row uses origin51 and62 timestamps. Result counts440 flow transitions and1760 aerodynamic evaluations agree with the reviewed fixed loops (44×10 and44×10×2arms×2conditioning modes). These totals are loop-derived/result declarations, **not independent per-forward runtime instrumentation**.

## Independently recomputed numerical evidence

From the saved 44×10×4 prediction/target arrays, finite residuals, four-force MAE and pooled RMSE were independently recomputed and agree with the reported overall summaries. For each of44 cases and both models, the first H1 and AR predicted force vectors are exactly identical.

| Method | Rear-Cl MAE | Rear-Cl pooled RMSE | Total-Cd MAE |
|---|---:|---:|---:|
| K1 true-field H1 | 0.0265460384012 | 0.0388264060924 | 0.0141079925678 |
| K1 free AR | 0.0386932160032 | 0.0542399828243 | 0.0169093010778 |
| K4 true-field H1 | 0.0265855153702 | 0.0388108517732 | 0.0141228853302 |
| K4 free AR | 0.0385882253708 | 0.0541970594876 | 0.0168978058479 |
| Stored-force persistence | 0.633210405078 | 0.721110763186 | 0.173677577891 |

These MAEs average equal-length ten-point windows over all44 trajectories. Pooled RMSE is not the mean of per-case RMSE. H1 here means true-field conditioning at each preceding endpoint; it is not a single common initial-state rollout.

AR rear-Cl MAE exceeds H1 by45.759% for K1 and45.148% for K4;29/44 cases worsen for each model. K4 AR improves only0.2713% relative to K1 AR overall. At lead10, K1 H1/AR rear-Cl MAE is0.0178495343/0.0413247842; K4 is0.0178959213/0.0411621833.

| Family | K1 H1 / AR rear-Cl MAE | K4 H1 / AR rear-Cl MAE |
|---|---:|---:|
| base20 | 0.032885355 / 0.041104067 | 0.032889900 / 0.041021145 |
| train8 | 0.026125873 / 0.062022449 | 0.026061323 / 0.061876348 |
| train16 | 0.018831975 / 0.024015035 | 0.018967130 / 0.023903014 |

The additional AR error is especially large on train8. Truth-conditioned error remains nonzero, and this comparison does not identify a unique global cause. Persistence is substantially worse, not a substitute accepted predictor.

## Resources

`resource_watch.jsonl` SHA256 `62d831068916d952b314e251cc3249231eec224147f077bc9cd630967296c1e0` contains26 host samples: minimum MemFree **22.5692176819GiB**, MemAvailable **111.0762176514GiB**. The actual completion guard reports25 samples, minimum CUDAfree **22.5717926025GiB**, MemAvailable **111.5352897644GiB**, exit_code0. Observed minima exceed the unchanged20GiB floors; sampled minima are not continuous-time measurements.

## Interpretation, limits and next hypothesis

All44 force-error cases remain in the denominator; terminal endpoint cost is available for36 and unavailable for8 due to recorded action constraints. This cost concerns frame61 only, not feasibility of the entire action sequence. Mixed62 statistics contain52 stored truth points and only10 predicted points, so their smaller errors cannot replace the ten-point prediction errors. Stored interpolated HDF forces/actions are offline descriptive data, not newly established online-causal measurements. No validation/frozen evaluation, optimizer, model save, PPO training or real-CFD control trial is established by this diagnostic.

The initial suggestion to add predicted-state exposure is superseded by source inspection: P026 already trains with equal H1 and H100 free-AR force losses. `scripts/p026_history_objective.py::chunk_force_objective` uses ten weighted chunks of `.5*H1 + .5*AR`; `train_fcp026_history.py` obtains those AR inputs through `train_fcp013_independent_force_fno.py::frozen_flow_states`. That helper explicitly requires frozen flow parameters, runs under no_grad, and repeatedly applies `(current + delta) * mask` with current/next stored actions. Missing predicted-state exposure is therefore not an established defect or novel intervention.

A more distinct provisional hypothesis is to update only the official flow model over H10 while holding terminal K1 aerodynamic weights fixed, using the same44 trajectories and exact1368 original starts, fixed171 optimizer updates, and a predeclared learning rate of1e-5, followed by unchanged full formal evaluation. This is scientifically motivated by the additional AR error, especially on train8, but P027 does not establish flow as the sole bottleneck or guarantee improvement. H1 force error remains substantial.

Before execution, specify the loss explicitly: masked normalized H10 field-state error targets must correspond to predicted endpoints1–10, use the original three-channel fluid-mask denominator, and preserve residual/mask/action ordering. A trainable flow rollout cannot reuse the frozen no_grad helper or detach between its ten transitions; full-H10 gradient/checkpoint equivalence and resource feasibility need bounded engineering checks. Frozen K1 aerodynamic predictions may serve as a non-updating diagnostic; if force loss is instead used to train flow, that is an additional objective choice requiring explicit preregistration, not an implicit part of the field-repair proposal. Preserve frozen aerodynamic tensor hashes and compare both fields and forces under the original H1/10/50/100 and window protocols; better field error alone is insufficient. This paragraph is a hypothesis assessment, not new execution approval.

Keep the physical mean-lift requirement `abs(mean rearCl)/baseline Cl-prime RMS <= 0.10` unchanged. P027 supplies prediction-error evidence, not evidence that a physical10% limit is inappropriate. Relaxing it to15% would not remove the already observed K1/K4 failures on four rotating branches' surrogate fluctuation-prediction criteria. No model is admitted to PPO from this review.
