# FC-P028 matched H10 diagnostic: independent terminal review

This is an offline, train-only diagnostic, not physical closed-loop or surrogate admission evidence. Independent review used saved JSON predictions, metadata, logs and runtime evidence; no HDF/checkpoint payload was reread and no model was executed.

## Actual execution

Unit `fluid-control-fcp028-h10-comparison-20261006`, invocation `74d9e3115791403ab96e55a5d06b8ffd`, was independently observed active/exited, MainPID0, success/status0. Result `artifacts/fcp028_h10_comparison_20261006/result.json` SHA256: `6146ea9276570981cc72c949e3e6fac46737c43aa83c561a37eaa0e54a4ab793`.

Saved container `97e3f2c00a64dac7f7caf6af7f2c55a942473c8ca087683e819921f06e81966f` used official image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`; start `2026-10-06T01:48:19.1449379Z`, finish `2026-10-06T01:49:24.397220546Z`, exit0/noOOM/PID0. The independently verified13-file source closure manifest is `e05d601b3bfef21663e4ea140f60b5cf52efec25c562af7bbc316ffb1963a2ae`, diagnostic `face46e3fe2239bbff07e365132607d4e4de8f7b79f120e4aae8c166b7152919`, role loader `538ee11626756ec7f7c7d251437b48db4f53b427b81ab733f7e6ffb126216627`.

All44 unique cases use origin51, H10 and recorded actions, with20 base,8 train8 and16 train16 cases. For every case, old P027 K1/K4 H1/AR and persistence prediction arrays and targets reproduce exactly. P028 H1 force predictions equal within-run K1 H1 exactly in all44 cases, as expected from the byte-identical frozen aerodynamic model. This is not a claim that P028 field predictions are unchanged.

Reported calls are440 parent and440 P028 AR flow transitions,440 parent and440 P028 truth-conditioned field forwards (1760 total flow calls),1760 parent and880 P028 aerodynamic evaluations. These counts agree with44×10 and the reviewed loops; saved results are not an independent hardware call counter.

## Recomputed force and field results

The following force metrics were independently recomputed from saved prediction minus target arrays over44×10 points. RearCl RMSE is pooled; total Cd is frontCd plus rearCd.

| Arm | RearCl MAE | Pooled rearCl RMSE | Total Cd MAE |
|---|---:|---:|---:|
| K1 H1 | 0.0265460384012 | 0.0388264060924 | 0.0141079925678 |
| K1 AR | 0.0386932160032 | 0.0542399828243 | 0.0169093010778 |
| K4 AR | 0.0385882253708 | 0.0541970594876 | 0.0168978058479 |
| P028 K1 AR | 0.0395074439053 | 0.0556543162316 | 0.0172835624353 |
| Persistence | 0.633210405078 | 0.721110763186 | 0.173677577891 |

P028 worsens overall AR rearCl MAE by about2.10%, pooled rearCl RMSE by about2.61%, and totalCd MAE by about2.21% relative to K1.24/44 individual cases improve rearCl MAE; this does not outweigh the aggregate deterioration.

Field values below are the arithmetic mean of saved per-case normalized field RMSE, **not pooled field RMSE**. Full field tensors are not saved here, so this review recomputed aggregation, not underlying pixel errors.

| Conditioning | Parent mean-case field RMSE | P028 mean-case field RMSE |
|---|---:|---:|
| H1 | 0.00848216044886 | 0.00735674953003 |
| AR | 0.0389092432081 | 0.0335438993167 |

Thus AR mean-case field RMSE improves about13.79% while force errors worsen overall. Improved aggregate state fidelity does not guarantee improved outputs of the frozen aerodynamic model.

Matched family rearCl AR MAE (K1→P028): base20 `0.0411040670247→0.0441859551957` worsens; train8 `0.0620224494720→0.0602103002602` improves; train16 `0.0240150354919→0.0233078766149` improves. Train8 Cd MAE nevertheless worsens `0.0214851655066→0.0242984920740`.

Canonical phase rearCl AR MAE: b00 (15cases) `0.0397201270424→0.0424708887376`; b02 (15) `0.0262181829933→0.0239470018470`; b04 (7) `0.0554499929347→0.0593527271816`; b06 (7) `0.0464681290090→0.0466557261135`. These phase aggregates combine families; they are not balanced per-family phase estimates.

## Resources and scientific limits

Host watcher33 samples: minimum MemFree23.11119842529297GiB and MemAvailable110.73434829711914GiB. Outer guard32 samples: minimum CUDAfree23.110191345214844GiB and MemAvailable110.74366760253906GiB, exit0. Recorded minima satisfy the unchanged20GiB floor.

This experiment demonstrates a tradeoff, not a successful repair: field rollout error decreases but frozen-model force accuracy does not improve overall. It does not establish flow as the sole cause, nor justify choosing a favorable family/phase after observing results. Original formal H100/window criteria remain decisive; mixed62 metrics with52 truth points are not admission evidence. Physical mean-lift10% and all other thresholds remain unchanged. No PPO, optimizer, model save, validation or frozen-test access occurred in this diagnostic. The next required test is the separately approved unchanged original formal evaluation, not a new parameter sweep inferred from this result.
