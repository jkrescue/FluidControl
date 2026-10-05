# FC-E028 — P018 complete evaluation: original admission FAIL

Independent Control/Evaluation review, 2026-10-05. This is a terminal evidence report, not a new protocol or approval.

## Operational completion and identity

The original formal service `fluid-control-fcp018-posteval-20261005.service`, invocation `ef589f7dbeff4fa0ab064309409971ad`, is retained `loaded`, `active/exited`, `Result=success`, `ExecMainCode=1`, `ExecMainStatus=0`, `MainPID=0`. No running status was mistaken for terminal success.

Output: `artifacts/fcp018_reduced_rate_training_20261005/posteval_fc_p018`.

- Complete receipt SHA: `d4d3f85a79e31d50866bb8dbd23ec90e0453cde39ef6b314e3db80344d33869c`.
- Development gate SHA: `f9fa6f8af0e3c506074505be001075f2d530b65a7c19315383b4f0c965133912`.
- Raw force-window result SHA: `4dff5983e9ddac687432270728c9c5a3ab921420552c48e5079cfa3c86060c61`.
- Resource guard receipt `../formal_supervision_r1/receipt.json` SHA: `e3c64cb430a6016601cd8cc6f51baa0f5a0268df646bc261864734c5c18ea64c`.
- Formal approval SHA: `7b137d4600aaa38458dccb8a1db4f13bd6b9e0140e23dcb9f409702fc2ed275b`; frozen-chain receipt SHA: `6a97e0b4792f2e15156694b91e148267f577ac6e4eaf0f134caad1dae50efe3f`.
- Candidate aerodynamic model/state SHA: `8a89f4774923afa698328e8e65ae337e7e0efefdae0bc9452d6b7a78758fb70d` / `d78d43d43738dd63b9556819e22f6b57c16993affaaff94dc3a29b6250994d6c`.
- Training result SHA: `5ca668810110676dc500f501fb602a3e2fbc1cd9a266ad6ad745484be403f926`; candidate audit SHA: `03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`; training completion SHA: `bce5fb4688e6fc784c309a4e63979a06603270c8b1e5b336572a1092306205bf`.
- Actual official CPU dual-reload receipt SHA: `d36258772d50db8ecdde1bd6437a0abee407df45f778f56830f038215391bdde`. Actual Docker identity/exit evidence is preserved in `FC_P018_CPU_DUAL_RELOAD_EXECUTION_20261005.json`, including two earlier cache failures.

All 18 complete-receipt file hashes, all three stage receipt subtables, and common model/lineage/approval/precision/chain bindings were independently verified. Re-running the frozen original development auditor (`ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412`) on the raw force-window result reproduced the saved gate exactly as a complete JSON dictionary. Numerical source remains `7216214b545fbbd50b2fb5ed866f231039b06b18` plus the reviewed five identity overlays.

Formal memory monitoring contains 1,017 finite samples: minimum MemAvailable **110.045368 GiB**, minimum MemFree **27.460377 GiB**; both remain above 20 GiB. Maximum elapsed sample is 2,189.84 seconds, below the three-hour limit. The guard reports completion with runner exit zero, not scientific admission.

Training was independently audited: 44 pinned HDF files, exact 1,368-window order, 171 actual AdamW steps, eight-window gradient accumulation, LR `1.5625e-7`, finite moments, unchanged frozen flow and two lifting biases. Protocol SHA is `310f0bdf8563a2a70b844a32852791fa1b1dc20278a3098418942e1dab204d2d`. Training host minima were MemAvailable 106.617924 GiB / MemFree 24.190212 GiB. Fixed-six train-panel initial rows exactly match the P015/P009-parent baseline: H1 objective improves 3.527%, AR worsens 0.126%; this is not full-dataset convergence evidence.

## Same-protocol validation10

Paths for comparison are `artifacts/fcp009_joint_force_row_candidate_20261005/posteval_fc_p009`, `artifacts/fcp015_window_accumulation_training_20261005/posteval_fc_p015`, and the P018 output above. Their validation10 stage files were rehashed; all 1,240 `(case,horizon,start)` records match, stride 25 / batch 4.

| Metric | P009 | P015 | P018 |
|---|---:|---:|---:|
| Rear Cl MAE H1 | 0.020124722557 | 0.041348166944 | 0.020234594436 |
| Rear Cl MAE H10 | 0.027990293622 | 0.041597291571 | 0.027894875564 |
| Rear Cl MAE H50 | 0.028401975790 | 0.031932615759 | 0.029492014911 |
| Rear Cl MAE H100 | 0.038740950316 | 0.036986703991 | 0.040269892293 |
| H100 pooled total Cd NRMSE | 0.005586713293 | 0.010489915624 | 0.006050547911 |
| H100 macro total Cd NRMSE | 0.005187365842 | 0.010445417520 | 0.005749087286 |
| H100 strict-start0 delta Cd MAE | 0.019206270576 | 0.020223498344 | 0.019042596221 |

The validation endpoint component passes for all three models. Pooled and macro errors are distinct; strict-start0 action differences are not the pooled endpoint statistic.

## Same-protocol dynamic6 and force-window admission

Dynamic6 has identical 3,858 `(case,horizon,start)` records, stride 1 / batch 8; referenced files were rehashed. Across both validation10 and dynamic6, all four horizon flow summaries (including u/v/p errors and relative L2) remain exactly equal across the three candidates.

| Dynamic6 H100 metric | P009 | P015 | P018 |
|---|---:|---:|---:|
| Pooled total Cd NRMSE | 0.018137902576 | 0.019293347776 | 0.017910925335 |
| Macro total Cd NRMSE | 0.015625640917 | 0.018346799729 | 0.015696077327 |
| Rear Cl MAE | 0.084296267150 | 0.085449025216 | 0.085070086672 |
| Strict-start0 delta Cd MAE | 0.010906636715 | 0.008182644844 | 0.010398447514 |

| Original force-window branch passes | P009 | P015 | P018 |
|---|---:|---:|---:|
| Total Cd | 6/6 | 5/6 | 5/6 |
| Rear Cl fluctuation RMS | 2/6 | 2/6 | 2/6 |
| Rear Cl mean prediction error | 5/6 | 2/6 | 4/6 |
| Joint | 2/6 | 1/6 | 1/6 |

P018 only passes jointly on `full40_dynamic_validation_b01_zero`. Its four rotating-branch rear-Cl fluctuation-RMS prediction errors, ordered b01 minus/plus, b05 minus/plus, are **0.0683246577 / 0.1226618394 / 0.0685902910 / 0.0813667155**. All exceed the unchanged approximately 0.0294 limits. Only the two zero branches pass the RMS criterion. The complete original development-admission verdict is **FAIL**; component passes do not authorize PPO.

## Interpretation and next action

The reduced-rate comparison improves some aggregated force metrics relative to P015, but does not repair the four rotating-window amplitude errors or satisfy complete admission. It does not establish an architecture-capacity limitation or justify a threshold change. No compatible new PPO, frozen-test evaluation, or surrogate-assisted real-CFD success occurred.

Physical `|mean rear Cl| / baseline Cl′RMS <= 0.10` is a real mean-load constraint, distinct from this surrogate's mean prediction-error criterion (`0.025` of same-window zero Cl′RMS). The conditional user review remains no earlier than **15:20 UTC**; this report changes neither criterion. Existing CFD-only PPO b00/b01 already satisfy 0.10 (ratios 0.01993 / 0.03867); compatible open-loop replay fails drag, not mean lift. Hypothetical 0.15/0.20 would not change these three physical verdicts.

Preserve FC-E028 as a negative complete-formal result. FC-P019 objective/statistic gradient diagnostic is being prepared only: not approved or executed. Require a reviewed protocol and separate Lead approval before any new GPU work. The final accepted-surrogate → compatible policy → constrained paired real-CFD goal remains incomplete.
