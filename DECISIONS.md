# DECISIONS

## 2026-10-07 — Explain actual methods and distinguish training from deployment

At 12:02 UTC, expand the existing technical solution with source-grounded explanations instead of a parameter glossary alone. Document the official FNO layer dimensions, frozen-flow versus trained-force networks, B's 256 windows and 32 optimizer updates, and E082 PPO's 32768 environment transitions and 512 optimizer updates. Explain 69 observations in physical terms, 24 actual reset states and sequential DummyVecEnv execution. Distinguish HydroGym surrogate training from frozen-policy real-OpenFOAM deployment: the successful deployment does not run online FNO/MPC or continue PPO training. Include a five-action MPC worked example with an exact offline copy of its existing 10-cycle JSON. Network parameter and forward-call totals are source-derived counts, not a new profiler/checkpoint experiment. Preserve all scientific conclusions and original deadlines; no new experiment or promotion is authorized by this documentation decision.

## 2026-10-07 — Add the top-level implementation framework before detailed modules

At 11:29 UTC, the user's requested technical solution is organized as blueprint → implementation work packages → staged acceptance → detailed data/model/control appendices. Preserve the existing B/E082/E114 chain and all scientific failures. `docs/TECHNICAL_BLUEPRINT_20261007.md` defines WP0–WP5 inputs, work, outputs, ownership and evidence handoffs; three linked technical-design documents explain existing APIs and proposed follow-on work. Integrate them in one offline HTML, linked from the results report. This is documentation only, not a new experiment, model promotion, deadline extension or permission to restart old queues. Any later experimental action still needs an explicit hypothesis, frozen protocol and resource budget.

## 2026-10-07 — Resume documentation only under the unchanged hard deadlines

At `2026-10-07T10:47Z`, resume work only to produce the complete Chinese illustrated report and to organize already-saved error decompositions. Keep the scientific-test deadline `2026-10-07T12:20:45Z` and archive deadline `2026-10-07T12:50:45Z`. Retain B, all physical and prediction thresholds, the existing experiment history, and the disabled `training-evaluation-watchdog.timer`. Do not infer a new result from a figure, restart old queues, or authorize training/PPO/CFD through this documentation decision. The prior closeout report remains a valid historical version; the illustrated edition adds explanations and source-labelled graphics only.

## 2026-10-07 — One representative256 train-panel fit; retain B pending fixed evaluation

Authorize one bounded training-only fit from retained B on the predeclared 256 real H1 points, using the exact reviewed producer, panel adapter and strict new consumer. Permit 0–300 closures because an already-fitted initial point is a legitimate zero-update terminal case; saving a candidate is not promotion. The actual approval SHA is `42e8c15d4835ea6696ecb12271246d9babba16b73f869b4802b57bbc8dabc38c`, and the actual invocation is `6bc6aca4e1bd48d5938b273d319fc716`. Keep the original prediction criteria and retained B unchanged. This decision does not authorize development inference, PPO, CFD, threshold relaxation, or any claim that the basic E114 closed loop has newly improved.

## 2026-10-07 — One B04 raw late-state trajectory only; conversion and training remain separate

Following independent review of the fixed801-point action contract and thin solver entry, authorize one train-only b04 raw OpenFOAM trajectory from120→200. Keep the exact repeated PRBS table, bound t120 backward restart/mesh, Re100/L/D5,16000 steps,8GiB/noSwap resource limits and exclusive output. The executed approval is [P064_B04_LONG_EXCITATION_RAW_CFD_APPROVAL_20261007.json](docs/P064_B04_LONG_EXCITATION_RAW_CFD_APPROVAL_20261007.json), SHA `f2bfcd8f6d1526af266946448ca452d3c271047466d77a9e439d843081338536`; actual invocation is `e2fa5d4e5fa4465a90a2dfbbdd402fc8`.

This decision does not authorize VTK, Curator/HDF conversion, normalization refit, model training, PPO or CFD control. Raw success must be independently checked before a separate case-local801-frame Curator authorization. The intended hypothesis is late feedback-like state coverage, but an eventual B comparison also changes profile/phase/action program/visited states and is not a pure causal decomposition. Basic B closed-loop delivery and E114 continuous-operation PASS remain unaffected.

## 2026-10-07 — Accept E114 same-case continuous reliability; keep prediction work separate

Accept [E114 independent review](docs/P064_B_CONTINUATION_328_408_TERMINAL_REVIEW_20261007.md), SHA `09e89416e4dc59494fdab362f7316b213762850c742a752a91a3a9b3856fd2e5`: after exact one-interval restart recovery, the frozen retained-B controller continued the same two E109 branches from328→408 for800 additional feedback cycles. The entire new80-D/U tail and each fixed20-D/U block, plus joined160/140-D/U windows, pass the unchanged2% drag/1.05 RMS/10% bias gates. Accept this as stronger same-condition continuous-operation evidence for the basic delivered chain. Do not call it an independent physical regime, surrogate prediction admission or proof of general control robustness; do not require online FNO/MPC as a new basic-case gate.

Full surrogate prediction quality remains incomplete. The B04 late-state coverage contract is accepted only as preparation: before any real data generation, a thin executable solver path, resources, source hashes and801-frame artifact protocol require separate review and approval; Curator conversion and model training remain separately unauthorized. Do not repeat additional B duration extensions in place of addressing prediction quality.

## 2026-10-07 — Extend the retained B case only after an exact restart replay

Accept the independently checked `327.9→328.0` recovery replay as technical evidence that E109's branch-specific backward state, current 69-vector and double-precision limiter state can be resumed by the existing controller/solver path. On that basis, execute one predeclared same-case continuation from `328→408` using the frozen B policy and unchanged 2% drag, 1.05 rear-lift RMS ratio and 10% bias criteria. Report the entire new80-D/U tail, every consecutive20-D/U block and the joined160/140-D/U summaries; do not discard the first new block as warm-up or change a window after seeing results. Preserve the original E109 six windows and source trees. This longer run is not a new condition, surrogate admission, training or policy selection. Full FNO prediction quality remains incomplete; online FNO/MPC is optional follow-on work rather than a newly imposed gate for the already delivered basic case.

## 2026-10-07 — Pressure sidecars accepted; one fixed pressure-aux H run only

Accept the mechanical sidecar conversion and its [independent review](docs/P064_FORCE_COMPONENT_SIDECARS_INDEPENDENT_REVIEW_20261007.md): all45 train-only trajectories/20,493 frames retain actual pressure, viscous and raw-total labels while the original HDF total, normalization, split and mix remain unchanged. This does not itself improve or admit a model. The one official-FNO out11 pressure-aux H run preserved the original7 outputs/B objective and used fixed lambda `.1`; R1 path failure had0 training and the path-only R2 completed correctly. Accept [R2 engineering review](docs/P064_TOTAL_PRESSURE_AUX_H_TERMINAL_REVIEW_20261007.md), but reject promotion: fixed-six H1 improves1.6023% while continuousAR100 worsens0.4026%, so the predeclared joint retention rule is false. Do not run80-endpoint dev merely to seek a favorable metric; its performance is unmeasured, not labelled FAIL. Do not retune lambda, sweep, or automatically launch PPO/CFD. Existing B remains the delivered default closed loop and unchanged gates still govern any later candidate decision.

## 2026-10-07 — E113 pressure-only coarse recovery is diagnostic evidence, not a model gate

Accept [E113 independent review](docs/P064_COARSE_ROI_FORCE_RECOVERABILITY_INDEPENDENT_REVIEW_20261007.md): the fixed27-point CPU readout and all saved-row arithmetic are reproducible, but the offset-ring pressure-only proxy has material rear-force and action-response errors. Do not treat it as exact surface traction, train a new proxy from these27 points, omit viscous force, or use its metrics as surrogate admission. No model, optimizer or CFD ran. Keep B and the verified real-CFD feedback delivery; full force-prediction quality remains incomplete. Any follow-up must first reuse existing near-wall velocity/traction or force-readout code and test one falsifiable representation hypothesis without a hyperparameter or architecture sweep.

## 2026-10-07 — E112 retention FAIL; preserve B and avoid unnecessary evaluation

Accept the [E112 terminal review](docs/P064_Y_REFLECTION_PAIRED_TERMINAL_REVIEW_20261007.md): the single paired-reflection run completed correctly, but both unchanged fixed-six objectives worsen B (H1+4.1991%, continuousAR100+1.4529%). The original AND selection condition is already false; do not launch80-endpoint dev merely to search for a favorable result. Its metrics are unmeasured, not retroactively labelled FAIL. No adoption, automatic continuation, CPUload/PPO/CFD, significance claim or change to original physical gates. The existing real-CFD delivery remains usable and B remains default. Next data diagnosis is preparation only, not an authorized scientific run.

## 2026-10-07 — One y-reflection paired-loss candidate, not a new physical truth set

Following the [bounded physical precheck](docs/P064_Y_REFLECTION_PHYSICAL_CPU_PRECHECK_REVIEW_20261007.md) and [CPU implementation review](docs/P064_Y_REFLECTION_CPU_IMPLEMENTATION_REVIEW_20261007.md), Lead authorized one fixed256-original/512-branch/32-update run. Actual launch is FC-E112, invocation `766ad5ca993d483bb6a42d0f2fc09bd8`, approval SHA `6e5ca18a42a1ad4410260fb9df4f14b457907ff2b5bceadfcfb0f3d621bb81e6`. The only scientific intervention is physical-y-reflection pairing at equal weights; it is augmentation, not new CFD truth or proof of discrete-grid/FNO equivariance. Keep original parent, schedule, H1/AR weights, normalization, frozen scope, final-only checkpoint and original diagnostic windows. Prespecified fixed-development plus fixed-six comparison remains necessary before any promotion; no loss-only claim or automatic new control experiment. B stays default, all earlier failures and original2%/1.05/10% physical gates remain. This research does not delay or invalidate the already demonstrated real-CFD feedback delivery.

## 2026-10-07 — E111 physical PASS does not replace B or erase prediction FAIL

Accept [E111 independent review](docs/P064_ABSOLUTE64_B01_CFD_TERMINAL_REVIEW_20261007.md), SHA `17d9328babbbbbe23f1c4bd830e7347626e1cdd055087bbec7ffac0dbecb4bbe`: all six original physical windows pass. Keep B default; absolute64 is a verified single-condition exploratory policy. Its primary drag/RMS/bias are better in this matched deterministic run, but this is not overall dominance: some early/full-window biases worsen and squared-action mean.25748 exceeds B.24045. No statistical significance, physical energy saving or independent generalization claim. Do not lower the10% bias gate or revise predictionselectionFAIL. No automatic new training/CFD is authorized by this terminal result.

## 2026-10-07 — E109 future-time physical validation accepted without changing gates

Accept the [E109 raw report](docs/P064_B_FUTURE_TIME_CFD_TERMINAL_REVIEW_20261007.md): fixed B zero228→248 then paired248→328 completed, all six original physical windows PASS. Primary drag reduction3.9948739165%, RMS ratio0.8165882753, bias2.8533042395%. Do not loosen bias thresholds to obtain a pass. This is additional deterministic-time evidence in the same physical setup, not a new independent condition, prediction-quality admission or authorization to convert/train. Absolute64 b01 remains a separately approved running exploration in the timestamped state; no outcome inferred. Retain B default and all earlier failures.

## 2026-10-07 — Absolute64 is mixed evidence, original selection FAIL; retain B

Accept the [absolute64 fixed-development independent review](docs/P064_ABSOLUTE64_DEVELOPMENT_REVIEW_20261007.md), SHA `d65af40479613570c935042c866a1dc2639b4c998b9376dcd62f8d6c724dcb32`. Update32 exactly reproduced historical B;64 updates improve both pooledH1 force MAEs and slightly improve pooledH5, but both fixed-six H1 and continuousAR100 retention regress. Apply the original rule unchanged: no automatic promotion or replacement of B, no claim of significance or overall prediction success. Preserve engineering R1guard failure and all historical scientific failures.

That separately approved same-budget canonicalH5 PPO completed as E110 and passed engineering audit, not prediction admission. Lead subsequently conditionally approved one fixed absolute64 b01 CFD; actual policy/Vec/result/review bindings and final metadata checks passed, and invocation `6fa0baeb90034371b3b49f3d8d51192a` launched. FixedB new-time CFD E109 is also running in the timestamped current-state snapshot. Neither is a newRe or independent generalization claim. User priority is timely real closed-loop exploration: complete surrogate prediction admission is not a prerequisite for these explicitly approved runs. B remains default; selectionFAIL and all historical failures remain. Original physical2%/1.05/10% criteria are unchanged. No automatic extension, ratio sweep, checkpoint selection or retry.

## 2026-10-07 — E106 engineering complete, no residual promotion; bounded absolute follow-up conditional

Accept [R3 independent engineering review](docs/P064_TEMPORAL_FORCE_DELTA_PROBE_TERMINAL_REVIEW_20261007.md), not a scientific candidate: one train-only window, each arm one update, no model saved or dev evaluation. Absolute total improves3.58%; zero-head residual total improves only.023% and remains far worse, H1 slightly worsens. This does not establish convergence, generalization, or that32updates are insufficient. Preserve R1/R2 failures and do not infer a container root cause from isolated migration checks. Lead conditionally approves preparing the original B absolute64-update follow-up, pending original runtime/spec verification; no execution is recorded here. No residual promotion, no automatic continuation/weight scan, no change to physical standards or retained B/G control evidence. P022 current-force input is not reopened by this distinct output-parameterization probe.

## 2026-10-07 — E105 limited train-fit improvement; prepare no-save temporal-delta probe

Accept [E105 independent audit](docs/P064_K1BG_B00_TRAIN_FIT_TERMINAL_REVIEW_20261007.md), not admission: B/G improve true-state pooled force fit on40 fixed b00 training points versus K1; G improves rearCl/totalCd MAE1.46%/8.08% versusB, but material residual errors remain. These points do not establish convergence/generalization. Retain B default, G's verified physical single condition, original prediction failures and physical thresholds.

Next is only [temporal force-delta preparation](docs/P064_TEMPORAL_FORCE_DELTA_PLAN_20261007.md): causal current-force skip plus learned temporal increment and explicit initialization/control comparison, not P022's already unsuccessful additional current-force input planes. This ledger authorizes no GPU probe, saved candidate or training. A future single-window no-save gradient/update/resource check is engineering evidence, not scientific admission; no automatic extension. Keep fixed development/retention rules and distinguish persistence-prior benefit from learning.

## 2026-10-07 — E auxiliary candidate not promoted after fixed development

[E097 independent evaluation](docs/P064_RESPONSE_AUX_E_DEVELOPMENT_REVIEW_20261007.md) verifies both principal pooled H1 force MAEs and both fixed-six retention objectives slightly worsen versus B. Lead rejects E promotion under the predeclared rule, retainsB and existing validated policies, and does not label the small differences significant. No continuation ofE, coefficient/data-ratio sweep or automatic PPO/CFD is authorized. The next single-factor H1-only training-objective comparison is preparation only, not execution. E095 basic closed-loop reproduction remains complete; original prediction failures and overall unfinished scope remain unchanged.

## 2026-10-07 — E095 basic closed-loop entry reproduced; retain broader accuracy gaps

Accept [E095 guarded-entry reproduction](docs/CANONICAL_B01_REPRODUCTION_TERMINAL_REVIEW_20261007.md) as completed engineering delivery:800 real feedback cycles, exact historical E085 actions/observations, all six unchanged physical windows PASS. It is not new holdout evidence or an improvement to the frozen model. Keep original10% bias, early/seed failures and full surrogate FAIL; no automatic training, CFD or ratio search follows. Separately approved auxiliary E training is now running under invocation618fcaf1d72742069d31a37393523349; initialization is not evidence of completed updates. The basic demo and the unresolved overall prediction-quality goal are separate; optional online FNO/MPC must not become a new mandatory prerequisite.

## 2026-10-07 — E094 closes the narrow precision question; prioritize safe canonical reproduction

The independently verified [paired precision diagnostic](docs/P064_FIRST_STEP_PRECISION_TERMINAL_REVIEW_20261007.md) exactly reproduced E073 high/TF32 first-step outputs, then found the selected local response reversals also under highest/noTF32. Do not explain them away as TF32 alone, extrapolate six cases into global causality, or launch another precision probe automatically.

Keep the successful bounded canonical real-CFD results and original failure records. D failed its prospective joint-H1 choice rule and is not promoted; C50/H25 rejections and full FNO precision FAIL remain. The immediate priority is a safe, reproducible existing canonical case, not indefinite model experimentation. Control-related force response and the original prediction-accuracy objective remain unresolved. No thresholds are relaxed and overall completion is not claimed; online FNO/MPC is optional rather than an invented prerequisite. Historical decisions below remain evidence of their own dates.

## 2026-10-07 — D25 not promoted after its predeclared fixed-development comparison

Lead rejects D promotion, retaining B and the demonstrated canonical controllers. [E092 independent review](docs/P064_B00_B02_COVERAGE_D_DEVELOPMENT_REVIEW_20261007.md), SHAee4258a9733074a8f25912298ab58d5814ef8b8176671a9526bd25d10fc37c58: pooled H1 rearCl MAE .1389983166→.1394301741 worsens while totalCd .03806537390→.03782491013 improves slightly. Both must improve; the joint condition fails. H5 and fixed-six H1/AR retention improve slightly and remain reported, not hidden or substituted for the main criterion. Frozen-flow field predictions are identical, not improved. D changes phase, behavior-policy/state-action coverage and start spacing; it is not a phase-only causal test.

No continuation, ratio sweep, new PPO or CFD follows. The next authorized work is read-only examination of existing teacher-forced and saved-array evidence about instantaneous force mapping, phase lag and action response; no new scientific computation is approved. This is neither project termination nor overall completion. Original physical2%/1.05/10%, earlier seed/startup failures, H25/C50 rejection and full surrogate prediction FAIL remain unchanged. The basic official-components/RL/real-CFD-feedback chain exists; online FNO/MPC remains optional.

## 2026-10-07 — Reject C50 promotion after the fixed development comparison

Lead decision after FC-E087/E088: C50 executed its equal-budget32updates/256windows training and fixed16×H5 development evaluation successfully, but both designated pooledH1 force errors worsen versus B, as do both metrics for each b01/b03 phase at all five leads. Flow arrays are exactly equal because flow was frozen. Execution/resource success is not prediction improvement. [Independent development review](docs/P064_CONTROLLED_DATA_DOSE_C_DEVELOPMENT_REVIEW_20261007.md), SHA `e26d6e49186616f3818c33f438055d707e3c66b881640696006e280a6bc4e091`.

C is not promoted to PPO or CFD. Retain B and the two specified-seed validated canonical controllers; no automatic proportion/seed/phase sweep. Existing original physical criteria2%/1.05/10%, early failures, complete prediction FAIL, rejected H25, negative oldseed and R1 engineering failure remain visible. Lead next authorized preparation only: one E082 frozen-policy b02 paired800 long controlled-data profile,106→186/primary126→186; it is not execution authorization and not a claim b02 lacked older short closed-loop data. A prospective later25%-total-controlled mixture across b00/b02 is distinct from another C50 sweep and is not approved training. No extra full-formal C run is implied; existing saved six-window retention evidence is limited. Overall goal remains incomplete.

## 2026-10-07 — Stop adding seed/phase trials; prepare one controlled-data-dose comparison

FC-E083 and FC-E086 independently confirm original primary physical criteria for the two predeclared canonical-coordinate seeds: seed20261007 drag reduction3.9567%/RMS ratio0.816543/bias1.0660%, seed20261006 3.7440%/0.773790/3.6741%. Fixed same-policy b01 FC-E085 also passes all six windows. Evidence: [E086 terminal review](docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_CFD_TERMINAL_REVIEW_20261007.md), SHA `24e9a5ad02b0b36b34910ebe32e72de4c8cc5692f6090e4efa0dcad8439ac7fc`. This supports these two specified seeds, not arbitrary-seed robustness. E086 early12.4 and first6.2 mean-bias11.2623%/21.1797% fail10%; first6.2 also exceeds20% sensitivity. E083 early failure and older negative results remain unchanged.

Lead decision: do not add seeds or phases now. Startup/transient behavior remains a later control-research issue, not silently accepted. Preserve original primary2%/1.05/10% and the complete FNO prediction FAIL. Current next work is preparation and review of ONE C50 controlled-data-mix comparison: same K1 parent, fresh optimizer,32updates/256windows/objective/resources, change controlled b00 dose25%→50% under the fixed sampling rule. C128 evenly spaced starts are not a strict superset of B64. No architecture change, new dataset, PPO training or CFD is implied.

C50 is not running or authorized by this document; final source/consumer/config preflight and separate execution approval remain required. Historical “no automatic next experiment” means no unapproved automatic continuation, not cessation of the project. Whole-goal completion is not declared; improving prediction remains required and physical success does not replace it.

## 2026-10-07 — Confirm canonical-coordinate control without weakening either acceptance standard

Lead decision after FC-E083: keep the original primary physical criteria (drag reduction >=2%, rear lift fluctuation RMS ratio <=1.05, mean-bias ratio <=10%). The completed canonical-coordinate b00 run has independently verified primary results 3.9567229236% / 0.8165429747 / 1.0659980225%; the first6.2 D/U mean-bias ratio17.5561% still fails. [Terminal review](docs/P064_B_SYMMETRY_CANONICAL_CFD_TERMINAL_REVIEW_20261007.md), SHA `cf7975dbd02da5dca41ddfafe409b86b3676c49e541dc2026988b02ca9f5408e`. Both the old second-seed negative result and the first-seed successful control remain archived; this is not proof of statistically significant superiority or general robustness.

The project surrogate prediction-admission FAIL is not a blocker for separately approved exploratory physical-feedback experiments. It remains FAIL, not a relaxed or renamed pass. This distinction preserves the overall high-accuracy-surrogate plus constrained-control goal, which is not complete. FNO is used for PPO training; deployed canonical-coordinate CPU policy feedback is not online FNO/MPC.

The next scope is only the predefined canonical-coordinate cross-seed replication (seed20261006, unchanged B/32768/H5/24-reset/reward) and fixed-policy b01 confirmation. Both are preparation only at this decision, requiring their own final execution approval and actual process evidence before being displayed as running. No seed sweep, checkpoint selection, new architecture, threshold change or automatic retry is authorized. Any startup failure must be recorded separately from scientific outcomes; historical FAIL records are retained unchanged.

## 2026-10-06 — Matched initial-policy control supports a bounded learning-contribution claim

The completed FC-E078 comparison changes policy weights only: exact seed20261006 initial tensors versus the trained B policy, with the same b00 restart148, 800 control intervals, reflection projection, single action filter and unchanged identity VecNormalize (normalization disabled). Independent raw review verifies all3200 force-file hashes,1600 solver segments and800 action decisions; six-window recomputation differs by at most4.44e-16. Both paired-zero force series match the trained-policy reference exactly across all16000 rows and columns. Evidence: [independent terminal review](docs/P064_INITIAL_POLICY_CFD_TERMINAL_REVIEW_20261006.md), SHA `8e66c497acfcc3483d760a77469330028bd12f9cd233ec36d445d15557c34d34`.

On the predeclared primary(168,228] window, initial versus trained B gives drag reduction −0.0075572161% versus +3.8952838833%, rear-lift fluctuation RMS ratio1.0008821412 versus0.8156230434, and mean-bias1.6502793569% versus1.1378146878%. The initial policy passes both lift criteria and fails only the original >=2% drag-reduction criterion; trained B passes all three. No threshold was relaxed.

Lead interpretation: 在本初始化下，仅保留相同投影/滤波并不能产生同等收益。This matched seed/phase experiment supports an actual contribution from the trained weights under the retained controller transformation. It does not establish superiority over all simple feedback controllers, success without reflection/filtering, cross-seed robustness, or untouched-phase generalization. The phase is already observed/development data. Preserve B's failed full surrogate prediction gate, the rejected H25 candidate, and all early-window physical failures; this evidence does not complete the entire high-accuracy-model goal or authorize further experiments.

## 2026-10-06 — Reject H25 candidate progression; prioritize the reproducible existing CFD loop

Lead decision after the completed same-six H100 comparison: do not train a new PPO or run new CFD for the bounded H25 flow candidate. Training R2 completed 32 updates/256 windows, but operational completion is not predictive improvement. The original numerical force-window worker, batch1/high-TF32, six HDF identities, times, applied actions and truth arrays match the saved B-parent comparison. All six cases worsen both mean-over100 and H100 velocity and raw-pressure relative L2. Across600 predicted force endpoints, rear-Cl MAE worsens .0623860029→.0830122845 and total-Cd MAE .0207225338→.0467238451; action-minus-zero window Cd error worsens .0156519935→.0201665627. Some lift-RMS errors improve locally, but this does not offset the broad regressions. Field means reported here are means of case-relative errors, not pooled field SSE. Evidence: `artifacts/p064_b_h25_quick_ar_20261006/result.json`, SHA `1b7bd2a2e99f9d02398df4cbcefc2d7dc5a486a02866d9a64856d0db9e9dafe0`; actual invocation `b34a1af84199467bad07b61758b92b49` exited0. Parent B result SHA remains `6195b21e6fc820382d580ae8339b47af3d4fb92598135d0d97c6c732e9516173`; it was not recomputed.

Preserve the existing successful frozen B-compatible projected PPO and its actual b00/b01/b07 physical feedback evidence. The immediate priority is a complete, reproducible case demonstrating the existing real-CFD → official FNO → HydroGym/SB3 PPO training → direct projected-policy OpenFOAM feedback chain, not unbounded architecture or hyperparameter iteration. The deployed policy is direct PPO with project reflection projection and one action limiter, not online FNO/MPC. Prior failures and the rejected H25 candidate remain archived; no automatic retry or further tuning follows this decision.

Do not claim the entire high-accuracy surrogate goal is complete: B's full development prediction gate still fails, and these physical phases were already observed/development data rather than independent untouched tests. Preserve original steady-window drag>=2%, lift-RMS ratio<=1.05 and mean-bias<=10% criteria, including b00/b01 early-window failures. No threshold relaxation is used to manufacture admission or hide this candidate's regression.

## 2026-10-06 — Continue fixed-policy physical validation alongside bounded prediction diagnosis

B-trained projected PPO has completed genuine online OpenFOAM feedback at b00 and b01. Their primary final60 D/U windows meet the unchanged physical criteria: drag reduction >=2%, rear lift fluctuation RMS ratio <=1.05, and absolute mean rear lift / paired-zero fluctuation RMS <=10%. Measured drag reductions are3.8952838833% and3.9275159299%. Both early first6.2 D/U windows still fail10% mean-bias. These are observed/development phases; b00 was used for training. Keep the old successful policy and do not claim meaningful superiority from tiny changes or statistically independent generalization.

The complete B prediction evaluation is also finished, not pending: R3 exit0 but `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`. Only2/6 force windows jointly pass; all four rotating branches fail lift-fluctuation prediction fidelity. Validation H100 force errors show retained tradeoffs and flow weights remain frozen. Evidence: `docs/P064_B_FORMAL_TERMINAL_REVIEW_20261006.md`, SHA `62a6234ed0e08ab70532a5252f34e6c8b203a22c6c6abeea5a67ba2635fc8cd6`; physical reports `docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md` and `docs/P064_B_PROJECTED_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md`.

Lead priority: advance the same fixed B policy to the next initial-phase real-CFD validation under separate execution approval, while preparing a signed H1 600-endpoint batch1 diagnostic in parallel. No model, policy, reward, physical threshold or geometry change is authorized here. The saved-JSON same-six H1/AR comparison has already completed (`docs/P064_B_SAME6_CACHED_H1_AR_REVIEW_20261006.md`, SHA `c7433e7b7bc364ded006d9ae17d1f1afea9465ce461385b57b926e60dc8ba1b7`); its absent signed H1 series and batch/precision differences preclude a causal attribution to autoregressive field error alone.

The project-added2.5% force-statistics prediction error requirement remains an unmet surrogate development standard, not a reason to block every separately approved exploratory closed-loop experiment. This distinction does not lower that requirement or grant surrogate admission. Original physical10% bias remains fixed; early-window15% is only a proposed sensitivity analysis, not an approved replacement or revised acceptance label. No new training is approved, no physical or numerical run is launched by this document, and the overall high-accuracy-surrogate plus constrained-control goal is not declared complete. Historical decisions below are retained unchanged.

## 2026-10-06 — P064 supports a bounded new-policy CFD experiment, not formal surrogate acceptance

Both A/B completed the fixed 256-window / 32-update budget and independent terminal checks. Independent saved-array review of the same opened b01/b03 development panel verified A result `c8b0242658a101120603514e6d2e5076c827c518965c92470810fe9693840fe6` and B result `47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665`. The predeclared descriptive B-versus-A H1 comparison is supported: rear-Cl MAE 0.1561168802 to 0.1389983166 and total-Cd MAE 0.0399521664 to 0.0380653739. H1 remains worse than persistence. H5 Cd improvement versus A is only about 0.096%, and B is worse than K1 on that metric. All velocity/pressure prediction arrays remain identical because flow weights were frozen.

Retained original six-window training diagnostics show a tradeoff, not uniform improvement: B H1 objective rises from 0.003488336884 to 0.003976855262 and AR objective from 0.008805384403 to 0.008946200483. Its true-state rear-lift tail-RMS error improves while AR tail-RMS error worsens. These are saved training diagnostics, not independent tests. Do not hide them or invent a post-hoc pass threshold.

Lead decision: proceed to prepare and separately approve one exploratory B-based fresh-initialized PPO run under the retained 32768-step / H5 / 24-reset / 69-observation / 62-sample reward protocol, then one paired 800-interval real OpenFOAM feedback validation using the same reflection projection and action filter. This directly tests whether the measured local force-prediction change yields control benefit; it does not assume that it will. Bind actual B model and resulting new policy identities before execution. Keep the previous successful K1-trained projected policy unchanged. A new execution still requires a complete, checked source/runtime/input bundle and the 20 GiB UMA reserve.

This is exploratory control evaluation, not formal scientific acceptance of B. The unchanged original formal requirements, historical H100 failure, original physical thresholds, and full goal remain in force. Neither better development error nor a single CFD result may be relabelled robust generalization, net energy savings, physical real-time control, or a completed project.

## 2026-10-06 — Preserve P064 A startup failure; continue only the approved same-source R2

A R1 failed before any training update at its first CUDA transfer and is retained as an engineering failure, not hidden or interpreted as a data hypothesis result. One exact 53-file cache-advice operation was separately approved and completed without global cache clearing or data mutation. The same immutable A source/argv R2 subsequently crossed CUDA initialization and at least 9/32 updates. This temporal recovery is useful operational evidence but does not prove the cache state caused R1; do not change learning rate, schedule, architecture, data, or admission criteria while R2 is live.

Arm B is only prepared: its complete CUDA-hidden dry-run passed, but execution remains unauthorized until A reaches a reviewed terminal state and Lead separately approves B. Both arms must restart from the identical K1 parent with fresh Adam and the same 32-update/256-window budget. Never initialize B from A, extend either arm post hoc, select a best intermediate checkpoint, or write running progress as scientific precision.

## 2026-10-06 — Investigate long controlled-state coverage without discarding successful feedback

FC-E063 independently establishes a concrete remaining failure: controlled-branch
true-initial-state H1 rear-Cl MAE0.186710 and total-Cd MAE0.048653 exceed their
matched persistence values0.088235 and0.017126. This cannot be explained solely
by long autoregressive accumulation. Its zero branch and fixed-action FC-E060
have different populations; do not use their errors as a matched improvement.
Existing P027 includes dynamic and prior direct-PPO training trajectories, so
"no changing-action data" and "no AR exposure" are not supported explanations.

Approve preparation, not scientific execution, of one data-coverage intervention:
inventory the already-saved b00 projected-policy 80D/U CFD and determine the
minimal official Curator/Reader conversion needed for a complete long controlled
training trajectory. Retain current architecture, successful frozen controller,
original physical criteria and all formal failures. Do not generate duplicate CFD.
Before any training approval, specify a matched original-data training control,
fixed optimizer budget, separate force/field evaluation and runtime memory budget.

If b00 is reused for fitting, declare the entire trajectory training data and
its already-opened eight-origin replay a development diagnostic, never a final
test. b01/b03 have also been inspected and cannot become untouched tests.
Any future b07 policy-CFD run must acknowledge that its fixed-action forecast
data were already opened. Do not randomly split adjacent frames. New model
weights require a newly compatible policy and real-CFD validation before replacing
the accepted finite-case controller; no such replacement is authorized here.

## 2026-10-06 — Authorize one fixed b03 physical confirmation, not retuning

Lead authorized and launchedFC-E061 at09:30:16UTC underapproval3ca5531815c48cff59fd1ca0d96e40e2305402435cb2e28d2adb26eeaf9328a6. Predeclaredb03/restart144 is not performance-selected. Preserve same frozen32768policy/projection and all800cycle/sixwindow/resource settings; originalprimary2%/1.05/10% criteria unchanged. No execution outcome is known at this entry. b03fixed-actionH5payloadwasalreadyopened; interpret a future result as an additional physical-policy confirmation, not an untouched finaltest or independent statistical replicate. Any failure remains evidence; do not tune or restart automatically.

## 2026-10-06 — FC-E059 confirms the original physical criteria in a second observed phase

Keep the unchanged projected32768 policy and original2%drag/1.05RMS/10%mean-bias definitions. b01 primary (150,210] independently measures3.92363725% reduction/RMS0.815785507/bias0.027300222, consistent with b00 primary benefit. Do not hide earlyfirst6.2 bias12.78%, equate historically usedb01 with fresh independent generalization, or overwrite K1 H100FAIL. No physical threshold relaxation is needed for these two primary windows. Further claims require their own evidence, not automatic retuning after the newly opened frozen short-horizon test. The actual online CFD feedback loop is complete; broad robustness and physical real-time deployment are not established.

## 2026-10-06 — Short-horizon confirmation is measured evidence, not permission to retune final-test performance

FC-E060 completed fixedsealed10×32×H1–H5 with independent numerical confirmation. H5 velocityrelativeL2 .009541481 and centeredpressure .026827415 withrearClMAE .022116273 quantify useful short-time behavior; no retrospectively invented accuracythreshold orH100override. Preserve fixedcandidate/testopening provenance andallcase outputs. Actual projected b00/b01 primary constrainedclosedloop passes are distinct physicalevidence, not proof of arbitraryaction predictionorallphase robustness. No newtraining, thresholdrelaxation orcheckpointselection is authorized by this result.

## 2026-10-06 — FC-E058 supports fixed cross-phase replication, not broader admission

The explicitreflectionwrapper aroundtheunchanged32768policy achievesb00primary3.89198%dragreduction/.815696centeredRMS/.011385meanbias underoriginalcriteria. Exactpairedzeroreproductionandunchangedphysics supportthematchedinterventioncontrast. Preserve earlyfirsthalf10%biasfailure andslightlyhigherRMSversusunprojectedcontroller; do not claim allwindows orallmodels improved. Next ispredeclaredb01/restart130withsamepolicy/projection/horizon/resources, subjecttoRootexecutionapproval; no thresholdtuning, checkpointselection ornewPPO. Even asecondphasepass isnotfreshindependentfinaltesting orrepair ofK1H100formalFAIL. Operationalchainandonephasephysicalcriteriaareverified; generalizationandformalsurrogateacceptanceremainseparate.

## 2026-10-06 — FC-E055 measured long-window benefits do not resolve mean-lift bias

The completed800-cycle pairedtrial establishes real direct-policy feedback over80D/U. Primary paired drag reduction2.377964895% and centeredliftRMSratio0.795798378 improve physical performance, butmeanbias0.322286722 failsboth10%and20%; inclusivecompanion agrees anddoesnotreplaceprimary. Thus neither blanketfailure ofallcontrolbenefit norcompletephysicalsuccess is accurate. Saturation404/800 and worsenedearlybias versus4096policy remain material. Preserve K1formalFAIL, originalCFD-onlybaseline, and source-bound rawproof. Do not relaxthresholds or automatically escalate PPO budget: fixed24surrogate reward improved only.002769 withmixedcases. Next scientific intervention must explain/test persistentbias or model/reward/observation limitations under a separatelyapproved fixedcomparison; no newrun is authorized here.

## 2026-10-06 — FC-E056 does not justify blanket budget escalation

Actual deterministic comparison of unique4096/32768policies on same24verifiedtrainH5 starts yields macroreturn gain only0.0027690371 (7better/10worse/7equal), trading improveddrag penalty for worsemean-bias/actuation/rate penalties. Keep allcase/component evidence; do not infer convergence, physicalgain or choose favorablecases. H5 versus62sample rewardhistory and omittedhistory in69observations remain possible explanations, not causalproof. No threshold/weight/architecture change follows automatically. Preserveoriginal10%physicalreference and await unchanged actual800cycleFC-E055; diagnostic is inferenceonly, not a new training run.

## 2026-10-06 — FC-E051 operational RL loop complete, physical objective incomplete

唯一终态PPO真实配对CFD124周期已完成并独立重算：全窗减阻+0.4118%，rearCl波动RMS增加10.59%，均值偏置为同窗zero RMS的52.7%。前/后窗同时保留，10%改20%仍全部不满足，故不调整物理标准、不据短窗宣布成功。原80D/U CFD-only成功与K1完整formal失败保持独立；结果SHA `4007493f22de5855cbd0574e0ec006ca715941b8396f4e48af6527dc11e03d47`。

124次请求均+.75，117个实际端点饱和；不将它仅归因于零起点覆盖。69观察缺少完整62点受力奖励历史，短回合自举、奖励稀释、代理偏差及训练/部署探针差异也可能贡献。Lead仅批准新隔离适配器及CPU验证：24固定真实train起点（4原零起点+20base-train frame62），相位环境轮换6起点；同K1/H5/奖励/PPO4096。检验reset分布这一单项干预，不扫权重、不根据结果选阈值，不修改既有正式100步wrapper。真实GPU训练和下一次CFD仍需单独批准。

## 2026-10-06 — 保留长H5汇总失败，批准仅离线恢复全部固定窗口

FC-E049实际124周期完成而末窗汇总失败，原因是legacy reader包含154.2左端点；不重跑CFD、不覆盖失败unit或伪造原result。Lead完整读审恢复源码及7项CPU测试后批准一次离线计算，明确采用预定 `(begin,end]`、无插值/其他删点，原force_metrics不变。新recovered_metrics SHA `1605604dc27f106acd05e6a721f26c4ba24527ac53996d6e65fbc70c601fa2b1`。

全窗减阻−0.6506%、前窗+4.1150%、末窗−5.4164%；不能选前窗作为成功证据。全窗meanCl小是两半异号抵消，lift波动降低不等于减阻目标。保留GPU/Curator前缀工程复现证据与真实负结果分离；不宣称80D/U准入，不自动延长/换权重/放宽阈值或启动新PPO。后续干预需独立明确审批。

## 2026-10-06 — 已启动固定H5的124周期探索，保留原正式评价

Lead已以批准SHA `03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75` 启动同K1/H5、同代价/五候选/动作约束、同初态的12.4D/U配对真实CFD反馈。实际unit/invocation为 `fluid-control-accelerated-long-h5-20261006.service` / `a601eec2da7649b4af6f9354a4deb470`。新变化是经已有帧/十状态验证的采样和GPU实现加速，以及124周期物理时长；不扫权重或预测时域，不重训模型/PPO。

Curator数组一致与GPU十状态动作/排序一致均已有实际证据；GPU使用显式加载后highest/no-TF32，不冒充checkpoint原精度协议。完整12.4、前6.2、末6.2窗口均报告正负结果，不挑选成功子窗。原80D/U物理判据及K1科学FAIL保持；本条是探索执行批准，不是正式准入。终态前不填CSV科学结果。持续MemAvailable、无swap、allocator及所属容器清理保护不放宽。

## 2026-10-06 — 探索性闭环优先，保留原正式评价

依最新用户指示，允许真实CFD每步重观测的短预测反馈不等待全部长AR门槛通过；不得将探索结果改称正式准入。当前无需放宽物理10%偏置：FC-E047的100个候选阶段该罚项均为0，修改它不能解决HOLD。下一项仅批准准备H5代替H2，保持模型、五候选、62点真实历史、代价权重、初态及10周期；独立测试审查后另行批准执行，不扫权重，不把预测收益当CFD收益。

官方 https://nvidia.custhelp.com/app/answers/detail/a_id/5728/kw/x79 明确Spark UMA的cudaMemGetInfo不含可回收内存。批准准备新的隔离GPU推理探针：物理MemAvailable至少20GiB，启动50GiB、运行22GiB提前退出，分配上限及连续记录；禁止全局drop_caches，保留旧协议及其失败。此条不授权完整训练，不代表已有GPU计算。

## 2026-10-06 — P028完整正式失败，不以短时流场改善批准PPO

FC-E041完整收据 `63fd75d4e90176dd94998f2844f2f70cb5a7e357bd59d5362591019ed8655154` 已独立核验。正常退出不等于科学通过：validation10动作差阻力误差 .02739353 超过原 .023；六窗联合1/6，Cd/RMS/mean各2/6，相比K1的5/6、2/6、4/6没有修复。动态端点子检查通过不能覆盖失败。报告 `docs/FC_P028_ORIGINAL_FORMAL_TERMINAL_REVIEW_20261006.md` SHA `ae98f3a63b195aca184ce348d2e1991f88ad2f416766105bb8bd5b788107ee44`。

下一项保留同P009父模型、官方FNO、44训练轨迹、1368窗/171更新和原正式评价，只检验预先固定50/50父模型尺度归一化的流场与受力联合目标。先取得实际父模型尺度及新反向图资源证据，再由Lead分别决定训练；不扫权重，不继承旧PPO，不把冻结受力模型输入梯度当作真实CFD改善。物理平均升力10%标准和代理误差门槛均不修改。本条是结果解释与下一步边界，不是执行批准。

## 2026-10-06 — 不将流场误差改善等同于控制能力改善

P028同起点H10实测全场误差改善13.79%，但后Cl MAE恶化2.10%、总Cd MAE恶化2.21%。保留全部负结果和原P027可复现对照。完成已批准的原完整正式评估；不以较好的单个工况或field指标批准PPO，不通过降低代理误差阈值获得通过。

若原完整评估仍失败，下一假设只改变flow训练目标：在官方现有FNO和固定K1受力模型上联合考虑流场与气动力误差，保持数据、父模型、训练预算及正式评价不变。先形成独立审查设计；新增损失属于项目训练代码，不冒充官方API。当前未批准P029训练、架构变化或新CFD数据。物理平均升力10%与升力波动/减阻要求不变；已有CFD-only控制通过这些物理标准，因此目前无证据需要放宽它们。

## 2026-10-06 — P028工程集成不等于训练批准

接受显式P028更新流场/冻结K1受力的加载类型，旧P026身份检查不放宽。90项canonical CPU回归通过；资源检查使用独立418文件副本，仍加载原父模型且不保存候选。下一步优先真实尺寸无更新资源检查；单次反向无Adam分配，不能当完整训练容量证明。启动前物理空闲不足额外30GiB时只考虑经核验项目文件的可回收缓存，不降低20GiB要求、不全局清缓存。

## 2026-10-06 — P027完成后优先分析预测状态下的受力误差

后续源码复核已确认：P026本身已有等权H1/H100 AR训练，故不执行“首次增加预测状态暴露”这一重复方案。当前转向设计固定K1受力模型、仅更新官方流场FNO的H10多步训练。此为针对额外AR误差的可检验假设，不能宣称流场是唯一问题；原H1误差仍在。

FC-E039独立核验表明真实流场条件下已有受力误差，连续预测使后Cl整体MAE增加约45%；增加历史长度只产生约0.27%的AR MAE变化。下一项实验必须针对已观测的误差来源，并先确认已有训练是否已包含预测状态，不能把重复训练包装成新方法。未批准新的GPU任务，未用训练集H10结果覆盖原正式H100失败。物理10%平均升力要求与代理误差指标均保持原定义；实际CFD只因前者失败时才适用用户允许的15%评估方案。

## 2026-10-06 — FC-P027一次只读诊断执行批准

实现、32项canonical CPU测试、独立来源和资源启动审查完成后，Root批准 `FC_P027_EXECUTION_APPROVAL_20261006.json` 的单次运行。复用原K1/K4候选与44条训练轨迹，比较真实流场条件H1与共享冻结flow的H10受力误差，不创建优化器、候选或新CFD。允许按批准清单只读校验模型与44HDF字节；不授权缓存清理或修改数据。900秒、12GiB容器、GPU0、0.06 allocator，持续保留20GiB物理free/available与CUDA余量。失败时保存证据、先诊断，不盲目重启或放宽限制。下一决策只由实际误差分解及独立复核决定；混合窗口不构成准入。

## 2026-10-06 — K4完整评估失败后不追加历史输入微调

后续执行范围更新：独立K4终态审查已完成；Root完整审阅FC-P027设计后批准隔离CPU实现及小型单元测试。保持offline stored-HDF目标标签，不开展可选全44轨迹raw因果审计，不加载真实HDF/模型、不运行GPU；实现与测试仍需独立复核再集成。此更新不授权训练或新候选。

实际K4 development gate `ce60621723ce364e4f8cdc165268a64ca5fd1a0185b7529bc21c1f91ac8aab9e` 与K1均为joint1/6；四个旋转分支升力波动预测仍失败。保留匹配对照负结果，不以部分端点精度改善批准PPO，也不将实际平均升力10%改成代理误差容限。

先完成独立终态审查与账本同步。仅准备FC-P027短时误差分解设计：固定现有训练轨迹起点、K1/K4、真实流场条件H1与free-AR H10及持久性对照。必须分别报告10个预测端点误差和52真实加10预测的成本误差，后者不可因真值占比高而被用作准入证据。原始受力时间因果性需单独证明；train16原H100窗口缺52点过去受力，不能假称复用原1368窗口。当前未授权GPU执行或新训练。该诊断服务于修复主线代理，不替代H100评估、兼容PPO训练及真实CFD反馈验证。

## 2026-10-05 22:40 UTC — 新K1正式终态不触发物理平均升力15%复议

重新核对现有两项配对真实CFD结果：b00
`physical_result.json` SHA
`3c9587d66482ca98f773a4bccc641895f84bacc930df8009bc590d9ba49e2c2c`，
`|rear mean Cl| / zero rear Cl-prime RMS = 0.01992845687`；b01文件SHA
`da42b6018f35677af8205ce9ba78cc9753ca348abf5f5994f2189554c4eaba36`，
对应比值`0.03866683099`。两者也分别实现4.2212%和4.2502%减阻，
rear-Cl-prime RMS比为0.935645和0.935974；原物理10%平均载荷要求、
RMS不增超过5%及减阻至少2%均已通过。

新增K1代理正式gate SHA
`32cc7cb438bb215847bb6f342e88e1d63c895b21b2f01a25a1c8d2c641499592`
是**代理预测误差**证据，不是新的真实CFD物理结果。其窗口均值误差门槛为
同窗zero RMS的2.5%（约0.0294），b05-minus误差0.0485904、b05-zero
误差0.0310912未通过；同时四个旋转rear-Cl-prime RMS预测误差
0.0683388/0.1222618/0.0682180/0.0813537均未通过，且b05-plus总Cd误差
0.0241096也未通过。因此K1正式FAIL并非“真实CFD只因平均载荷失败”，
不能用物理15%方案覆盖。

决策保持不变：物理准入继续使用
`|rear mean Cl| / zero rear Cl-prime RMS <= 0.10`；代理mean-error与
RMS-error门槛保持独立且不放宽。只有未来真实配对CFD同时满足减阻、
波动和动作要求而**唯一**失败项为物理平均载荷10%时，才条件性复议15%，
且必须并列报告原10%结果。K4目前只有运行中训练、尚无物理结果，不进入
本次判断。

## P026完整对照设计通过Lead审查；仅准备工程实现

采用 `docs/FC_P026_HISTORY_COMPARISON_PLAN_20261005.md` 的单帧K1/四帧K4对照设计：各消费同1368训练窗、171次更新及原优化器，保持原完整评估。原流场模型冻结，受力模型均从P018终态开始；必须分别绑定两个模型来源，不能沿用旧程序把两者都从P009初始化。K4增加288个输入投影系数，其额外容量和小学习率敏感性明确披露。有限预算失败不证明所有短历史方法无效。

目前只批准CPU历史目标函数、测试与无更新资源试验程序的准备，不批准GPU执行或完整训练。实际GPU试验须源码/数据/父模型检查和独立审查。保存重载及所有正式评估调用的历史输入适配必须在完整训练前完成，但不阻塞先测共享目标函数的内存需求。已有13项CPU测试、官方小网格模型验证和六窗真实HDF实跑结果可作为各自范围的证据，不外推全数据精度。

## P025终态决定：结束局部输入系数/统计损失路线

独立终态报告 `docs/FC_P025_TERMINAL_REVIEW_20261005.md` 已完成；Root核验同一实际unit终止及结果SHA `7648df661439f530984504538bfaef3d1a602c1b960fd914fcba2f016fc4e74c`。固定统计监督改善部分波动预测，但未同时改善均值及原始目标，不能准入控制器。遵守实验前约定，停止此路线的进一步尺度/权重扫描；不把0.1473%的局部均值误差退化解释为物理平均升力10%约束太严。

批准下一阶段仅隔离CPU工程：保留既有官方FNO和冻结流场模型，新增可选短历史数据适配器；单帧K1与四帧K4将以相同动作语义、数据、预算进行后续对照。历史只来自当前和过去状态，AR使用已预测状态滚动，不能取未来真实状态。已有44条轨迹先复用，不追加CFD。缺少三帧历史的68个窗口使用明确标识的首帧填充，与1300个完整历史窗口分项报告且不剔除总分母。当前批准不包含GPU运行或训练；正式预算、指标、不可变源码和资源试验需后续独立审查。短历史收益尚未证实，不替换已有模型或已有CFD-only成功基线。

## 2026-10-05 17:32 UTC 一小时复评：当前保留10%，保留条件性15%方案

已到约定时间。Root与独立评价再次核对既有真实CFD配对结果：b00平均后Cl绝对值0.0235684593/基准波动RMS1.1826535021=0.01992845687；b01为0.0453378586/1.1725258436=0.03866683099。两例减阻4.2212%/4.2502%，波动RMS比0.935645/0.935974，已满足原物理要求。它们是CFD-only PPO结果，不是本小时新增成果或代理辅助闭环成功。

证据为`artifacts/direct_cfd/directppo2048_b00_eval80_v1/physical_result.json` SHA `3c9587d66482ca98f773a4bccc641895f84bacc930df8009bc590d9ba49e2c2c`，b01同路径命名文件SHA `da42b6018f35677af8205ce9ba78cc9753ca348abf5f5994f2189554c4eaba36`。Root直接读取两文件并重核SHA；独立评价还确认兼容开环回放的平均升力也通过10%，失败在减阻。

本小时P022有限训练已完成但未满足局部精度改进要求；P023参数隔离CPU工程验证完成、真实训练程序正在准备。没有新增合格代理或代理辅助真实CFD结果。P018仍主要受四个旋转窗口的升力波动预测误差限制。将真实平均载荷要求从10%提高到15%不会改变上述已有物理判定，也不能解决预测误差。因此当前不修改该指标，不把时限到达解释成自动通过。

用户允许适当放宽的授权保留：后续若真实配对CFD结果同时满足减阻、波动与动作要求，仅平均载荷超过10%，再评估15%版本，并列报告原10%结果。15%是项目工程折中而非文献通用标准。此授权不用于放宽代理均值或RMS预测误差。下一步按9b8f8f5计划准备P023两步长、仅新增96系数对照，实施和独立审查并行；不重复无依据训练或改变主线。

## 用户再次授权的一小时复评（2026-10-05 16:32 UTC / 北京时间10月6日00:32）

用户再次要求保留10%继续尝试一小时，仍不能解决时适当放宽。新复评时间为2026-10-05 17:32 UTC（北京时间10月6日01:32）。该安排追加于此前15:21 UTC的结论，不回写历史。当前目标仍未完成；P021真实尺寸资源检查正准备，不冒充训练或控制结果。

10%是本项目的平均横向载荷要求，不是不可修改的物理常数。若到时真实CFD控制的主要失败确为平均升力偏置，优先评估放宽到15%，并列报告10%与15%的结果；20%只做敏感性分析，需收益与平均载荷证据支持后另行决定。这里的15%是待验证的工程折中，不是文献公认标准。不得因一小时时限到达自动把失败改成通过。

当前既有CFD-only PPO两例比值约1.99%/3.87%，已经满足10%；其通过不等于代理辅助闭环成功。当前主要问题仍是代理的升力波动预测，不能把放宽物理平均载荷当成解决预测误差。减阻至少2%、升力波动RMS比不超过1.05、动作限制和双20GiB资源要求均不变。继续P021已评审准备路线，GPU执行仍须既定资源核验；复评时根据新增真实结果决定，不凭计算利用率或工程测试宣称科学进步。

## 2026-10-05 UTC — FC-E031因果输入审计与P021 CPU-only范围

接受独立复核的六窗时间审计：606个端点每圆柱均有唯一raw同时间力；base/train8共200/505 HDF帧含微小未来插值，train16精确端点，无实际未来依赖。六个初始力精确，AR初值基线不变；旧HDF-lag H1基线不得称严格因果。新显式输入sidecar只使用这些精确端点并绑定来源/窗口/时间/归一化，保留原HDF目标。不得把六窗证明外推全数据。

Root CPU工程授权`225d99f`及`docs/FC_P021_CPU_ENGINEERING_APPROVAL_20261005.md`仅允许隔离暂存CPU模块与测试；不允许GPU资源试跑、训练、候选保存、部署或PPO。官方坐标映射证据已记录，仍须测试原6物理列保留、旧坐标6:8移至10:12、新力6:10列为零，以及完整H100因果梯度不在checkpoint边界detach。数据审计不证明条件化有效性，P020/P018失败与数值门槛不变。

## 一小时复评结论：暂不修改物理平均升力限制（2026-10-05 15:21 UTC）

已到用户授权的15:20 UTC复评时间。P018原完整评估已独立核验失败（FC-E028）：Cd预测5/6、升力脉动RMS预测2/6、平均升力预测4/6、联合1/6。四个旋转分支的波动误差均仍超限；修改真实CFD的平均横向载荷限制不能解决这个预测问题。

真实CFD-only PPO的b00/b01平均升力比为0.01993/0.03867，已满足原0.10；兼容open-loop回放失败原因是减阻。假设放宽至0.15/0.20不会改变这三项物理判定。具体来源与原协议比较见`docs/FC_P018_INTERIM_EVIDENCE_20261005.md`和`docs/FC_P018_TERMINAL_REVIEW_20261005.md`。这些结果不是新的代理辅助CFD成功。

因此当前保留物理平均升力0.10、波动比1.05、减阻2%及原动作要求；不是认定0.10是不可修改的物理定律，而是现有失败证据不支持把它作为本轮改进手段。用户允许适当放宽的意图保留：若后续真实配对CFD结果主要受平均横向载荷限制，再依据实际收益/载荷权衡提出明确的新版本，同时保留旧标准结果。不能把这一权限转用于放宽代理预测误差。

P019固定训练面板诊断已经完成：局部多窗口原目标下降方向与连续预测升力RMS误差存在冲突，且各单窗口自身方向并非都冲突，支持进一步研究跨窗口相互作用。该证据不证明AdamW轨迹或泛化原因。下一P020为原目标与对称尾窗统计监督的16步双臂对照，当前仅准备/测试，尚无GPU执行批准或改进结论。

## 用户授权的一小时指标复评（2026-10-05 14:20 UTC / 北京时间22:20）

用户要求：继续尝试一小时；若仍不能解决，可适当放宽“平均升力偏置不超过基准波动10%”的要求。决策时间不早于2026-10-05 15:20 UTC（北京时间23:20），本小时保持原指标和当前P018正式评估不变。

该授权仅针对真实CFD控制验收中的绝对后圆柱平均Cl / 基准Cl′RMS <=0.10，不自动改变代理预测误差门槛、升力脉动RMS比<=1.05、减阻>=2%、动作/内存限制。先区分实际横向平均载荷与代理预测均值误差，以及波形/幅值误差。若主要失败仍是升力波动预测，放宽平均载荷要求不能视为解决该问题。

一小时后基于同初态配对CFD和预测误差证据评估修订幅度；如采用放宽标准，明确登记新版本，旧10%结果仍并列报告，不回写历史PASS，不把验收标准变化称为模型精度改善。新标准下的最终通过仍须真实CFD闭环验证。用户此次授权优先于历史文件中禁止任何门槛修订的笼统约束，但不授权任意扩大其他指标。

## FC-P015 更新协议决定（2026-10-05；执行后索引既有预声明）

预声明位于`docs/FC_P015_WINDOW_ACCUMULATION_PLAN_20261005.md`（66d591e），启动前审批为`docs/FC_P015_EXECUTION_APPROVAL_20261005.json`（b0c326a）。本段在启动后补充索引，不冒充新的事前记录。

P014表明原固定六窗H1/AR目标均退步，并非纯偏移；原1368次preclip梯度均超过clip1，但这不证明唯一原因。拒绝仅做全训练集bias标量校准，因为常数偏移不能修复仍失败的Cl′RMS。选择同P009亲本、同数据顺序/loss/官方架构的8窗梯度平均更新协议，171次更新；不改LR、不加轮数、不选择中间模型。更新次数、Adam moments与累计weight decay随协议共同改变，结论不得归因于单独降低噪声。固定终态仍须原完整formal与真实CFD闭环，P013失败不被覆盖。

本文件于2026-10-04重建。旧决定的记载是事后整理，不伪装成历史预注册。新决定先登记再实验。原始失败与证据不覆盖。

| ID | 决定/状态 | 原因和证据 | 后续检验/改变条件 |
|---|---|---|---|
| D001 | 保持固定Re100、L/D5、后圆柱旋转；accepted | 当前数据和已验证CFD控制均为该场景；不能把用户举例的多U当已有数据 | 当前闭环验证完成后再单独批准跨Re数据设计 |
| D002 | 以整体减阻及升力约束为目标；accepted, reaffirmed | 真实CFD既定>=2%减阻、RMS<=1.05、偏置<=0.10 | 不因候选失败而放松；目标变更需独立记录 |
| D003 | 保留官方二维FNO和官方数据组件；accepted | 当前存在可复现数据/模型/控制接口，未证明结构容量是根因 | 仅在误差分布和受控对照支持时提新架构实验 |
| D004 | 当前优先控制相关幅值/动作响应，不只看field loss；accepted | train16动态端点Cd通过但力窗口失败；加权分支幅值动作方向错误 | 完整后评估验证paired λ0/10是否真正改善 |
| D005 | 拒绝当前λ10 paired-stat干预；rejected by FC-P001 | 同亲本/同数据/同协议下，λ0与λ10均只有2/6 zero分支通过force-window；λ10将四个旋转分支Cl′ RMS误差均值仅降1.68%，个别phase有改善也有退化，仍约为限值的6倍 | 保留两支失败证据；完成FC-P002失败图后，只提交一个Lead批准的FC-P003单因素假设，不启动代理PPO |
| D006 | 不推倒已有CFD-only PPO，不强制从MPC重开项目；accepted | 已有真实反馈减阻约4.2%的有效基线 | 可在代理通过验证后加入有限时域MPC诊断对照；MPC也受代理误差影响，不能绕过精度要求 |
| D007 | 不新增统一加权reward替代既定物理约束；accepted | 权重混合目标可能掩盖升力超限；已有动作与物理验收合同 | MPC/RL比较必须同观测/动作/起点/预算/物理约束，reward改变另做消融 |
| D008 | 跨Re/OOD与主动采样暂列后续，不声称现已具备；deferred | 当前不足是固定Re下动作/时序预测；未知不等于已定位跨Re泛化问题 | 先明确训练参数支持、误差图及不确定性校准，再设计新CFD采样 |
| D009 | 论文价值待实验支持；accepted | CFD-only收益不能归因FNO；b00/b01非独立工况；尚无样本效率/净电能收益证据 | 增加冻结测试、多seed、CFD预算与收敛/稳健性证据后再扩大主张 |
| D010 | 以持久状态和实验台账协调代理；accepted now | 多次流程结束、后处理失败与旧状态导致进展混淆 | 每关键节点更新并独立复核；阶段完成不得改project_goal_complete |
| D011 | 区分港理工相关研究背景、独立实现与严格论文复现；accepted now | `docs/POLYU_ZHAO_2024_SOURCE_AUDIT_20261003.md`记录部分原文参数尚未核实，而旧`docs/PAPER_REPRODUCTION.md`含较具体及较早阶段描述 | 不把旧文档更肯定的说法当证据；建立原文页码/公式对照后才宣称一致。当前整体减阻+升力约束是项目目标，不能冒称论文原奖励 |

## D012 — 补全 canonical 代理验证的数值生成程序

2026-10-05 Asia/Shanghai，accepted before FC-P003/P003B完整后评估。
独立源码和历史审计确认：旧window/dynamic receipt仅有consumer字段合同，
没有历史数值producer。采用2026-10-04已经固定的development误差标准补全，
不降低阈值、不将今日定义追认成历史预注册，也不把共享证据的receipt描述为
独立科学实验。实现须从校验后的原始评估证据重算，而非改名PASS布尔值。
详见`docs/CANONICAL_SURROGATE_PROTOCOL_COMPLETION_20261005.md`。
只授权CPU producer/诊断；PPO及最终真实CFD物理验收要求不变。

## D013 — 先恢复已完成的后评估阶段，技术探针不代替科学实验

2026-10-05 Asia/Shanghai，accepted operational decision。FC-P003B首次后评估在validation10 GPU推理完成后因audit挂载路径合同不一致失败。保留原失败，只允许不可变恢复程序在重算SHA、checkpoint、有限性、计数和数据合同后复用完整推理pair；成对文件缺失、哈希不符或未知故障必须停止，不得为赶进度重跑或改阈值。端点组件PASS不得越过dynamic6、force-window或development gate。

true-state paired-force GPU工程探针只被批准检查固定真实train-only pair上的causal索引、chunked gradient等价性和峰值内存；无optimizer、无权重保存、无validation/frozen数据。v1因原容器身份下mode-600输入不可读而在forward前失败，不对更底层rootless/user-namespace原因作未验证声称。Lead后续明确批准的v2只修复容器UID/挂载可读性并使用新独占输出，数值合同不变；它已通过梯度等价与资源检查，且模型parameters/buffers未变。该技术PASS不是候选代理、PPO准入或闭环成果；正式损失实验仍须等FC-P003B完整结果并单独批准。

## D014 — 拒绝FC-P003B并仅批准FC-P003C工程实现

2026-10-05 Asia/Shanghai。FC-P003B在与FC-P003相同的亲本、regular顺序、λ10、update数、epoch数和评价协议下，只将static16配对监督内容改为dynamic8×2。四个旋转分支rear Cl′ RMS误差较FC-P003均值仅降2.94%，force-window仍2/6、development FAIL，因此该干预不足以支持代理PPO。true-state H1的旋转rear-Cl MAE仍为0.156--0.192，说明失败不能只归因于自回归累积。

Lead以`docs/FC-P003C_APPROVAL.md`批准单因素下一步的工程实现和CPU测试：保持所有数据、顺序、模型、归一化、超参数及验收不变，只将配对损失替换为true-state每端点action-minus-zero四力误差。固定`w=[1,1,4,1]/7`和λ10不意味新旧损失梯度等强；必须报告通道损失/梯度贡献，不得看到validation结果后再改权重。完整GPU训练需另行批准，原准入门槛不变。

## D015 — FC-P003C若完整失败，先分解训练拟合与泛化缺口

2026-10-05 Asia/Shanghai，conditional，只在FC-P003C完整后评估仍失败后执行。epoch1 `training_history.json`显示true-state paired的rear-Cd/rear-Cl加权贡献为0.0589506/0.00223312（约26.40倍），16个组合更新的pre-clip norm均大于1（最小11.7792）。源码证明这16个组合更新占用1368个regular batch中的预定位置，每epoch总optimizer step仍为1368，不是1384。这些是诊断现象，不足以把clipping定为因果。

若触发本决策，优先用相同true-state H1协议比较现有train8与dynamic6数据，并按动作/相位分组，区分训练拟合失败与泛化失败。不盲目增大λ或网络，不改准入门槛，不授权新readout拟合或GPU实验。

补充的CPU-only动作端点覆盖审计（receipt SHA `eb19fb2aff2628e2383707b23376201994dd17f29b8651c156d3310d1d1e4074`）在train8的1600个transition与dynamic6的1200个transition之间找到1004个精确局部动作特征匹配，归一化最近距离最大为0.07511075。该审计只读`omega/time`，不支持“明显的局部转速幅值/步长覆盖缺口”解释，但未测state/phase/history/force联合覆盖，不能证明泛化，也不得把ω距离单独当作模型失败的因果。审计产生时FC-P003C仍在epoch2训练，该条本身不触发失败后实验。

时域索引另须明确：train8 action HDF每条201帧，zero HDF每条801帧；paired DataPipe只额外监督索引0–100中的targets 1–100，而常规train8 H100/stride2的408个训练窗口仍覆盖到target200。D015已经在FC-P003C完整gate失败后按固定合同执行，分成`train_paired_window` targets1–100（800 delta）、`train_late_window` targets100–200（808 delta）和`validation_late_window` targets100–200（606 absolute/404 delta）三个面板；target100重叠，不是独立重复证据。targets101–200未获额外paired监督，但不是“未训练”。

结果拒绝了“只有验证分布泛化失败”这一单一解释：rear-Cl delta MAE在额外paired监督窗口、同轨迹late窗口、validation late窗口分别为0.11880/0.10753/0.17185；即使训练前缀也没有被准确拟合，同时validation相对train late仍更差。front-Cd delta MAE仅0.00069/0.00128/0.00133，误差明显集中在后柱受力，尤其rear-Cl。下一项若批准应首先针对train-only rear-Cl拟合/优化暴露做单因素检验，而不是扩大网络或降低门槛；本诊断本身不授权训练、PPO或frozen访问。

## D016 — 拒绝单纯增加delta-only曝光，仅批准absolute-paired CPU实现

2026-10-05 Asia/Shanghai。固定C亲本的128/64有界校准使paired/late rear-Cl delta MAE仅下降3.02%/2.57%，但absolute MAE与u/v/p场误差均退化；zero rear-Cl MAE约恶化3–4倍。由于action与zero共享的force偏差会在delta loss中严格抵消，Lead批准一个单因素CPU实现：将paired项替换为`0.5 * (weighted action absolute MSE + weighted zero absolute MSE)`，其余亲本、数据、regular loss、预算、λ、权重、学习率、seed、clip及门槛不变。若未来获批执行，必须分别报告action、zero、delta四力与paired/late场误差；zero改善而action/delta无一致改善，或force改善伴随field退化，均反证该机制。当前commit `5cb65bb`仅为CPU-tested代码，不授权GPU、正式后评估或PPO。

该实验已按固定合同执行并触发反证条件：zero rear-Cl误差显著恶化，delta改善很小，且所有field通道退化，因此absolute监督分支也被拒绝，不做正式后评估。现有64个paired update中，四通道加权贡献占比约为front-Cd 2.83%、front-Cl 0.31%、rear-Cd 93.01%、rear-Cl 3.85%，64个pre-clip norm全部大于1（中位26.46、均值30.96）。action与zero的有符号bias变化在每个通道上数值接近，说明存在共同移动的相关模式；但这与贡献/裁剪统计都不是因果证明，不授权继续增大λ、曝光量或模型。

## D017 — 固定特征读出显示fit窗可读但病态，先做train-only ridge稳定性诊断

2026-10-05 Asia/Shanghai。v1在默认TF32下因为先空间平均再做仿射与原逐点仿射再平均的非结合数值差异而fail-closed；数值probe中原逐点FP32顺序可bitwise复现，禁用TF32/highest FP32后换序差降至2.38e-7。v2保持原`2e-5`容差，在该独立数值协议下完成。CPU从cache重求的1600行float64 least-squares系数逐值相同，所有action/unique-zero/delta和逐pair指标精确复现。prefix四通道action MAE降84.6%–93.8%，但矩阵条件数为`1.9016e5`、系数L2为`717.87`，late rear-Cd MAE增加10.15%且rear-Cl仍有`0.08337`。因此不将fit窗低残差解释为已修复优化、不生成部署checkpoint、不启动PPO。

Lead只批准下一个CPU-cache诊断：在prefix做4-fold leave-one-phase-out，固定`alpha={0,1e-8,1e-6,1e-4,1e-2,1}`，每fold仅用fold-train标准化及对称zero加权，以归一化四通道等权mean-MSE选alpha（并列选较大alpha）；选定后在全prefix重拟，只查看一次late。不为每个alpha扫描late，late仍是同train轨迹时间检查而非独立验证。该诊断只检验高方差/病态假设，不允许把失败直接归因于覆盖不足。

该cache诊断已完成（result SHA `dbeee783…50e0f`）：prefix四相留出选定`alpha=1e-6`，aggregate normalized MSE从alpha0的`0.0064395`降至`0.00359875`（-44.1%）。选定后唯一一次late检查的physical rear-Cd/rear-Cl action MAE为`0.03876/0.06274`，优于OLS的`0.05602/0.08337`，但仍不足以支持控制准入。独立CPU复算的六个alpha score最大差`6.4e-12`，所有selected prefix/late指标复现。结果支持系数不稳定为重要贡献，但late仍高、且全部为train-internal，不允许宣称唯一根因、泛化或PPO准入。

## D018 — 批准FC-P008全train family读出校准的实现与CPU测试

2026-10-05 Asia/Shanghai，implementation and CPU tests approved；GPU执行另审。FC-P008只检验固定FC-P003C表征上的四力末层读出，不改变FNO架构、场输出、数据、归一化、reward或既有科学门槛。它使用全部44条train-only轨迹、19648个H1端点；base20/train8/train16的family share固定为既有regular sampler比例`(720,408,240)/1368`，每端点权重为`share_f/N_f`，其中`N=(16000,1600,2048)`。四折必须按真实source phase划分并以全局endpoint weight聚合OOF，禁止先将缺family的fold等权。

source mapping artifact `57ed2a25…3b92`把四个canonical phase固定为b00/t148、b02/t106、b04/t120、b06/t134；train16只可凭source case/time和curated frame-0 identity并入b00/b02，不能按episode名或标签顺序推断。数值协议固定为正式default TF32/high，不沿用highest-FP32诊断路线。alpha grid、并列取较大值、fold-train-only统计、全train单次refit均预先固定；先前train8 late端点在本次属于全train的一部分，但不得用既往late结果事后调整grid或选择规则。最终只允许改新checkpoint的四个force rows，其余tensor须字节不变；default TF32的严格`2e-5` wiring采用captured pointwise-head→mask-mean执行顺序，ideal pooled-affine与native差异只逐通道报告，不假定严格代数等价。当前批准不包含GPU提取、候选执行、validation/frozen、PPO或真实CFD；完整科学准入仍需后续独立评估。

实现已通过独立CPU审查：9项测试及真实44轨迹inventory验证family权重、缺family的fold聚合、fold-train统计、误导性train16名字不决定phase、去标准化、official fresh reload和force-row confinement。Lead随后只批准一次受守卫的Main全train校准执行；这项执行可生成train-only候选和原生replay证据，但不能自行开启formal evaluation或PPO。执行结果须从真实feature cache独立重算CV与native指标后再决定是否申请后续评估。

该次校准已完成且独立重算通过：固定规则选中`alpha=0`，OOF physical rear-Cd/rear-Cl MAE为`0.012857/0.022035`；全train native rear-Cd/rear-Cl MAE由亲本`0.025576/0.064053`降至`0.010462/0.017277`。同时default-TF32的ideal/native rear-Cd bias仍为`-0.004621` physical，说明不能用理想仿射拟合替代真实执行。Lead据此批准一次原封不动的formal suite；该批准是“值得测”而非“已通过”，不得改`alpha=0`、阈值或跳过force-window/development gate，PPO仍未授权。

formal suite已完成并反证准入：总receipt SHA为`14fd24d9…edcb5`，development FAIL，force-window仅1/6联合通过。四个旋转分支rear-Cl′ RMS误差相对C均明显下降，但dynamic delta-Cd、validation rear-Cd以及多个window mean-Cl/Cd指标退化；因此不能将“升力RMS局部改善”改写为整体控制准确度成功。FC-P008不进入PPO。后续FC-P009只研究train-only free-AR隐藏特征及matched-weight H1对照，不能改变或追认本次阈值。

## D019 — 批准FC-P009 free-AR隐藏特征的train-only CPU实现与测试

2026-10-05 Asia/Shanghai，implementation and CPU tests approved；GPU提取另审。FC-P009检验固定FC-P003C/default-TF32/high模型的free-AR训练窗口隐藏特征能否改善四力末层读出，不改变模型架构、数据、归一化、reward或任何科学门槛。`alpha`固定为`0`，不运行六alpha选择；四相OOF只评价固定读出，不能用于选参。输入仍为既有regular sampler的1368个H100训练窗口、每窗100个相对时刻，共136800行隐藏特征；它们只覆盖19648个唯一真实CFD端点，同一端点在不同AR起点/relative horizon下的隐藏状态不得冒称独立物理样本。

family顺序固定为base20/train8/train16，share仍为`(720,408,240)/1368`，而不是按端点数分配；对应window-step行数为72000/40800/24000，唯一CFD端点数为16000/1600/2048。四折按canonical source phase进行，OOF须按预定全局row weight汇总，不能在缺family的fold内重新等权。原regular sampler暴露保持，但这不是严格单因素消融：相对FC-P008，隐藏状态从true-state H1换成free-AR，同时同一真实端点因AR起点/relative horizon重复出现，row权重结构也随之改变。为分离该权重混杂，CPU阶段必须从现有P008 H1 cache将相同19648个目标映射成同一136800个window-step目标、保持目标值与phase逐项相同，并以固定`alpha=0`计算matched-weight H1对照；AR fit只能与这个对照和原P008两者并列解释。两种fit必须共用从同一HDF raw四力按P008 canonical NumPy-float64规范化后转float32得到的标签；官方DataPipe的torch-float32规范化另作同raw端点审计并记录两条算术路径的差异，不能用直接`array_equal`误判舍入差，也不能放宽容差掩盖case/step错配。不得读取validation/frozen选参，也不得因validation10已FAIL而事后改变阈值或权重。实施先限于一次cache-only特征提取及CPU拟合；生成候选、native replay、formal和PPO均须另审。

FC-P009已完成且未生成候选。completion/result/cache/CPU交叉分析SHA分别为`0893ec75…c214f`/`1321c30a…91daf`/`fc1b84fd…8ca84`/`ffd48eba…7516`。固定`alpha=0`下，free-AR fit在held free-AR域的all-step rear-Cd/rear-Cl MAE为`0.013660/0.039128`，优于matched-H1 fit的`0.026794/0.056831`；H100为`0.013683/0.043003`对`0.032974/0.067989`。但同一free-AR fit在held H1域的all-step误差为`0.022449/0.053706`，明显差于H1 fit的`0.013412/0.022520`；H1-step为`0.020044/0.042925`对`0.011691/0.019504`。因此证据支持状态分布特异的可读性，而不是可部署的共同受力头；不生成P009候选、不申请formal或PPO。

后续固定50/50共享头与受限候选构建分别经独立审批执行；这不是对上述专用头结论的追认。候选原formal receipt SHA `ac5c0dd0…e231c`的18项文件SHA一致，validation10和dynamic6端点组件通过，但force-window仅两个zero分支联合通过。四个旋转分支rear-Cl′ RMS误差`0.06750/0.12501/0.07045/0.07987`仍全部超限，故FC-P009 development FAIL，不进入PPO。唯一批准的后续是从既有train-only cache按原固定共享头重算100步及尾62步的mean-Cd/mean-Cl/Cl′ RMS并按family/case/phase分组；该CPU诊断不调alpha/mix、不读取validation/frozen、不形成新候选。

上述CPU诊断已完成（SHA `f6c122a6…b626`）。joint full/phase-OOF在free-AR尾62步的Cd/mean-Cl/Cl′ RMS MAE分别为`0.00671/0.01984/0.01760`与`0.00751/0.02195/0.01985`，说明phase留出只解释小部分退化。train8 family pooled RMS为`0.02936`，最差既有train PRBS/PPO cases已达`0.046–0.058`，所以时间窗幅值误差在训练profile内部异质存在；但这既不证明覆盖是唯一原因，也不证明default-TF32数值误差放大了formal失败。下一步若做native-vs-ideal诊断，必须冻结当前候选、case/window和计算协议，且只能作机制定位，不能修改formal门槛或追认PPO。

冻结候选的4窗native-vs-ideal检查已完成（SHA `2dea49fa…b1ef`）。runtime hidden features与cache逐值相同，pointwise wiring为0；保存float32头的native-minus-pooled-affine尾62步Cl′ RMS差为`-0.001942/-0.000292/-0.000851/+0.000152`，而相应native真值误差为`0.00794/0.12561/0.08839/0.06647`。除zero基准外，执行顺序差异远小于旋转窗误差，故不再把TF32/native归约作为该失败的主要解释，也不继续此诊断分支。后续只评估一个明确针对train-only尾窗幅值的监督干预；它仍须保留完整field/force formal协议，不能用训练窗改善替代准入。

Lead随后只批准一个有界CPU-cache判别：将相同136800行、相同canonical targets/source phase/原row weight的H1与free-AR特征各自归一化为总质量1后乘`0.5`，在每个source-phase fold中只用fold-train联合数据计算一个共享scaler，并拟合一个共享`alpha=0`仿射头。held phase必须在两个域分别报告all-step与H1/H10/H50/H100的逐通道物理误差；不得搜索mixture或alpha，也不得用单个平均数建立新准入阈值。该诊断只判断一个共享头能否兼顾两种隐藏状态分布；不要求它逐项支配两个分别优化的专用头。即使训练内改善，也仍须另行批准candidate/native replay并通过原formal gates。

共享头诊断已完成（SHA `931fcd2d…f2b0bc`）。H100 AR域rear-Cd/rear-Cl/total-Cd MAE为`0.01612/0.04610/0.01607`，相对C亲本`0.01637/0.07392/0.01637`改善但不及AR专用头`0.01368/0.04300/0.01367`；H1域为`0.01593/0.03035/0.01572`，优于C亲本`0.02622/0.06333/0.02580`但不及H1专用头`0.01432/0.02447/0.01411`。这支持折中头的train-only可表达性，不证明validation或控制收益。Lead只批准最小候选实现和CPU测试：复用缓存系数，官方加载C-e2，只改四个force rows/bias，保存为独立epoch0后重载并逐tensor核confinement；default-TF32/high下仅用固定base20首个train batch的H1作`2e-5` pointwise wiring sanity和有限性检查，pooled/native差只报告。候选GPU构建、原formal suite与PPO仍分别需要明确批准。

## D020 — FC-P010尾窗幅值监督显示多域权衡，不构建线性头候选

FC-P010在固定P009 train-only cache上保持H1/free-AR逐步rear-Cl损失各占一半，只新增free-AR尾62步centered-RMS项，并仅优化rear-Cl行。结果SHA `d69033fd…8f94`的输入绑定和指标已独立重算：free-AR四个phase OOF RMS误差均改善`1.45%–7.34%`，但H1有三个phase恶化约`13%`；full fit为free-AR改善`9.02%`、H1恶化`10.51%`，Cd因其它三行冻结而逐值不变。全部五个LBFGS fit都用尽200次且梯度未达容差，因此不能宣称已收敛或找到最优头。

该结果不要求每项支配作为新gate，但其方向一致的跨域代价不足以支持候选构建。停止继续扫描固定线性头、loss mixture或相位权重；下一假设必须是有界的train-only官方FNO训练干预，之后仍用原field/force/dynamic/window formal协议裁决。不得把训练窗改善、有限完成或CPU诊断状态写成PPO准入。

## D021 — FC-P011训练完成但固定train诊断为权衡，以原formal裁决而非训练指标挑选

FC-P011两臂均按同一1368-window顺序完成且scope auditor通过。A只训练rear-Cl行，free-AR rear-Cl逐步/RMS约改善1.5%，但true-state H1退化；B额外训练最后decoder hidden linear层，free-AR rear-Cl逐步/RMS反而恶化6.65%/3.37%，同时free-AR field与H1 RMS改善。该结果既不支持按训练指标直接接受任一臂，也不允许把A→B之外的变化归因于scope。两臂原formal现均已完成并FAIL：A/B窗口都只通过两个zero分支，旋转rear-Cl′ RMS误差分别为`0.070312/0.122247/0.068076/0.083214`和`0.062584/0.119208/0.075209/0.076236`；B的validation10 delta-Cd还以`0.024318>0.023`失败。故A、B均拒绝进入PPO，原门槛保持不变。

## D022 — FC-P011后先做train-only梯度分解，不自动切换路线或增加训练

FC-P011的局部scope干预未修复formal窗口失败，但训练记录显示A从不clip、B每步都clip；这些统计不能单独证明是field还是force梯度造成。下一优先只准备一个no-optimizer、no-save、无validation/frozen/PPO的train-only诊断：固定既有6个窗口，在P009亲本与P011B终态上分别重建相同H100 loss graph，独立计算field与weighted-force梯度的范数、夹角和合成前后clip尺度，并按允许tensor组报告。它只用于判断B的scope权衡是否伴随梯度竞争/尺度失衡，不构成新gate，也不从梯度相关性宣称因果。用户最终目标和force/world-model→PPO→真实CFD路线不变；任何训练干预仍须另行审批并通过原formal。

canonical surrogate的reward warm-up与direct-CFD reset不一致属于接口缺陷而非新科学变量。commit `962c165`只让canonical路径从同一绝对restart时钟恢复真实62点prehistory，并严格绑定来源；32项CPU回归及两次独立复核均PASS。该修复不追认历史候选、不放宽门槛，也不等于控制执行授权。

## D023 — FC-P012不支持以强梯度尺度失衡或反向冲突解释FC-P011B

FC-P012按预声明在P009亲本和P011B终态各复算相同六个train-only H100窗口的field与`0.2×balanced-force`梯度。五个非zero-action-history窗口中，两模型的hidden组范数比`>10`计数均为`0/5`，cosine`<-0.2`计数也均为`0/5`，未触发既定的`4/5`或`3/5`解释条件。zero窗另列而不混入计数。故不根据该诊断调整loss权重、clip或重启训练；它也不证明不存在局部或其它参数组的优化问题。

result/completion SHA为`4142cdc5…d814d`/`86f9d931…b26c`，12行均有限且模型tensor前后相同。direct-total与组件和的relative residual为`3.09e-5–9.84e-5`，按预声明仅作observational、没有数值等价阈值，不能从COMPLETE推断等价PASS。下一representation-capacity诊断或训练方案须独立批准，原formal门槛和PPO阻断不变。

## 新决策格式

## D024 — FC-P013独立受力FNO正式训练与独立诊断调度

2026-10-05 06:13 UTC。资源探针v2完成且未更新任何模型后，Lead按`FC_P013_INDEPENDENT_FORCE_FNO_PLAN_20261005.md`批准一次固定1368窗口训练。冻结P009的流场递推，另一个同架构官方FNO从相同P009初始化，训练完整可训练气动力表征；H1/free-AR等权，四力均方误差与rear-Cl均方误差等权。它是组合系统实验，不是梯度冲突已获证实或单独容量因果实验。

最终源码`1634c05`通过24项CPU回归；此前核心数学、官方冻结参数和双模型契约已有独立审查。三名代理均反复因模型服务容量错误退出，因此Root接管最终trainer恢复和审查，并如实记录额外独立复核未完成，不能据此宣称科学准入。原固定六窗物理指标转移到独立只读评估，严格保持原数据/窗口/H1-H100/tail62定义，不用于模型选择。正式训练只保存终态并要求官方双模型fresh reload。

训练4小时上限，统一内存守卫20 GiB、allocator<=0.45，实际训练集只读挂载；无validation/frozen/PPO。执行批准SHA `1bdcfcf7…1a120`。训练结束仍需核验实际1368条记录和所有来源SHA，完成固定诊断和原完整formal suite。若失败，保存负结果并分析下一可检验假设；不改科学门槛，不自动把训练损失下降写成减阻成功。

ID、记录时间、状态、待检验假设、对应实验ID、所依据证据/协议、可选方案、取舍原因、保留的不确定性、撤销/调整条件。只有读取过的产物可作为事实；代理口头报告是待核信息。
# 2026-10-06 — P028资源实测之后的训练决策

R3真实H10反向完成，模型未更新。最低物理空闲20.9028GiB，仅比20GiB要求多约0.9GiB；不能忽略至少0.35184GiB的Adam动量和其他训练临时内存。先核对已结束且无写入的项目CFD案例，必要时仅用既有POSIX_FADV_DONTNEED工具释放这些明确文件的缓存；不删除数据、不全局清缓存、不改变物理目标或模型规模。训练仍需独立执行配置及连续资源守卫，保持原1368窗口/171更新。实际证据及前两次工程失败见 `docs/FC_P028_RESOURCE_TERMINAL_REVIEW_20261006.md`。原完整代理验收和真实CFD闭环要求不变。
# 2026-10-07 — B04 raw generation accepted; conversion remains separate

Lead accepted the independently audited B04 `120→200` raw trajectory as the one planned late-state coverage source. This is a train-only data-generation milestone, not surrogate improvement or admission. The next permitted work is preparation and independent review of a case-local, raw-read-only VTK and official Curator pipeline that reuses the existing normalization bytes and real `t=120` force source. No VTK/Curator/HDF conversion, model training, PPO or CFD follow-on is implied by the raw result.

## 2026-10-07 — B04 late-state coverage receives one fixed-budget training test

The VTK/Curator train-only conversion passed independent Reader, label, finite-array and endpoint Mesh checks without changing the original normalization. Lead therefore authorized one fixed I comparison: 192 unchanged original windows plus 32 existing b00 and 32 fixed-PRBS b04 windows, still 256 windows/32 updates and the original K1 initialization, objective, LR, precision and frozen scope. The intervention jointly changes source profile, phase and visited-state coverage, so a result cannot isolate any one as the cause.

The actual training invocation is `7c1d1d634af94321b638ba6da2febe45`. Its launch and early finite progress are engineering facts, not evidence of improved prediction. Candidate decisions remain prospective: require the unchanged fixed-six retention first, then the unchanged fixed development comparison; on failure retain B and do not tune this run or auto-launch PPO/CFD. The already accepted B real-CFD closed loop remains the bounded delivery while full surrogate accuracy is unresolved.

## 2026-10-07 — I is not promoted despite fixed-six improvement

I completed its fixed training and both prospectively required evaluations. The fixed-six H1 and continuous-AR objectives improved by about 4.14% and 2.05%, respectively, so the retention half of the rule passed. The full fixed development panel contradicted that local result: pooled H1 rear-Cl and total-Cd MAE both increased versus B, and pooled H5 was also slightly worse. Therefore the original joint promotion rule fails.

Retain B as the delivered controller/surrogate baseline. Do not tune I, choose a favorable phase/lead after seeing results, or automatically launch I PPO/CFD. The result supports only the bounded statement that added b04 late-state coverage improved the original six-window diagnostic but did not generalize to the fixed opened development panel. It does not isolate source profile, phase, or visited-state coverage as a cause. E114 already passes the physical delivery gates, so no physical threshold relaxation is needed; complete surrogate accuracy remains a separate unresolved research target.

## 2026-10-07 — Short saved reward evidence does not admit a controller

Keep the original prediction FAIL and B as the deployed default. The saved six-branch diagnostic shows that short-horizon drag direction is often recoverable, but exact canonical cumulative reward still has fixed action-ranking errors and drag-only choices carry material rear-lift cost. Do not tune reward weights, lower physical gates, or repeat the old K1 H5 experiment unchanged. The next approved work is preparation only for a scientifically distinct current-B H5 held-action MPC engineering run at start148 for ten feedback cycles. Its selector retains the frozen canonical objective and action/rate limits and has independent action authority; nonfinite prediction, state-bound failure, or bounded timeout is fail-stop in this initial run, with no new PPO-fallback dependency. Ten cycles cannot establish physical benefit; a paired 80-D/U run remains separately reviewable and must retain the original `2%/1.05/10%` criteria.

## 2026-10-07 — Temporal-increment auxiliary is not promoted

The fixed intervention completed its only authorized 256-window/32-update run and independent engineering audit. H1 changed by `-0.0137219195%`, but continuous AR100 changed by `+0.0191286021%`; therefore the unchanged two-metric nondegradation AND fails. The effect sizes are recorded without being promoted as meaningful improvement or used to alter thresholds.

Retain B and the already accepted E114 PPO/OpenFOAM delivery. Do not run development inference, PPO, CFD, a weight sweep, or another temporal-increment training variant from this result. The next representation/fit evidence review is preparation only until separately approved and executed; complete surrogate precision remains unresolved.

## 2026-10-07 — Forty train points do not establish a fit plateau or surrogate admission

The fixed 40-point LBFGS diagnostic materially reduced its exact accepted-point train loss but exhausted the predeclared 300-closure budget before any of the four normalized RMSE channels reached `.01`. Because the last accepted points were still improving, do not label this as optimizer stagnation, insufficient capacity, insufficient inputs, or convergence. Line-search trial values are never candidate-best evidence; only the 140 returned, remeasured accepted points are reported. No model was saved.

Keep the already delivered B PPO/OpenFOAM E114 result and its original physical gates unchanged. Complete surrogate precision remains a separate unmet target. The next authorized activity is preparation only for the same bounded fitting question on the original 256 H1 schedule (192 original plus 64 controlled windows), with unchanged force weights; it is not yet a GPU execution authorization and must not be used to lower prediction or physical standards.

## 2026-10-07 — Representative256 train fit reduces error but misses every fixed RMSE target

The approved 256-point fit completed and passed engineering/checkpoint audit. It reduced the fixed-panel objective by about 70.84%, with the last ten accepted points still decreasing, but exhausted 300 closures while all four normalized RMSE values remained above `.01`; front-Cl physical MAE increased. Therefore do not label this run as fitted, converged, capacity-limited, uniformly improved or admitted.

Retain B as the delivered default and E114 as the bounded real-feedback control result. The already frozen fixed-six evaluation may separately measure training retention before the science deadline, but it is not an independent development set and does not authorize dev, PPO or CFD. No threshold, model architecture or physical gate changes are justified by this train-only result.

## 2026-10-07 — Reject Representative256 after fixed-six H1 and AR both regress

The matched `highest/no-TF32` fixed-six check removes a precision-comparison ambiguity and shows substantive regression against the same-precision B: H1 `+15.816903%`, AR100 `+14.270292%`. Training-panel loss reduction therefore did not preserve the original continuous force objectives. The predeclared nondegradation AND fails.

Retain B. Do not execute Representative256 development inference, PPO or CFD, and do not lower prediction or physical thresholds. This negative result closes the candidate within the time-bounded closeout; it does not alter E114's already accepted real-feedback physical result.
