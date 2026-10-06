# Experiment ledger

## FC-E047 — Canonical causal-history H2 real feedback: HOLD, zero benefit

Code c75bf01; approval SHA b273716de8edb19b8517126d3d8232cc50cc5b2ab5a7af04cb16f96aeae1e5a1; immutable driver ace9871ac27e2b92de0d90010aa1db3f4cd037cdcf3e49ce5f3f87deb32f06cc. Compared with FC-E046, only scoring changed to canonical causal62-history H2 stage average. Invocation 6f554f10e87e4b9f9d6b6ed8b555c548 completed05:48:59–05:52:01UTC, exit0. Result artifacts/exploratory_causal_history_h2_real_cfd_20261006/result.json SHA74a28d45dce9b84ec5044700fe470390cde899a2fcf40a0b893c1b28817d99ca; independent report docs/EXPLORATORY_CAUSAL_HISTORY_H2_TERMINAL_REVIEW_20261006.md SHA9ceb4d58a66b549faa86834d15d0444d57c8fdcc2f2635f18859440a679d2098.

All10 actions HOLD;200 samples/branch exactly equal. Mean totalCd2.413592168615, mean rearCl.887113848192, fluctuationRMS.3953805314509264; paired drag reduction0, RMSratio1. All50 candidate/100 stage costs independently reproduced. Mean-bias/RMS penalties0 throughout; modeled drag gains smaller than action/rate costs. Minimum sampled MemAvailable122066739200bytes. No GPU, optimizer, new PPO or long-window admission. Negative benefit result, not project completion. Next prepare prospective H5-only comparison with all other factors fixed.

## FC-E046 — Actual paired ten-cycle official-K1 H2 MPC feedback (exploratory)

Executed code `d7f737e`, immutable driver SHA `c8260b21d742f54814bf04e81bb13eeff491a971df4d177f3a384c8f8c940e3f`; approval `docs/EXPLORATORY_PAIRED_H2_APPROVAL_20261006.json` SHA `1679f6bee293eae200c95bc078a20db5873e81afcd54d81b81bb06445015f8b2`. Actual unit invocation `e3b9eb7b58724a1c9ec4e64d63ac7bbe` completed05:34:11–05:37:14UTC, exit0; two owned solver containers removed. Result `artifacts/exploratory_paired_h2_real_cfd_20261006/result.json` SHA `45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb`; independent review `docs/EXPLORATORY_PAIRED_H2_TERMINAL_REVIEW_20261006.md` SHA `8c600368d836e8c34c09e4ac3be1129ffcfb67fd3587e48e4e4feba33d711dcd`.

Real CFD current fields drove ten H2 decisions; only each selected first action was applied, with a same-start zero branch. Actual200 force samples per branch at148.005–149.0 were independently re-read: every saved force metric recomputes exactly. Mean totalCd2.4192140666805 versus2.413592168615 gives **−0.2329265954% drag reduction** (worse). RearCl mean.857989757705/.887113848192, fluctuationRMS.372887030442/.395380531451 (ratio.943109235), peakabs1.390924178/1.468057611. Actions increased.05 each cycle to.5, respecting.75 magnitude/.10 rate bounds. Ten matched next-endpoint rearCl predictionMAE.0122973621; CPU decision latency3.625–4.414s, not real-time. Minimum sampled MemAvailable121930932224bytes; no model updates, new PPO or HydroGym solver.

Outcome: **EXPLORATORY_FEEDBACK_COMPLETE_NOT_ADMISSION; SHORT_WINDOW_DRAG_WORSE**. This is actual FNO-assisted closed-loop execution, not completion of the long-window scientific goal. OneD/U is below a shedding period and cannot establish original80D/U criteria or alter physical10% mean-lift tolerance. K1 formalFAIL remains. The fixed current cost's H2variance+mean² equals H2meanCl²; review its drag/lift tradeoff before extending duration. Lead-authorized preparation only: one canonical62-actual-history/H2-stage-average cost change, with no future truth, no threshold change and no new execution yet. CSV holds one descriptive paired drag-reduction row; its evaluation hash binds the actual driver, not a fabricated evaluation manifest.

## FC-E045 — One existing current-frame bridge equivalence (engineering only)

Approved CPU-only invocation `1e908f9cd20d4430ad9de94226c4ea60` sampled the existing train b00-zero frame at148.0 and compared official Curator output with official HDF5Reader frame0. Result SHA `4530afee4aa093713f1d22b50c2200de79c9a79c3b86e096d8299829d872ac84`: mask/grid/time equal; physical and normalized valid-field maxabs/RMSE0 over97,020 values; packed-input maxabs/RMSE0 over196,608 values. Source/model-independent runtime identities and limits are recorded in `docs/ONLINE_CURRENT_FRAME_ONE_FRAME_REVIEW_20261006.md` (SHA `5ea30b2bd6ae72e71e9d31c184739a3b7a6d589f90044d286b065efe5c50b3ac`). PhysicsNeMo2.2.2/Curator0.1.0 CPU paths, unchanged sampler and byte-bound normalization were used; no model, solver, optimizer or control action ran. Retained journal reports1.1G peak/0Bswap; requested4GiB/2CPU limits are distinguished from subsequently collected unit properties. This proves one stored-frame engineering equivalence, not live CFD stepping, prediction fidelity or physical/PPO admission. Next preparation: two independently approved solver intervals in fresh isolated cases, exact-time field extraction and shadow-only input bridge; preserve existing baselines and all gates.

## FC-E044 — P030 start0/H100 train diagnostic: short-lead field gains reverse

Actual r2 invocation `4b89d3cb85c540448eeebd8ef5c7c3b3` completed with exit0/noOOM at04:25:41UTC. Status **DIAGNOSTIC_COMPLETE_NOT_ADMISSION**. Result SHA `b1042b94fde60aed135c60d348431aa1c6177b1b9b12b1ae8ba56bc9fab6ee7f`; approval `docs/FC_P030_RECOVERY_EXECUTION_APPROVAL_20261006.json` SHA `6c4edae1e955268dcae718c4eb2f6b24a216b8961189d56a09f4d4da7c091378`; frozen17-source manifest `73ac42127e0ace741c675cb7a5a53a6339171565462d0771c11665124d0f08e6`. Executed driver/launcher/core bytes were independently matched to code commit `ba96948`, not a later documentation commit. Independent report `docs/FC_P030_RECOVERY_TERMINAL_REVIEW_20261006.md` SHA `4e21fc05f543b5c90a74e318e9e7ae999b27b8350d7573817b49c4c6670dc5b4` recomputes all grouped summaries exactly and independently checks primary pooling to floating roundoff. Minimum observed CUDA free21.3438GiB; all resource guards unchanged.

Exactly44 train trajectories (base20/train8/train16), each start0 with uninterrupted H100, were compared using K1 versus P029 flow and the same frozen K1 aerodynamic model. These are44 early windows, not the entire1368-window training population, prior origin51 H10 panel, or formal validation population. CSV records30 measurements: two arms × five at-leads1/10/25/50/100 × velocity relative L2/rear-Cl MAE/total-Cd MAE. Threshold cells are intentionally empty; no diagnostic PASS criterion is invented. Cumulative prefixes are distinct and are not substituted for at-lead metrics.

At lead10, pooled velocity relative L2 slightly improves `.0144896780→.0143547827`, but rear-Cl MAE worsens `.0235259831→.0348705419`. Velocity error worsens in **44/44 cases at each lead25/50/100**; at lead100 pooled velocity is `.0508497210→.0629025821`, rear-Cl MAE `.0556205714→.0664759759`, and total-Cd MAE `.0156524378→.0189485685`. All44 lead1 forces are exactly equal, consistent with common initial input/readout. Exceptions remain explicit: train16 lead100 rear-Cl improves `.04961730→.03957725`; train8 lead100 total-Cd improves `.01832665→.01711290`; phase b02 rear-Cl improves while b00/b04/b06 worsen. These observations support investigating short-horizon training exposure, not a unique causal explanation or majority-vote admission.

CSV checkpoint_sha256 denotes the flow archive: K1 `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`, P029 `2f71e25532b3002955396e7e97a2714d2e8ff46a0bbaef4cc7606783312658aa`; shared aero is `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5`. Config columns bind the actual execution approval; evaluation_manifest_sha256 binds the frozen source manifest, with selection/raw hashes retained in result. The first attempt's post-rollout aggregation failure and immutable source are preserved; it produced no usable numerical result and supplies none of these metrics. A conditional H25 training plan is a next design hypothesis only, not approved or completed execution. P029's original full formal FAIL remains authoritative; no PPO, physical tolerance or surrogate-error threshold change follows.

## FC-E043 — P029 original full formal: endpoint subsets pass; complete admission FAIL

Actual invocation `85ae29a422fc48739136418317de8ca5` completed normally at03:47UTC. Receipt SHA `96e207af491ef4abe0c9e9c85983672111d86d70fe88b2d88551b29d0739a334`; approval `docs/FC_P029_FORMAL_EXECUTION_APPROVAL_20261006.json` SHA `b339d1175ea17dfd110763c044b0d9a2a15ce088062b1834c3627969f279fec5`. Independent report `docs/FC_P029_ORIGINAL_FORMAL_TERMINAL_REVIEW_20261006.md` SHA `020042bb6846e9be14ccb10e36035bba7c5d3fa4d6e164c81ee5d027f9a62527` verifies35 outputs/411 sources/eight exit0-noOOM containers; minimum host free21.200443GiB. Scientific status remains FAIL, with no PPO admission.

Validation10 endpoint action-difference Cd MAE `.020433813333511353` passes original `.023`; dynamic6 strict delta-Cd MAE `.01296532154083252` also passes. Full tail-window joint/Cd/RMS/mean counts are **2/6,6/6,2/6,4/6** (K1:1/6,5/6,2/6,4/6; P028:1/6,2/6,2/6,2/6). Rotating b01-minus and b05-plus gain joint/RMS passes, but both zero-action RMS passes are lost; b01-plus and b05-minus RMS errors worsen. Subset PASS and improved counts cannot replace the unchanged all-six rule. Development gate SHA `aa7dd557bc516e898339655517eba8bf16cf27b4579751c6b9f75a4de9153c53`.

Executed source is numerical base `7216214b545fbbd50b2fb5ed866f231039b06b18` plus seven reviewed overlays and external orchestration. The seven overlay and three P029 orchestration bytes were independently matched to commit `d6138a1`; CSV code_commit records that overlay/orchestration commit, not current documentation HEAD or a claim that every base file came from it. Frozen source-chain receipt SHA `3e86ce83523eaeb6335efe6943817a069a1be83a0e1ca7fd6b58a0fdf12dad43`. CSV config_path binds the actual formal approval; its config_sha256 is the approval-file SHA. The approval separately binds training YAML `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`, P029 flow model `2f71e25532b3002955396e7e97a2714d2e8ff46a0bbaef4cc7606783312658aa` and candidate manifest `72b52ff2b6c702c9d100cc5fbade0952875289d25e826bb6b88020cb042d9b16`. No threshold changes or new execution are authorized by this ledger entry. Earlier partial/running observations remain historical.

## FC-E042 — P029 matched H10: force improves, field trade-off remains

The actual train-only 44-case, origin-51, H10 comparison completed under invocation `f7291c51f1f1423fa4093b08e71371fd`. The approved execution source is `docs/FC_P029_H10_COMPARISON_EXECUTION_APPROVAL_20261006.json` SHA `6a3aaa1aa22b02c24d470992017b28bc446e009211291e550a3ee81b1a49b771`; result SHA is `aed040eb22766f8f47b1f6093f50bc1aa753dd748cd90b8d68ac9db1095b329d`; independent report `docs/FC_P029_H10_TERMINAL_REVIEW_20261006.md` SHA is `824b17f9b615c9d086735a7a8f285902e1749ddc7f8b81a782fa43206c69f49e`. The current repository state at recording is `d57c1f0`, while the actual evaluated diagnostic/source closure derives from reviewed commit `ab1c0c5` and immutable 13-file manifest SHA `44edc4acd9a490402407319d9c0bf023b239e130f54ee3860dd7f4fe911fc44b`; the CSV therefore records `ab1c0c5` as the executed code identity rather than the later documentation state.

Under identical 44 origins, recorded actions, normalization and horizon, K1/P028/P029 H10 AR mean-case field RMSE is `0.038909243208102205 / 0.033543899316679344 / 0.03793723525648767`; rear-Cl MAE is `0.03869321600319711 / 0.039507443905313265 / 0.03335770925861487`; total-Cd MAE is `0.01690930107777769 / 0.017283562435345214 / 0.0153860518200831`. Thus P029 improves all three metrics versus K1, and improves the two force metrics versus P028, but its field error remains 13.10% worse than P028. The unchanged K1/K4/persistence arrays reproduce exactly and all 44 P029-versus-parent H1 force-equality flags are true.

The force improvement is not a uniform fluctuation repair. P029 rear-Cl RMS absolute error is better than K1 for train8/train16 but worse for base (`0.016772501092118808→0.01885180255257461`); by physical phase only b04 clearly improves, while b00/b02/b06 worsen slightly. This diagnostic is `COMPLETE_NOT_ADMISSION`: it used no optimizer, saved no model, accessed no validation/frozen data and authorizes neither formal admission nor PPO. The separately running original formal suite remains the decisive unchanged evaluation.

## FC-E041 — P028 original full formal complete: scientific FAIL

Invocation `eb4e12507302498bb8944373e0717a25` completed normally;35 output hashes,411 source hashes and8 containers exit0/noOOM independently verified. Receipt SHA `63fd75d4e90176dd94998f2844f2f70cb5a7e357bd59d5362591019ed8655154`; report `docs/FC_P028_ORIGINAL_FORMAL_TERMINAL_REVIEW_20261006.md` SHA `ae98f3a63b195aca184ce348d2e1991f88ad2f416766105bb8bd5b788107ee44`. Minimum host free28.029789GiB. No PPO/frozen-test access.

Validation10 action-difference Cd MAE .0273935347795 exceeds .023 (K1 .0191296935081). Dynamic6 endpoint diagnostic passes at delta-Cd MAE .0157500505447 and pooled H100 Cd NRMSE .0255184459841, but does not override complete tail-window failure. Joint/Cd/RMS/mean pass counts **1/6,2/6,2/6,2/6**, versus K1 **1/6,5/6,2/6,4/6**. Development SHA `770f004f3a9ea1072abd66719fe31aa34d2e2643206de6839fb9515121f439ff`.

Same-protocol H10 field MAE improves .00632490→.00471157, while H100 field MAE worsens .0191041→.0197010 and rear-Cl MAE .0401863→.0693762. All six mean errors worsen; two rotating RMS errors improve but still fail. Next requires separately approved P029 parent-scale/resource evidence before fixed-budget training. No execution or admission is authorized by this entry; earlier running states below are historical.

## P029 conditional CPU preparation — no scientific result

Same original parent/data/order/171-update budget; the sole planned intervention is a fixed50/50 parent-normalized field/four-force objective through the frozen K1 aerodynamic model. Official architecture is unchanged. Root canonical68new CPU tests and133legacy/1skip pass; additional host collection lackingPhysicsNeMo is explicitly not included. Independent423-source closure verification and launcher review completed; mapSHA `c6584b8fdbbb38cf7149ad2f086526169f44174a800ef49a32fcda08a678627a`. No actual P029 scales, resource probe, training, evaluation or PPO yet. No scientific results.csv row is warranted. See `docs/FC_P029_CONTROL_AWARE_FLOW_PLAN_20261006.md` and `docs/FC_P029_CPU_PREPARATION_REVIEW_20261006.md`.

## FC-E040 — P028 matched H10: field improvement, force deterioration

Actual 44-case origin51/H10 comparison completed under invocation `74d9e3115791403ab96e55a5d06b8ffd`. Result SHA `6146ea9276570981cc72c949e3e6fac46737c43aa83c561a37eaa0e54a4ab793`; independently recomputed force arrays and field aggregation in `docs/FC_P028_H10_TERMINAL_REVIEW_20261006.md`. All old P027 K1/K4/persistence arrays reproduce exactly; P028 true-field-conditioned force equals frozen K1 exactly.

Mean-case normalized full-field AR RMSE improves0.0389092432081→0.0335438993167 (not pooled RMSE). Rear-Cl AR MAE worsens0.0386932160032→0.0395074439053; pooled RMSE worsens0.0542399828243→0.0556543162316; total-Cd MAE worsens0.0169093010778→0.0172835624353.24/44 cases improve rear-Cl, but base20 and overall performance deteriorate. No scientific admission; train-only H10 cannot replace original formal H100/window criteria.

Original full formal is now actually running: invocation `eb4e12507302498bb8944373e0717a25`, output `artifacts/fcp028_original_formal_20261006`; actual observation01:57UTC isvalidation10, not a final scientific result. P029 is conditional design preparation only, not approved training. Older entries below describe historical states.

## P028 training and official independent CPU reload complete

Actual result `74bc0d491d82da8c3b897a330e1397ac7db2e92465221801ae1868a57114840d` verifies1368windows/171updates; source/protocol/role metadata and actual terminal container independently checked. Actual candidate audit `dd3390d0ac09f8f8e4673ed8eb48d2fbe47293a1dd1689a7abae29972ddddaea`; official CPU reload `685d55a9a2116d1554f14c26e54ce2d2913a4f9c47553c70407f3c4e25d3aecd` verifies saved flow tensorfac5f298...5406 and unchanged aero b0ec7405...80eb. No matching-evaluation improvement claim yet; complete report `docs/FC_P028_TRAINING_TERMINAL_REVIEW_20261006.md`. Next matchedP027 H10 and unchanged full formal. Formal preflight passed without numerical execution; no PPO admission.

## P028 fixed flow-rollout training — actual running, not terminal

Unit `fluid-control-fcp028-flow-train-20261006.service`, invocation `c46c60f3c2634802b2646bb094f9d201`, is actively computing under approved spec `655f4d924036ef1e23857c4bc1892b8f0bbce40af1a1b22f8bd4657eca097837`. Observed37/171updates and302/1368windows. No accuracy result or terminal model yet. Inputs/source/protocol exactly match the actual R3 probe; only training mode enables the predeclared optimizer. Review deadline03:26:54UTC. Formal source receipt `fb5fd1ef87a09188d78453d0c5f93e49cf1a795dc7fa9fcee5fd77bf14cc910d`; no held-out execution yet. Scientific results.csv remains unchanged until measured same-protocol results exist.

## P028 actual resource check R3 — engineering milestone only

Actual official-container H10 forward/backward completed, exit0/noOOM, on original train window816. Result SHA `e808095f9c4de77f838c7132615427ba76985f78d3c0c7803b1d8528008a40f1`; independent review `docs/FC_P028_RESOURCE_TERMINAL_REVIEW_20261006.md`. Thirty finite/nonzero gradient norms; zero optimizer steps, unchanged flow/aero tensors, no saved candidate. Minimum host/CUDA free20.9028/20.9051GiB. Adam moments alone add0.35184GiB: full-training capacity is not established by this no-update measurement. R1 missing helper dependencies and R2 wrong read-only mount destinations are retained operational failures, not scientific results. R3 used the corrected421-file source and exact data aliases. No scientific CSV improvement row is warranted. Next: sufficient safe memory headroom, finalized unchanged formal evaluation support, then the predeclared171-update flow-only comparison.

## P028 engineering preparation — no new scientific result

Canonical runner/updated-flow loader and legacy suites90PASS; official-model objective10PASS; old/new resource-launcher mocks24PASS. Source-only418-file preparation completed; no HDF/model/GPU access. See `docs/FC_P028_RUNNER_LOADER_CPU_REVIEW_20261006.md`. Next actual task is one no-update H10 resource check after memory clearance and bound approval, then fixed171-update experiment only after actual capacity and evaluation compatibility are established. This does not change E039 or admit K1/K4.

## FC-E039 / P027 — actual short-horizon diagnostic complete

Result SHA `7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366`; actual official-b40 container exited0/noOOM. Independent review reproduced rear-Cl MAE directly from44 unique case prediction/target arrays. K1 true-field/free-AR MAE0.026546/0.038693; K4 0.026586/0.038588; persistence0.633210. Corresponding pooled rear-Cl RMSE: K1 0.038826406092363715/0.05423998282426294; K4 0.03881085177318823/0.054197059487591556. Each model worsens under AR on29/44 cases. Train8 K4 MAE rises0.026061→0.061876. First-lead H1/AR outputs match exactly for both models/all cases.

Protocol: train-only44 trajectories, origin51, ten recorded-action transitions; K1/K4 fixed frozen-flow parents and unchanged targets; no optimizer/checkpoint/validation/frozen/PPO. Shared440 flow transitions and1760 aero evaluations follow the audited loop and completed case count, not independent hardware counters. Mixed62 costs include52 truth samples and are not admission evidence; eight terminal costs unavailable, with cases retained in force metrics. Independent report: `docs/FC_P027_TERMINAL_REVIEW_20261006.md`. Next hypothesis is robustness of the force model to predicted flow inputs, subject to checking existing training exposure before approving a controlled intervention. Original full formal failures remain authoritative.

## FC-P027 execution preparation — not a scientific result

Root and independent reviews completed for diagnostic f398c86f and launcher aa8e337d. Canonical32 CPU tests passed0.64s. Frozen414 source manifest `ffdd7e623f9c8da0b39ccb167ca0b74c9013313fdf9cddc37266e884df578c1d`; independent source/metadata checks read no HDF/model payload. Approval `docs/FC_P027_EXECUTION_APPROVAL_20261006.json` SHA `fe218527b6f85daf08999673b9525a2b93144e1722235f5769dc2ea055e567a5` authorizes one bounded read-only44-origin/H10 diagnostic. Dry-run command preparation succeeded without Docker/GPU. No scientific CSV result until actual output and independent review; no policy training follows automatically.

## FC-P026 K4 original formal evaluation — terminal scientific failure

Observed 2026-10-06: actual invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e` exited0/PID0. Receipt SHA `729f9ce1f307d5462307470af20491806f6cfe31b5c81fc74ec284a2461841d9`; development SHA `ce60621723ce364e4f8cdc165268a64ca5fd1a0185b7529bc21c1f91ac8aab9e`. Original validation10/dynamic6 endpoint components passed, but six-window joint count is1/6, Cd5/6, rear-Cl fluctuation RMS2/6, rear-Cl mean4/6. Four rotating RMS errors are0.06823553510506697/0.12234179725403527/0.06830637102663029/0.08125565316417882. K4 does not repair the matched K1 admission failure; no PPO or frozen-test access. Minimum observed free/available memory20.770393/110.100964GiB. Independent terminal review is being completed before the final scientific CSV entry and milestone push.

Independent terminal review is now complete: all35 output and411 source hashes agree; eight containers exited0/noOOM; unchanged auditor under Python3.12 reproduces the complete gate dictionary exactly. Review: `docs/FC_P026_K4_FORMAL_TERMINAL_REVIEW_20261006.md`; scientific ledger FC-E038. The preceding review-in-progress sentence is the earlier observation.

Next: FC-P027 separates true-field-conditioned force error from recorded-action H10 autoregressive error on existing training trajectories. Root approved isolated CPU implementation and synthetic engineering unit tests only; no real-data scan/model load/GPU execution yet. No new training, architecture, CFD generation, or relaxed admission is authorized by this negative result.

## FC-P026 K4 original formal evaluation — actual running observation

Observed2026-10-05T23:38:24Z: unit `fluid-control-fcp026-k4-formal-20261006.service`,
invocation `d5d2201c8e2c4bf2ab40201cca0dcb1e`, PID1156613, activating/start.
Officialb40 container `0be58547d0ccd85f1d569a336e921fa712af255c12589d5a74da6ca784f84b64`
is running original validation10 H1/10/50/100 stride25/batch4 with p026_k4.
Actual approval SHA `03f6880893a730307a6a6acf0ed19276ca8dc958fc3266ebf14e8e2a1323cbb8`
and readonly source/candidate/data mounts verified. Unit start23:36:49UTC;
container start23:36:54.985072171UTC. No numeric result, completion, scientific
admission or PPO authorization follows; no CSV scientific row added.
Evidence: `docs/FC_P026_K4_FORMAL_RUNNING_OBSERVATION_20261006.json`.

## FC-P026 K4 training integrity and official CPU reload milestone — not a scientific result

Actual retained K4 training invocation `eee5a6fbad40411cac2f05e00520b079`
completed1368 windows/171 updates, success/exit0/PID0. Independently rehashed
all7 candidate and6 execution files against candidate audit SHA
`423ad58a3d441d26f174174bc68824a59ccd453b2e49f0577888530a81083b0b`.
Actual official CPU-only reload containerc84f3e5f…50df24e exited0/noOOM;
receipt SHA `491d6e4e8868edd0c0a222ceb1e1ed5cc1c5b2c3a8b88f0f4a053895aed1a729`
matches audit/tensors/7filemap/416-source closure. Internal/host/guard free
memory minima20.803394/21.006046/21.071632GiB remained above20GiB.
Four fixed training-panel aggregates were independently recomputed from JSON;
small terminal improvements are descriptive train-only evidence. K4 formal
results, scientific admission and PPO remain pending/unapproved. This entry
adds no scientific CSV row. See `docs/FC_P026_K4_TERMINAL_REVIEW_20261006.md`.

## FC-E037 — FC-P026 K1 original formal evaluation: terminal development FAIL

The actual K1 formal unit `fluid-control-fcp026-k1-formal-20261006.service`,
invocation `c039836ab63246ff8772dad66e1b46e5`, exited successfully after the
unchanged validation10, dynamic6, force-window6, and development-gate sequence.
Receipt SHA is `f2f7a50a26177c65ee048b58fb20df0aee7f4cfa42aed0edd857f08d911ef948`;
all 35 receipt-named outputs rehashed exactly and all eight terminal container
records are exit0/non-OOM. Minimum external MemFree/MemAvailable were
28.046733856/110.021461487 GiB.

Endpoint evidence was positive: validation10 H100 delta-Cd MAE 0.0191297 passed
the 0.023 limit with sign8/8 and ordering20/20; dynamic6 strict delta-Cd MAE
0.0103930 and pooled H100 total-Cd NRMSE 0.0178932 also passed. The unchanged
62-point window gate nevertheless passed only 1/6 branches (Cd5/6, rear-Cl RMS
2/6, mean4/6). Four rotating RMS errors were
0.0683388/0.1222618/0.0682180/0.0813537 versus fixed limits near0.0294. Status
is `DYNAMIC_FNO_DEVELOPMENT_ADMISSION_FAIL`; scientific admission/PPO/frozen
access are false. The candidate seven-hash map equals the approval and prior
audit maps, but this review did not reread candidate payload bytes. Full report:
`docs/FC_P026_K1_FORMAL_TERMINAL_REVIEW_20261006.md`.

## FC-P026 K4 matched full training — actual GPU execution started

Independent approval commit `ef2ddd3`, SHA
`fca9c5a1106c85fb55a54590784453bd5246d55c21bbaa8ded1e3c72562b3e86`;
approved dry-run passed. Actual user unit
`fluid-control-fcp026-history-k4-20261006.service`, invocation
`eee5a6fbad40411cac2f05e00520b079`, MainPID941365 started at22:00:26UTC.
Container `79a39768f1398bc5ff9d1085f027a4b1802978120107ca5f9008ffdd300a75e3`
uses official image `b40d5888…a22e` and explicit `--history-k 4`. Actual
preflight observed CUDA free32.9703GiB and MemAvailable115.0164GiB; GPU later
reached96%. This is a running observation only. Completion, audit/reload,
unchanged formal evaluation, admission, PPO, and real-CFD benefit remain
unknown.

## P026 formal source freeze — actual artifact verified

Exclusive artifacts/fcp026_formal_source_20261005_immutable contains408BASE721
files plus7reviewed c6ce0b1 overlays (411finalsourcefiles) and pinned config07e55.
Sourcechainreceiptff8b742aac29f86ae10b4202c04df5c2ef3b8de6217fefa540d240fdf682fe24;
originalarchiveSHA9b81ee07b20b2d09833dfb383c56d395c60cd4e4c43004d477d76107bd61a1fa.
Root independently checked all411actualfile hashes, regenerated originalarchive
digest and checkedconfig. Summary docs/FC_P026_FORMAL_SOURCE_FREEZE_20261005.json.
This is immutable evaluation preparation only; no formal execution/heldout/model
reload or scientific admission. K1training continues under its existing service.

## P026 formal runner — reviewed integration, not executed

Runner afb6144fdb289dc8db7c296358a5f20d9c327d4ba6eb41edc8b64f16b0c27a22;
tests480f055e8bb43ef4da532098ffefca33c9cbe9d80a743309fa47e02571830323;
dev30 identity overlayb44407ab3e828d65f99fd079b8b0a0701706849c4f211da021a36b25d682a94b.
Root34canonical tests0.11s and independent57combined tests pass. Newloader
validation precedes preflight; numerical source is exactBASE721 plus7reviewed
overlays. Main-only retaineduser units are deliberate scope. NoHOMEoverride;
CPUaudit containers explicitlyexcludeGPU. Report
docs/P026_FORMAL_RUNNER_INDEPENDENT_REVIEW_20261006.md documents limits.
K1actualtraining keepsrunning; no formal/GPUlaunch by this preparation milestone.

## P026 K1 full matched training — actual GPU execution started

Sourcebe4fc7e, approval755cd99/SHA1088285e4e13c7e3553009dd511436e9a5da976e93eed44d6bb0c28ab9515aa3.
Dry-run identity/protocol/dependency checks passed before execution. Actual unit
fluid-control-fcp026-history-k1-20261005.service, invocationb3759e7e1acc4de7a1aa9f6e8d38de9a;
containered4f0ad7a42621712b6689d3f694ed90067f154ed7bf78c72e0993bbad930f65
started2026-10-05T19:27:46.093783477Z. Exact44training HDFbyte checks completed.
Earlyactual5window events/GPU96% confirm computation, not merelyservicecreation.
Root registered real unit+approval+launcher+log in dashboard; training progress
does not imply admission. Immutable copies and dual20GiBguards remain active.
No formal evaluation/PPO/realCFD launch authorized by this running observation.

## P026 formal history callers — actual-main CPU fixtures verified

Evaluator daef4a3a6eb1b9190f5cde728581b0c1b4b9656f56e865cf61a53deb1f37de88;
force-window eee1f59fa269a22556048f3fde8fc7ecbc31bb913571e49075dead92618a8576.
Six actual-main fixtures pass Root1.66s/independent1.70s; official loading is
mocked, so no scientific model acceptance or real validation is claimed.
K1 actualmetrics identical; K4 future-observation poison cannot affect predictions.
Report docs/FC_P026_FORMAL_CALLER_REVIEW_20261005.md records coverage and recovery
of60new synthetic plots to repo-external engineering_quarantine. Final tests force
temporary cwd and outputs. Training execution review next; noGPUstarted yet.

## P026 role-aware production loader — integrated, software checks only

Loader3343dba367dd6e45fdc914fc321e90b94efc2d8a00553c61b025ef7d776fc2a8;
focusedtests6eef48fa7f0c386995fd477ada90fe0ca30a4de227c2b4703bd0a70b36aa4047.
Independent23focused/baselegacytests and earlier35legacytests pass; Rootcanonical
33focused/base/history tests0.64s plus23P015/P018 regressions0.61s pass.
Explicit role architectures and actual imported history module hashes are enforced.
Report docs/FC_P026_ROLE_LOADER_REVIEW_20261005.md limits these claims to software
compatibility; no realcandidate/heldout/HydroGym admission is implied. Additional
official smoke is engineering-only and does not exercise full production loader;
its scratch-cleaning script is not synchronized. Canonical earlier parent-bound
checkpoint verifier remains the retained save/reload evidence.

## P026 monitoring preparation — actual training events supported

Root added registered progress parsing for history_training window/update events,
with contiguous sequence, arm and 8-window/update consistency checks. Matched
systemd oneshot activating/start with a verified live PID is running, not stopped.
Eighteen registered-dashboard/P023 tests pass. This prepares the local-only UI;
current registration remains the completed resource probe until a real training
invocation exists. No training or scientific gain is claimed by UI changes.

## P026 actual official CPU training inventory — verified

Container2fa6d157 exited0/noOOM, official pinned image, noGPU, readonly
train-only inputs and exclusive output. Receipt76c84cae2e08595d5326159926396ed8b1bc31a6c1fdd09a92bbae5b9ef2a526
independently verifies all1368 unique identities, original order177ebd95,
family counts720/408/240 and warm1300/padded68.
Effective protocol hashes K1 daf22b2464744509260f1eb9e0b20d3b80da484c8985f3a22887293bbb40cb30;
K4 72b3638c4f0fbad687bcc4b216365e8c68935611778f7be562386abaa1db7a3d.
Report docs/FC_P026_TRAINING_INVENTORY_CPU_REVIEW_20261005.md includes launcher
static review and18CPUtests. Metadata/sampler only: no field batches, training,
new accuracy, PPO or scientific admission. Full44 bytes remain checked at launch.
Next actual role-loader/history-caller integration, then separate execution review.

## P026 full trainer — independently reviewed, not executed

Final trainer SHA562d268545ba5cd2559374f4bd8bd34bf59a2e49f2e886e4e286bb12aac5d49e;
tests fc32801e1c873285d4b2ef81df4e58664683744f74924592e34da88f84262332.
Independent 41 CPU tests passed in1.16s; Root44 trainer/history tests passed in1.09s.
Report docs/FC_P026_TRAINER_CPU_REVIEW_20261005.md records coverage and limits.
Matched1368/171 training, separate parent identities, original objective and
terminal checkpoints are implemented; warm/padded fixed-panel reporting does not
alter selection or acceptance. No full training or formal evaluation executed.
Next: actual inventory/order preflight, role-aware production loader/callers,
reviewed resource launcher and immutable execution approval.

## P026 checkpoint engineering terminal review complete

Independent retainedcontainer/image/mount/log inspection and all6savedfile hashes
match receipt8b1b4893; report docs/FC_P026_CHECKPOINT_CPU_REVIEW_20261005.md.
No reviewer rerun or model load; exact fresh tensor equality was enforced by the
reviewed successful producer. Three artifacts remain engineeringfixtures, never
accepted candidates. Fulltraining/formalcallers still under implementation.

## P026 official CPU save/reload — executed, independent terminal review pending

Script433a2f8bd5e2c3a27a59c5e4c3a1426dcccfefa0b98379fbb83e6269a0627708,
Root5CPUtests0.56s/independent5CPUtests0.52s. Actual containerdb6cb7dd exited0,
officialb40d image, noGPU,8GiBcontainer, onlyexclusiveengineeringoutput writable.
Receipt8b1b48930eeab818921a1a287d69b02769635faa912c643c219150af03af4665
records three officialfixtures flowK1/aeroK1/aeroK4 with exactfreshloadedtensors,
epoch/metadata and parentidentity checks. Nooptimizer/training/candidate. This
roundtrip does not exercise allformal/HydroGym loaders or certify accuracy.

## FC-E036 — P026 resource terminal audit passed, engineering only

Independent result/provenance/precision/parent metadata checks complete;
full report docs/FC_P026_RESOURCE_TERMINAL_REVIEW_20261005.md. Both observed host
floors exceed20GiB, guard/container0. Initial saved predictions and objectives
exactlyequal K1/K4; singlewarmwindow only. No newscientificaccuracy, optimizer,
candidate or control result. Officialsave/load CPU fixture and formal caller
integration are next; original physical and prediction requirements remain.

## P026 no-update production-size GPU resource check — terminal, audit pending

Actual invocation8667f3c9c82146d6ab861d634c460107 exitedsuccess/PID0; sourceee13932,
approval40abd5f. Result8e1113efc903c6c95cd24755d8c9b081fb098a01ce60aa3c2477e932052f3589.
One real warmtrainwindow816,100frozenflow calls, twoarms each10pairedchunk forwards
and backwards. K1/K4 elapsed3.2774/3.0470seconds, peakallocated3.64/3.73GiB.
Added288gradient norm0.0462021, finite; initial K4/K1 predictedforce difference0.
Internalminimumfree29.121925GiB/available107.817810GiB. Fullrun22.6785seconds.
Nooptimizer/update/checkpoint/heldout/PPO. Independentterminal/resource audit
pending. Resource observation is singlewindow only, not fulltraining accuracy or
guaranteed identical timing. Next officialreload/historyformalcallers integration.

## P026 shared force objective — CPU equivalence verified

Source4d27fb53/tests907e25bd; Root20PASS0.78s and independent20PASS0.75s.
PinnedP013 K1 loss/outputs/gradients exact on batch1/2 CPUfixtures. K4 chunked
gradients agree with full-window fixture; no aerodynamic field feedback or
futureH1truth in AR. Final targetalignment and288history coefficient gradients
tested. See docs/FC_P026_HISTORY_OBJECTIVE_CPU_REVIEW_20261005.md. NoGPU/model
update/admission. Next no-update resourceharness preparation and explicit
history-aware formalcaller integration; no legacy6channel K4 bypass permitted.

## P026 real-HDF integration prerequisite — independently reviewed

Actual official CPU container a8c735d1 exited0. Six realtrain windows cover
base0/20, train8 0/4, train16 0/4 with originalstrides; exact originaltargets,
metadata and K1inputs, correct K4 paststates/actions and padding. ScriptSHA
17ad02b3a9162fa8f746e3ed5671f0a8c50d71c7e0ec1fad1548742e59989179;
report docs/FC_P026_REAL_HDF_CPU_REVIEW_20261005.md. No model/GPU/datawrites/
heldout or fullfieldhash. This is bounded integration evidence, not accuracy.
Next shared history forceobjective tests then separately approved no-update
resource probe. Full44 K1/K4 design recorded without GPU execution approval.

## P026 engineering prerequisite — synthetic official CPU check, not admission

Adapter2b5b37dc / tests145929fc / verifier647de8e9 are separately named project
glue around official FNO and existing official HDF5Reader. Thirteen tests pass
in Root and independent reviews. Actual pinned official CPU container9e4b793a
exited0 withoutGPU; tiny4x4/modes2 K1/K4 zero-history-weight output differences0,
finite nonzero historical and shifted-prediction gradients. No optimizer or
candidate. Report docs/FC_P026_OFFICIAL_CPU_EXECUTION_REVIEW_20261005.md records
actual provenance and limitations. Next prerequisite: realHDF integration then
separately reviewed matched full44 training protocol/resource check. No claims
about scientific accuracy or full-project completion follow from these tests.

## FC-E035 — P025 isolated statistical supervision complete, unsupported

Result SHA `7648df661439f530984504538bfaef3d1a602c1b960fd914fcba2f016fc4e74c`;
source4d12c75, approval6fb2f810, protocol7f6cc9c6. Same six train windows,
P018 parent and exact P023 HIGH initial predictions/precision. Fixed sixteen
updates of96 new coefficients with J0 plus5/16 times four normalized mean/RMS
terms. Independent audit confirms96 backwards,36 evaluation windows, exact
initial and zero-input reproduction, frozen parent, finite loss accounting.
H1 bias squared worsens0.1473296% vsinitial; H1 J0 and AR bias also fail their
P023 HIGH comparisons. RMS and centered error improvements do not satisfy
the predeclared joint local test. No saved candidate, heldout or PPO.
Host minimumfree28.2702827454GiB; internal28.2385482788GiB, guardexit0.
Interpretation: this fixed statistical intervention did not repair the tradeoff;
not a proof that all statistical objectives or the official FNO family fail.
Next: close isolated-block/statistical-loss branch; staged CPU engineering for
matched K1/K4 causal history, preserving original full admission and CFD criteria.


## P025 running — fixed statistical supervision on isolated96 input coefficients

Source4d12c75; approval6fb2f810, protocol7f6cc9c6; actual invocation
124d6521045a41cd9dcf5f35edff6712 observedrunning18:17UTC,4/16updates.
Fixed16updates/96backwards/36evaluationwindows, unchanged six realtrainwindows.
Four normalized rearCl tail62 terms withfixed5/16, fullH100 recurrent gradients,
original weights frozen, storedP023HIGH original-loss control. Exact initial
rawpanel/source/precision comparability enforced beforeupdates. Root/independent
20CPUtests passed; no terminal result/candidate/PPO. Both20GiB guarded.
Actual container/mount evidence docs/FC_P025_RUNNING_EXECUTION_20261005.json.

## FC-E034 — P024 fixed response-scale mechanism diagnostic complete

Result6eecbd75c5b18cd821d6cf814c6d9319f755bf8dc70f2eb0c0e7e9326bf8ba0b,
source81a45c6, actualinvocationb85dcf9d70e443fe92ae4ef5d72d3c76 terminalsuccess.
Independent60journalwindows/10panels/96vectors/source/resource checks complete;
scale0/1exactlyreproduceP023 before-1/8/64. No optimizer or candidate.
Positive scaling improves some statistics but trades against others;64H1centered
MSE+19.096%,ARbias²+2.093%; no prescribednonzero scale meets all requirements.
No optimum/capacity/fundamentaltradeoff inference. Minhostfree30.915GiB.
See docs/FC_P024_TERMINAL_REVIEW_20261005.md. Next P025CPU-only boundedstatistical-
loss implementation with frozenparent, not another scale sweep or thresholdchange.

## FC-E033 — P023 isolated96-input finite comparison complete

Resultadfdd9a86cedf75019b655fe360b64b09d1aa51ce3de2f9166d0d80e296007cb;
sourcef13a6a0. Independently verified32updates/192backwards/72endpoint-and-ablation
windows, source/optimizer/frozen hashes and12repeatedpanel aggregates. Both20GiB
floors held. HIGHlocal_support=false solely H1bias²+0.070729%vsinitial;
ARbias²−0.199089%,RMSerror²−0.083577%,centeredMSE−0.249289%. Not admission.
Bothterminal zero-input ablations exactly reproduce initial raw predictions;
actualHIGHblocknorm.001553586, output effect remains small. No candidate/PPO.
See docs/FC_P023_TERMINAL_REVIEW_20261005.md. Next: separately specified response-
scale mechanism preparation, no automatic full training or acceptance change.

## P023 running — isolated current-force input coefficients

Protocol docs/FC_P023_INPUT_BLOCK_COMPARISON_PLAN_20261005.md; approval SHA
714db1f9d09b0ee7037953d6b807b3341cf7a736889d5c5fa98e5ae8c0116cf0,
source f13a6a0. Actual invocation39aec740a9914226bb1f74c2d29e7917 observed
running17:38UTC, LOW4/16completed/HIGH0; no terminal evidence yet.
Fixed six real train windows, both causal,96new coefficients only; LOW1.5625e-7
versusHIGH1e-5, originalJ0/fullH100.32updates/192backwards/72endpoint-and-ablation
window evaluations. Root and independent25CPU tests passed before execution.
Output artifacts/fcp023_input_block_20261005. No model saved/heldout/PPO.
Next: independent terminal review using original local conditions and frozen
parent checks. Do not infer acceptance from update count or training loss.

## FC-E032 — P022 causal force-input finite comparison (2026-10-05)

Operationally complete and independently reviewed; local_support=false. Result
SHA `69e5d4a8a93b8036187a18a5c40cb270aec53462a49ddab94417fa0b908096d8`.
Two matched arms, sixteen updates each,192 fullH100 window backwards; original
objective and fixed six training windows. A/B initial rows and endpoint repeats
exactly agree; complete aggregate/check dictionaries independently recomputed.
Current-force B improves four mean/amplitude squared errors versus zero-input A,
but relative to initial H1bias+0.52877%, H1centeredwaveform+11.3133%, ARbias+0.69550%;
AR original objective also worse than A. Not a candidate, admission or PPO result.
Both20GiB floors held; external minfree26.807617GiB. Source074d979; full report
`docs/FC_P022_TERMINAL_REVIEW_20261005.md`. Next action is targeted learning-scale
and update-scope analysis, not automatic repetition or full training.

## P021 engineering prerequisite completed; P022 scientific comparison pending

P021 r2 actual full-size100-step forward/backward passed independent engineering
review; result SHA `975fc40bbb88d5d0ab3d239ee0ce994bc0635aabcad9b7d74568bf73f1a8ce7a`.
Each zero/current-force arm100forward+100recompute,28finitegradients, no optimizer
or changed parameters. Minimum external MemFree28.595875GiB; source and recovery
records in `docs/FC_P021_RESOURCE_TERMINAL_REVIEW_20261005.md`. First startup failure
is preserved, not omitted from the record. This is not an accuracy experiment.

Next P022 CPU preparation is approved under
`docs/FC_P022_CAUSAL_CONDITIONING_PLAN_20261005.md`: six fixed train windows,
16updates/arm, same original objective and fresh AdamW, only zero versus causal
current-force inputs differ. Full-gradient100-step recurrence, no saved candidate,
no validation/frozen/PPO. GPU execution not yet approved; no results to report.

## FC-E031 — Six-window causal force-input audit (2026-10-05)

Descriptive CPU data/source audit only, no model training. Actual persisted evidence is `artifacts/causal_force_input_audit_20261005/{persistence.json,timestamp_audit.json,official_source.json}`. Primary timestamp SHA `72d9117922ef5dbbd3b9f9a5ae193d01c19ac44a39aea30b4dbe5eaf3189d4e2`. Independent in-memory recomputation from actual raw files matches the full timestamp result, including274 source/config hashes. All606 endpoints per cylinder uniquely match nominal time exactly; restart-source fallback occurs only selected initial frames975/1077/1233.

Actual future interpolation affects200/505 base/train8 HDF frames; train16's101 frames use exact endpoints, not its hypothetical stored-time interpolation. Six initial forces are exact: AR-initial persistence unchanged, HDF-lag H1 persistence is not strictly causal. H1 persistence's smaller mean/RMS-amplitude errors do not fix waveform error (five-nonzero rearCl MAE0.107195 versus FNO0.038682). Remedy is an explicit exact-raw six-window input sidecar with unchanged targets/normalization, not recuration or a full-data claim.

Root225d99f authorizes P021 staged CPU engineering only; no GPU, training, deployment or admission. Full report `docs/CAUSAL_FORCE_TIMESTAMP_AUDIT_20261005.md`. Preserve all earlier records below.

## FC-E030 — FC-P020 symmetric tail-statistic finite-update comparison (2026-10-05)

Operationally complete: identical P018 initial tensors and fresh AdamW for both arms, fixed SIX training windows, 16 updates per arm, six raw gradients averaged before one clip. All192 unique window backwards and32 update records verified. Four endpoint panels repeated twice; all repeat rows and A/B initial rows exactly equal. Source/dependency/candidate/approval hashes, journal44-HDF mapping, resource guard exit0, finite output and aggregate/criterion recomputation verified independently. Result SHA `a7c0d0c41b35391e22d07fb223a5ed243891ccdd4759815e9bf08b82670b5042`.

Predeclared interpretation: `LOCAL_CONDITIONS_NOT_MET`. B improves all four five-nonzero tail statistics versus A, but its H1 bias MSE increases4.8015515e-6 and H1 centered residual MSE increases0.0002363738182 versus the shared initial state. Both six-window original domain objectives decrease; that does not override the failed conditions. Repeat spread zero is observational, not a rigorous error bound. This finite-update comparison is not a full-training, capacity, admission or physical-control result.

Host382 samples minimum MemAvailable106.712757/MemFree24.620411GiB; innerguard384 samples minimum CUDAfree24.622280GiB. No saved candidate, heldout, PPO or threshold change. Next scientific direction awaits Root analysis and separate approval; current-force conditioning is not approved execution. Full provenance and numbers: `docs/FC_P020_TERMINAL_REVIEW_20261005.md`. Preserve earlier entries below.

Next-action clarification: Root authorizes preparation only for current-force-conditioning data/causality/persistence-baseline investigation, not GPU execution or architecture implementation.

## FC-E029 — FC-P019 local cross-window gradient interference (2026-10-05)

P018 terminal, six fixed train windows, original mixed20 chunk objective; five gradient kinds each repeated twice,60 unique passes. No optimizer/update/candidate save/held-out/PPO. Exactinvocation275365360254440aba18ed96aac58630 retainedexited0; result SHA `1bd66e3cbf7c1200ff0af96d23bd62803d59422129eec5c5dddf7615fbcb183f`, source6de42a9. Approval/source/candidate identities and44-HDF startup map verified. Every original objective exactly reproduces P018terminal in both repeats. Host minima available102.286327/free20.551254GiB; internalCUDAfree20.544643GiB, guardexit0.

Five-nonzero aggregate AR-RMS squared-error derivative along the negative original gradient is+0.859184292/+0.859184299 across repeats; against six-window original gradient +0.468013701/+0.468013706. Other three aggregate derivatives are negative. All six individual-window AR-RMS derivatives along their own negative objective gradients are negative: this supports local cross-window interference, not universal within-window conflict or actualAdamW behavior.64summary algebra checks finite and consistent; fullvectors not saved so no independent full-vector re-dot. Repeat spread is not a rigorous uncertainty bound.

See `docs/FC_P019_TERMINAL_REVIEW_20261005.md` and Root-owned running evidence SHA `5719e608b59b61de03206714128079b08e0bf52efd1a85392c385ac25626adab`. P018 admissionFAIL remains. P020 two-arm16-update fixed-six-window loss diagnostic averages all six training-window gradients; its statistical summary separately focuses on five nonzero windows. It is PREPARATION ONLY, not GPU approved/executed. No threshold change or scientific admission.

## FC-E028 — FC-P018 reduced-rate complete formal rejection (2026-10-05)

Same P009 parent, 44 real train trajectories, fixed 1368-window order and eight-window accumulation; the sole optimizer override relative to P015 is learning rate1.5625e-7. All171 actual AdamW steps, finite moments, frozen tensors, protocol and data hashes were independently checked. Actual official CPU dual reload passed; earlier cache failures are preserved.

Original complete evaluation finished on invocation `ef589f7dbeff4fa0ab064309409971ad`, retainedactive/exited with success/0 and MainPID0. All18 receipt hashes and original raw numerical auditor output exactly match. Complete receipt SHA `d4d3f85a79e31d50866bb8dbd23ec90e0453cde39ef6b314e3db80344d33869c`. Verdict **FAIL**: joint1/6, Cd5/6, rearCl fluctuationRMS2/6, rearmeanCl4/6. Only b01zero passes jointly. Four rotating RMS errors0.0683247/0.122662/0.0685903/0.0813667 remain above approximately0.0294. Formal memory minima available110.045368GiB/free27.460377GiB; all1017 samples satisfy both20GiB floors.

Same-protocol dynamic6 pooledH100CdNRMSE0.0179109253 is lower than P0090.0181379026/P0150.0192933478, but H100rearClMAE0.0850700867 remains worse than P0090.0842962672. Flow metrics exactly equal the frozen parent. Do not merge pooled/macro/start0 quantities or claim componentPASS as admission. See `docs/FC_P018_TERMINAL_REVIEW_20261005.md` for full identities and comparisons.

No new PPO, frozen-test access or surrogate-assisted physical success. P019 objective/statistic gradient diagnostic is preparation only, not approved/executed. Physical mean-lift0.10 remains separate from surrogate mean-error limits; no threshold change before the conditional15:20UTC review and none made here. Preserve all earlier results below.

## FC-E025 — FC-P015 complete formal rejection (2026-10-05)

Eight-window gradient accumulation completed171 updates over the original1368
windows from the same P009 parent. Official dual reload and complete formal
evaluation finished successfully as processes, but scientific admission failed.
All18 receipt artifact hashes and the original numerical auditor output were
independently verified. Receipt SHA
`353004afa2aa5a35b912205751a100bc6d37d0572ce5431108dca3ed2ea95a5b`.
Joint1/6, totalCd5/6, rearCl-primeRMS2/6, rearmeanCl2/6; only the b05 zero branch
passes jointly. Dynamic H100 pooledCdNRMSE0.01929335 and rearClMAE0.08544903 do
not improve the P009 values0.01813790 and0.08429627. Frozen-flow metrics remain
identical. Preserve the negative result; no compatible PPO or frozen test ran.

FC-P016 is now preregistered for a bounded fixed-six32-update local fitting
probe, with implementation and independent review underway. It is not executed
or a scientific result. See `docs/FC_P016_FIXED_PANEL_FIT_PLAN_20261005.md`.

## FC-E024 — FC-P014 exact-objective residual decomposition (2026-10-05)

Result SHA `5550140b818a3d9028073b913cf994da99d9d3e917ea24896e4a9d477b8e95a7`, source59215be, approval e26473d; actual service invocation6bba81dba45b46638643f441ba21082f exited0. Six original train H100 windows, parentP009 and terminalP013, original mixed-batch20 chunk objective with force train mode and no_grad. No backward/update/model-save/held-out/PPO. Tensor identities unchanged. Independent reviewer recomputed all48 residual decompositions; maximum identity residual6.94e-18. All44HDF hashes were checked before launch; host minima MemAvailable109.60GiB/MemFree29.02GiB.

H1/AR/total normalized objectives each worsen6/6. H100 rear-Cl AR centered residual MSE increases6/6; H1 centered residual MSE increases3/6 and decreases3/6, while H1 bias-squared increases6/6. Terminal12 domain means and analytic bias-coordinate derivatives are positive. There is no compensating H1/AR improvement on this panel; mixed bias/waveform deterioration cannot be explained as a pure constant offset. This is not proof of full-dataset convergence or a unique optimizer cause.

Lead decision: do not allocate another full1368-window pass solely to scalar bias correction, which cannot repair the failed centered-RMS criterion. Develop one bounded optimization intervention capable of changing the waveform, retaining the official architecture and original objective/protocol. Do not sweep parameters or lower admission criteria.

## FC-E023 — FC-P013 complete formal rejection (2026-10-05)

The unchanged validation10/dynamic6/force-window/development suite completed, service invocation `7235b2f06282435a89b84964e384c60f`, retained exited status and exit code0. Completion is not scientific success. Receipt SHA `2733c3cb1061837da28db83bb8c0d16e534017211b514885f960255b48aa2e7a`; all18 referenced file hashes independently match. The original numerical auditor SHA `ca6da0afdce5859be1c060eb48ba2cdd1ccc5ee3aeb2570d9c9b53067d5bc412` was rerun and exactly reproduces the saved admission FAIL.

Force-window joint0/6, total-Cd5/6, rear-Cl fluctuation RMS2/6, rear-Cl mean0/6. P009 counts under the same protocol were joint2/6, Cd6/6, RMS2/6, mean5/6. Four rotating RMS errors are0.047608874/0.128188129/0.070502560/0.053709452 versus fixed limits about0.0294. Mean-Cl errors span0.062781407–0.109959183, all above their original limits. Validation10 start0 delta-Cd MAE improves0.01920627→0.01554142, while H100 rear-Cl MAE worsens0.0387410→0.103710; endpoint improvement does not override failed temporal force prediction. No frozen access or new PPO.

Decision: reject P013 for control training. Execute separately approved P014 fixed-train objective/residual decomposition (no parameter updates), then choose a new hypothesis from its evidence. Do not lower thresholds or relabel historical CFD-only PPO as surrogate-assisted.

## FC-E022 — FC-P013 terminal fixed-six train diagnostics (2026-10-05)

The preregistered P013 r2 fixed pass completed1368 updates; terminal integrity and exact Docker/journal exit were independently verified. The read-only original six train windows then completed on P009 and terminal P013 with unchanged tensor hashes, optimizer0, no validation/frozen/PPO. Diagnostic result SHA:`8e0255c955c1b26fdff240a0854fc0a92d3bd247cc38ed6c268fc1d397cec873`; training source1634c05 and original diagnostic source from immutable23a4ec020ed6 chain. These are train diagnostics, not a formal admission threshold or converged optimization claim.

H1 rear-Cl MAE worsens6/6; free-AR worsens5/6. The sole AR improvement is b00PRBS0.100626→0.100346. The zero-window H1 MAE worsens0.011493→0.082078 and AR mean-Cl error0.000669→0.081711. H1 tail62 RMS error worsens2/6 and AR4/6, so amplitude changes are mixed. All H1/AR u/v/p field metrics are exactly unchanged. This pass does not demonstrate improved force fitting. Absolute-error summaries do not establish a common signed offset or its cause.

The original complete formal suite is now executing under separate approval`4ad097c`; no checkpoint reselection or threshold change. Service`fluid-control-fcp013-posteval-r2-20261005.service`, invocation`7235b2f06282435a89b84964e384c60f`. Its results are not yet recorded as completed. Next diagnostic hypothesis, if formal admission fails, concerns signed residual means and same-window optimization evidence; no architecture expansion or weight sweep is authorized by this observation.

This is the human-readable index for `experiments/results.csv`. Stable completed-history IDs use `FC-E###`; proposed work is intentionally excluded and uses the separate `FC-P###` namespace.

The entries below are retrospective reconstructions from immutable artifacts and receipts; they are not presented as historical preregistrations.

## Recording rules

- One CSV row is one `experiment × protocol × metric` observation. Protocol names are deliberately distinct: epoch-internal validation, validation10 H100, dynamic6 H100, force-window6, and paired 80D CFD are not interchangeable.
- Blank numeric values mean unknown. `NOT_EVALUATED` means the protocol has not produced verified evidence; it never means zero.
- `checkpoint_sha256` is the model/policy identity. `artifact_sha256` identifies the cited receipt when one exists; `evaluation_manifest_sha256` binds a validation or dynamic panel where one manifest applies. A blank commit or hash is preserved as unknown rather than inferred from the current tree.
- Epoch-internal metrics are training diagnostics only. Admission uses independently produced post-evaluation evidence and unchanged gates.
- FNO development admission and the final physical CFD gate are separate. Passing an endpoint metric cannot override a failed force window, and surrogate metrics cannot substitute for paired OpenFOAM verification.
- The 16 matched training pairs contain only four distinct initial states. Frozen10 is materialized and sealed in full40, but excluded from the development dev30 view and was not opened for this ledger.

## Historical experiments

### Active FC-P013 — 2026-10-05 06:13 UTC

Approved train-only independent aerodynamic official FNO experiment, implementation `1634c05`, execution `a6ca463`, approval SHA `1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120`. Hypothesis and immutable inputs are in `docs/FC_P013_INDEPENDENT_FORCE_FNO_PLAN_20261005.md` and the source-bound execution approval. The P009 flow model is frozen; the independent force FNO uses the same parent and fixed 1368-window order, one terminal checkpoint, equal H1/free-AR supervision and no validation/frozen/PPO access. Root observed update 8 at 06:13 UTC; this is an active run, not a completed result or an FC-E scientific acceptance row. Exact six-window physical diagnostics are retained as a separate read-only stage under `docs/FC_P013_DIAGNOSTIC_SCHEDULING_20261005.md`. Outcome, same-protocol comparison and interpretation remain pending; next action is terminal record/checkpoint/source verification, then unchanged formal admission. No improvement is claimed from live training losses.

| ID | Experiment | Hypothesis | Evidence and outcome | Interpretation | Next action |
|---|---|---|---|---|---|
| FC-E001 | B5 immutable parent | Static-train FNO may transfer to controlled trajectories. | validation10 recorded; dynamic6 Cd NRMSE 56.78% and action-delta Cd MAE 0.11461 both fail. | Useful immutable parent and negative control, not a controlled surrogate. | Retain unchanged. |
| FC-E002 | Dynamic train8 H50 | Train-only action trajectories at H50 improve controlled response. | dynamic Cd gate passes; action-delta 0.03009 exceeds 0.023. | Partial endpoint improvement does not admit PPO. | Preserve as H50 development evidence. |
| FC-E003 | Dynamic train8 H100 | An H100 training regimen may improve long-rollout behavior. | validation action-delta 0.023258 and dynamic action-delta 0.030907 both fail 0.023. | The near-threshold static result is still a failure. H50 used four epochs and H100 two, so this is not a pure horizon comparison. | Used only as an immutable parent. |
| FC-E004 | Control train16 Main | Genuine direct-PPO train-only trajectories improve action response. | dynamic6 passes, but static action-delta and 6.15D/U force-window development gate fail; rear Cl-prime errors dominate. | Good endpoint agreement is insufficient for trustworthy control optimization. | Evaluate a predeclared matched-pair statistic intervention. |
| FC-E005 | Control train16 lift-balanced | Increasing rear-lift channel weight improves force-window fidelity. | dynamic6 passes, but static action-delta and force-window development gate fail. | Channel reweighting did not solve the window mechanics. No claim of superiority over Main is made across different diagnostics. | Retain as controlled negative comparison. |
| FC-E006 | Paired-stat lambda0 | Loss-off with the same parent, sampler, and batch is the within-experiment control for lambda10. | Formal validation10 and Dynamic6 endpoint gates pass, but the force-window joint gate passes only the two zero-action branches (2/6); all four nonzero branches fail rear-Cl-prime RMS and the overall development admission fails. | Better endpoint/field metrics do not repair controlled-window mechanics. This is the within-experiment control against which lambda10 is judged. | Keep PPO blocked; retain as the FC-P001 loss-off reference. |
| FC-E007 | Paired-stat lambda10 | A train-only matched-pair statistic loss improves force-window mechanics. | The identical formal suite also fails: validation10 and Dynamic6 endpoint gates pass but force-window joint pass is 2/6. Mean rotating-branch rear-Cl-prime RMS error falls only 1.68% versus lambda0 while individual phases mix improvements and regressions; it remains about six times the per-phase limit. | The intended matched-pair loss effect is small and inconsistent and does not satisfy admission. | Reject the current lambda10 intervention; build the FC-P002 failure map before a Lead-approved single-factor FC-P003 experiment. |
| FC-E008 | Direct real-CFD PPO training | SB3 PPO through official HydroGym interfaces plus the project OpenFOAM adapter can learn in a genuine online CFD loop. | 2048 transitions, two environments, eight rollout/update rounds completed; each PPO round contains multiple minibatch optimizer steps. | This establishes CFD-only RL closure, not PhysicsNeMo-surrogate success. | Freeze the final policy and use paired CFD for physical claims. |
| FC-E009 | Direct CFD PPO b00 | The frozen final policy improves the same-start physical response. | Final 60D/U: drag reduction 4.2212%, rear Cl-prime ratio 0.93565, mean-bias ratio 0.01993; joint pass. | A valid training-phase CFD-only baseline. | Keep policy and VecNormalize immutable. |
| FC-E010 | Direct CFD PPO b01 | The frozen policy transfers to an unused start time. | Final 60D/U: drag reduction 4.2502%, rear Cl-prime ratio 0.93597, mean-bias ratio 0.03867; joint pass. | b01 was not used for training, but its start is only 18D/U (about three shedding periods) from b00; statistical independence and broad generalization are not established. | Add genuinely separated starts before broader claims. |
| FC-E011 | Uniformly interleaved paired-stat lambda10 | Spreading the same 16 paired updates uniformly through each epoch prevents later regular updates from washing out control-force supervision. | Immutable post-evaluation completes. Validation10 endpoint and Dynamic6 action diagnostics pass, but force-window joint pass remains 2/6. The four rotating rear-Cl-prime RMS errors are 0.165159/0.213692/0.178018/0.157702, essentially unchanged from frontloaded lambda10; development admission fails. | The intervention is not supported. Small mixed endpoint/field changes do not repair the control-relevant window or authorize PPO. | Retain the negative result and await FC-P003B's independently approved dynamic-pair comparison before choosing another single-factor experiment. |
| FC-E012 | Dynamic-pair interleaved lambda10 | Replacing static matched pairs with eight train-only dynamic action/zero pairs, each used twice per epoch, improves control-force fidelity. | SHA-verified training, transfer and complete post-evaluation finish. Validation10 and Dynamic6 action components pass, but the force-window joint gate is still 2/6. Rotating rear-Cl-prime RMS errors are 0.160242/0.206943/0.172678/0.153707: a consistent but only 2.53--3.16% branch reduction (2.94% mean) from FC-E011 and still about 5.9 times the limits. True-state H1 rear-Cl MAE remains 0.191916/0.156482/0.157575/0.191730. | Dynamic pairing gives a small window-level improvement but does not repair the one-step action-force mapping or meet development admission. | Reject FC-P003B for PPO; implement and CPU-test the separately approved FC-P003C direct per-endpoint paired-force objective without changing data, model, gates or reward. |
| FC-E013 | True-state per-endpoint paired-force lambda10 | Replacing the nine-statistic paired term with direct true-state action-minus-zero endpoint-force supervision repairs the controlled lift-window error. | Training and immutable post-evaluation receipts verify one epoch-2 checkpoint. Validation10 and Dynamic6 endpoint components pass, but force-window joint pass remains 2/6. Rotating rear-Cl-prime RMS errors are 0.164160/0.206895/0.173657/0.158384, about 1.37% worse in mean than FC-E012; full field/force changes are mixed. | The single-factor intervention does not repair the key lift-window bottleneck. Endpoint PASS remains insufficient for control admission. | Reject FC-P003C for surrogate PPO; perform only the approved D015 train-fit versus generalization diagnosis after CPU validation. |
| FC-E016 | Fixed-feature affine force-readout diagnostic | The frozen C hidden representation contains force information accessible through a different affine readout. | v1 preserved as an operational wiring failure under default TF32. The bounded highest-FP32 v2 passes unchanged `2e-5` wiring tolerance, is full rank 129, and is CPU-reproducible from its cache. Prefix action rear-Cd/rear-Cl MAE falls from 0.04408/0.11869 to 0.00475/0.00733, but the retained condition number is 1.90e5; late rear-Cd worsens 10.15% and rear-Cl remains 0.08337. Prefix-only phase-blocked ridge selects alpha `1e-6`, lowers held-phase MSE 44.1% from alpha0 and improves late rear-Cd/rear-Cl to 0.03876/0.06274, but both panels remain train-internal. | The fixed features are linearly readable on the fit window and regularization reduces coefficient-instability effects, but temporal force errors remain material. This does not establish a unique cause or alter the formal default-TF32 model, field/window gates, or PPO status. | Do not continue hyperparameter scanning on train8; any next feature audit must be separately predeclared on broader existing train-only evidence. Keep PPO blocked. |
| FC-E017 / COMPLETE — DEVELOPMENT FAIL | FC-P008 full-train force-row calibration | A phase-blocked, family-weighted readout fit over all existing train-only H1 endpoints can yield a more stable four-force head than the bounded train8-only diagnostic. | Receipt `14fd24d9…edcb5` binds the complete unchanged suite. Validation10 pooled H100 total-Cd NRMSE is 0.01664246 and start0 delta-Cd MAE is 0.04284067; dynamic6 all-rolling pooled H100 Cd NRMSE is 0.03056368, while the distinct six-start0 development endpoint pooled value is 0.01386137 and delta-Cd MAE is 0.03346293. Force-window joint pass falls from C's 2/6 to 1/6: all four rotating rear-Cl-prime RMS errors improve by 37.5--61.1%, but several mean-Cl/Cd errors fail and b05-zero newly fails mean-Cl. Validation/dynamic field metrics are exactly unchanged from C. | The fixed force-row fit found a real lift-RMS improvement but redistributed error into drag response and mean lift, so it did not produce a jointly admissible surrogate. A partial channel improvement cannot override the fixed multi-metric gate. | Reject FC-P008 for PPO. Preserve it as the force-row negative result and evaluate only separately approved train-only hypotheses without changing thresholds. |
| FC-E019 / COMPLETE — DEVELOPMENT FAIL | FC-P009 fixed 50/50 H1/free-AR joint force-row calibration | One shared train-only force head can retain P008's RMS improvement while restoring drag/action and mean-lift fidelity. | Receipt `ac5c0dd0…e231c` binds the unchanged suite and 18 artifact hashes. Validation10 endpoint passes (pooled H100 Cd NRMSE 0.0055867; start0 delta-Cd MAE 0.0192063); dynamic6 also passes its endpoint diagnostic. Field metrics are numerically unchanged from C. All six window Cd branches and five of six mean-Cl branches pass, but only the two zero branches pass RMS and joint admission. Rotating rear-Cl-prime RMS errors are 0.067499/0.125012/0.070447/0.079871 versus fixed limits near 0.0294. | The joint head corrects important P008 force tradeoffs and remains much better than C on rotating RMS, but controlled-window lift amplitude is still inaccurate. Component endpoint passes cannot override the 2/6 window result. | Reject FC-P009 for PPO. Use only the approved cache-based 100-step versus trailing-62-step train-window failure map before another hypothesis. |

## Dataset identities

| Dataset/evidence | SHA-256 | Scope |
|---|---|---|
| dev30 manifest | `5213c7bb07c824c6e601636c3cfa974b80ec7c051633b0c8368a045cea41ddd2` | train20 + validation10 |
| dynamic train8 manifest | `a0bd0e3b79d4d43ace407292c31a595d552dbd6e1a5de1fd149b1ca446f53f35` | train-only |
| direct-PPO train16 manifest | `7c62dab94e317442cecf7ea3be547cf6e03c594a2c7ec2a1a695caba2445cf5b` | train-only; reused already-generated CFD interactions |
| matched-pair manifest | `15bfa7a47e3195afad59884f96fd4e305c8ba0001dd6eb043be2412b2b9ce2b7` | 16 action/zero pairs but four unique initial states |
| fixed normalization | `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1` | train-only normalization shared by listed FNO candidates |
| direct-PPO baseline summary | `b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7` | b00/b02 training baselines |

Frozen10 is materialized and sealed in full40, but is excluded from dev30 development and was not opened here. No frozen metric appears in the CSV.

## Current boundary

Both lambda0 and lambda10 now have SHA-verified complete FC-P001 receipts and both fail development admission. Their validation10 and Dynamic6 endpoint passes are retained as component results and do not override the shared force-window failure. Lambda10 changes the four rotating-branch rear-Cl-prime RMS errors in mixed directions and reduces their mean by only 1.68%; this does not support the intervention hypothesis. Epoch-internal metrics remain training diagnostics only; no surrogate PPO is authorized and the project remains incomplete.

The D012 numerical compatibility producer (`1a7c8f4`), candidate-aware PPO launcher (`0e05aa4`), and candidate-to-OpenFOAM evidence adapter (`6af0c9e`) are software-chain milestones, not new scientific experiments. D012 recomputation keeps lambda0/lambda10 at canonical-window FAIL (2/6 branches) even though the separate canonical dynamic assertion passes. Consequently no candidate policy or FNO-assisted CFD loop is authorized by these commits; FC-P003 and FC-P003B remain active surrogate experiments whose immutable post-evaluations must finish first.

FC-P003 Main is now recorded as FC-E011. Its two approved epochs and immutable v3 post-evaluation completed, but the development gate failed because only the two zero-action force-window branches passed. The complete post-evaluation receipt SHA is `7b7f55b3593b150578d65ab6c403e5a84c678a5dc6d043f702c4027334cadabb`; the canonical compatibility reconstruction separately records window FAIL and dynamic PASS. Uniform interleaving therefore did not support its hypothesis and does not authorize surrogate PPO. FC-P003B remains a separate running experiment. Commit `a4399f3` adds observational wall timing to a future eligible real-CFD feedback run; it is code-ready instrumentation, not a CFD experiment or control result.

FC-P003B is now recorded as FC-E012. Its initial post-evaluation failed after complete validation10 inference because the legacy audit expected `/workspace/gatedata` while the report recorded `/workspace/devdata`; that failure remains preserved. Immutable recovery commit `2331301` reused the inference pair only after SHA/count/finite/checkpoint checks, completed the unchanged protocol, and the Worker-to-Spark transfer is verified by receipt SHA `bc665a9a7bd468393ce2a32ceaaf1df7e8b27540ea4297f46ffb09e74d856fcc`. The complete post-evaluation receipt SHA is `98f3d336b79a7816f17c204bfd521c1bdbcc83eed1b5ddf5df59c04995314258`. Endpoint components pass, but canonical window SHA `6f2c829de2d7bf5fb6a2fe02b7609ef6f38379a90cc99ce0e17eb58285d48410` and development gate SHA `066ebcf85d066446be05e3474a160942d9d2bda1fc1a7ff85a2bdc9f1a639ce4` both record failure. No PPO was launched and frozen data were not accessed.

The 606-endpoint same-state diagnostic (SHA `618483f1898a74b6e0118fd8b9ac0a4ce62a9eb51fbb1e703f5b04aebbc808f9`) supplies each recorded CFD state to H1 and finds large rotating rear-Cl errors while zero-action errors are only 0.009827/0.006964. This rules out a purely autoregressive explanation for the observed failure, but does not prove a unique cause. FC-P003C holds the Main-e2 parent, data/order, model, normalization, seed, learning rate, lambda, weights, update count and two epochs fixed, while replacing the old nine-statistic paired term with direct normalized per-endpoint action-minus-zero force supervision. Training completion SHA is `8411d422d373de1d7bad6098304bc3543c4abc677345d948ecc89333e014e3b8`; complete post-evaluation SHA is `0ee2468b193b25e09cf2bc1b2c71a43fc918888788db92b2f115149f21cb8338`. Validation10 and Dynamic6 endpoint components pass, but the force window remains 2/6 and development admission fails. The four rotating rear-Cl-prime RMS errors average about 1.37% higher than FC-E012, so direct endpoint supervision does not repair the target bottleneck and PPO remains blocked.

At 2026-10-04 21:11 UTC, epoch-1 internal validation recorded selection score 0.0265716 and terminal total-Cd pooled NRMSE 0.0132990; these are training diagnostics, not admission metrics, and are not added as a completed FC-E result. The separate train-only 16-position gradient diagnostic failed closed before completing position 0 because the regular-total gradient decomposition residual ratio was `3.9183e-5`, above its unchanged `2e-5` implementation-consistency tolerance. Its debug receipt records no optimizer step, saved candidate, or validation/frozen access. The diagnostic failure did not interrupt or alter the completed formal training and does not establish whether FC-P003C improves fields, forces, or control windows.

The interim epoch-1 visualization receipt (SHA `3c6877f45493fc341dbd6e16ae9bca7c9cdb3a31dfac9ca2ecd55bddbce4b4e9`) binds only `b01_plus`, start 0, at H1/H10/H50/H100. Velocity relative-L2 is 0.002099/0.020483/0.063993/0.092529; at H100 pressure relative-L2 is 0.282826 and rear-Cl MAE is 0.124421 for one endpoint. The long rollout images show small-scale spatial error, but this preview cannot identify its cause and is neither a force-window RMS result nor a multi-branch formal evaluation. It is not used to choose an epoch, change a gate, or authorize PPO.

Independent review verified every SHA in the complete FC-P003C receipt and re-ran the strict complete validator with absolute candidate/output paths. The CPU-only C→D012 adapter then produced external compatibility receipts without modifying the source bundle: canonical window SHA `07a2ceaf4345b75032c7f54f8972ee8fab88ce888f99eef9fd69b1119c34c376` remains FAIL, while dynamic SHA `551275aa225d249d0de44513415ab6af0c6076cfbc3d0fd0ec015d2b6b8ae186` is PASS; neither authorizes PPO. D015 engineering implementation is approved, but any GPU execution remains contingent on CPU validation and independent review.

The true-state paired-force backward probe is an engineering preflight rather than a scientific experiment. Its first execution stopped before model forward because a mode-600 manifest was unreadable under the original container UID/cap-drop configuration; the evidence does not establish a deeper rootless/user-namespace cause. The subsequently approved operational-only v2 kept all numerical inputs and tolerances fixed and changed only the exclusive output plus reviewed non-root mount identity. It passed: T20 monolithic/chunked losses agree within the fixed tolerance; the maximum parameter-gradient absolute difference is 1.86e-9; H100 gradients are finite/nonzero; peak CUDA allocation is 5.886 GiB and minimum unified MemAvailable is 105.921 GiB. Model parameters/buffers are byte-stable and no optimizer step/checkpoint/validation/frozen access occurred. Result/completion SHAs are `773af049…f9c8`/`eb23c661…12d4`. This establishes only gradient-accumulation and memory feasibility for one fixed train-only pair; it does not create an FC-E scientific result or alter the development gates.

The strongest verified physical result remains the frozen CFD-only PPO baseline at two start times. It does not show that an FNO surrogate is accurate enough for PPO, that PhysicsNeMo contributed to the physical benefit, or that the policy generalizes statistically.

The bounded D015 train-fit calibration is complete but rejected as a mechanism experiment. Its completion/result SHAs are `7703e2b1…3bfa`/`f2397fed…e423`; it executed 128 regular-loss updates with 64 paired delta updates and eight complete dynamic8 passes, with no validation, frozen, or PPO access. Rear-lift delta MAE improved only about 3%, while absolute rear-lift and every field channel regressed; the deduplicated zero-branch rear-lift MAE increased from about 0.006 to about 0.025. The CPU-tested absolute-paired mode in commit `5cb65bb` is a proposed single-variable follow-up, not an executed experiment or admission result.

The corresponding absolute-paired calibration was subsequently executed once and rejected. Receipt/result SHAs are `20c19387…8ed`/`38687ccc…fdb`; its sample order and initial readout are identical to the delta calibration. Action rear-lift MAE improved about 5%, but zero rear-lift MAE worsened 145–175%, delta improved only 1.8–2.2%, and all field channels regressed. The weighted paired loss was 93.01% rear-Cd and only 3.85% rear-Cl, with all 64 pre-clip norms above one. These observations do not establish causality and do not authorize formal validation or PPO.

FC-E018 records the completed FC-P009 cache-only representation diagnostic. It used the fixed FC-P003C/default-TF32/high parent and fixed `alpha=0`, with 1368 H100 windows and 136800 weighted rows representing 19648 unique train-only CFD endpoints. The completion/result/cache/cross-domain SHA values are `0893ec75…c214f`/`1321c30a…91daf`/`fc1b84fd…8ca84`/`ffd48eba…7516`; parent tensors are unchanged and no candidate, validation, frozen access, PPO or formal evaluation occurred. Free-AR features support a much better held-free-AR fit than the matched-weight H1 head, but the same head is substantially worse on held H1 features. This is a domain tradeoff, not admission. The only approved follow-up is a fixed 50/50 CPU-cache joint fit with one fold-train-only scaler, one shared alpha-zero head and separate held-domain reports; there is no mixture search or new scientific threshold.

The fixed 50/50 joint diagnostic subsequently completed with result SHA `931fcd2d…f2b0bc`. Its single shared head improves the unchanged C parent in both held hidden-state domains, while remaining worse than each separately optimized specialist. At H100, physical rear-Cd/rear-Cl/total-Cd MAE is `0.01612/0.04610/0.01607` on free-AR states and `0.01593/0.03035/0.01572` on H1 states. This is a Pareto compromise and train-only representability result, not a new gate. Minimal candidate implementation and CPU tests are approved, but GPU construction, native formal evaluation, PPO and real-CFD control are not.

FC-E019 records the resulting joint force-row candidate's complete formal evaluation. The candidate altered only the four force rows/bias and preserved all field tensors. Its formal receipt SHA is `ac5c0dd047c90fddba884b77cd82bbe4f5147123d0f2f4455fb1b938607e231c`; all 18 bound files independently hash-match. Validation10 and dynamic6 endpoint components pass, while the unchanged force-window gate passes only the two zero branches. The four rotating rear-Cl-prime RMS errors remain 2.3--4.3 times their fixed limits, so development admission fails and PPO remains blocked. This result replaces the earlier candidate-engineering-pending state; it does not invalidate the train-only representation evidence.

The bounded train-cache window diagnostic (SHA `f6c122a6…b626`) uses the same fixed joint head and reports both full-fit and phase-held OOF statistics without selection. On free-AR trailing-62 windows, joint rear-Cl-prime RMS MAE is 0.01760 for the full fit and 0.01985 for phase OOF; by family it is 0.01002/0.02936/0.02035 for base20/train8/train16. Individual train PRBS/PPO cases reach 0.046--0.058. Thus heterogeneous temporal-amplitude error already exists within train profiles and phase holdout is not the sole explanation. This CPU evidence does not quantify a TF32 contribution, alter FC-E019's formal failure, or authorize a candidate/PPO.

The four-window native-versus-pooled-affine diagnostic (SHA `2dea49fa…b1ef`) closes that numerical question at the tested scope. Runtime and cached features are identical and pointwise head wiring is exact. Native-versus-ideal trailing-window Cl-prime RMS differences are at most 0.00194, versus native truth errors of 0.12561/0.08839/0.06647 on the three rotating train windows. No optimizer, save, validation, frozen data, or PPO was involved. This is sufficient to stop treating default-TF32/native reduction order as the main explanation for FC-E019, but it does not prove a unique learning or data cause.

FC-P010 is the completed CPU-only tail-amplitude mechanism diagnostic (result SHA `d69033fd…8f94`). It changed only the rear-Cl affine coefficients in memory and saved no model. Across the four held phases, free-AR trailing-window RMS MAE improved by 1.45--7.34%, while H1 worsened by 13.26--13.58% in three phases and improved by 2.08% in one; the all-train fit improved free-AR by 9.02% and worsened H1 by 10.51%. All five fixed-budget LBFGS fits reached 200 iterations with final gradients above tolerance. This is a non-converged, train-only multi-domain tradeoff, not an FC-E admission result; no candidate, formal evaluation, validation/frozen access, GPU, or PPO was produced.

FC-P011 is the approved two-arm decoder-scope comparison from the same P009 parent. Both arms consumed the exact same 1368-window train order once with batch size one, fixed loss/optimizer/seed and no validation selection. Arm A trained only the rear-Cl output row/bias; Arm B additionally trained the existing final decoder hidden linear layer, so only A-to-B supports a scope attribution. Both terminal receipts report 1368 optimizer steps, guard exit zero, fresh official reload and exact allowed-tensor confinement. A completion/result/model/state SHA are `6d3ecb91…b149`/`1c0bb91b…8960`/`cbfd0c7b…5ed1`/`2a4f4fb9…a2ea`; B are `499b3b6c…c686`/`d05f1d07…873e`/`5102e83e…00d8`/`c3c8c92e…ae6b`.

The predeclared six-window train-only diagnostics show tradeoffs rather than admission: from step 0 to 1368, A mean free-AR rear-Cl MAE changed `0.0691683→0.0681227` and free-AR tail-RMS error `0.0636410→0.0626839`, while true-state H1 rear-Cl MAE and H1 tail-RMS worsened `0.0339308→0.0350658` and `0.0208365→0.0228801`. B changed the same free-AR metrics to `0.0737654` and `0.0657843` (worse), while free-AR field relative-L2 improved `0.1121373→0.1068521` and H1 tail-RMS improved `0.0208365→0.0194299`. These are fixed train diagnostics, not validation or checkpoint selection.

Both unchanged formal suites subsequently completed and failed. Arm A receipt SHA `256a65c7…52a07` records validation10 delta-Cd PASS but force-window joint pass only 2/6; rotating rear-Cl-prime RMS errors are `0.070312/0.122247/0.068076/0.083214`. Arm B receipt SHA `e6c0a171…cebcc` fails validation10 delta-Cd at `0.024318>0.023` and also passes only the two zero branches in the window; its rotating errors are `0.062584/0.119208/0.075209/0.076236`. B passes all six mean-Cl checks but only three Cd checks, so neither arm satisfies joint admission. These results reject both FC-P011 arms for PPO; they do not establish a unique gradient cause. A bounded train-only gradient decomposition may inspect field-versus-weighted-force scale and alignment on the fixed six windows, but it is a diagnostic rather than training or a new threshold.

The canonical PPO reward-startup compatibility patch is commit `962c165`. It reuses one strict raw-force history reader for direct and surrogate paths, preserves raw float64 values, binds canonical HDF/config/manifests and starts at the absolute CFD restart with 62 causal points already window-ready. The 32-test implementation suite plus independent Root/SOTA reruns passed. This is software compatibility evidence only; it neither authorizes PPO nor changes any force, field or closed-loop acceptance threshold.

FC-E021 records the completed FC-P012 train-only gradient diagnostic. Its result/completion SHAs are `4142cdc5c67ff04d50f2921887e01034a9ea7d09a2865ac286b52d1c911d814d`/`86f9d93178e32a9e2769444bb327f9eaf494b290c77590efff1b1726d204b26c`. For both the P009 parent and P011B terminal, none of the five nonzero-action-history windows has hidden-group field-to-weighted-force norm ratio above 10, and none has cosine below -0.2. The separately reported zero window has ratio/cosine `10.4045/-0.0983` for P009 and `4.5051/-0.1605` for P011B. The direct-total/component-sum relative residual range `3.09e-5–9.84e-5` is observational without a predeclared equivalence tolerance. Model tensors are unchanged and no optimizer, save, validation, frozen access or PPO occurred. This result does not support a loss-weight sweep and is not an admission experiment.

## 2026-10-06 — P026 terminal tools engineering integration

The independently reviewed K1/K4 terminal candidate auditor and official CPU dual-reload verifier are integrated as preparation for actual training completion. The auditor checks retained execution identity, actual 171-step AdamW state, 1,368-window order, causal history, frozen tensors and bound data/source bytes; the separate verifier exercises the production role loader against externally pinned evidence. Canonical tiny CPU tests passed 38/38 in 1.89 seconds with CUDA hidden and exclusive temporary outputs. See `docs/FC_P026_TERMINAL_TOOLS_REVIEW_20261006.md` for exact hashes, independent review and limits. K1 was still running during this integration; no actual candidate audit, archive/HDF scan, full-size reload, new training or formal evaluation was executed. This is an engineering entry only, with no scientific CSV result, admission, PPO authorization or threshold change.

## 2026-10-06 — P026 formal receipt-schema engineering integration

The separately executed formal runner now validates the actual P026 terminal-audit and CPU-reload receipt schemas in addition to their externally approved SHA values. It checks exact arm, 171/1368/8 counts, unit/invocation, seven-file map with literal `candidate/` prefix normalization, saved/reloaded tensor identity, audit-to-reload binding and explicit no-admission/CPU-only flags. Root and implementation each passed 82 staged CPU tests; all numerical commands and source-chain functions remain unchanged, and the frozen numerical tree was not modified. See `docs/FC_P026_FORMAL_RECEIPT_SCHEMA_REVIEW_20261006.md`. Only software fixtures were read; no actual candidate audit, model/HDF operation, GPU/formal run or scientific CSV result accompanies this preparation.

## 2026-10-06 — P026 K1 bounded clean-cache maintenance

Lead executed one reviewed exact44 train-file same-descriptor SHA256/clean-cache-advice pass; no data writes, global cache clearing, model operation or restart. Independent JSONL review verified all44 digests/stat identities,90 ordered records,7.45135s completion and unchanged running K1 invocation b3759e7e. Receipt SHA792d0402237346b6f22b114f895f68e0bae9e3210c99576d2657b55247399537; pass minimum free21.252487GiB/available106.689064GiB, overall watcher minimum free20.936733GiB at869 samples. Observed free21.256115→26.729771GiB is not attributed exclusively to advice given concurrent activity. The existing dual20 floor, immutable training protocol and all scientific gates remain unchanged. See `docs/FC_P026_K1_CACHE_ADVICE_OPERATION_20261006.md`. This operational entry is not a scientific CSV result and does not authorize another pass, PPO or admission.

## 2026-10-06 — P026 cache receipt-name engineering amendment

Reviewed helper94d3c43b adds only explicit fixed r1/r2/r3 receipt-name choices under the same K1 output, with r1 default and exclusive creation. All original44-file/hash/stat/memory/deadline checks remain unchanged; no scheduling or automatic pass. Implementation and Lead each passed11 software-fixture tests. See `docs/FC_P026_CACHE_RECEIPT_AMENDMENT_20261006.md`. Lead separately authorized r2 for Lead execution after canonical hash confirmation; this entry records no r2 execution or success, and r3 remains unauthorized. Original r1 source/proof remains preserved; no scientific CSV result or protocol change.

## 2026-10-06 — P026 K1 separately authorized r2 cache maintenance completed

Lead executed r2 once,8.66831s/exit0. Independent read-only review verified90 rows/44 exact approved train-file digests and unchanged stat identities; receipt SHA c8f3292ce93231e2c7ef26b000a2cb51f579047f0fbb4dbfe4df8fc3871f313a. Pass minima free21.717365GiB/available106.596531GiB; overall watcher minima20.936733/105.834663GiB at1412 samples. Same K1 invocation b3759e7e,MainPID598666,containered4f0ad7a426 remained running. No model/protocol/data changes or second execution by reviewer; r1 proof unchanged and r3 absent/unauthorized. Operation report appended with actual evidence; no scientific CSV result, admission or automatic future pass.

## 2026-10-06 — P026 HydroGym explicit-history runtime CPU engineering integration (reviewed)

The reviewed six-file runtime/readiness change is present in the canonical worktree after Root verified all six production SHAs and reran the canonical tests; only the final commit remains. It adds explicit, candidate-bound K1/K4 history state to the P026 surrogate path while preserving the legacy/direct-CFD branches, the successful CFD-only baseline, and all numerical gates. MPC remains out of scope. Fresh canonical CPU tests were deliberately split into separate processes to prevent fake-HydroGym module-cache contamination: the implementation run passed 15 runtime tests and 60 readiness plus existing legacy tests; Root independently reran the same groups with 15 passes in 1.32 seconds and 60 passes in 1.21 seconds. The one staged-only layout assertion was deliberately omitted after canonicalization; no scientific behavior assertion was removed.

A separate retained user unit `p026-hydrogym-actual-core-canonical-20261006.service` then passed one actual HydroGym `PDEBase`/`FlowEnv` lifecycle test in 0.022 seconds under a 1 GiB/no-swap/2-CPU cap. Root independently verified invocation `3f7675223b9244518ccbbb812ab56253`, terminal success, and equal before/after source-list SHA `d185a68f…6131`. The fixture uses a tiny mock K4 network and synthetic temporary HDF/raw-force data; it does not load an official PhysicsNeMo model, run CFD/PPO/GPU, or establish scientific readiness. The exact log, unit evidence and tested source copies are archived read-only at `artifacts/fcp026_hydrogym_actual_core_canonical_cpu_20261006/`; manifest SHA is `b0ff32a762b13736da9f22a5f2ebeb4b476fc7ac6dc01e5576a94bd29a49674d`. This engineering result does not alter the running K1 training, admission thresholds, or PPO authorization.

## 2026-10-06 — P026 K1 actual terminal integrity and official CPU reload

The original K1 invocation `b3759e7e1acc4de7a1aa9f6e8d38de9a` completed 1,368 windows and 171 updates, retained success/exit0/PID0. Independently reviewed actual candidate-audit SHA is `fa26bf47b6eb448e36046973a479e771b2d37eb605d9630b9022b392ce30d944`; actual official CPU dual-reload receipt SHA is `980698fd335a7a536e358ced26b9f69236e8d101d1f038a46eaf6a0c84f67028`. Seven candidate file identities, both tensor digests, protocol and runtime source bindings agree. Actual pinned-image CPU container `a6529eb902e2d93b283a61b8091491002bbc36f66a0e8471b27e37f80e3b88c4` exited0/noOOM under 8GiB/2CPU/noGPU/network-none/read-only-input restrictions. Training minimum CUDA free was20.733688GiB; host/internal free minima were20.936733/20.740211GiB.

Both earlier audit failures remain recorded: strict JSON clip-norm spelling and Python float-sum roundoff. Reviewed narrowly scoped compatibility corrections preserved original approval/candidate/trainer/data hashes and all scientific gates; failed attempts were not relabelled as successes. See `docs/FC_P026_K1_TERMINAL_REVIEW_20261006.md` and immutable proof archive `artifacts/fcp026_k1_terminal_review_20261006/` (manifestSHA `3cc28047d08d93dceb8a42beacdc031109b915821372f1303487c14a79e822de`). HydroGym runtime integration `23711cc` and runtime-image preparation `e542ff1` are now committed, superseding the historical pending-commit note above. No formal-evaluation launch/result, accepted surrogate or PPO is claimed here; formal execution needs its separate approval and actual execution identity. No scientific CSV row or threshold change accompanies this milestone.

## 2026-10-06 — FC-P026 K1 original formal evaluation started

The Lead-approved unchanged formal suite is actually running under Main user unit `fluid-control-fcp026-k1-formal-20261006.service`, invocation `c039836ab63246ff8772dad66e1b46e5`. At the recorded observation it was in validation10, with actual container `4a9cea6bd3a181840855db1ee92934c783030a279eacf840f1123d6fca2543d2` using official image `b40d5888…a22e`, GPU0 and the fixed 20 GiB guard. Approval SHA is `2d15c323…0000`; the immutable 411-file formal source receipt, runner and terminal runtime manifest are `ff8b742a…fe24`, `03c5862e…c0f3` and `277ec97a…1f70`. This entry records only launch and point-in-time provenance. No terminal receipt, gate result, admission or PPO authorization exists yet; Root retains monitoring/UI ownership.
