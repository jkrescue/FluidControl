# Experiment ledger

## P064-B original formal evaluation — R2 engineering start, no numerical outcome

Actual unit `fluid-control-p064-b-formal-r2-20261006.service`, invocation `01806bdc150841f7b9efd04360a441f2`, started12:41:45 UTC active/running. Approval SHA `cd58fd47e991ec6dac200bd82d414347f72b778ea78415377427e430dfd47478`; output `artifacts/fcp064_arm_b_formal_20261006`. Precision stage completed; validation10 entered CUDA/PhysicsNeMo initialization, not yet an evaluation result. R1 `6d7cea57ea7740c991223448dfc45e0a` exited2 from wrong cwd before opening the script/container/GPU; R2 preserves identical approval/source with explicit correct cwd/absolute paths. Resource correction after actual-source inspection: outer .06 guard is startup accounting, NOT an enforced allocator cap; evaluator/forcewindow actually set torch allocator fraction .15 (approximately18.25GiB). Available50/22GiB and72GiB/noSwap remain; Lead explicitly authorized continuing the same R2 with actual .15 and observed Available109.23GiB, without mutation/restart. Data, numerical commands,75tests and scientific thresholds remain unchanged. No numerical CSV row, no training, no inferred admission or replacement of historical H100 failures. Original launch evidence: `docs/FC_P064_ARM_B_FORMAL_LAUNCH_20261006.md` SHA `ac3dc4b3262a943128277b4acf3f53e014dfd3d2a604f99968d3a3ce36c85cda`; original approval preserved and resource supplement recorded separately.

## P064-B fixed b01 replication — actual start, no outcome yet

One approved CPU paired800 run started 2026-10-06 12:37:56 UTC: unit `fluid-control-p064-b-projected-ppo-b01-long-cfd-20261006.service`, invocation `432c12de32b0444d9a6f6626e12616d1`, initial observed14/800 cycles. Same actual B policy and identity VecNormalize as FC-E069; fixed historical b01 restart130 through210, primary(150,210], inclusive companion[150,210], unchanged six windows/physical thresholds/projection/single slew filter. Source SHA `4b8fa43f8ac020521ae8cde1047b6ec2f80d35ec0512bc7606d1050835010619`; approval SHA `3a19e326ebfd37df24060b8b5717b5af08b405ed63165e4034974aeb7b0abcb6`. All20 restart/source metadata file identities and final serialized consumer preflight passed;6 synthetic CPU tests independently passed. Controller8GiB/noSwap4CPU, two solver8GiB/noSwap2CPU, Available50/22GiB,3600s inner3750s outer,120s stop. No new model training, no online FNO, no automatic retry or scientific admission. No scientific CSV row before terminal independent review.

## FC-E069 — P064-B fresh-policy projected b00 paired800 terminal

Actual unit `fluid-control-p064-b-projected-ppo-long-cfd-20261006.service`, invocation `3a078c62ed9e4f7b8876f0f166bdb510`, completed800 cycles with exit0. Approval `5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99`; executed driver83e08d66; result `8b31091d5e69edfbfd5ea78ba99dd7709623e6eeb0bd4f13c54e984b7fc28907`. Independent report `docs/P064_B_PROJECTED_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `7b453d9c529d9d5c52988608c89d61050204510d41549cbb2be620fcdcebbe04` verifies3200 raw hashes,1600 clean solver logs,800 projection/filter steps, all six fixed windows and cleanup.

Primary(168,228],12000 points: drag reduction3.8952838833%, centered rearCl RMS ratio.815623043405, mean-bias ratio.011378146878; unchanged2%/1.05/.10 criteria met. First6.2 bias.135464513 fails10%; full80 peak rearCl exceeds zero despite primary peak reduction. All six windows are appended to CSV without discarding negative results. New policyf7644639 differs from old successful policy; paired zero time/Cd/Cl arrays match exactly. Primary drag gain over old trial is only.0033039437 percentage points, not evidence of significant improvement. This b00 trajectory entered B training, so this is in-sample confirmation, not independent generalization. No formal/H100 admission or replacement of old policy. Same-new-policy b01 preparation is the next bounded check, not yet an outcome.

Operational complement: independent official B CPU dual reload succeeded in R3 after preserved R1 Warp-cache and R2 PhysicsNeMo-cache read-only failures. Model/source/image unchanged; root-cache tmpfs and receipt0444 publication are engineering repairs only. Proofs: `artifacts/p064_arm_b_official_cpu_proof_20261006_r3/candidate_audit.json` SHA976e020151e136a2563fbdeb9470776209f9f8df65fbbad0e3a0ba1e9051c36d; `official_cpu_reload.json` SHA497e1164cf534263d2cfe612793fd43263b0904cc27cda727f300d8220fad87c. No forward/optimizer/GPU. See `docs/P064_B_OFFICIAL_CPU_PROOF_REVIEW_20261006.md`.

## FC-P064 — B PPO terminal verified; actual projected paired800 CFD running

The fresh B-candidate PPO completed on unit/invocation `fluid-control-p064-b-ppo-32768-20261006.service` / `f613395cbf1140549dc60e7b046e0f6b`, PID0/exit0. Independent review verified32768 timesteps,256 PPO updates,512 optimizer records, exact24-reset/H5/69-observation/62-reward protocol,74source/192runtime closure and unchanged FNO tensors. Result/policy/Vec SHA are `3c70e21327baae98f682fc0982ca3c3910cf6d1902f3d62175f980fd685817b3`, `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`, and `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`. Review `docs/P064_B_PPO_TERMINAL_REVIEW_20261006.md` SHA `0cc1494286f85b930b43b1a11713ae6ac3719d8ff599cfa0780a8d9fe71b0d65`; terminal training is not CFD benefit or formal admission.

Lead then launched the separately approved paired800 physical trial as `fluid-control-p064-b-projected-ppo-long-cfd-20261006.service`, invocation `3a078c62ed9e4f7b8876f0f166bdb510`, output `artifacts/p064_b_projected_ppo_long_cfd_20261006`. Approval SHA is `5fc8ab36e69e7e6ea27ed7c4d60ae207bc67be3c9513e89800cedccf46970a99`. At this record point the same handle is active/running; no terminal drag/lift metric or CSV row is claimed. A preceding CLI probe omitted `--execute` and was rejected before the execution body, so it launched no CFD and is not a scientific attempt. The preserved prior successful policy is not overwritten.

## FC-P064 — Actual B-candidate fresh PPO launch; CFD remains unbound

After independent source/import/proof review, Lead launched `fluid-control-p064-b-ppo-32768-20261006.service`, invocation `f613395cbf1140549dc60e7b046e0f6b`, with output `artifacts/p064_b_diverse_h5_32768_ppo_20261006/payload`. Approval `docs/P064_B_PPO_APPROVAL_20261006.json` SHA `ae327fee310bad562aceef35029595d20c9a3421d5d5be3dc3b68bb82649e9fe` binds the actual B manifest `92766915cb11ca75d313608a5f75e61a218371dcc789f5b44725f0a8260e7891`, training result `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`, and independent engineering receipt `d78f87d041fd907c50ad6b2ca8880498bf8f5ad80e5b6916270ec585105b2915`.

The run retains the previously successful 32768-step/24-reset/H5/69-observation/62-reward/seed PPO protocol but starts a fresh policy against the explicit P064-B runtime candidate. It does not replace the preserved successful policy. Launch is not terminal training success, CFD benefit, or formal admission. The paired800 CFD file remains preparation-only with null future policy/result hashes; it requires actual PPO terminal artifacts, independent review, and a separate execution approval. No scientific CSV row is added while PPO is running.

## FC-P064 / FC-E068 — A/B terminal development comparison and retention tradeoff

Both32-update/256-window arms completed independently verified engineering checks. B actualunit450ef57c25c14ec38e722cbd597ffb50 normalexit0; CPUreview f28d193f5a1a40ddb0f616cab8903679 under8GiB/noSwap/CPU1/CUDAhidden passed434sources, exactB64replacement schedule0..700,32×8records and28finiteAdamstates allstep32, frozenbias2 and copiedflow bytes. Receipt SHA `d78f87d041fd907c50ad6b2ca8880498bf8f5ad80e5b6916270ec585105b2915`, trainingresult `9167e8d811f64cf001cc87bfd45d9ed2d48f5c588a19b951f7be2c826637b980`; training sampledAvailable106.103218GiB/peak3521921024B. Officialreload is producer per-role proof, independent tensor inspection is not model construction.

A development inv68ef4125cc5b494dad9b52a320c143c2, approvale42e28f1b4651ccb13d30c2508e9ef65e6a76c3a24d0b0c7517bc2ed2c36eeb6; B inv6613b11d351f48fba27ba33fdff54534, approval5e05d12d415959523bcdfd473990dacf61cef902aa1a1874ded8c79e983db590. Bothnormalexit0; resultsA c8b0242658a101120603514e6d2e5076c827c518965c92470810fe9693840fe6 / B47e7d4c6931fadc62730790500bb9a8a07f44792d1bef82c900d10c36f9a8665. Independent16NPZ/80endpoint×allphase/origin/pooled arithmetic each,440sources192runtime8inputs; minAvailable120453775360/121369776128B. Import-only repairedworker0d6f1ac8981d85d092c0e970dcbf1597fa2074d749a75b56f3b82d0cb3bf5af0 preserves original numericalbytes; failed initial PENDING preparation retained, no failedGPU evaluation hidden.

Predeclared BvsA pooledH1 rearCl .156116880→.138998317 andCd .039952166→.038065374 strictly decrease10.9652%/4.72263%. H5 rearCl improves6.8593%, Cd only.0956%; B H5Cd remains.7884% worse thanK1 andb01 H5Cd .6848% worse thanA. H1stillworse thanpersistence. Flowarrays/truth/actions/time K1/A/B exactlyequal. Original6trainwindow H1/AR objective B .00348834/.00880538→.00397686/.00894620 worsens; true-state tailRMS improves while ARtailRMS worsens. This is local development support plus retention cost, not comprehensive repair/admission. Fullalllead/phase tables in `docs/P064_AB_DEVELOPMENT_COMPARISON_REVIEW_20261006.md` SHA2fa7e5d71e4b163bfff2261ec2b38ef43c1c5aa75acc7cea7d47c1f3296c225a. FC-E068 CSV records measured A/B sameprotocol metrics, no arbitrary PASS thresholds.

Lead next direction: separately reviewed exploratory B-based fresh32768PPO thenpaired800realCFD with unchangedprojection/filter/physicalcriteria. Bindactualnewmodel/policy; preserveoldsuccessfulpolicy. This nextchain is not yet executed by this milestone. No10% relaxation/H100override/posthocretentiongate.

## FC-P064 — A terminal engineering verified; independent B training running

A R2 invocation `50de1d8b43ce42ac923752fad76ca4d9` completed normal exit0. Result SHA `03951fee3c1ba66ae48d451fe35aeb7735f092e40ef0b76deff90738c9f8e6a1`, manifest SHA `049d29df94a410ba3dbcedf52484447e828cde7787593ad837cbe38be7abd778`. Independent bounded CPU checker invocation `8b65dab971bf48ce92268e8d6056d570` exited0 under8GiB/noSwap/CPU1/CUDAhidden; receipt `artifacts/fcp064_arm_a_terminal_cpu_review_20261006/receipt.json` SHA `2d51c4f84b53fa9b29748c77200cac279ab5a3a4b8ebb5aa6309b4ae9a45f52a`. Verified434sources,256scheduled journal windows,32updates×8records, finite statistics/recomputed means/clip, source/norm identities,28Adam states allstep32, frozenbias2 exact and originalflow bytes. Training sampled minimumAvailable106.844551GiB, cgrouppeak9884188672B; reviewerMemoryPeak unavailable, notzero. Producer official per-role fresh reload evidence is distinct from independent weights-only tensor checks, not an independent model forward/reload. Report `docs/FC_P064_ARM_A_TERMINAL_ENGINEERING_REVIEW_20261006.md` records boundaries.

Root separately launched B unit `fluid-control-fcp064-aero-arm-b-20261006.service`, invocation `450ef57c25c14ec38e722cbd597ffb50`; independently observedPID528728 active/running. Exact approval SHA `a1e79d108f5067027742f08f3e04b2d73cb059286e2bd433e53f4e0d51247b29`. Same originalK1/freshAdam/32updates256windows, with64mechanical b00 replacements; B does not continue A. No B terminal or accuracy claim, no CSV scientific row.

The first A development-PENDING preparation failed before publication because the434training closure lacks the historical selector module. No evaluation/GPU occurred; minimal source-bound import correction is separately reviewed and original failure retained. Existing A R1 CUDA startup failure also remains. Engineering completion is not scientific admission or evidence that augmentation improves generalization. Same fixed development comparison remains prospective; originalH100FAIL, three-phase physical results and10% constraint are unchanged.

## FC-P064 — A R1 启动失败、精确缓存建议完成、A R2 已越过首批更新

Canonical P064 source/tests/plan/approval files are committed as `977a027d51c3ae03ade858deae0030d194a79eb8`. A R1 (`fluid-control-fcp064-aero-arm-a-20261006.service`, invocation `cfc40285f8ec49fdba9a99defe2960cc`) exited1 at the first CUDA transfer with **0 optimizer updates and 0 consumed windows**; no candidate was written. Its approval SHA is `c1872be6e816a2f058a111084594882ebfd9d3f0222e642ba58bfba71487ee1e`. This is an engineering startup failure, not a scientific A-arm outcome.

The separately approved exact-53-file cache-advice unit (invocation `1364cefc6aa14d75a6574dd25931f154`) exited0 without modifying file bytes, permissions, sysctls or global cache policy. A same-source/same-argv R2 then started as `fluid-control-fcp064-aero-arm-a-r2-20261006.service`, invocation `50de1d8b43ce42ac923752fad76ca4d9`, approval SHA `773daa7329a46930b532586d884502d0c4170ea2c3a689ba73b2fb3d5da8b568`. The preserved milestone observation is at least 72/256 windows and 9/32 updates on the same live handle. This only establishes recovery past CUDA initialization and real optimizer progress; it does not establish cache causality, terminal success, improved precision, or any scientific metric. No scientific CSV row is added while R2 is running.

Arm B pending SHA `d55a43ae5a68a23296508a555896bb2d8a0c980da558fff0ce13f1974e9f7cba` contains the complete Root-tested no-GPU argv, but remains preparation-only and unauthorized. B must start independently from the same K1 parent with fresh Adam, never from A, and requires a separate post-A approval. Fixed budgets remain 256 windows / 32 updates per arm; no early stopping or best-checkpoint selection.

## FC-E067 — Independent terminal baseline metrics retained without admission

Actual K1 result `9ea3e0e781e76265bbc65ea52d6fec92ebe5b5cb7a93d91c3c9addb454f7c4de`, report `docs/P064_K1_DEVELOPMENT_TERMINAL_REVIEW_20261006.md` SHA `a15a6699cf20d0d3b76a8569357bd3580e1bb11fe232207164bff5cc08ab2fef`. Samea39f8106 PID0/exit0, supervisorerrornull. Independent reviewer rehashed16NPZ,378source/192runtime/5inputs and recomputed all80 field/force/persistence endpoints plus phase/pooled summaries. Largest field-sum reduction-order absolute difference2.614e-8, agreement relative1e-12/absolute1e-10; force metrics agree. Runtime12GiB/noSwap, minimumAvailable120994258944B; unchangedmodel/zerooptimizer source/runtime evidence, no independent model reload.

| Phase | Lead | velocity relativeL2 | pressure relativeL2 | rearCl MAE / persistence | totalCd MAE / persistence |
|---|---|---:|---:|---:|---:|
| b01 | H1 | .010296 | .031852 | .167390 / .080909 | .046173 / .020139 |
| b01 | H5 | .043053 | .136114 | .189925 / .400050 | .033153 / .103384 |
| b03 | H1 | .010336 | .032233 | .150352 / .099758 | .033088 / .020498 |
| b03 | H5 | .043027 | .137900 | .169775 / .479915 | .027891 / .098731 |
| pooled16 | H1 | .010316 | .032042 | .158871 / .090334 | .039630 / .020318 |
| pooled16 | H5 | .043040 | .137006 | .179850 / .439982 | .030522 / .101058 |

Exact H1/H5 perphase+pooled model/persistence values are preserved in CSV with coefficient units for force MAE, ratio units for field relativeL2, no invented force percentages. Field pooling sums physical masked SSE/reference before sqrt, not average per-case L2. Persistence holds each origin's initial state/force. Both phases' mean H1 force errors exceed persistence before recurrent field accumulation; H5 beats the stale baseline on average but retains material errors. No causal state-coverage explanation or scientific admission follows. Already-opened development, no new CFD, no altered10% physical constraint or H100 label; same fixed panel/precision for prospective A/B comparison only after separate approval.

## FC-E067 — K1 fixed development baseline executed; independent metrics review pending

Lead launched unit `fluid-control-p064-k1-development-h1-h5-20261006.service`, invocation `a39f8106aa9b4559b1fb1587e7a8e38c`, initialPID425486. Same-handle observation now PID0/active-exited/ExecMainStatus0. Exact approval SHA `05891a360ffaa17d61e6854a8dce632d818a3ab1ec5b4a616244d4abb6a85b79`; executed source SHA `e86ef8ea3c55be63287b0f9d5e9e0cf0034e7fc6959df25e53e49c0fe23d90a8`. Canonical promotion preserves these worker bytes; only test import-path portability changed,6CPUtests PASS0.06s. Output `artifacts/p064_k1_development_h1_h5_20261006`.

Fixed panel: already-opened b01/b03 controlled development,8 origins each,5 leads each,16starts/80endpoints. Exact original K1manifest7adca21e/config07e55/normf1b460, existing validated dual loaderd769; no dependence on unfinished P064 A/B loader. Same FC-E063 numerical rollout and metric core, officialHDF5Reader, pre-load6GiB cap and post-loadhighest/noTF32 override. Requested12GiB/noSwap/GPU0/CPU1, hostAvailable50startup/22runtime,600s supervisor/630s outer. All inputs/source/runtime bound in approval; no new model optimization, CFD, or checkpoint selection. Array/metric/resource verification by an independent reviewer remains pending, so no scientific CSV rows or accuracy certification are recorded here. Original formal H100 failure remains unchanged.

## FC-E065 — b00 whole-train conversion and dedicated view verified

Originalf5ee31 exited0 at10:38:17UTC after1240s. ResultSHA `f24f2fbc8b283c0781b2a01189d291b43da0203e9ede28208ec575173bbe0bd9`; HDF `45041e79e70043838763e5dd8da0fe9c02dba4cfe6df01356f8dcd1389e32d8f`. Independent1614 source hashes,801 official-reader/HDF finite arrays, actual progress time/action/four-force endpoints and48 retained packets match.753 temporary new packets were deleted, so no independent post-hoc NPZ comparison/resampling claim.16 exporters exit0/OOMfalse/absent; root-owned scratch retained.20571 samples minimumAvailable121431912448B, actual12GiB/noSwap/CPU1, peak2883301376B.

Exclusive `artifacts/b00_controlled_train_dataset_view_20261006` retains source inode permissions and original norm bytes. Manifest97a82e2a…929a5f is train-only with actualCFD endpoint labels; no normalization refit/development leakage. Actual official DataPipe yields701 H100 starts, exact0→100 and700→800 field/101action/100force targets. This hardlink view requires read-only use, not an OS ROmount; source SHA checked afterward. CPU auditR2 166133d841f24be2a409caaec40e0b89 passed4.37s after preserved audit-only inventory-schema failure; conversion not rerun. ReportSHA `c265cc7254d1cd423626bce84d0b064e90a2686bff4c651021017e09952657eb`, viewreceipt41e75928…d1ef9e. Engineering milestone only; no new inference/training/CFD or scientific admission. Candidate A/B still requires separate approval.

## FC-E066 — Independent development conversion terminal, no scientific metrics

Actual invocation `8ac389027a954b79bc1d766f89029a6b` completed PID0/exit0. Approval `c82e4a865d0d6e0927faee026baa84ebca2fcabb0707daa00f988174349dd5a3`; result `1da29262fa8cbfa7c41687439ad34cc1967ce8c7d9faa35c435668705ca9351f`; independent report `docs/DEVELOPMENT_PHASE_H5_CONVERSION_TERMINAL_REVIEW_20261006.md` SHA `aa1417d9d6d05d282e2b00606985c31b9070bdb31b45378d6dba8dcc746e1c94`. Independently rehashed6source/approval/selection,216originalfiles,16HDF and compared96 sampled arrays plus actual force/action endpoints. Max nominal-time difference6.103515630684342e-6, preserved VTK FP32 rounding. Two exporter containers exited0/noOOM and exact CIDs are absent; runtime resource span154.883s, minimumAvailable121511297024B; systemd12GiB/swap0/CPU1/15min verified.

Scope is already-opened b01/b03 development,8 fixed starts each,6frames/start,80 next-step endpoints. No new CFD/model/optimizer, no accuracy measurements or CSV scientific rows. Root-owned export scratch retained intentionally. Progress16/16 retained its old CONVERTING label; actualunit/finalreceipt establish terminal status. Original physical criteria and H100 rejection are unchanged; these inputs enable later separately approved same-protocol development evaluation, not checkpoint selection on an untouched test.

## FC-E066 — Actual start: fixed b01/b03 controlled development conversion

Root launched `fluid-control-development-b01-b03-h5-conversion-20261006.service` at2026-10-06 10:27:34UTC, invocation `8ac389027a954b79bc1d766f89029a6b`, PID379249, independently observedrunning. This is CPU conversion of already saved U/p, not new CFD, model inference or training. Fixed starts0,100,…700 plus5followingframes at b01/base130 and b03/base144 select48controlledframes each; intended96frames/16six-frameHDF/80endpoints. These opened development phases cannot be relabelled independent finaltests.

Approval SHA `c82e4a865d0d6e0927faee026baa84ebca2fcabb0707daa00f988174349dd5a3`; immutable driverSHA `cb7fdb83e620903be89c85f38286a0be0afbb4b95792adb7b7179065f85e8a9c`, selectorSHA `26c085d1da91c3452fc514cda9c06332afe75e600be7770f8d476c1029986cb0`. Eachphase selectedsourceinventory has108files, including96U/p plus12constant/system. Originalsources are read-only; R2 converter/helper source remainsunchanged. New two-file immutable closure avoids duplicate large source trees.9fixturetests independentlypassed beforelaunch; canonicalpromotion9PASS0.09s afterlaunch, exactproductionbytes retained. Tests prove engineering contracts only.

Resources:12GiB/noSwap/CPU1,900s,Available50startup/22runtime/20reserve,serial4GiBfoamToVTK exporters; retainroot-ownedscratch. Output `artifacts/development_b01_b03_controlled_h5_conversion_20261006` is exclusive. No terminal metrics entered in CSV. A different reviewer will verify actualHDF/arrays/sourceposthash/cleanup atcompletion; this entry does not claim conversion success or scientific admission. b00 FC-E065 remains a separate running train-data conversion.

## FC-E065 — b00 controlled train trajectory conversion actually running

Launched by Lead at10:17:37UTC, unit `fluid-control-b00-controlled-train-conversion-20261006.service`, invocation `f5ee31dd92624f0980a509084de9c756`. Independent same-handle observation: PID348063 active/running,59/801 written frames; official-reader terminal verification remains false. Approval SHA `4fa7192e13bf7ad3a141bffb483710e2400fd8ee243caa60a6e67ab695927686`; immutable executed adapter SHA `f96c882a90e7ecaf4a2f8a5fc327764909ab42b4e6bbb11cde8e99205488b075`. Output `artifacts/b00_controlled_train_conversion_20261006`; exact1602 U/p plus12 metadata files are SHA-bound, original source read-only.48 controlled cached frames are checked against old receipt-bound HDF arrays/actions/forces;753 remaining frames use unchanged official Curator sampler in16 batches, then one801-frame physical HDF is roundtripped through the official HDF5Reader.

Whole b00 is train-only, no random-frame split, no zero trajectory addition, no normalization refit. Actual endpoint coefficient labels differ in provenance from older interpolated HDF labels. This is CPU data engineering, not FNO/PPO training, new CFD, model repair, or admission. Source tests13PASS0.54s before launch; these synthetic checks are not proof of actual conversion completion. Requested12GiB/noSwap/CPU1 and3600s, hostAvailable50startup/22runtime; retained batch exports avoid invalid unprivileged deletion of root-owned VTK.10GiB planning disk budget with20GiB startup headroom. No scientific CSV metrics while running; failures and prior results remain unchanged.

## FC-E064 — Existing train-cache action coverage, descriptive not causal

One approved JSON-only execution completed at10:07:34UTC under unit `fluid-control-train-cache-coverage-20261006.service` / invocation `08658ea0d6a5435a847df4de1223d21f`, PID0/exit0. Source SHA `1fbd532caae13240f62a45922e499f02cf6402ad1c860828cd1a100350288317` unchanged on canonical promotion; approval SHA `87ede7d20203c2cb5bb77e5f9af96c1f1d8b53c0355fec2180098fb47aa5fdc3`; result `artifacts/train_cache_coverage_20261006.json` SHA `34ec16db0540c0dabb2f44caf3240ae386893ea85bd28bc1436e8820212b039d`; independent report SHA `582f5d04ef7fcab10349a858cd4c618ddc98e44bcf9eb99da84de552d48a8754`.

All44 P027 origins51 and offsets1–5 retained. Independently recomputed44×5×3streams×5channels and all group summaries exactly; targets52–56/current51–55, persistence fixedforce51. TotalCd sums signed errors beforeabs. Actual source/three JSONmanifest/phase-map/cache hashes verified. New information is mechanical action-window coverage plus manifest duration, not another inference. Constant21=20base+1train8; reversing18=16train16+2train8; changing5=train8. Reversal means nonzero increment-direction change, not necessarily omega crossing zero. Group five-offset rearClMAE H1/AR/persistence: constant .03889449/.04534176/.34568305; reversing .02087019/.02324315/.35321556; changing .04321395/.06027287/.34027185. This does not support a simple reversal-causes-error story; family/state/phase confounding is explicit.

All cached elapsed starts are5.1D-U despite fulltrajectory durations80/20/12.8. Neither absence of dynamic actions nor adequate late controlled-state coverage follows. Old high/TF32 and stored interpolated HDF labels differ from current controlled no-TF32 replay; no matched causal attribution or OOD measurement. Exact maxstored delta=.1000061035 retains rounding, no clamping. MemoryMax1GiB/swap0/CPU1 verified; MemoryPeak unavailable, notzero.8canonicalCPUtests PASS0.03s (independent stage8PASS0.02s). Status `OFFLINE_DIAGNOSTIC_COMPLETE_NOT_ADMISSION`; no HDF/model/GPU access or optimization, no threshold change. CSV executed-source SHA identifies the staged executed bytes rather than inventing a prior Git commit. Future intervention remains separately approved.

## FC-E063 — Realized-action H1–H5 replay complete; controlled-state accuracy gap remains

Fixed K1, both saved b00 projected-PPO/zero branches, starts0,100,…700, leads1–5:16 origins/80 endpoints, no training or new CFD. Actual unit `fluid-control-projected-policy-h1-h5-inference-20261006.service`, invocation `627cb6b8b59a40aca3bb9159fb617eb3`, exited0. Approval `99f7c348afbd5b094018dd3d3cb0e0ee9b7bda50ab7e28bcc4e44b2a57f401fa`; immutable executed driver `94e583182b453d8200a91700f1e68a997e47a9033debf1a9e08ed37cf297841f`; result `247af0405d9e622f0b3b3b5dbc64e46c20890439fcd5216682b8d973957fd00d`; independent report SHA `197b385617420e5f0e9cb7c8dfb25d957e98f85ea93f280bb2d6c0effc898faa`.

Independent saved-array recomputation verifies all16 NPZ hashes, four force channels, physical masked SSE/reference/persistence,80 endpoints and separate eight-origin branch summaries. Controlled H1/H5 velocity relativeL2=.009955/.042127, rearCl MAE=.186710/.149926, totalCd MAE=.048653/.035954; respective force persistence=.088235/.423392 and .017126/.084684. Controlled H1 is worse on average than persistence for both forces; only3/8 rearCl and1/8 Cd origins improve. H5 improves against persistence at8/8 and7/8 but retains materially higher errors than zero-branch H5 velocity=.007050/rearCl=.010806/Cd=.007414. All controlled origin/lead force errors are retained in the report; no performance-based selection.

Status `DIAGNOSTIC_COMPLETE_NOT_ADMISSION`: retrospective realized-action conditional replay, not online FNO/MPC forecasting. 378source/192runtime/5input hashes verified, minimum sampledAvailable121518190592B,12GiB/noSwap,6GiB allocator; high/TF32 verified-load then highest/noTF32 inference, unchangedmodel/zerooptimizer. Canonical promotion after execution preserves exact driver bytes;7 CPU tests PASS0.59s (Root stage7PASS0.61s). CSV code_commit explicitly uses `executed-source-sha256:94e583…7841f` because the executed source was an immutable staged file, not a claimed historical Git commit. Original failed conversion and H100 rejection remain preserved. A separately approved train-only true-state force/action-history diagnostic is a hypothesis, not yet an executed fix or a new acceptance gate.

## FC-E062 — Fixed 96-frame saved-CFD conversion complete; inference not run

R1 invocation `45e422939f5a477a85f359cd60c5047e` failed at the first sampler view because protected-hardlink rules rejected `os.link` on root-owned0644 exported VTU bytes. It produced no successful sample, model load, optimizer step or CFD trajectory. The failure remains preserved. The separately reviewed R2 changed only that filesystem operation to source-SHA → `shutil.copy2` → destination-SHA equality; converter `304fece8b7c205dbd8182b9cdae24701fc917b102884406b54734f7e0d0cad68`, approval `da1fb69dc51cde9f955a6e9fe2a69dfd32afa00c488fb6ae04fe42579a77d9ed`, unit/invocation `fluid-control-project-policy-h1-h5-conversion-r2-20261006.service` / `60aaab28df8d46508bdcb483a2c63074`.

R2 exited0 and independently verified96 packets,16 six-frame official-reader HDFs and80 endpoints over fixed starts0,100,…700 in both saved branches. All216 source identities,16 HDF hashes, FP32 applied-action/four-force arrays and one common mask/grid identity match the sealed inputs. Result SHA `a22c3aa67509e9b3a342071398ae85da2ce4e07c74a3cbbd87e2a493c2b248bf`; review `docs/PROJECTED_POLICY_H1_H5_CONVERSION_R2_REVIEW_20261006.md` SHA `4172fb77b3e0afe774e3595537f3217e211102ef4fb7e53a3d5ae5e08b89781e`. Sampled VTK times retain float32 rounding (maximum nominal-grid deviation `6.103515630684342e-6`). Both export containers are absent; minimum of1192 MemAvailable observations was122120433664B.

This is an engineering conversion milestone only: model-loaded/optimizer/CFD/scientific-admission flags are all false. The planned replay conditions on future recorded applied actions unavailable to an online forecast. It neither measures K1 accuracy nor reverses H100 failure; separate frozen-source approval and actual inference remain required.

## FC-E061 — Verified b03 projected-policy physical confirmation complete

Actual invocation `47612677a9f64dfc968917fada5e9ba8` completed 800 paired CFD intervals, PID0 / exit0. Independent review rehashed all 3200 coefficient files and recomputed all six windows from four exact 16000-sample streams; maximum metric discrepancy 8.88e-16. Primary (164,224], 12000 samples per branch: drag reduction **3.8971402137%**, centered rear-Cl RMS ratio **0.8148587790**, mean-bias ratio **0.01688858081**. All original physical criteria pass without relaxation; all six windows pass for this trial. Historical inclusive [164,224] remains a separately labelled 12001-point companion. Original H100 FAIL and prior unsuccessful controllers are unchanged.

Result SHA `d4d755faf913d393ca1466ee74614c0fb11f33b8f662a76a1cb7e4de3dd17a2f`; independent report SHA `0d48a7e914ec82ad682d374e6531f2aab6a305aa81dad2dde6cd5c23dca48a0f`. All 800 projection/filter decisions checked; all 1600 solver logs clean, max Courant0.245351545. Source restart and bound code unchanged; owned containers absent. Their idle-process exit137 reflects intended cleanup, not failed CFD. Minimum MemAvailable121825087488 bytes; runtime1099.822s. This is actual frozen-policy online CFD feedback, not a new training run or a real-time guarantee. b00/b01/b03 are three observed phases of the same physical configuration, not statistically independent generalization; b03 fixed-action H5 exposure is disclosed. Original10% remains the criterion and20% only sensitivity.

### Preserved launch record

Started2026-10-06 09:30:16UTC; sameunit `fluid-control-exploratory-projected-32768-ppo-b03-long-cfd-20261006.service` / invocation `47612677a9f64dfc968917fada5e9ba8`, PID4013554 observedrunning. Fixedrestart144→224,800cycles,primary(164,224]12000points plusfiveotherpredeclaredwindows. Uniquechange fromb01 is phase/restart metadata; same frozenpolicy/projection/onefilter/CFD/resources/originalcriteria. Source6516456f07d765728055f58036bed97a5e37036f205c4d1ee2bf2456be8cc3b0; approval3ca5531815c48cff59fd1ca0d96e40e2305402435cb2e28d2adb26eeaf9328a6.6CPUtests independentlypassed beforelaunch; these are engineering evidence, not physical outcomes.

No scientific CSV metric is added while running. Output `artifacts/exploratory_projected_32768_ppo_b03_long_cfd_20261006`; allsix raw windows and cleanup require terminal independent review. b03H5fixed-actiondataalreadyopened; this trial is not universally unseen or statistically independent. No tuning from96frame replay/H5 results, no thresholdrelaxation, no automaticretry; existingK1H100FAIL remains.

## FC-E059 — Fixed b01 projected-policy replication completed

Actual R2 invocation `9ef43959e065431490bd4725fa8fb7fe` exited0 after800 paired cycles, restart130→210. All3200 force-file hashes, six predeclared windows,800 projection/filter equations and1600 clean solver segments independently verified; both owned containers absent/OOMfalse. Primary (150,210]12000points: drag reduction3.9236372469%, centered rearCl RMS ratio0.81578550745, mean-bias ratio0.02730022153. Original2%/1.05/10% criteria met, without relaxation. First6.2 secondary bias0.127807798 stillfails10%; allsixwindow evidence retained. Full80D/U:3.81531467%/0.825013365/0.007191961.

ResultSHA `961e1bc3ccb7a9f9dae4b54e9f8233c906507c794cff9a497d806391e0fc5c37`; independent report `docs/EXPLORATORY_PROJECTED_32768_PPO_B01_LONG_CFD_TERMINAL_REVIEW_20261006.md` SHA `1b59fdfdb9d698fd2c0622085c19bf4d59a670070ea7434cb469ac19c72a297b`. Same frozen policy/projection/physical protocol asb00; historically usedvalidation b01 is a second observed phase, not statistically independent generalization. FirstCLI failure omitted--execute and performednoCFD; separatelyapprovedR2 preservedit. Wall1125.320s/800, minimumAvailable117067710464B. No model retraining, MPC substitution, H100 admission or real-time latency claim.

## FC-E060 — Frozen H1–H5 prediction confirmation completed, no training

Actual authorized7a887ed792ec4d60a429f4a7a3660b3a exited0 after1600 fixedendpoints (10b03/b07cases×32starts×5horizons), failed0/nonfinite0. H1/H5 pooledvelocityL2 .0022426952/.0095414810; pressureL2 .0064234466/.0268274147; totalCdMAE .0070819531/.0106288686;rearClMAE .0191755319/.0221162728. Independent reviewer reproduced savedJSON field/force/coverage andstart0pairedresponse calculations (FP32subtraction rounding≤1.3e-11). OfficialHDF5Readeractualexecution; highTF32checkpointidentity thenhighest/noTF32override;6GiBallocator/12GiBnoswap;minimumAvailable117072576512B.

Result77ab4fb85f238d1e76b6e5a20c18f8992182d24a82b38d7b4b5a52483f9fce20; report docs/SHORT_HORIZON_FROZEN_CONFIRMATION_TERMINAL_REVIEW_20261006.md. This finaltestopening followed candidatefreeze andseparateLeadapproval; nooptimizer/CFD/selection. No posthoc accuracyPASS threshold, H100FAIL remains, and physical10%isnotpredictionMAE. Both projected b00/b01 primaryphysicalpasses are separate CFD evidence; b01 ledger follows independentownerhandoff.

## FC-E059 — Fixed b01 projected-policy replication running

After independent raw verification of FC-E058, Lead approved one fixed b01 replication with the same frozen32768 policy, reflection projection, one existing action filter,800 cycles,six relative windows and resources. The only preregistered physical difference is validation phase b01 at restart130 through210, with primary `(150,210]`; this is not favorable phase selection or an independent final test.

Actual r2 unit `fluid-control-exploratory-projected-32768-ppo-b01-long-cfd-r2-20261006.service`, invocation `9ef43959e065431490bd4725fa8fb7fe`, initial PID3509955, is active/running. Approval `790bb12fae2f5df729efda98ef59e5d99f75521d220cb9b39e8caf04808e4c59`; immutable driver `8b653f43bd1ffc69b6285dd10523199898d88d74aebe4279f65a66fb4807c741`; output `artifacts/exploratory_projected_32768_ppo_b01_long_cfd_20261006`.

The first invocation `dfa8ba1412e34522a1ed1385e28b2df7` exited1 before the execute body, model loading, output creation or CFD because the systemd command omitted required `--execute`. Root verified the operational cause and approved only a fresh r2 unit with that flag added; source, spec, policy, numerics and resources are unchanged, and the failed journal is preserved. No b01 terminal outcome, cross-phase success or admission is inferred while running.

## FC-E058 — Reflection projection meets unchanged b00 primary physical criteria

Actualafa5cde4daec474eb52b61c08f86746f completed800cycles,1097.174830s,PID0/exit0. Sameunique32768policy/pairedrestart148/80D-U/resources asFC-E055; onlyrequestedaction=.5(pi(o)-pi(Ro)) beforeoneexistingfilter. Approval87944e807a68caab6ce7a46e01207e1d7ba432b86ccd639b6f47424de481c64c;driver5c3f40728cd383913a256a2f46b6bfaf0b02cc7d91586e198c354a007fcb9e76.

Primary(168,228]12000:drag3.8919799397%,RMSratio.8156957479,bias.0113852067; alloriginal2%/1.05/.10criteria met. Companion12001:3.8918146724%/.8156948719/.0112901058;full80:3.7516809558%/.8243555266/.0136988610. Earlyfirst6.2bias.1354617484fails10%while20%sensitivitypasses. Maxomega.65537,0saturation/2ratelimits. Comparedoriginalprimary,drag/biasimprove butRMSratio.795798→.815696slightlyhigherstillbelowzero; reportalltradeoffs.

Independent3200rawhashes/6windows/all800formula-filter/zeroallcolumnsidenticalFC-E055,9sources8inputs;1600cleansolversegments/minAvailable121917501440/ownedcontainersabsent. Result199127979c6cb43e6304c60fc3373a2b1a8465476ffdd265d30c108dfffd0ca6;report docs/EXPLORATORY_PROJECTED_32768_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md. Singleb00physicalsuccessnotK1formaladmission/broadrobustness. Fixedb01replication separatelyapproved only; noautomaticretraining/thresholdchange.

## FC-E058 — Actual projected-policy paired800 CFD running

One authorized execution is live: `fluid-control-exploratory-projected-32768-ppo-long-cfd-20261006.service`, invocation `afa5cde4daec474eb52b61c08f86746f`, initial PID3059882. Approval `87944e807a68caab6ce7a46e01207e1d7ba432b86ccd639b6f47424de481c64c`; immutable driver `5c3f40728cd383913a256a2f46b6bfaf0b02cc7d91586e198c354a007fcb9e76`. Final persisted preflight verified nine source and eight input identities, image identity and exclusive output.

Relative to FC-E055, only the requested action is replaced by `0.5*(pi(o)-pi(Ro))` before exactly one existing filter. Same frozen32768 policy, restart148, pairedzero,800cycles, six windows, physics, criteria and resources. No terminal outcome or CSV result is recorded while running; no automatic retry/admission.

## FC-E057 — Actual fixed-prefix policy reflection-defect audit complete

Unit `fluid-control-policy-reflection-defect-audit-20261006.service`, invocation `dff10f4049dc4b4da84d7d05818d1c9b`, exited0. Fixed680-cycle snapshot SHA `e2e4b0d6363d3ec36f530ad673d92f2d72638be993cf79fc30ce979958c405c5`; result SHA `b0c48354f85a2f3e0b6ccf9f41079e2eded7e33fe4a422f7f3a0c61310fc6809`. Final32768 policy mirror-defect mean/RMS is −0.6702486725962338/0.8046740447610694; the4096 policy is −0.3195020545493154/0.3197158563363626.

Projected-action filter counts are conditioned on recorded original-trajectory previous omega, not a counterfactual rollout. This is a CPU read-only diagnostic with zero optimizer steps and no CFD/model writes; it motivates but does not prove FC-E058.

## FC-E055 — Completed 800-cycle direct PPO / real-CFD paired evaluation

Actual unit `fluid-control-exploratory-diverse-32768-ppo-long-cfd-20261006.service`, invocation `285bea88ff234cd5acfb9cb03c2b3cf3`, exited0 at08:15:05UTC after800 pairedcycles (1099.09s worker). Approval `e103288a0558c10784a43a199a3c4d731ffc0e6509753646da7bb6930cb4dc12`; executeddriver `17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa`. Final32768policy `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a` directly controls CFD; no onlineFNO orMPC substitution.

Primary (168,228]12000points: dragreduction2.377964895%, centeredrearClRMSratio0.795798378, meanbiasratio0.322286722. Companion[168,228]12001points:2.378337717%/0.795798492/0.322366798. Full80D/U:2.441019142%/0.803539927/0.343010009. Allsixwindows showdrag/fluctuationbenefits, butallfail10%and20%meanbias. Earlyzero rawarrays exactlyequalFC-E053, enablingpairedcomparison: earlydrag improves0.505760%→2.522740%, butbias worsens0.123721→0.368591. No fullconstraint success orformalK1admission.

Independent3200rawhashes, nine sources/seveninputs, allsixmetrics recomputed;1600clean20stepsegments,404/800saturatedendpoints,255ratelimited, minAvailable120469553152bytes, bothownedcontainersabsent/OOMfalse. Result `artifacts/exploratory_diverse_32768_ppo_long_cfd_20261006/result.json` SHA `b425bd28ea6e1ca6786ee6ea38dd3a09e13191849a5778270b987a830584e827`; report `docs/EXPLORATORY_DIVERSE_32768_PPO_LONG_CFD_TERMINAL_REVIEW_20261006.md`. Preserve originalbaseline and predeclaredprimary; no threshold/budget sweep authorized by this outcome.

## FC-E056 — Actual fixed24 H5 deterministic policy comparison, engineering only

Unitfluid-control-diverse-policy-h5-comparison-20261006 invocationf4ba411c1bcf4cae9ebd178ad0c30a1f exited0. Approvala9c3eaf77f9dc2a444459e2a17da3dc8ea556ce0eb0d952e09cae1880e2fd08f, worker630bc478ed17475539b21e206242598d7bbaa044b3acaf00829d95eb1424008c. Result3f8c6f7e5b03877601a3b25b26409e9d6f943fcbaec62600fa51343995922b9f. Same24train packets/frozenK1/H5/precision/canonicalreward; deterministic final4096vs32768 policies, allcases equalweight. Meanreturns−3.6902918374361167→−3.6875228003375486,delta+.0027690370985678316;7better10worse7equal. Dragpenalty improves whilebias/actuation/rate worsen. No optimization, newCFD, heldout or checkpointselection; no convergence/physicalbenefit claim.

Independent47source/192runtime hash and24×5×2 telemetry/component recomputation passed. Executedpolicy/FNOtensor checks unchanged, not independent tensorreload. MinimumAvailable120472039424bytes; peakGPUallocated755589120bytes. Supervisor22.019s total versus7.069s workerpostsetup; actualevaluation240s limit, inheritedtrainingprotocol1800/32768 onlyprovenance. FiveCPUtests passed. Reportdocs/DIVERSE_POLICY_H5_COMPARISON_REVIEW_20261006.md; sourcecapture afterexecution. FC-E055 longCFD untouched.

## FC-E055 — Actual800-cycle final32768-policy paired CFD running

Root launched `fluid-control-exploratory-diverse-32768-ppo-long-cfd-20261006`, invocation `285bea88ff234cd5acfb9cb03c2b3cf3`, initialPID2588512. Approvale103288a0558c10784a43a199a3c4d731ffc0e6509753646da7bb6930cb4dc12; immutable driver17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa, source captured afterlaunch. Fixed800cycles148→228, same69observation/CPUdeterministicpolicy/slew/ramp/pairedzero/solver. Sixpredeclared windows retain early124 comparison and final60primary12000 versus explicitlydifferent historicalinclusive12001. Originalcriteria unchanged;20%sensitivity labeled. No extra124trial was executed and no physicaloutcome inferred fromlaunch. Same8G+2×8G/noSwap/Available50,22; time-only3600inner3750outer120stop approved following cumulativeprobe-read estimate. Four independentCPUtests passed.

## FC-E054 — Independent32768 terminal review complete

ActualR2a19900b2bfa64d8d8372b67bc0564139 exited0, resultff3532a604b6816fb3ad4c7a11edfcd579bcb924445abca52a2fdab8ea4dcf20; finalpolicy5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a. All6artifacts/45source/192runtime hashes rechecked by independentreviewer;32768rows/256epochs/512optimizerhooks andeachphase274+273×5resets. First4096transitions match previous4096run exactly in actions/forces/reset/ledger/reward, notwallclock. Policychanged, executedFNOtensorchecksunchanged; no independent modelreload. Wall575.563s/minAvailable119260291072bytes. Report7bbb772513337bbd67588aa59454fcb1272014d81b52b1a7e3785f4fe7701d19. Preserved firstc406pretrainingJSONfailure and correctedR2approval; trainingcompletion is not physicalsuccess/convergence.

## FC-E054 R2 — Actual retry running after persisted approval preflight

Root launched `fluid-control-exploratory-diverse-h5-32768-ppo-r2-20261006`, invocation `a19900b2bfa64d8d8372b67bc0564139`, initialPID2560906. Final persisted1cd1d518 approval passed actual frozen validate_spec and exact protocol types beforelaunch. New exclusive_r2 output; failedc406 evidence unchanged. Mandatory future practice: validate final serialized approval using its actual frozen consumer, not just draft JSON. No training result inferred from a live handle.

## FC-E054 — Initial launch failed before training; exact JSON compatibility correction

Root launched `fluid-control-exploratory-diverse-h5-32768-ppo-20261006`, invocation `c406bd4af9a84219840027961119e6ab`, under approval5f6485493aa51db0b14b15d37bdf530684864a919bc78878fd0b266be988bb21. It failed before model/training because approval generation converted ent_coef0.0 to integer0, rejected by strict JSON protocol comparison. Original failure/output/approval preserved. R2approval1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132 fixes only that literal and exclusive_r2 output. Actual frozen validate_spec on persistedR2 JSON and all protocol value types passed. No scientific thresholds or numeric algorithm changed.

Immutable trainer4d681771736b63b628712d3b62fcdde831601e80221aef6f1fd78a4b6840ff01 and supervisor8363986dc675cc8efe09285b60c25557f1307c7ffcf49a54e0e4388ce3ff9a47 unchanged. Six independent CPUtests verify exact source delta/accounting. Same24 packets/K1/H5/reward/seed/hyperparameters, fresh policy/optimizer; expected256epochs/512steps, final-only policy. Same12GiB/noSwap1CPU/Available50,22/1800inner1950outer. No scientific CSV outcome for failed startup. Different reset-distribution losses cannot establish convergence by direct comparison.

## FC-E053 — Matched final diverse-policy CFD complete, not admission

Actual464de68ee1114eea8e8ae214d18dc045 completed07:33:58–07:36:26UTC exit0. Result8c909aa4bd0b73e3cf570dd55cb2a1abd7346a9c424695a5e0056b4e5e833bdc; independent report31a338bfc81e4ece0adeee943c074686e7065e57048d5783697d062856aa4e53. All496 raw hashes verified and complete zero arrays exactly matchFC-E051. Full/first/trailing drag+0.5057597%/+0.6872659%/+0.3242481%; rearCl RMS ratios.9737074/.9605220/.9864395; bias ratios.1237210/.1071973/.1402443. No saturation;124 distinct requested actions, maxmagnitude.20272. Improved bias/full fluctuation and removal of prior constant+.75 behavior do not imply dominance: trailing drag benefit decreased. All drag benefits<2%; original10%bias fails,20%sensitivity passes biasonly; not80D/U admission.

620resource rows minimumAvailable122083807232bytes, bothownedcontainers absent/noOOM; all248solversegments completed. Original model formalFAIL and earlier CFD-only success stay separate. Next singlefactor is optimization budget, not posthoc threshold/weight changes. Prior running entries below remain historical.

## FC-E053 — Actual diverse-final-policy paired CFD running

Root launched `fluid-control-exploratory-diverse-ppo-cfd-20261006`, invocation `464de68ee1114eea8e8ae214d18dc045`, initialPID2474346. Approval `67fda1a404f844d89b986442a4a9000561b02757417d5a8d366fdf9f9e6033db`; executed immutable driver `89e0d8bea92440babd3d647eed31758db9cfc2a43e31ed6e9d9bdf5047b77b6e`, captured in Git after launch. Final diverse policy `8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b` alone supplies deterministic CPU actions; no MPC or onlineFNO. Same paired148→160.4/124cycles, physical69 observation, constraints, full/halves open-left metrics and10% physical reference. Outer8GiB/noSwap/4CPU, two8GiB/noSwap solvers, Available50/22,1950s/stop120. No scientific CSV outcome while running; not80D/U admission.

## FC-E052 — Independent terminal training review complete

Result `cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640` and all6 listed artifacts rehashed. Independently counted4096 rows with per-env six-slot steps `[174,170,170,170,170,170]`, matching205 resets `[35,34,34,34,34,34]`,204 complete episodes plus4 partial steps. Actual64 optimizer hooks/32epochs; policy changed and reviewed trainer checked frozenK1 tensors unchanged.172 memory rows minimumAvailable119470489600bytes. Applied actions now span±.75 and both lift penalties are active, but these training diagnostics do not establish CFD benefit. Report `docs/EXPLORATORY_DIVERSE_H5_PPO_TERMINAL_REVIEW_20261006.md` SHA `c93b1e6b7d0127c204a5dd8795b080e11068fb990c19e559314920f26c80964e`. Sourcecapture0e38e49 was afterlaunch; actual identity is immutable trainer SHAaae8c9a4. Historical observed/pending entries below remain unchanged.

## FC-E052 — Training terminal observed; direct-policy physical outcome still pending

Same `3a34c4d621244e4bbacdf1816b5b1374` exited0. Actualresult `cd5775e4647280b77803de9a5ced6abdf6378cded4f676f935bd9836350c3640` records4096steps/32epochs/64optimizersteps and eachphase resetcounts `[35,34,34,34,34,34]`. Finalpolicy `8dc8cabf2104654345f270e3fb86edca7752cf4c883112c0a4cbd3a181acea9b`; VecNormalize `6988d4d161bc69c8bbd89d477e9320ad9ef264d35c9dee0bbf63954d4cdfce70`; wall83.807s. No final-policy selection or FNO update. Independent terminal resource/artifact review is assigned separately; training completion is not CFD benefit.

The new directCFD consumer accepts actual training JSON and unchangedprotocol/packetproof, with7independent CPUtests passing. Its paired124 solverloop/numericalhelpers remain byte/AST-identical toFC-E051 apart from new producer identity. Actual CFD execution still requires Lead authorization and terminal proof. The running entry below is historical.

## FC-E052 — Actual running: fixed24 diverse-real-reset exploratory PPO

Verified user unit `fluid-control-exploratory-diverse-h5-ppo-20261006`, invocation `3a34c4d621244e4bbacdf1816b5b1374`, started2026-10-06 07:30:12UTC, active/running PID2463028. Approval `760e1f9e81494bdd8c3742cd0ce77e168b0df2bf288bd04546412096721f41e2`, exact immutable trainer `aae8c9a4311112439251b695001c7601ffd3d7cd3010937f86bd9a0bfebf3040`; new4file manifest `2f9a1e5de2c99fa153cd8316fd2ecdbf96c70a735c0c7fc6a4a07621f8ec21ca` with old39source dependencies unchanged. Canonical source capture occurs afterlaunch, not represented as launchHEAD.

Only reset distribution differs fromFC-E050: phase00/02/04/06 each cycle originalzero0 then m075/m0375/zero/p0375/p075 frame62. Actual CPU24packet receipt `f85f84a4b82e0c21eaf011281e0b98b570bfaa083805c604e1fbe04aaa14583b`; source-bound realfield/action/rawcausal62force/grid identities must reproduce beforeenvironment use. Same frozenK1/H5/69obs/canonicalcost/actions/seed/PPO4096,64expected optimizersteps,final-only policy. Independent13trainer tests and10adapter tests plus2actualHydroGym/SB3 synthetic lifecycle tests passed. No GPUdata probe or policyselection added.

Output `artifacts/exploratory_diverse_h5_ppo_training_20261006`;12GiB/noSwap1CPU/.06allocator, physicalAvailable50startup22runtime reserve20,1800inner1950outer. This running entry provides no new scientific CSV outcome. Resetcoverage is a hypothesis, not proven solecause ofFC-E051 saturation; history-dependent reward and surrogate/short-return biases remain. Subsequent directPPO realCFD requires actualsuccessful terminalpolicy and separateapproval.

## FC-E051 — Actual direct final PPO / paired CFD completed; not physical admission

Same invocation `fd6d92f7ea9946b49c91c07e21f1d74b` completed07:10:18UTC, exit0,124cycles from148 to160.4. ResultSHA `4007493f22de5855cbd0574e0ec006ca715941b8396f4e48af6527dc11e03d47`; approval `7ace192519a08795fe9217473fae33941fc5edbb1075daeeb3701e672c521cb3`; immutabledriver `44b488a97a2882e1325da8871d3ac4905cdae2a6f2cbb17202ced91afc58b91a` (later source capture `f473abe`, not launchHEAD). Finalpolicy `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1` acts directly, without onlineFNO/MPC substitution.

Independent496raw-file hashes and force recomputation agree with result to1e-14. Full/first/trailing counts2480/1240/1240, open-left fixed windows. Drag reductions +0.004117553020020259 / −0.01635856133306368 / +0.02459427885158949; rearCl fluctuationRMS ratios1.1058626225931032 /1.1998320940691536 /.9909864238426835; pairedzero mean-bias ratios .5273107117128877 /.4172662478614581 /.6373532395708328. All10%/20% bias sensitivity checks fail, also using original fixed train-b00 reference. No threshold change or favorable-window selection.

Every requested action+.75;117/124 applied endpoints saturated. Full RMS worsens10.59% despite small drag benefit. Operational FNO-trained PPO→CFD loop is complete, constrained physical control is not;12.4D/U is not original80D/U. MinimumAvailable122930147328bytes; both exact containers absent/noOOM. Independent report `docs/EXPLORATORY_FINAL_PPO_CFD_TERMINAL_REVIEW_20261006.md` records scope and raw proof.

Next preparation only: fixed24real-start reset panel, four original train-zero frame0 plus20base-train frame62, sameK1/H5/reward/PPO4096. Reset coverage is a falsifiable hypothesis, not established cause:69observations omit full62rewardhistory, and short-return/model bias remain. No new GPU launch authorized here. Prior running entry retained below as history.

## FC-E051 — Running: final surrogate-trained PPO directly controls paired real CFD

Actual user unit `fluid-control-exploratory-final-ppo-cfd-20261006.service`, invocation `fd6d92f7ea9946b49c91c07e21f1d74b`, started2026-10-06 07:07:51UTC and independently observed active/running withPID2346139. This is a real launch, not a terminal outcome. ApprovalSHA `7ace192519a08795fe9217473fae33941fc5edbb1075daeeb3701e672c521cb3`; executed immutable driverSHA `44b488a97a2882e1325da8871d3ac4905cdae2a6f2cbb17202ced91afc58b91a`. Git source capture follows launch and is not represented as launchHEAD.

The single finalPPO policySHA `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1` and identityVecNormalizeSHA `54a08a438501aac0663e50da931f41aabb63cdeb8255aa80051e7af1b4eaaba2` are bound to actual4096-transition training resultSHA `138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4`. No policy selection, onlineFNO, or MPC substitution. DeterministicCPU inference consumes actual69-channel physicalCFD observations before applying one canonical rate limit and the existing linear boundaryramp. Probe coordinates/channel semantics match training; grid interpolation versus rawCFD probes is explicitly not asserted numerically exact.

Predeclared pairedzero148→160.4,124cycles; report all2480 solver-step samples and full/first/trailing windows using fixed open-left bounds. Original10% mean-lift reference unchanged; exploratory12.4D/U is not formal80D/U admission. Outer8GiB and two8GiB solver caps/noSwap, CPUonly, MemAvailable50/22GiB, internal1800s/external1950s/stop120s. Source-only independent10CPUtests passed; actual policy/CFD outcome remains pending in `artifacts/exploratory_final_ppo_real_cfd_20261006`. No scientificCSV result is inserted while running.

## FC-E050 — Actual exploratory frozen-FNO / HydroGym / SB3 PPO training completed

Same invocation `21cb82da66214924b38f120eb30723e5`,06:51:48–06:53:11UTC, exit0/PID0. Explicit exploratory approval `8aa44f5f177d6c7d831a1efc55abf9d8db4b700d640cb640c5fcbb709cc44569` permits H5 while leaving canonical100-step admission unchanged. Read-only39-source manifest `730c9f315234a59c381fefa176d46a124eb310d41886176547496bce8b998856` and trainer `8806682ec66d0a1b8dde30b0f361bb77e115f9add85f207651d9d01d1d8d4100` identify actual execution; baseHEAD76f13dc did not yet contain these new scripts. Later Git capture is not execution HEAD.

Exactly4096 transitions (1024 per fixed train-zero b00/b02/b04/b06 start),816 completedH5 episodes,32 epoch updates/64 actual optimizer steps. Policy tensor digest changed; official K1 tensors stayed frozen/no-grad and outside the PPO optimizer. Only one final policy saved, no reward selection. Result `artifacts/exploratory_h5_ppo_training_20261006/payload/result.json` SHA `138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4`; final policy SHA `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1`. Independent artifact/counter review is `docs/EXPLORATORY_H5_PPO_TERMINAL_REVIEW_20261006.md`.

Final value loss0.0210340086/approxKL0.0106169721 are finite; all episode lengths5, no divergence,93.04% actions rate-limited. Worker80.902s; minimum sampledMemAvailable119542509568bytes. This proves actual surrogate policy training, not CFD benefit or repair of K1 formalFAIL. Four-start short resets, diluted62-sample reward and timeout bootstrap can hide long-return/model bias. Next step is separately approved direct-policy pairedCFD, not MPC substitution; no reward/physical threshold changes based on training metrics.

## FC-E049 — Accelerated fixed H5 real CFD124 cycles; failed summary, offline recovered negative drag result

Execution is identified by immutable driver `4cca28757f44e80f693d2d4a33c15cea0ea5bc74368eda669aead292b37464bf`, approval `03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75`. Source capture commit `aae121a` was created after launch, not HEAD at launch; CSV code_commit identifies that reproducible capture. Actual unit invocation `a601eec2da7649b4af6f9354a4deb470` ran06:28:56–06:40:19UTC, completed124 CFD cycles but exited1 at trailing-window summary: legacy inclusive-left reader returned1241 instead of1240. Original failed state and absent result.json preserved.

Approved offline recovery source `5003424f0e8abf4e89733f62d6515a72239eb98a01690188813cadd08d542e7b` used original metric function and explicit intended `(begin,end]`, no new CFD/model. Recovered artifact `artifacts/exploratory_accelerated_long_h5_real_cfd_20261006/recovered_metrics.json` SHA `1605604dc27f106acd05e6a721f26c4ba24527ac53996d6e65fbc70c601fa2b1`; report `docs/EXPLORATORY_ACCELERATED_LONG_H5_TERMINAL_REVIEW_20261006.md`. Raw windows2480/1240/1240 yield paired drag reduction full−0.006505988003726815, first+0.041150445558386095, trailing−0.05416384473057234; rearCl fluctuationRMS ratios0.8322411751/0.9135812066/0.7003219964. All windows are reported; no favorable-window selection. This is exploratory12.4D/U, NOT_ADMISSION and not original80D/U completion.

First10 actions and all4 raw-force prefixes exactly reproduce priorCPU H5. Original restart rehashed unchanged; both containers absent/noOOM. Resource minimumAvailable120401592320bytes. PredictionMAE rearCd0.05848863/rearCl0.09056808 and38 saturated endpoints motivate further separately reviewed investigation, not causal proof or automatic rerun. Original10% criterion unchanged; relaxing bias cannot remove recorded full/trailing drag worsening.

## Active — Accelerated fixed-H5 paired124-cycle real-CFD trial (no outcome yet)

Actual user unit `fluid-control-accelerated-long-h5-20261006.service`, invocation `a601eec2da7649b4af6f9354a4deb470`, PID2131269 was observed active/running with5/124 completed cycles. Approval SHA `03e2bac8f55c4bbd09e377b60bfef849515b53a6d418d490ec682b2cde95bc75`; driver `4cca28757f44e80f693d2d4a33c15cea0ea5bc74368eda669aead292b37464bf`, sequencer `c1011780b4e72f45e43127f3d7a728cf546d90b07dd1a7d41e57cf0dfc750688`; output `artifacts/exploratory_accelerated_long_h5_real_cfd_20261006`. No scientific CSV outcome is recorded before terminal evidence.

Same FC-E048 K1/H5 canonical-history controller, candidates, weights, constraints and restart148. Persistent Curator exact-array replay completed21 comparisons; GPU highest/no-TF32 replay on all10 stored H5 states preserved all selected indices/actions/ranks, max force difference2.2649765e-6 (not zero). GPU result SHA `3568870390df28c4f63145317ad89f941ff7d1c6073f5a756b06ab2a46273a69`, independent review `95dca2e1941b2aff46d6c7510cce0ce74ad49d3fcdc439cd1fdb62064eb69524`; Curator result `d8786fde2de19f609df320c0d5df21a8cffa91f91cca6a02d55d83ce53bbcddc`.

Real duration is extended to12.4D/U, with full, first6.2 and trailing6.2 windows reported separately. This is not original80D/U evaluation, model/PPO training, robust-control success or surrogate admission. Terminal raw forces/actions, solver health, identities and cleanup determine the eventual result; earlier short-window negatives remain preserved.

## FC-E048 — Canonical-history H5 actual feedback: nonzero control, no demonstrated drag benefit

Code `8f5afb3`; executed driver SHA `0917cd5c62e43fc3f7b2cdc23900aa9d0932dff9ef4842a155bcf4288524b14d`; approval SHA `f927f6b956f847766d899745ebc0e679a7638db63f27cfb1f9a489eb69fb6c0e`. Same K1 and canonical-history cost as FC-E047, horizon changed to H5, CPU retained. Actual invocation `6adc59fae65344d2b49b57cbe5b30f70`,06:08:12–06:12:13UTC, exit0. Result `artifacts/exploratory_causal_history_h5_real_cfd_20261006/result.json` SHA `d4c3ad8198f69199606c0fa7a6c1a668c9a0f581e3b52bbeca99e2b0bd902c5e`; independent report `docs/EXPLORATORY_CAUSAL_HISTORY_H5_TERMINAL_REVIEW_20261006.md` SHA `e6b7496b3a59b0bfe5a6b4365bbb9076d892ec651b3cf75fa212075b98dbe79f`.

Ten actual actions `[.1,.2,.25,.25,.2,.1,0,-.1,-.2,-.3]`; raw200 samples per branch over148.005–149.0 reproduce mean totalCd2.4137825099145 vs zero2.413592168615, paired drag reduction−0.0000788622460641264 (0.0078862% worse), rearCl fluctuationRMS ratio0.9836105589246837. Selected one-step MAEs `[.00011434555,.00133591443,.00599833131,.00971178710]`. All50 candidate/250stage costs reproduced; mean-bias penalty0/250, so the10% term did not block selection. Both owned containers removed; noOOM. Engineering complete, NOT_ADMISSION: oneD/U is insufficient for original80D/U physical criteria or causal attribution of the small drag difference.

Next preparation only: fixedH5 paired124cycles/12.4D/U, report full and trailing6.2D/U windows, retain cost/candidates/constraints. Accelerated sampling and explicit GPU precision require engineering assessment and separate execution approval. No new execution, GPU training, PPO, or threshold change authorized by this entry.

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
