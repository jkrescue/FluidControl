# FC-E067 — Independent K1 controlled-development baseline review

Status: RETROSPECTIVE DIAGNOSTIC COMPLETE, NOT SCIENTIFIC ADMISSION.

Actual unit `fluid-control-p064-k1-development-h1-h5-20261006.service`, invocation `a39f8106aa9b4559b1fb1587e7a8e38c`, independently observed PID0/exited/exit0; supervisor error=null/returncode0. Approval SHA `05891a360ffaa17d61e6854a8dce632d818a3ab1ec5b4a616244d4abb6a85b79`, executed worker SHA `e86ef8ea3c55be63287b0f9d5e9e0cf0034e7fc6959df25e53e49c0fe23d90a8`.

Result `artifacts/p064_k1_development_h1_h5_20261006/result.json` SHA **`9ea3e0e781e76265bbc65ea52d6fec92ebe5b5cb7a93d91c3c9addb454f7c4de`**. This is K1, not a newly trained A/B candidate. Fixed b01/b03 controlled trajectories, eight mechanical starts0,100,…700 each, H1–H5:16 origins/80 endpoints, all retained. Both phases are already opened development data, not fresh heldout confirmation.

## Independent checks

Rehashed actual driver/base,378 source entries,192 runtime entries and5 inputs; all match approval. All16 NPZ hashes, shapes and finite values passed. NPZ actual times/actions/four-force targets match the conversion selection's precise per-phase clocks (130/144 bases). Predictions in NPZ equal saved JSON predictions. No HDF/model/CFD rerun was used for this review.

Independently recomputed every origin/lead's three masked physical field SSE/reference sums, fixed-initial-state persistence SSE, all four force MAEs and signed-sum-before-absolute total-Cd error; then recomputed both phase and pooled summaries. All agree within relative1e-12/absolute1e-10; maximum absolute summation difference2.614e-8 occurred in large field sums from reduction order, not force metrics. Persistence is each origin's initial field/force held fixed across all five leads, not rolling truth.

Actual receipt records verified-load high/TF32 followed by explicit highest/noTF32 inference, allocator6GiB, zero optimizer steps, unchanged model tensors/no gradients. These are runtime checks and source-bound observations, not an independent model deserialization in this review. K1 manifest remains7adca21e…acc7.12GiB/noSwap cgroup verified in supervisor;22 memory samples over10.51s, minimumAvailable120994258944B, comfortably above22GiB. No new CFD or policy run occurred.

## Pooled all16 origins

| Lead | Velocity relativeL2 | Pressure relativeL2 | Rear-Cl MAE / persistence | Total-Cd MAE / persistence |
|---|---:|---:|---:|---:|
| H1 | .010316 | .032042 | .158871 / .090334 | .039630 / .020318 |
| H2 | .019808 | .061678 | .152044 / .179995 | .029367 / .041096 |
| H3 | .028415 | .089014 | .153251 / .267660 | .026036 / .062011 |
| H4 | .036140 | .114071 | .167256 / .353433 | .028167 / .082367 |
| H5 | .043040 | .137006 | .179850 / .439982 | .030522 / .101058 |

## Phase separation

| Phase / lead | Velocity relativeL2 | Pressure relativeL2 | Rear-Cl MAE / persistence | Total-Cd MAE / persistence |
|---|---:|---:|---:|---:|
| b01 H1 | .010296 | .031852 | .167390 / .080909 | .046173 / .020139 |
| b01 H5 | .043053 | .136114 | .189925 / .400050 | .033153 / .103384 |
| b03 H1 | .010336 | .032233 | .150352 / .099758 | .033088 / .020498 |
| b03 H5 | .043027 | .137900 | .169775 / .479915 | .027891 / .098731 |

H1 rear-Cl improves over persistence only3/8 origins in each phase; total-Cd improves1/8 b01 and3/8 b03. H5 improves rear-Cl7/8 in each phase and total-Cd8/8 in each phase. Thus average H1 force prediction is worse than persistence in BOTH phases, even though average H5 beats the increasingly stale persistence baseline. This is not evidence of uniformly accurate short-horizon forces.

Origin0 rear-Cl H1/H5 is small: b01 .014295/.000535, b03 .014665/.012400. Later examples are much larger: b01 start400 H1 .486311, start700 H5 .369690; b03 start500 H5 .428715. Every origin/lead remains in the immutable result; these examples illustrate heterogeneity, not selected evaluation populations. The H1 error is present before recurrent prediction accumulation, so long-AR drift alone cannot explain it. No causal attribution to action reversals, precision or state coverage is established by this replay.

## Meaning and next use

This is a frozen K1 baseline for subsequent separately approved A/B comparisons on exactly these development records. Future commands are the actually realized commands from prior real-CFD feedback; they are not available prospectively at an online origin. Therefore this is conditional retrospective model replay, NOT online FNO/MPC forecasting and NOT a new closed-loop physical test. The measured three-phase physical control benefit, original10% mean-bias criterion, and K1 H100 FAIL all remain unchanged. No threshold relaxation, candidate selection or GPU training is authorized by this report.
