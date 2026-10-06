# K1 saved GPU/CPU inference comparison — 2026-10-06

This independent review compares existing saved outputs only; no model, GPU, CFD, or optimization was executed for the comparison. The GPU probe is not a replacement-controller validation. H5 remains a CPU-only single-factor experiment.

## Evidence and matched identities

- GPU: `artifacts/k1_uma_gpu_inference_probe_20261006/worker_result.json`, SHA256 `7e3c373d895ae839ad360369988c49aa52a1ecc14d0f6d94b4793b83efb0559c`.
- CPU: step 1 of `artifacts/exploratory_paired_h2_real_cfd_20261006/result.json`, SHA256 `45fcab568ed7456e521ed17c4469f44716d231ca4ec5c08820864803ae856fbb`.
- Executed GPU probe source SHA256 `56c17443245c60f51b4e4c8f15b1851136a091c73441e5ecd76eebe0695f5e74`; CPU driver SHA256 `c8260b21d742f54814bf04e81bb13eeff491a971df4d177f3a384c8f8c940e3f`.
- Shared selector SHA256 `ec22561f60c183dc7fa9dfe68b32477de60faff9a41a5e051dccdad48d74f587`.
- Shared K1 manifest SHA256 `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`; flow archive `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`, aero archive `e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5` (producer identities, not independently reloaded in this comparison).
- Identical t148 current sample SHA256 `b9bdf87db1b4744633fb3c175050449952a6661ec95676a055d7f62111f80bc9`; previous action zero; five held-action sequences `[-0.1,-0.05,0,0.05,0.1]`, each repeated for H2.
- Configuration SHA256 `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`; normalization `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`; baseline `b5b7923f30600eba25c837f3b8d6781f37afbb6fcb1649e0c0d410f32a101ed7`. These small input files were rehashed against the common historical approval `1679f6bee293eae200c95bc078a20db5873e81afcd54d81b81bb06445015f8b2`.

## Recomputed differences

Each channel statistic uses all ten predictions (five candidates × two leads). Differences are GPU minus CPU; the maximum column uses absolute differences. All three saved GPU decisions are exactly identical to each other.

| Physical force channel | Maximum absolute difference | RMSE |
| --- | ---: | ---: |
| Front Cd | 0.00001537799835205078 | 0.000012237749527655312 |
| Front Cl | 0.00027680397033691406 | 0.00025482721524059754 |
| Rear Cd | 0.0008111000061035156 | 0.0006521266743594412 |
| Rear Cl | 0.013054072856903076 | 0.012471641130405749 |

Rear-Cl maximum differences at lead 1 and lead 2 are respectively `0.013054072856903076` and `0.012162715196609497`. After subtracting each device's HOLD prediction, the maximum discrepancy in candidate-relative effects is `[0.0000015497207641601562, 0.000033736228942871094, 0.00014781951904296875, 0.000354081392288208]` in the channel order above. Thus much of the discrepancy is shared across actions, but it is not purely an identical offset.

| Held action | CPU H2 cost | GPU H2 cost | GPU minus CPU |
| --- | ---: | ---: | ---: |
| -0.10 | 1.1982172909912878 | 1.204780655014974 | 0.00656336402368618 |
| -0.05 | 1.1921244662430552 | 1.1986106615606102 | 0.00648619531755501 |
| 0 | 1.1885765733367701 | 1.1950178796453297 | 0.006441306308559547 |
| +0.05 | 1.1875701675893993 | 1.193775218474452 | 0.006205050885052632 |
| +0.10 | 1.1891021926691046 | 1.1954090220226388 | 0.006306829353534216 |

Both rank candidate indices `[3,2,4,1,0]` and select `+0.05`. The winner-to-runner-up margin is CPU `0.0010064057473708754` versus GPU `0.0012426611708777902`. Saved state bounds and feasibility agree for these candidates.

## Interpretation and limits

The observed approximately `0.013` rear-Cl device discrepancy is comparable in magnitude to the first CPU trial's ten selected one-step rear-Cl prediction MAE (`0.012297362089157104`), although these are different aggregations and not a formal equivalence test. Matching action ranking at one state does not establish matching predictions, trajectories, other states, or physical control performance. GPU inference therefore cannot silently replace CPU inference without further numerical assessment.

No arbitrary tolerance or equivalence PASS is introduced. The source enables high matmul precision and TF32 flags on the GPU path; hardware/backend/TF32 differences are hypotheses, not an identified cause from these saved outputs. No additional run was performed to isolate the cause. The probe's three repeated decisions demonstrate repeatability only within that recorded GPU execution. Original model-admission failures and physical criteria remain unchanged.
