# P027 training-cache coverage: independent terminal review

2026-10-06. JSON-only diagnostic complete, not admission or a new prediction experiment.

## Actual identity and audit

User unit `fluid-control-train-cache-coverage-20261006.service`, invocation `08658ea0d6a5435a847df4de1223d21f`, PID0/exit0; start and exit both recorded at 10:07:34UTC. Actual limits: CPUQuotaPerSecUSec=1s, MemoryMax=1073741824, MemorySwapMax=0. MemoryPeak is not retained (`[not set]`), so this review does not invent a measured peak or continuous resource minimum.

Output `artifacts/train_cache_coverage_20261006.json` SHA256 `34ec16db0540c0dabb2f44caf3240ae386893ea85bd28bc1436e8820212b039d`; executed source `1fbd532caae13240f62a45922e499f02cf6402ad1c860828cd1a100350288317`; approval `docs/TRAIN_CACHE_COVERAGE_APPROVAL_20261006.md` SHA `87ede7d20203c2cb5bb77e5f9af96c1f1d8b53c0355fec2180098fb47aa5fdc3`.

Independently checked the original P027 cache SHA `7785ebb92ca932b4fb572175b4bd66497fc3b7495f6ecdb534f3b587a0096366`, source-phase map `57ed2a25eed41ff25f75aad529e4186f66d952053ddd7adcfff9517291693b92`, all three manifest SHA bindings and executed source. Recomputed all44 ×5 offsets ×3 streams ×5 error channels from the original saved prediction/target arrays, without importing the analysis implementation. Per-row errors and all group aggregates match exactly (maximum difference zero). Exact44 membership, action clocks, group membership and counts match; no dropped cases. No HDF/model/GPU/CFD was accessed for this review.

Current action is frame51; next actions and targets are frames52–56. `k1_h1` remains a fresh truth-conditioned one-step prediction at each offset, whereas `k1_ar` evolves freely from51. Persistence is the original producer's fixed frame51 force, not rolling observed forces. Total-Cd error is the absolute signed sum of front/rear Cd errors, not their absolute-error sum. The source review independently ran8 CPU fixtures successfully before Root's one actual analysis.

## New coverage information

| Action category | base20 | train8 | train16 | Total |
|---|---:|---:|---:|---:|
| Constant | 20 | 1 | 0 | 21 |
| Changing, no increment-direction reversal | 0 | 5 | 0 | 5 |
| Increment-direction reversing | 0 | 2 | 16 | 18 |

“Reversal” is a sign change between nonzero action increments, not necessarily an omega sign crossing. Categories use exact stored signs and can include rounding artifacts; they were not selected from errors. Maximum stored absolute omega=.75; maximum absolute delta=.100006103515625. This last value reflects stored precision and must not silently be clamped or interpreted as proof every action meets the exact online bound.

Manifest trajectory lengths are base801/80D-U, train8 201/20D-U, train16 129/12.8D-U. **All cached origins are only approximately5.1D-U after each trajectory start** (stored range5.0999984741–5.1000061035); targets extend to approximately5.6. Long manifest duration therefore does not imply that these cached predictions cover late states of those trajectories. No state-distance or OOD statistic was measured.

## Error evidence

Rear-Cl MAE below averages the fixed five offsets and equal-weight cases within each group; H1/AR denote conditioning, not two different evaluation populations.

| Group | Cases | True-state H1 | Free AR | Fixed-force persistence |
|---|---:|---:|---:|---:|
| All | 44 | .03201176 | .03799814 | .34814962 |
| Constant | 21 | .03889449 | .04534176 | .34568305 |
| Changing | 5 | .04321395 | .06027287 | .34027185 |
| Reversing | 18 | .02087019 | .02324315 | .35321556 |
| base20 | 20 | .04022540 | .04586562 | .34239015 |
| train8 | 8 | .03274422 | .04620459 | .35996936 |
| train16 | 16 | .02137849 | .02406056 | .34943910 |

At the first offset, reversing rear-Cl MAE=.02379181 versus constant=.04199404. At offset5, reversing truth-conditioned/free-AR=.01747960/.02692052; changing=.04246832/.07336521; constant=.03279928/.04522541. Thus this cache does **not** support the simple claim that action reversal makes force error larger. Increment-reversing cases are dominated by train16 while constant cases are dominated by base20: family, state, phase and action category are confounded. The five-case changing group cannot establish a causal action-history mechanism either.

All-case total-Cd MAE at offsets1→5 is .01544764→.01419120 for truth-conditioned prediction, .01544764→.01550768 for free AR, and .02460255→.15923607 for persistence. All per-offset four-force and total-Cd values/member identities remain in the saved output, not only this compact table.

## Interpretation and next decision boundary

This adds mechanical action-window coverage and trajectory-duration context beyond the old P027 overall/family ten-step report. It rules out asserting that the cached training examples contain no changing/reversing actions. It does not establish adequate coverage of later closed-loop states, prove the cause of the controlled FC-E063 errors, or support tuning a model on the opened replay panel.

P027 uses stored-HDF interpolated labels and high/TF32 inference; FC-E063 uses sampled controlled CFD trajectories and highest/no-TF32. They are neither precision-matched nor state/action-distribution-matched. Correlation in these groups is not causal evidence, and manifest duration is not state-distribution coverage. Any next train-only true-state/readout versus recurrent-flow experiment needs a separately approved fixed design; this output authorizes no new training, model selection, CFD, threshold change or H100 admission. Existing physical closed-loop successes and formal H100 failure remain separate facts.
