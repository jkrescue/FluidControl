# FC-E055: independent 800-cycle direct PPO / real-CFD terminal review

**Operationally complete; constrained physical control remains incomplete.** The 32768-step surrogate-trained PPO achieved measured paired drag reduction and reduced lift fluctuations over the predeclared primary window, but mean lift bias fails both the original 10% reference and the requested 20% sensitivity. No K1 formal surrogate admission is implied.

## Actual execution and identities

User unit `fluid-control-exploratory-diverse-32768-ppo-long-cfd-20261006.service`, invocation `285bea88ff234cd5acfb9cb03c2b3cf3`, ran 2026-10-06 07:56:43–08:15:05 UTC and independently reports PID0/exit0. Worker wall time: 1099.090493 s. All 800 paired cycles completed; no restart or extra 124-cycle trial was run.

- Approval: `e103288a0558c10784a43a199a3c4d731ffc0e6509753646da7bb6930cb4dc12`.
- Executed driver: `17060dda570ead4fdc8e33920fcc559b5bb8ad8d640a7e154579f8a795507afa`.
- Result: `artifacts/exploratory_diverse_32768_ppo_long_cfd_20261006/result.json`, SHA `b425bd28ea6e1ca6786ee6ea38dd3a09e13191849a5778270b987a830584e827`.
- Final policy: `5ab92ebe04459419bc724b48c6e20bde2464d7b6d880396e504406aa08806d4a`; VecNormalize: `3161ba46c65bac3bc23fa4ccb300c52c95fda63ee4190d9f30d2f0bd4b9040ec`.
- Actual R2 training result: `ff3532a604b6816fb3ad4c7a11edfcd579bcb924445abca52a2fdab8ea4dcf20`, invocation `a19900b2bfa64d8d8372b67bc0564139`.

Independent byte checks matched all nine source bindings and seven approval inputs. Direct deterministic CPU PPO consumed physical 69-channel observations and applied the existing action/rate filter; no online FNO inference, MPC substitution, policy update or policy selection occurred.

## Independent raw recomputation

All 3200 result-listed coefficient files were independently rehashed. Each body/branch contains exactly 16000 strictly ordered samples on (148,228] at 0.005 spacing. Recomputed every branch's drag/lift means, centered and total RMS, and rear-lift peak, plus paired metrics in all six windows; agreement with saved output is within 1e-14. No model or HDF was loaded for this review.

| Predeclared window | Samples/branch | Drag reduction | Rear-Cl centered RMS ratio | Mean-bias / paired-zero RMS |
|---|---:|---:|---:|---:|
| Early (148,160.4] | 2480 | 2.52274047% | 0.842337960 | 0.368591292 |
| Early first (148,154.2] | 1240 | 2.85345095% | 0.846137323 | 0.361900443 |
| Early trailing (154.2,160.4] | 1240 | 2.19202011% | 0.838468076 | 0.375282023 |
| **Primary (168,228]** | **12000** | **2.37796490%** | **0.795798378** | **0.322286722** |
| Historical-boundary companion [168,228] | 12001 | 2.37833772% | 0.795798492 | 0.322366798 |
| Full (148,228] | 16000 | 2.44101914% | 0.803539927 | 0.343010009 |

The primary total Cd is 2.24290496773565 versus paired zero 2.2975396541652002. Rear Cl mean is +0.38113820242083973; centered RMS is 0.941115914069231 versus zero 1.1826059717635977. Primary peak |Cl| is 1.720667761 versus zero 1.647306236: lower fluctuations do not imply a lower absolute peak. The companion includes exactly one additional t=168 sample; it is not substituted for the primary denominator.

All six windows exceed 2% paired drag reduction and reduce lift fluctuations; all six fail both 10% and 20% mean-bias references. The original train baseline rear-Cl RMS 1.1826535012844825 gives primary bias about 0.322274, also failing both references. These are real paired physical benefits, not completion of all constraints or proof of generalization across starts.

Early zero arrays (all raw columns, both bodies) exactly match the prior FC-E053 4096-step policy trial. Relative to that trial's early drag reduction 0.505760% and RMS ratio 0.973707, the new early drag/RMS improve, while mean-bias ratio worsens from 0.123721 to 0.368591. Historical restart boundary representations differ, so this review does not claim byte-identical restart provenance to the older successful CFD-only baseline merely because the companion statistics nearly coincide.

## Actions, solver health and cleanup

404/800 endpoints are saturated; 255/800 are rate-limited. Maximum |omega| is 0.75 and maximum applied delta is 0.10000000000000003 (floating-point representation of 0.1). This is not the previous constant-action policy, but saturation remains substantial.

All 1600 recorded solver segments ended cleanly with 20 solver steps each; maximum Courant number is 0.245582471 and maximum absolute per-step global continuity error is 8.9009479e-13. The 4004 resource samples have minimum MemAvailable 120469553152 bytes; MemFree is not the UMA capacity criterion.

Both exact owned container IDs were independently absent from `docker ps -a` after completion:

- `ac162bc6fad997631b7d567eb5905773130e13dd87428a18b4fee8c81a15b6f3`, terminal receipt SHA `5bd978544dbfaf7dfc4a21045aac5a1b7fb93e6103b7e2bcaccbcee5c35f3447`.
- `017f75352f06a4a7fc0332fe70e1febde799b83df4d99fc3727d0753ea0684e6`, terminal receipt SHA `9a0757904ec4f553b5e463012a9dccd4846e50c896680dda4d3dda0547cfa149`.

Both retained terminal states report OOMKilled=false; exit137 belongs to stopped sleeping container processes, not failed solver segments. The driver reports the original restart tree unchanged; this reviewer did not reread its large field payloads.

## Decision scope

The genuine FNO-training → HydroGym/SB3 PPO → direct real-CFD feedback chain has now completed an 80D/U paired run. Its primary mean-lift bias still fails both references. Keep the successful original CFD-only baseline and failed K1 formal surrogate evaluation separate. Do not relax thresholds, call this full project success, or infer that increasing PPO budget alone solved transfer. The fixed24 H5 diagnostic found only +0.002769 mean surrogate-return change with mixed per-case effects. Next work requires a separately reviewed hypothesis addressing persistent bias and model/reward/observation limitations, not an automatic budget or weight sweep.
