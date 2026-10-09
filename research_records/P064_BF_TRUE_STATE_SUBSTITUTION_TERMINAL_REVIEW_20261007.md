# P064 B/F True-State Substitution Terminal Review (2026-10-07)

## Disposition

**Independent engineering and numerical review: ACCEPTED as a bounded diagnostic only.**

This run compares the B and F aerodynamic FNO branches on the same saved physical CFD states and realized actions. It is not training, CFD, admission evidence, or an additive causal decomposition of force error. In particular, the difference between teacher-forced and autoregressive errors is descriptive: replacing the input state changes the point at which a nonlinear model is evaluated.

## Bound execution

- Unit: `fluid-control-p064-bf-true-state-substitution-20261007.service`
- Invocation: `c9ea8b5a18d54bc1bb5f3e20631c17f0`
- Terminal state: `MainPID=0`, `ActiveState=active`, `SubState=exited`, `ExecMainStatus=0`
- Approval SHA-256: `387404745ecbbb1e517464e37aeda9779c0ae55181304e276901eeb208591dae`
- Frozen worker SHA-256: `9331850503b7a6425dd0d2746b6954d67a4a043dc0b71855fc5793a95c3cf99d`
- Result: `artifacts/p064_bf_true_state_substitution_20261007/result.json`
- Result SHA-256: `cf891b9c476f6b5dee842061209d75b44ea02589c280017ad0680c144f028633`
- Supervisor-result SHA-256: `a67afff63904112d648cdfba28a7bffbcc44ce66b105b7cde3527bdfc7a275ff`
- Actual calls: 160 aerodynamic, 0 flow, 0 optimizer
- Memory ledger: 33 samples over 16.015 s; minimum `MemAvailable=119930249216` bytes (111.69 GiB)

The actual unit used the reviewed 12 GiB/no-swap/CPU1/Tasks64, 600 s inner/630 s outer/20 s stop, 6 GiB allocator and 50/22/20 GiB Available/reserve contract. No retry or second execution was used.

## Independent validation

The review independently checked, without loading either model again:

1. All 160 rows are present exactly once: two models, two phases, eight fixed starts and five leads.
2. Every B/F pair has identical physical-current-state SHA, normalized-input SHA, current/target time, current/next action and truth force (80 paired points).
3. All 32 referenced NPZ files were rehashed and reread. Each row's current-state SHA, saved AR force, truth force, action clock and physical time were reproduced from its NPZ.
4. All 90 stored metric blocks (two models × three phase groupings × five leads × three predictors) were independently recomputed for four force components plus total drag. Stored MAE, RMSE and signed bias agree to floating-point roundoff.
5. All 32 H1 model rows reproduce the original saved H1 force bitwise. This is the expected identity because H1 starts from the same physical initial state.
6. B and F tensor digests are unchanged before/after inference, both use the same frozen flow tensor digest, and every parameter gradient remains absent.
7. The result explicitly records `scientific_admission=false` and `descriptive_nonadditive_noncausal_not_error_decomposition`.

The frozen consumer's `CheckpointIdentity` does contain `model_sha256` and `state_sha256`, and `DualFNOIdentity` contains the `flow` and `aerodynamic` identities used by the terminal receipt; no deferred identity-attribute failure is present.

## Main force results

Values are MAE. `TF` means the aerodynamic model receives the saved true current CFD state; `AR` is the already-saved autoregressive result. Rear-cylinder lift and total drag are shown.

| Scope | Lead | Model | TF rear-Cl | TF total-Cd | AR rear-Cl | AR total-Cd |
|---|---:|---|---:|---:|---:|---:|
| pooled | H1 | B | 0.138998 | 0.038065 | 0.138998 | 0.038065 |
| pooled | H1 | F | 0.136574 | 0.033974 | 0.136574 | 0.033974 |
| pooled | H2 | B | 0.141053 | 0.036412 | 0.131030 | 0.028274 |
| pooled | H2 | F | 0.141146 | 0.032267 | 0.131668 | 0.025662 |
| pooled | H3 | B | 0.153665 | 0.036397 | 0.132374 | 0.025320 |
| pooled | H3 | F | 0.153647 | 0.031843 | 0.135255 | 0.023067 |
| pooled | H4 | B | 0.167884 | 0.037575 | 0.148797 | 0.027976 |
| pooled | H4 | F | 0.166238 | 0.032597 | 0.150899 | 0.028224 |
| pooled | H5 | B | 0.180160 | 0.039123 | 0.165745 | 0.030762 |
| pooled | H5 | F | 0.176119 | 0.033823 | 0.170507 | 0.031422 |

At pooled H5, F's true-state MAE is 2.24% lower than B for rear lift and 13.55% lower for total drag. However, true-state H5 remains worse than the corresponding saved AR result for both models: B is +8.70%/+27.18% and F is +3.29%/+7.64% for rear-Cl/total-Cd. This rules out a simple claim that autoregressive flow-state drift is the sole source of H5 force error.

The improvement is not uniform by lead. F's pooled H2 rear-Cl is slightly worse than B (0.141146 versus 0.141053), and H3 is effectively unchanged. H1 is identical to the original evaluation by construction, with F better in the pooled value but not in every phase.

## Phase findings

| Phase | Lead | Model | TF rear-Cl | TF total-Cd | AR rear-Cl | AR total-Cd |
|---|---:|---|---:|---:|---:|---:|
| b01 | H1 | B | 0.155280 | 0.044178 | 0.155280 | 0.044178 |
| b01 | H1 | F | 0.147754 | 0.040206 | 0.147754 | 0.040206 |
| b01 | H5 | B | 0.173646 | 0.038923 | 0.180412 | 0.033761 |
| b01 | H5 | F | 0.169877 | 0.032146 | 0.186979 | 0.032872 |
| b03 | H1 | B | 0.122716 | 0.031953 | 0.122716 | 0.031953 |
| b03 | H1 | F | 0.125394 | 0.027743 | 0.125394 | 0.027743 |
| b03 | H5 | B | 0.186674 | 0.039322 | 0.151078 | 0.027764 |
| b03 | H5 | F | 0.182362 | 0.035500 | 0.154034 | 0.029973 |

- On b01, F improves true-state H5 rear-Cl and total-Cd versus B. Relative to each model's own AR result, b01 teacher forcing improves lift for both models and improves F drag slightly, but worsens B drag.
- On b03, F improves true-state H5 versus B, yet true-state substitution substantially worsens both models relative to their own AR force errors. F also worsens b03 H1 rear-Cl while improving total-Cd.
- Therefore neither a pooled average nor the H5 F-versus-B direction should hide the b03 behavior.

## Bounded conclusion

F changes the instantaneous aerodynamic mapping in a favorable H5 pooled direction, especially for drag, but not consistently at every lead or phase. Supplying the true CFD state does not generally reduce force error and often increases it, most clearly on b03. The evidence does not establish an additive split between flow-state error and aerodynamic-readout error, does not support “longer training” by itself, and does not promote F over B. Any subsequent short-H5 training hypothesis must remain a separately approved experiment with the already-open development panel unchanged.
