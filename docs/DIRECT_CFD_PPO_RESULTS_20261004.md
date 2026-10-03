# Direct real-CFD PPO stage results

## Outcome

The frozen final policy from the fixed 2048-transition direct real-CFD PPO run
passed the unchanged paired physical checks at both the b00 training start and
the held-out-time b01 validation start. Each comparison used two new
80 D/U OpenFOAM branches from an identical restart and only the final 60 D/U
for statistics.

| phase | role | total mean Cd | rear Cl' RMS | rear mean Cl |
|---|---|---:|---:|---:|
| b00 train | PPO | 2.200543 | 1.106544 | -0.023568 |
| b00 train | zero | 2.297526 | 1.182654 | -0.019378 |
| b01 validation | PPO | 2.202531 | 1.097453 | -0.045338 |
| b01 validation | zero | 2.300298 | 1.172526 | -0.041090 |

The acceptance quantities and unchanged targets were:

| phase | drag reduction (>=2%) | rear Cl' ratio (<=1.05) | mean-bias ratio (<=0.10) | joint |
|---|---:|---:|---:|---|
| b00 train | 4.2212% | 0.935645 | 0.019928 | PASS |
| b01 validation | 4.2502% | 0.935974 | 0.038667 | PASS |

The policy SHA-256 was
`99065e2e524ceda2221d5905e31db78cbfcdb0b82fc0bcb82b03ad0fefa96204`;
the matching frozen `VecNormalize` SHA-256 was
`d6cfc3cffec6e26abcde0155af2e1bd381e0a58ecb9bfa4fd99a3c1ba1d789a4`.
Both evaluations used deterministic actions, `training=False`, and
`norm_reward=False`; policy tensors and observation statistics had identical
before/after fingerprints.

## Evidence

- b00 result: `artifacts/direct_cfd/directppo2048_b00_eval80_v1/physical_result.json`
- b00 rollout and raw actions: `artifacts/direct_cfd/directppo2048_b00_eval80_v1/rollout/rollout_result.json`
- b00 independent audit: `artifacts/direct_cfd/directppo2048_b00_eval80_v1/audit_receipt.json`
- b01 result: `artifacts/direct_cfd/directppo2048_b01_eval80_v1/physical_result.json`
- b01 rollout and raw actions: `artifacts/direct_cfd/directppo2048_b01_eval80_v1/rollout/rollout_result.json`
- b01 independent audit: `artifacts/direct_cfd/directppo2048_b01_eval80_v1/audit_receipt.json`
- preserved OpenFOAM cases and per-segment journals reside under the corresponding
  run directory and `cfd/tandem_cylinders/cases/direct_cfd_<run>_*`.

## What this stage does and does not establish

Implemented and demonstrated here are an SB3 PPO policy evaluated through the
pinned upstream HydroGym `FlowEnv` API and the project-specific OpenFOAM adapter
in a genuine online CFD loop, actual causal
force history, constrained rear-cylinder rotation, and a beneficial frozen
policy at one training start and one untrained-time validation start. b00 and
b01 are separated by 18 D/U, about three nominal shedding periods; this is not
evidence that they are statistically independent physical samples or a broad
phase-generalization study. The final2048
policy is therefore retained as the validated CFD-only baseline. An optional
`std=0.075` continuation path is implemented and tested but was not run, because
the frozen policy already produced useful physical behavior.

This does **not** establish that FNO-surrogate PPO has passed its separate gate,
and the physical benefit is not attributable to FNO. It does not yet provide a
frozen-test phase, multiple RL seeds, uncertainty intervals across independent
training runs, hardware closed-loop evidence, CFD computational-efficiency
evidence, or a standalone novelty claim. The b01 result must not be used for
training or checkpoint selection. PPO is supplied by Stable-Baselines3, not by
HydroGym; the OpenFOAM transport is a project adapter, not an upstream HydroGym
CFD backend.

## Fixed-sequence discriminator at b01

A predeclared post-hoc replay applied the immutable 800 `applied_omega`
endpoints from the b00 feedback run open-loop from the same b01 restart. It used
the same linear ramps, 0.1 D/U decisions, 0.005 D/U solver step, 80 D/U horizon,
and final 60 D/U statistics. All endpoints were reproduced exactly and all 800
solver segments passed numerical checks.

| b01 branch | total mean Cd | change relative to zero | rear Cl' ratio vs zero | mean-bias ratio |
|---|---:|---:|---:|---:|
| frozen observation feedback | 2.202531 | -4.2502% | 0.935974 | 0.038667 |
| fixed b00 action sequence | 2.317412 | +0.7440% | 0.810540 | 0.033997 |
| zero | 2.300298 | 0 | 1 | — |

Thus the fixed b00 sequence did not reproduce the feedback drag benefit at this
b01 start; feedback Cd was 4.9573% below the open-loop-sequence reference. This
supports added value from observation feedback for this single held-out-time
comparison. It is not proof over statistically independent starts, arbitrary
phases, disturbances, or multiple policies. Evidence is under
`artifacts/direct_cfd/b00seq_b01_openloop_v1/`; the immutable audit receipt is
`audit_receipt.json`.

## Secondary ideal fluid-power diagnostic

The post-hoc b00 diagnostic at
`artifacts/direct_cfd/directppo2048_b00_eval80_v1/hydrodynamic_power_diagnostic.json`
uses the existing rear-cylinder `CmPitch` and linearly ramped angular velocity.
Following the coefficient-file header axis and OpenFOAM moment normalization,
its mean ideal actuator-to-fluid coefficient is 0.010975. This is excluded from
the canonical gate. It is not motor electrical input, drivetrain loss, net
energy, or proof of energy savings; the fixed-cylinder CFD problem has no
literal towing-energy bill.
