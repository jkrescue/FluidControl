# Direct real-CFD PPO stage results

## Outcome

The frozen final policy from the fixed 2048-transition direct real-CFD PPO run
passed the unchanged paired physical checks at both the b00 training phase and
the independently reserved b01 validation phase. Each comparison used two new
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

Implemented and demonstrated here are an official HydroGym/SB3 policy connected
through the project OpenFOAM adapter to a genuine online CFD loop, actual causal
force history, constrained rear-cylinder rotation, and a beneficial frozen
policy at one training and one independent validation phase. The final2048
policy is therefore retained as the validated CFD-only baseline. An optional
`std=0.075` continuation path is implemented and tested but was not run, because
the frozen policy already produced useful physical behavior.

This does **not** establish that FNO-surrogate PPO has passed its separate gate,
and the physical benefit is not attributable to FNO. It does not yet provide a
frozen-test phase, multiple RL seeds, uncertainty intervals across independent
training runs, hardware closed-loop evidence, CFD computational-efficiency
evidence, or a standalone novelty claim. The b01 result must not be used for
training or checkpoint selection.

## Secondary ideal fluid-power diagnostic

The post-hoc b00 diagnostic at
`artifacts/direct_cfd/directppo2048_b00_eval80_v1/hydrodynamic_power_diagnostic.json`
uses the existing rear-cylinder `CmPitch` and linearly ramped angular velocity.
Following the coefficient-file header axis and OpenFOAM moment normalization,
its mean ideal actuator-to-fluid coefficient is 0.010975. This is excluded from
the canonical gate. It is not motor electrical input, drivetrain loss, net
energy, or proof of energy savings; the fixed-cylinder CFD problem has no
literal towing-energy bill.
