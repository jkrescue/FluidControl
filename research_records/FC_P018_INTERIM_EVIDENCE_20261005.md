# P018 intermediate evidence — not scientific admission

Observation: 2026-10-05 14:39 UTC. The formal evaluation remains live under
`fluid-control-fcp018-posteval-20261005.service`, invocation
`ef589f7dbeff4fa0ab064309409971ad`, PID 56992. Validation10 is complete;
dynamic6 and force-window conclusions are pending. Do not use this note as a
terminal acceptance decision.

## Completed validation10 comparison

Independent review rehashed all four P018 validation10 receipt files and
checked approved model, state, dual manifest, precision and frozen-chain
bindings. The 1,240 sampled case/horizon/start identities match P009 and P015
(stride 25, batch 4). This is repeatedly used development validation, not an
independent frozen test.

| Metric (lower is better) | P009 | P015 | P018 |
|---|---:|---:|---:|
| Rear Cl MAE, H1 | 0.0201247225574 | 0.0413481669442 | 0.0202345944359 |
| Rear Cl MAE, H10 | 0.0279902936221 | 0.0415972915711 | 0.0278948755644 |
| Rear Cl MAE, H50 | 0.0284019757903 | 0.0319326157592 | 0.0294920149109 |
| Rear Cl MAE, H100 | 0.0387409503162 | 0.0369867039909 | 0.0402698922934 |
| H100 pooled total-Cd NRMSE | 0.00558671329285 | 0.01048991562436 | 0.00605054791083 |
| H100 macro total-Cd NRMSE | 0.00518736584209 | 0.01044541752037 | 0.00574908728566 |

The field summaries at all four horizons are identical, as expected for the
frozen flow model. All three validation endpoint checks pass, but P018 H100
rear-Cl MAE is worse than both references. Lower learning rate has not yet
demonstrated a repair of control-relevant long-horizon lift prediction.
Pooled and macro aggregation must not be interchanged.

P018 step receipt:
`artifacts/fcp018_reduced_rate_training_20261005/posteval_fc_p018/step_receipts/validation10.json`
SHA256 `699334e0b7c12839017198f853736816c6db7b3c3c81f1d19e9fc1e0c903f4e0`.
Evaluation SHA256 `171e909f6a408ddb719ad5ff5e37e4706438ef08a81fc6406f6e398162f652bd`.

## Physical mean-lift criterion: preparatory sensitivity check

The user authorized a conditional review of the physical mean-lift limit
after one further hour, no earlier than 15:20 UTC. No limit is changed here.
The quantity below is actual CFD `abs(mean rear Cl)/baseline rear Cl fluctuation RMS`,
not the surrogate's mean prediction error.

| Existing real-CFD run | Mean drag reduction | Rear Cl fluctuation RMS ratio | Mean-lift ratio | Joint result at hypothetical 0.10 / 0.15 / 0.20 mean limits |
|---|---:|---:|---:|---|
| CFD-only PPO b00 | 4.221212% | 0.93564500 | 0.01992846 | Pass / Pass / Pass |
| CFD-only PPO b01 | 4.250159% | 0.93597372 | 0.03866683 | Pass / Pass / Pass |
| b00 action sequence replayed open-loop at b01 | -0.744022% | 0.81053959 | 0.03399713 | Fail / Fail / Fail: drag |

Other criteria in this sensitivity calculation remain drag reduction at least
2% and fluctuation RMS ratio at most 1.05. b00 uses [168,228]; b01 and the
open-loop replay use [150,210]. Each compares control with its matched zero-action
reference over the same 60-time-unit statistics window. These examples do not identify physical mean
lift as the dominant failure. They do not establish surrogate-assisted control,
frozen-test robustness, or net-energy savings. b00/b01 are development phases.

Source files and SHA256:

- `artifacts/direct_cfd/directppo2048_b00_eval80_v1/physical_result.json`:
  `3c9587d66482ca98f773a4bccc641895f84bacc930df8009bc590d9ba49e2c2c`
- `artifacts/direct_cfd/directppo2048_b01_eval80_v1/physical_result.json`:
  `da42b6018f35677af8205ce9ba78cc9753ca348abf5f5994f2189554c4eaba36`
- `artifacts/direct_cfd/b00seq_b01_openloop_v1/result.json`:
  `5a18f282c44ba803b1544d00fc6f5096ad116cebc20b8e0aac427c4a59b022c1`

Next action: complete original P018 dynamic6 and force-window evaluation,
independently review the full numerical result, then decide the next measured
intervention. Do not reinterpret a change in engineering limits as an accuracy
improvement or replace the requested surrogate-assisted CFD closed loop with
the already successful CFD-only baseline.
