# HydroGym multi-start PPO and real-CFD feedback audit — 2026-10-02

## Scope

This milestone connects the existing real OpenFOAM tandem-cylinder dataset and the accepted 20-epoch official PhysicsNeMo FNO checkpoint to a pinned HydroGym PPO environment, then executes the frozen policy against observations produced by segmented real OpenFOAM solves. It is an engineering and scientific audit, not a control-benefit claim.

## Reproducible configuration

- PhysicsNeMo checkpoint: epoch 20, previously accepted by the independent surrogate screen.
- HydroGym source: commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`.
- PPO training: 8 real-CFD-derived starts, 8 vector environments, 8192 total timesteps.
- Checkpoints: 0, 2048, 4096, 6144 and 8192 timesteps.
- Fixed checkpoint panel: 4 validation plus 4 test starts, 16 surrogate steps each.
- Final held-out audit: 24 validation/test evaluations. The common t=80 frame is counted once per split, giving 9 unique starts per split.
- Device: GPU 0 in the pinned project container. The resource guard recorded a minimum system `MemAvailable` of 110.945 GiB, comfortably above the required 20 GiB unified-memory reserve.

## Surrogate PPO result

The fixed-panel mean reward moved from -2.842426 at step 0 to -2.836082 at step 8192. On the authoritative independent audit:

| Split | Unique starts | Mean reward change | Positive starts | Mean Cd change | Mean Cl RMS change |
|---|---:|---:|---:|---:|---:|
| validation | 9 | +0.025348 | 8/9 | -0.004236 | -0.006102 |
| test | 9 | +0.017810 | 7/9 | +0.000131 | -0.008301 |

This is evidence that the PPO software path learns a small average improvement inside the FNO surrogate. It is not sufficient evidence for physical flow control because the FNO force rollout error grows at longer horizons and performance is start-dependent.

## Real OpenFOAM closed-loop result

The policy was first executed for 3 feedback intervals from the real t=80 restart. Each interval read 64 real wake-probe velocities plus real rear-cylinder Cd/Cl, inferred one deterministic policy action, changed the rear-cylinder rotating-wall boundary and ran the next OpenFOAM segment. The run completed cleanly and its mean objective difference from the matched zero-action reference was -0.002397. This short run only established integration.

The same chain then completed 32 feedback intervals:

| Metric | Result |
|---|---:|
| Mean objective difference from zero | +0.003855 (worse; lower is better) |
| Improved intervals | 8/32 |
| First 8 intervals, mean difference | -0.002174 |
| Last 24 intervals, mean difference | +0.005864 |
| Mean Cd difference | +0.010697 |
| Mean Cl² difference | -0.034315 |
| Action range | -0.02373 to +0.10360 |
| Action RMS | 0.04526 |
| Maximum Courant number | 0.2436 |
| Maximum continuity residual | 6.17e-13 |

The CFD integration remained numerically healthy, but the controller did not sustain a benefit. Drag increased enough to dominate the reduced lift contribution.

## Root cause and corrected gate

All expanded trajectories share the physical t=80 restart at frame 0. The final PPO policy's held-out surrogate reward change at that start was -0.003373, even though the mean over diverse starts was positive. The earlier gate accepted the mean and therefore did not protect the specific physical restart used by the real replay.

The real-CFD runner now requires both:

1. positive aggregate reward with at least 6/9 positive starts in each independent split; and
2. positive, mutually consistent held-out reward at the shared t=80 restart.

The current policy intentionally fails the second condition. Another expensive t=80 real-CFD feedback run is prohibited until a new controller passes it.

## Evidence

- `artifacts/hydrogym/tandem_ppo_multistart_8192_spark/audit_independent.json`
- `artifacts/hydrogym/tandem_ppo_multistart_8192_spark/checkpoint_evaluations.json`
- `artifacts/hydrogym/tandem_ppo_cfd_feedback_multistart_8192_01/result.json`
- `artifacts/hydrogym/tandem_ppo_cfd_feedback_multistart_8192_32step_01/result.json`

## Next experiment

Train a controller with an explicit t=80 objective while retaining diverse real-CFD starts, select checkpoints on both the t=80 gate and the independent multi-start panel, and only then repeat the real-CFD replay. The medium-grid constant-rotation case should finish first so numerical sensitivity is available alongside the next control result.
