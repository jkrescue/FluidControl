# FC-P015 terminal status and next stage

Observed 2026-10-05 11:17 UTC. The overall online closed-loop objective is incomplete.

## Completed checks

- Exact retained training invocation `7842742926284d0c94b0383163d5dc0b` exited successfully: 171 optimizer updates, 1368 original-order training windows, accumulation size 8.
- Root and an independent reviewer recomputed the complete candidate audit, including 44 real HDF inputs, saved official model archives, optimizer states, frozen flow model and all fixed diagnostic panels. Both audits agree.
- Completion SHA: `9c27e5ebe104a8988f274b9c3e8cd0d6728838c1d6aa34daff9d25d310605fdf`.
- Actual CPU reload of both FNOs using the official pinned image and shared dual adapter succeeded. No forward, optimizer, GPU, or model save. Reload receipt SHA: `925a7dc0c1a1be0157afd5838018b1a3bccd74f0083b2d70718e8a65e8c2b18c`. External Docker evidence is recorded separately in `FC_P015_CPU_DUAL_RELOAD_EXECUTION_20261005.json`.

## Accuracy remains unproven

The fixed six training windows are diagnostics, not validation or checkpoint selection. Initial-to-terminal mean objective increased from 0.00624614 to 0.00730327; only 1/6 windows improved. The H100 centered rear-lift residual MSE improved in 5/6 H1 windows and 4/6 autoregressive windows, while squared mean bias worsened in 6/6 and 5/6 respectively. Partial waveform improvement does not establish accurate total lift or control benefit.

## Ordered continuation

1. Execute the unchanged complete formal evaluation: validation10, dynamic6, six force windows and original combined criteria. Fixed terminal only; no intermediate checkpoint selection.
2. If rejected, preserve the complete negative result and design the next testable correction from the measured errors. Do not lower thresholds or blindly extend training.
3. Only an accepted surrogate proceeds to a newly compatible HydroGym/SB3 PPO policy, then actual paired OpenFOAM feedback under the original drag/lift/action requirements.
4. Preserve the existing CFD-only PPO baseline. Do not describe it as surrogate-assisted, or describe an offline surrogate as online FNO planning.

The four-layer architecture remains CFD truth → official PhysicsNeMo surrogate → controller → real CFD feedback. PhysicsNeMo Curator/DataPipe prepare real data; HydroGym provides the environment interface; SB3 provides PPO. Project-owned coupling code is not an official library API. Online simulation feedback is not yet a demonstrated hardware real-time controller.

Final acceptance remains mean total drag reduction ≥2%, rear lift fluctuation ratio ≤1.05, normalized mean rear lift magnitude ≤0.10, with the original rotation/rate constraints and paired 80 D/U simulations (discard first 20, measure final 60). Training completion is not project completion.
