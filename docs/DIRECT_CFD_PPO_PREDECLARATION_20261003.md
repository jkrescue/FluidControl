# Direct real-CFD PPO commissioning predeclaration

This track demonstrates genuine reinforcement learning and online control in
the real tandem-cylinder OpenFOAM solver. It does not replace, relax, or bypass
the separate PhysicsNeMo-surrogate readiness gate.

## Fixed first-run protocol

- Geometry/solver: Re=100, L/D=5 tandem cylinders; rear-cylinder rotation;
  pinned OpenFOAM v2512 image.
- Interface: official HydroGym `FlowEnv`, `PDEBase`, and `TransientSolver` at
  commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`; the OpenFOAM adapter is
  project-local and is not represented as an official HydroGym backend.
- Training sources: train-only phase restarts `b00` and `b02`. Validation and
  frozen-test cases are neither opened nor enumerated.
- Observation: canonical 69D vector: 32 `(u,v)` probes, front `(Cd,Cl)`, rear
  `(Cd,Cl)`, and applied rear-cylinder angular speed.
- Action: `|omega| <= 0.75`, control interval 0.1 D/U, and
  `|delta omega| <= 0.1` per decision.
- Reward: unchanged `canonical_joint_v1` costs. Each episode begins with real
  OpenFOAM force samples from `t0-6.1` through `t0`, never future samples, so
  the 6.15-D/U trailing causal ledger is immediately defined.
- PPO: two parallel environments, 128 decisions per episode, CPU SB3 PPO,
  `n_steps=128`, batch 64, four optimization epochs, seed 20261003, and 2048
  real-CFD transitions. Policy observations and training rewards use persisted
  `VecNormalize` statistics; the unnormalized physical reward and every cost
  component are journaled every transition.
- Checkpoints: every 256 transitions. Every episode case is preserved as a new
  generation during the first run; there is no episode-case deletion/recycle.
- Safety: at least 20 GiB host `MemAvailable`, clean 20-step solver segments,
  `max Courant < 0.8`, absolute global continuity error below `1e-5`, finite
  observations, and absolute force coefficients below 10.

Before the full run, a 20-transition zero-action runtime/transport probe must
complete under the identical two-worker architecture. The probe does not train
PPO and makes no control-benefit claim.

## Process isolation

Two host workers each own one persistent, pinned OpenFOAM container and a
private AF_UNIX socket. The read-only HydroGym/SB3 policy container connects
through those sockets; it receives no Docker socket and cannot launch host
containers. Shutdown targets only the exact run-owned PIDs and container names.

## Interpretation

The 6.15-D/U reward ledger is a training diagnostic. Rear mean lift in a short
window can be phase-sensitive, and its threshold-normalized squared penalty can
dominate the drag signal. Therefore reports retain each raw reward component
and reward distribution; a short-window joint-pass fraction is never described
as final physical qualification.

The `omega^2` and `delta omega^2` terms are control-amplitude and smoothness
regularizers only. They are not a rotary-energy or net-power measurement: this
protocol does not compute cylinder torque times angular velocity. Physical
claims are therefore limited to the separately paired total-drag and lift
statistics, never net energy savings.

After training, a frozen policy still requires a predeclared, phase-matched,
paired real-CFD evaluation over the final 80 D/U. The unchanged physical gate
is total drag reduction at least 2%, rear `Cl'` RMS no more than 1.05 times the
zero branch, and absolute mean rear `Cl` no more than 0.10 times the zero-branch
rear `Cl'` RMS. This commissioning run is not guaranteed to pass that gate.
