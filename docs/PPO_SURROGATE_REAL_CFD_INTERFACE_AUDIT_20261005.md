# Surrogate PPO / real-CFD interface audit (2026-10-05)

This is a read-only control-interface audit. It does not authorize PPO, change
the reward, or claim closed-loop benefit.

## What the code already keeps consistent

- `run_candidate_full40_canonical_ppo.py` fixes each surrogate episode to 100
  control decisions. The four `DummyVecEnv` members always reset to frame 0 of
  the train-only b00/b02/b04/b06 zero cases. The default 8192 PPO transitions
  therefore span repeated 100-step episodes; they are not one 8192-step FNO
  rollout.
- `TandemFNOStepper` feeds the predicted field back at every step. It does not
  read a target state or apply teacher forcing. This agrees with the candidate
  lineage requirement of zero teacher forcing.
- Both observation paths use
  `32*(u,v), front Cd, front Cl, rear Cd, rear Cl, omega`, exactly 69 physical
  values. The surrogate de-normalizes its internal field before bilinear probe
  sampling and de-normalizes its predicted forces. The OpenFOAM path reads the
  same probe coordinates and physical force coefficients. PPO `VecNormalize`
  is explicitly identity (`norm_obs=false`, `norm_reward=false`), so the
  persistent real-CFD policy process is correct only under that bound identity
  contract.
- Both paths enforce `|omega| <= 0.75`, `dt=0.1 D/U`, and
  `|delta omega| <= 0.1`. The FNO receives `(omega_now, omega_next)`; the real
  OpenFOAM boundary condition linearly ramps from the previous to the applied
  endpoint over the same control interval.
- Existing read-only 69D parity artifacts report raw-OpenFOAM versus curated
  probe errors below 0.0045 and identical force/action channels on the audited
  legacy validation/test frames. This supports channel order and scale, not
  policy performance.

The scope remains fixed tandem cylinders at Re=100. Rear-cylinder lift
fluctuation is a force statistic on fixed cylinders, not measured VIV. The
`omega^2` reward term is an actuation proxy, not real energy consumption.

## Blocking semantic mismatch before surrogate PPO execution

The two environments call the same `canonical_joint_v1` cost and use the same
6.15-D/U trailing-window formula, but their reset histories differ:

- `TandemSurrogateFlow.reset()` clears its force history and inserts only the
  frame-0 force. The canonical point-as-interval helper is not ready for the
  first 60 decisions and becomes ready after decision 61 (62 samples spanning
  6.1 D/U plus one 0.1-D/U sample interval). The physical drag/lift gate terms
  are zero during that warm-up; only action and rate costs remain.
- `DirectOpenFOAMFlow.reset()` requires 62 real, causal 0.1-D/U force samples
  ending at the restart. Its window is ready before the first controlled step.

Thus the PPO training reward timing is not currently the real-CFD reward
timing. Passing H100 FNO gates does not resolve this mismatch.

## Minimal compatible repair and data source

Before any candidate PPO execution, seed each surrogate train environment with
the same 62-point causal force prehistory used by the direct-CFD worker. The
source must be the pinned raw OpenFOAM restart case and exact restart time from
each train case's `case_config.json`; do not synthesize values from a baseline,
repeat frame 0, or interpolate missing samples.

A read-only inventory confirmed that all four current train resets have every
required endpoint in both `forceFront` and `forceRear`:

| phase | restart | required interval | missing front/rear |
|---|---:|---:|---:|
| b00 | 148.0 | 141.9--148.0 | 0 / 0 |
| b02 | 106.0 | 99.9--106.0 | 0 / 0 |
| b04 | 120.0 | 113.9--120.0 | 0 / 0 |
| b06 | 134.0 | 127.9--134.0 | 0 / 0 |

All four resolve to the existing `tandem_backward_dt005` raw force history.
The exact raw files are:

- `cfd/tandem_cylinders/cases/tandem_backward_dt005/postProcessing/forceFront/0/coefficient.dat`,
  SHA-256 `bce88443ce3d19a6411c31b9266af16a3dd4e992f7adb1ddabc659238a1e88d1`;
- `cfd/tandem_cylinders/cases/tandem_backward_dt005/postProcessing/forceRear/0/coefficient.dat`,
  SHA-256 `654c5bbf64b758505bdfe4d379e153f460862902309e96d2b01974e45536680b`.

The b00/b02/b04/b06 `case_config.json` SHA-256 values are respectively
`cab6d5ee02153fc10a617876641ae9fc9b54d6c0776d2ca748f2fe3fb728db24`,
`76f84672c5588855f57206d547b43325962e0727cb6f0603d56d9e2ba0bb30a8`,
`9c4706a7457ab0a1c6126100fd901da7818c973ded6417f4ed8e170785f07c20`,
and `b88c73d6bfd83a7d2b560a368a6268f3bab0d301ee68c258d55bb6d9250bea41`.

The repair should reuse one shared, fail-closed prehistory reader or an exact
equivalent of `direct_cfd_openfoam_worker.actual_causal_prehistory`, bind the
raw files and case metadata by SHA, and add tests that:

1. surrogate and direct resets are window-ready with identical 62 timestamps
   and forces for a fixed source;
2. a missing, conflicting duplicate, future, or non-finite sample is rejected;
3. the first-step canonical reward components match for the same force/action
   sequence; and
4. no validation or frozen source is opened while constructing train PPO
   environments.

Also bind an actual matched-start frame-0 69D parity receipt and verify that
the identity-VecNormalize policy path and the persistent raw-observation
inference path produce the same action on representative 69D inputs.

## Horizon limitation

The surrogate PPO and its surrogate validation run exactly 100 autonomous FNO
steps. The predeclared real OpenFOAM feedback run is 800 decisions and uses a
fresh real 69D observation after every 0.1-D/U segment, so it does not roll the
FNO for 800 steps. Nevertheless, policy state-distribution coverage and
long-horizon closed-loop stability beyond decision 100 are unverified until
the guarded 800-step paired real-CFD run completes. H100 passage must never be
reported as 800-step autonomous stability.

Finally, the surrogate online reward uses 0.1-D/U endpoint samples over 6.15
D/U, whereas the real-CFD final acceptance reads the dense 0.005 force history
over the predeclared final 60 D/U. The latter is intentionally a higher-fidelity
physical outcome evaluation, not the same statistic as the PPO reward.

## CPU implementation approval

Lead approved a narrowly scoped compatibility implementation and CPU tests.
The canonical surrogate must run on the source restart's absolute CFD clock,
atomically replace its one-point reset history with the verified 62 unique
causal endpoints, and recreate that history from immutable values on every
reset. An identical duplicate raw row may be merged exactly as in the direct
CFD reader; a conflicting duplicate must fail closed. The last timestamp and
four-force row must equal the current frame-0 state, and any different reset
state or time must be rejected. The implementation must also bind the HDF5
embedded case configuration and the raw force-file hashes. This approval is
for CPU implementation and regression tests only; it does not authorize PPO,
FNO training, CFD execution, or a change to any scientific gate.
