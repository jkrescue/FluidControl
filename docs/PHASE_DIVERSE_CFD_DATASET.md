# Independent-phase OpenFOAM augmentation

The expanded-v1 dataset has broad rear-cylinder rotation coverage but every
trajectory starts from the same uncontrolled `t=80` field. This augmentation is
designed to measure and reduce that initial-phase bias without introducing any
synthetic labels.

## CFD scene

- Solver: OpenFOAM v2512 `pimpleFoam` in the existing isolated container.
- Flow: two fixed-centre tandem circular cylinders, `D=1`, `L/D=5`, `Re=100`.
- Actuator: angular velocity of the rear cylinder, with `|omega| <= 5`.
- Time step: `0.005`; field output interval: `0.1`.
- Forces: front and rear `Cd/Cl`; 32 wake-velocity probes are retained.
- Grid: existing 19,290-cell coarse discovery mesh.

## Phase panel

Four trajectories restart from the validated, uncontrolled baseline at physical
times `82`, `84`, `86` and `88`. The uncontrolled shedding period is approximately
`6.154`, so the restart spacing samples materially different vortex phases. Each
case runs for 24 time units, approximately 3.9 shedding periods, and writes 241
field frames.

| Case | Split | Restart | Schedule |
|---|---|---:|---|
| `phase_train_82_multisine` | train | 82 | smooth multisine |
| `phase_train_84_ramp` | train | 84 | rate-bounded random ramp |
| `phase_validation_86_multisine` | validation | 86 | independent multisine |
| `phase_test_88_ramp` | test | 88 | independent random ramp |

All action tables start at `omega=0`, remain inside the existing action support and
have `|domega/dt| <= 1.6`, matching the expanded dataset's rate envelope.

## Generation and validation

The generator refuses to overwrite cases and writes the exact restart provenance,
action table and numerical settings to each `case_config.json`.

```bash
python3 cfd/tandem_cylinders/make_phase_diverse_control_dataset.py \
  --audit artifacts/tandem_cylinders/phase_diverse_v1_action_audit.json

bash cfd/tandem_cylinders/run_phase_diverse_case.sh phase_train_82_multisine
```

The first case is a numerical/runtime pilot. Remaining cases may be run concurrently
only after the pilot ends normally and system memory remains above the 20 GiB
reserve. A separate curator will keep this phase panel identifiable; it must not be
silently merged into the original test split.

## Intended use

1. Audit FNO rollout accuracy from independent shedding phases.
2. Add the two training trajectories during rollout-aware fine-tuning if the phase
   audit exposes a material gap.
3. Keep the validation/test phase cases held out for model and policy selection.

This panel does not replace policy-distribution CFD. Once a candidate HydroGym
policy exists, additional trajectories must be collected from states and actions
actually visited by that policy.
