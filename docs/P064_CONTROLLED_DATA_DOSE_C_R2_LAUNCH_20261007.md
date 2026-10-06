# FC-E087 C50 training launch record

Date: 2026-10-07

This is an engineering launch record, not a terminal scientific result.

## Frozen scientific contract

- Arm C uses the same frozen P026 K1 parent, seed `20261003`, fresh AdamW,
  learning rate `1.5625e-7`, objective, precision, and 28 trainable aerodynamic
  parameters as arms A/B. Flow parameters and both lift-bias parameters remain
  frozen.
- The fixed budget is 256 windows and 32 accumulated updates. Each group of
  eight contains four controlled-b00 windows at positions `0,2,4,6` and four
  original-44 windows at positions `1,3,5,7`. The 128 b00 starts are a denser
  mechanical schedule, not a strict superset of the earlier 64 starts.
- Only the final checkpoint is eligible for later review. Development inference,
  PPO, and CFD are not authorized by this training approval.

## R1 operational failure

- Unit: `fluid-control-p064-controlled-dose-c50-20261007.service`
- Invocation: `9595da6d9092476298f597ef8ed08fb2`
- Exit status: 1
- Failure occurred at the first project import, before model loading, GPU use,
  output creation, or training: `ModuleNotFoundError: fluid_control`.
- Cause: the systemd environment omitted the immutable project's `PYTHONPATH`.
  The failed unit and journal are retained; its output path remained absent.

## Authorized R2 recovery

- Approval: `docs/P064_CONTROLLED_DATA_DOSE_C_TRAINING_R2_APPROVAL_20261007.json`
- Approval SHA-256: `6d0d0f8dbdbae17a89d3b7dcc1717145b8e5a44464e928b5cb1a4e6debf2800f`
- Unit: `fluid-control-p064-controlled-dose-c50-r2-20261007.service`
- Invocation: `d80f61c62da64497bf378b6c7fd9c917`
- Initial PID: `480866`
- Output: `artifacts/fcp064_controlled_aero_arm_c50_20261007_r2`
- Immutable source: `artifacts/p064_controlled_data_dose_c50_source_20261007_immutable`
- Source manifest SHA-256: `b4448691052571202178e756369778a420bbfda4efd063253ea45fe5799c5578`

R2 changes only the outer process environment:

```text
PYTHONPATH=/workspace/fluid_control/artifacts/p064_controlled_data_dose_c50_source_20261007_immutable/src:/workspace/fluid_control/artifacts/p064_controlled_data_dose_c50_source_20261007_immutable/scripts
```

The actual systemd unit retains `MemoryMax=12GiB`, `MemorySwapMax=0`,
`CPUQuota=800%`, `TasksMax=2048`, `RuntimeMaxSec=3660`,
`TimeoutStopSec=20`, `KillMode=control-group`, and `OOMPolicy=stop`.
The runner retains the 50 GiB startup and 22 GiB runtime MemAvailable guards,
20 GiB physical reserve, `.06` allocator fraction, and 3600-second inner limit.

Before R2, the exact final environment imported `fluid_control.dual_fno`,
`fluid_control.p064_controlled_aero_abc`, `train_tandem_fno`, and
`p026_state_history` from the immutable closure and matched every source SHA.
R2 then produced genuine `training_window_complete` events and
`accumulation_update_complete=1`; therefore it passed model initialization and
performed a real optimizer update. Terminal success and model quality remain
pending independent review.
