# FC-P026 PPO image metadata/source-byte fingerprint

Date: 2026-10-06

Status: bounded CPU metadata evidence, not PPO readiness or scientific admission.

The retained container `fcp026-ppo-api-fingerprint-20261006` exited zero in the existing image `sha256:2e45b4e1ac9553ea86aa9148455be9aae30688446039fdee6255a637603acb2c` (`fluid-control-physicsnemo-hydrogym:2.2.2-4ab9854`). Docker inspection confirms `runc`, no network, read-only root, no mounts or device requests, 256 MiB memory/swap, one CPU, 64 PIDs, `CUDA_VISIBLE_DEVICES=` and `NVIDIA_VISIBLE_DEVICES=void`.

The command used `importlib.metadata` for package versions and read two installed source files as bytes for SHA256. It did not import or execute PhysicsNeMo APIs. Observed metadata:

- NVIDIA PhysicsNeMo `2.2.2`
- PyTorch `2.13.0a0+8145d630e8.nv26.6.54250401`
- Stable-Baselines3 `2.7.1`
- Gymnasium `1.2.3`
- `physicsnemo/models/fno/fno.py`: `e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9`
- `physicsnemo/utils/checkpoint.py`: `0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e`

Those source hashes match the training pins, but this alone does not prove runtime API compatibility. No model, project source, HDF, GPU, PPO policy, environment, or CFD was loaded or executed. The actual-candidate preflight and remaining conditions in `docs/FC_P026_HYDROGYM_RUNTIME_CPU_INTEGRATION_20261006.md` remain mandatory.
