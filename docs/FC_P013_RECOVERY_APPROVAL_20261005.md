# FC-P013 operational recovery approval — 2026-10-05

Lead: Root. Status: one recovery attempt approved, not scientific admission.

The original run stopped reporting after update 592 at 06:52:29 UTC. At 07:00–07:04 the trainer still consumed one CPU core while GPU utilization was about 1%; process I/O counters did not advance. A bounded GDB inspection found the main thread inside CUDA device-to-host copying. Kernel logs at 06:54:29 recorded NVIDIA NV_ERR_NO_MEMORY context-allocation failures. Docker reported OOMKilled=false. The image transfer ran 06:52–06:59, concurrent with this fault; association is observed, causality is not proven. High MemAvailable did not exclude this driver-level allocation failure.

The task-owned hung container was stopped after preserving kernel, container, process and memory evidence under the original output's operational_failure directory. No host reboot, GPU reset, driver change, global cache flush or unrelated-process termination was performed. A fresh container using the same pinned image then passed CUDA matrix multiplication, synchronization and device-to-host copying; CUDA free memory was 41,036,009,472 bytes.

## Unchanged experiment

Original approval: FC_P013_TRAINING_EXECUTION_APPROVAL_20261005.json, SHA256 1bdcfcf71d1581bbae66fc6551dde61a500bd95a464d9f61d510cbac1721a120.
Original immutable trainer/source, pinned image, parent, seed, 1368-window order, optimizer, data, normalization, precision, terminal-only checkpoint and evaluation criteria remain unchanged. There is no recoverable terminal checkpoint from the interrupted run. Restart from the original parent, not from an invented step-592 state. This is an operational retry, not a second model-selection trial.

## Execution and recovery controls

- Fresh output: artifacts/fcp013_independent_force_fno_training_r2_20261005. Preserve the first attempt untouched.
- Service/container use r2 identifiers. Maximum runtime remains four hours.
- No concurrent image export/import or other bulk transfer on Main during training.
- Original >=20 GiB MemAvailable guard remains; additionally require >=30 GiB MemFree and >=80 GiB MemAvailable at startup.
- Host watcher samples every 10 seconds and stops only this container if MemAvailable or MemFree falls below 20 GiB, or log age exceeds 300 seconds. These are conservative operational safeguards, not changes to scientific acceptance.
- A watchdog stop requires diagnosis; do not loop automatic restarts indefinitely. Root continues the active research task.
- Training completion still requires actual 1368-step and saved-model audits, fixed six diagnostics, separately approved original formal evaluation, then compatible newly trained PPO and real-CFD validation. No PPO, held-out evaluation or success claim is authorized by this recovery.

Extra independent agent review was retried but unavailable due model-service capacity. Root performed fault diagnosis; this is not independent scientific acceptance.
