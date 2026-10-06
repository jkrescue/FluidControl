# FC-P064 Arm B original formal launch record — 2026-10-06

This records execution identity only; it is not a scientific result or admission.

- Approved evaluation: `docs/FC_P064_ARM_B_FORMAL_APPROVAL_20261006.json`, SHA-256 `cd58fd47e991ec6dac200bd82d414347f72b778ea78415377427e430dfd47478`.
- Immutable formal source receipt: `artifacts/p064_formal_source_20261006_immutable/receipt.json`, SHA-256 `c5081a03d166e0983e8152ca0d050cb94e97f36f61d503bfd080ebc9574e461e`.
- Numerical runner: SHA-256 `03c5862e34a648a1254284d1709bd74c3b995d3a91ae06c4a6f92a945029c0f3`.
- P064 identity wrapper: SHA-256 `aac728bcf7f568073b723fb640a84f2fcad1e671b55b68ba8a331b77856089c9`.
- Resource adapter: SHA-256 `ac1b00563f7af30bc350421d0938fb61d9bf1cea5f8b0fff065e3e75f87e4e05`.
- Candidate audit / official CPU reload: SHA-256 `976e020151e136a2563fbdeb9470776209f9f8df65fbbad0e3a0ba1e9051c36d` / `497e1164cf534263d2cfe612793fd43263b0904cc27cda727f300d8220fad87c`.

The first operational unit, invocation `6d7cea57ea7740c991223448dfc45e0a`, exited with status 2 before Python opened the driver because the systemd working directory was `/home/USER while its argv used relative paths. It created no output directory, Docker container, or GPU work.

The approved path-only correction launched `fluid-control-p064-b-formal-r2-20261006.service`, invocation `01806bdc150841f7b9efd04360a441f2`, with an explicit repository working directory and absolute paths. Its systemd bounds are 72 GiB memory, zero excess swap, 800% CPU, 1024 tasks, a 10,800-second runtime maximum, 120-second stop timeout, control-group kill mode, and stop-on-OOM. The adapter separately enforces two 50 GiB `MemAvailable` startup checks, a 22 GiB continuous runtime floor, and the reviewed outer `.06` GPU-guard accounting profile while preserving the original seven numerical commands and scientific gates. The unchanged numerical command also contains its historical inner `.15` guard. In the pinned `spark_gpu_guard.py`, these fractions only calculate startup `MemAvailable` requirements; they do not call `torch.cuda.set_per_process_memory_fraction` and therefore are not effective CUDA allocator caps.

At the first post-launch check the unit was active/running with MainPID `1222599`; the precision container had completed and been removed, and `validation10` was running. The formal output remains non-admitting until the complete terminal receipt and all original gates are independently reviewed.
