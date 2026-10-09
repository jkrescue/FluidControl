# P064 B official CPU proof — actual execution review

R3 completed normally: user unit `fluid-control-p064-b-official-cpu-proof-r3-20261006.service`, invocation `2e08e113b35340f9a7b57a9d5edc80ef`, PID0/exited/exit0. This is model integrity/reload evidence, not accuracy or scientific admission.

Exact immutable CPU source manifest: `artifacts/p064_formal_cpu_proof_source_20261006_immutable/source_manifest.json`, SHA `5649b7d5cd50849fa79d974d2f0547cf4fbdcb088c619c998839c1d93ba25eac`,10 files including actual dataset import dependency. Audit source fd43293466c51a905bc2165d8fa4528d234bd1e97591e301367a54b0fc06a6b5; verifier5d461c5793e3e6f3d7cb3442bab678ffe70f5b7b78de796d660a535833015191. Independent preparation review:8 synthetic CPU tests PASS, full P026 diff reviewed. Production P064 role loader83ac4e41 remains strict.

Actual image `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`, runc, networknone, readonly source/model/root,8GiB memory/8GiB memory+swap (no additional swap), CPU1, no GPU devices. Actual full imports and official checkpoint implementation SHA were verified before load. Official dual CPU loading then reproduced flow tensor `89ce3b37dfa64f6c4f1cff556fbba21cd05374ed4c8e48b69c6127ba4243a8bb` and aerodynamic tensor `3cdac90fe62da9d483a6d38eb2a28b23a52b007028c1dc11c28f049f7f77b851`. No forward, optimizer, save or GPU occurred. High/TF32 historical loading precision is preserved.

Output `artifacts/p064_arm_b_official_cpu_proof_20261006_r3/`:

- candidate_audit.json SHA `976e020151e136a2563fbdeb9470776209f9f8df65fbbad0e3a0ba1e9051c36d`.
- official_cpu_reload.json SHA `497e1164cf534263d2cfe612793fd43263b0904cc27cda727f300d8220fad87c`.
- container_created.json SHA `f74659dae54f657f0f621ff844066786a86c623562452a6f00cb368d89e1e2f0`.
- container_terminal.json SHA `c5136407335edcffaa41cc4d95b4ff5981f30fb8ae84c7221c5392013835c2f2`.
- supervisor_result.json SHA `a77a0b5ecb9d9f97c5b2a2902efee71ef4f36a5b981bc684b4b2e19f6fa67b4e`.

Post-run host audit independently rehashed all seven candidate files and checked equality of audit/reload/result tensor maps and all nonadmission/no-forward flags. Actual container `ffc0b267c47dab6381b736357b65c51d0b0b482e5c0b7b278c90433f739b65c7` exited0/OOMfalse and is absent after exact-CID cleanup. Minimum MemAvailable121,167,958,016 bytes; supervisor error=null. Launcher SHA `ced53fc7b9a94bea7019f29e7b068bb802e1f51d95c437e5d730e819f4ef9d4b`.

Preserved failures: R1 invocation2836e100d7c34671b1f581191177dcff failed before model load because Warp attempted readonly `/root/.cache/warp`; actual b40 Warp source confirmed `WARP_CACHE_PATH` support. R2 invocation26df799f7abb4554bd7e70184d74347a passed full imports, then official loader failed at readonly `/root/.cache/physicsnemo`. Both containers were inspected and removed, no OOM or forward. Separately approved R3 retained `WARP_CACHE_PATH=/tmp/warp` and added1GiB `/root/.cache` tmpfs, preserving source/model/image/caps. No dependency repinning or model changes occurred.

Reused verifier published its receipt via root-owned0600 tempfile. Root separately approved a metadata-only same-image helper binding only that new receipt to chmod0444. Before/after SHA both equal497e1164…fad87c; helper was removed. This changes readability only, not proof bytes or original source/model permissions. R1/R2 outputs are retained, not overwritten.
