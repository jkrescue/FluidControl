# P026 official CPU execution — independent review

Engineering check completed successfully. This is a synthetic, reduced-grid CPU fixture, not real-CFD accuracy, full-horizon training, GPU resource evidence, or scientific admission.

## Observed execution

- Container: `fluid-control-p026-official-cpu-20261005`.
- Actual ID: `9e4b793a92d5ef747f031a6e697821dcba173babd4c0ef0e4b9d34a2b7b26d84`.
- Actual image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Docker start: `2026-10-05T18:31:18.185906976Z`; finish: `2026-10-05T18:31:27.426840662Z`.
- Retained state: exited, PID 0, exit code 0, OOMKilled false, RestartCount 0.
- Actual command: `python verify_fcp026_official_cpu.py --adapter-sha256 2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c`, under the image's `/opt/nvidia/physicsnemo_env.sh` entrypoint, working directory `/work`.
- Isolation: network none, readonly rootfs, only `/tmp/p026-review.ifqR2N:/work:ro` bind; `/tmp` tmpfs 512 MiB; memory limit 2 GiB, NanoCpus 2000000000. Docker MemorySwap was 4 GiB (not a no-swap claim).
- CPU-only evidence: Runtime runc, no device requests or devices, `NVIDIA_VISIBLE_DEVICES=void`, empty `CUDA_VISIBLE_DEVICES`, and script-reported `cuda_available=false`. Driver-unavailable warnings in logs are consistent with this setup.

## Source and retained-output identities

- Adapter SHA256: `2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c`.
- Unit tests SHA256: `145929fc36dc6f971b9bc1247a0e5a03eb097eb6495963c8c575ec879c8f4e93`.
- Verification script SHA256: `647de8e9a6f286e3f95a3d2542da9fa66dc6ec2291199b80fa8e2f0116720bd8`.
- Official FNO source SHA256, checked inside the script: `e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9`.
- Retained `docker inspect` stdout SHA256: `2059217d1d08df5dee31012cf87f63406f2ea1686a15802764d79b6ebe830707`.
- Retained `docker logs` stdout SHA256: `57f38d03666a4d035af9ba05cbb988b77aa0b222428ecba08e7dda6e7ce37268`.
- Retained `docker logs` stderr SHA256: `08cc1d42a3a3d6ffdbb0b6335bb601c5dab182a48663585d3cd47f56276d17c3`.

These hashes were read from the retained container and current stage. The bind was readonly to the container, not an assertion of host-side immutability. Adapter/script hashes emitted by the successful run agree with the inspected files. Raw Docker outputs remain in the retained container; this report records their hashes rather than claiming separate raw archives were created.

## Results and interpretation

The script constructs official FNOs with five layers, latent width 48, two modes and a synthetic 4x4 grid. Strict state loading uses the reviewed K1/K4 mapping. Actual K1 and K4 warmstart output maximum differences were both 0. K1 asserted exact equality; K4 asserted rtol 1e-5/atol 1e-6, so the zero observation must not be represented as a universally guaranteed bitwise identity.

For a deliberately nonzero synthetic history-column fixture:

- Historical-state gradient norm: `1.0686881068977527e-7`.
- Predicted-state gradient norm through a two-step shift: `1.9973190319433343e-6`.
- Initial-history gradient norm: `1.06402602284561e-7`.

All checked gradients were finite; model state digests remained unchanged during forwards, and frozen parameter gradients remained absent. The script created no optimizer or saved candidate. Its synthetic history-column assignment is fixture construction, not learned weights.

Independent CUDA-hidden regression rerun on the formatted sources: **13 tests passed in 0.60 seconds**. No container, GPU job or training run was launched by this reviewer.

No blocking issue was found for this CPU engineering scope. Remaining boundaries are real-HDF integration, an actual production rollout caller using predicted rather than target history, full-horizon gradient/resource behavior and scientific comparison. Stored prescribed action timing remains the existing explicitly qualified convention; this smoke test does not certify exact-time commands or authorize further execution.
