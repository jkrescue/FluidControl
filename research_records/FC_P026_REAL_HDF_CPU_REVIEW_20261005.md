# P026 six-real-window CPU integration — independent review

The retained execution completed successfully. Its evidence supports adapter compatibility for six representative train windows only, not a complete 44-file audit, learned-model accuracy, GPU readiness or scientific admission.

## Actual execution

- Container: `fluid-control-p026-real-hdf-cpu-20261005`.
- ID: `a8c735d1a2b7d1ca500ebface7e45488eb29b63ba0b2195a5e500a0c2669929c`.
- Actual image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Docker start: `2026-10-05T18:34:16.407494448Z`; finish: `2026-10-05T18:34:29.228999865Z`.
- State: exited, PID 0, exit code 0, OOMKilled false.
- Command: `python /workspace/probe/verify_fcp026_real_hdf_cpu.py`.
- Actual isolation: runc, user `1000:1000`, network none, readonly rootfs, all capabilities dropped, no device requests, 4 GiB memory limit and two CPUs. The successful script also checked CUDA unavailable.

All eight bind mounts were readonly: staged probe at `/workspace/probe`; immutable P013 source at `/workspace/project`; train-only metadata views for base/train8/train16; and the corresponding three curated train directories overlaid at their `/train` paths. No held-out directory was mounted as a bind; the script additionally rejected visible validation/test/frozen_test paths in each data view.

## Sources and retained evidence

- Script SHA256: `17ad02b3a9162fa8f746e3ed5671f0a8c50d71c7e0ec1fad1548742e59989179`.
- Adapter SHA256: `2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c`.
- Existing loader SHA256: `c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae`.
- Official HDF reader SHA256 recorded inside execution: `cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0`.
- Shared normalization SHA256 checked by the script: `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`.
- `docker inspect` stdout SHA256: `f1f5ea49e2b449264f68320e5ef459aafecfce0b83942768bc2045e1dc320827`.
- `docker logs` stdout SHA256: `f199310c299340fa7845360cd18d6230a1d7dab7216fa7fd8a1b426245b0fc9c`.
- `docker logs` stderr SHA256: `08cc1d42a3a3d6ffdbb0b6335bb601c5dab182a48663585d3cd47f56276d17c3`.

The reviewer inspected the complete producer, retained container configuration and successful JSON log. Adapter/loader and three manifests/normalizations are checked against constants before reads. The official reader hash is recorded, not independently compared against a separate constant by this script. Raw Docker evidence remains with the retained container; this report does not claim separate raw archives or new full-HDF hashes.

## Six verified windows

| Family and case | Start | Original stride | Family-local dataset index | K4 frames | Padded frames |
|---|---:|---:|---:|---|---:|
| base: matched_start_acquisition_train_b00_zero | 0 | 20 | 144 | 0,0,0,0 | 3 |
| same base case | 20 | 20 | 145 | 17,18,19,20 | 0 |
| train8: dynamic_train8_b00_prbs | 0 | 2 | 51 | 0,0,0,0 | 3 |
| same train8 case | 4 | 2 | 53 | 1,2,3,4 | 0 |
| train16: direct_cfd_directppo2048_v1_env0_ep0009_b00 | 0 | 2 | 105 | 0,0,0,0 | 3 |
| same train16 case | 4 | 2 | 107 | 1,2,3,4 | 0 |

All six rows report and enforce exact original sample tensors, targets and metadata; exact history states/actions against the existing official reader; and exact K1 legacy input equality. Original sample tensors and K1/K4 inputs are explicitly checked finite. K1 shapes are `[6,128,256]`, K4 `[18,128,256]`. K4 current-state and current/next-action slices are checked against the original sample. Start-zero padding is explicitly `[true,true,true,false]`; warm-history windows have no padding.

Targets remain available through the original dataset for equality checks, but are not fed into the constructed history input. The next prescribed action follows the existing stored-sample convention; this is not new certification of exact nominal-time commands. No model or optimizer was created, no data were modified, and no GPU or held-out execution occurred.

## Scope and verdict

No blocking issue was found in this bounded integration evidence. It covers three representative files at two starts each, not all trajectories or all 1,368 training windows. Dataset construction can inspect index metadata across its train files, but no full-field-byte audit was performed. Full rollout/model behavior, resource use and scientific comparison remain separate work requiring their own approval. This review performed no rerun or additional data scan and changed no thresholds.
