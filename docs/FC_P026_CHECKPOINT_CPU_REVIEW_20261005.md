# P026 official checkpoint CPU roundtrip — independent review

Engineering execution completed successfully. Three deliberately non-candidate fixtures were saved and freshly reloaded by the official API. No optimizer, training, forward/backward computation, GPU execution or scientific admission is established by this check.

## Actual execution and provenance

- Retained container: `fluid-control-p026-checkpoint-cpu-20261005`.
- ID: `db6cb7dde8455bec7ea9b33b00ae151dd5b867f4785af6484d3d4236a7ac3f6b`.
- Actual image: `sha256:b40d5888b59975a56bb536437c6e27dc94d9af5a182a55bb3a83803d41f8a22e`.
- Docker start/finish: `2026-10-05T18:55:41.481153357Z` / `2026-10-05T18:55:57.042793785Z`.
- Retained state: exited, PID 0, exit code 0, OOMKilled false.
- Runtime runc, no GPU device requests, two CPUs, 8 GiB memory and 8 GiB memory-plus-swap limit, network none, readonly rootfs, all capabilities dropped. Script reports CUDA unavailable.
- Stage, immutable source, P018 parent and configuration are readonly bind mounts; only the dedicated output directory is writable. Actual command has in-container `timeout -k 10 600` and explicit `--execute`, targeting `/workspace/output/engineering_fixtures`.

Receipt: `artifacts/fcp026_cpu_checkpoint_engineering_20261005/engineering_fixtures/engineering_receipt.json`.

Receipt SHA256: `8b1b48930eeab818921a1a287d69b02769635faa912c643c219150af03af4665`.

Reviewed producer SHA256: `433a2f8bd5e2c3a27a59c5e4c3a1426dcccfefa0b98379fbb83e6269a0627708`; history adapter `2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c`; resource helper `b806ded8258c787807e67ccb42b5166dbd06e36fc40eba5025d9bda0769eedc6`.

Execution pins official FNO source `e64eb9bef031bfdae5d84f0ed35a1ebb27915f18aed4a333b2dd985a083c71a9` and official checkpoint implementation `0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e`. P018 parent manifest SHA is `91cc2c9a295a1ace5eadcffd8242234b93b73d14959bb902e17b59655f4acf13`; base configuration SHA is `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`.

## Independently rehashed fixture files

Each directory contains exactly the two files listed below; all six byte hashes were recomputed by this reviewer and match the receipt.

| Fixture | FNO.0.1.mdlus SHA256 | checkpoint.0.1.pt SHA256 |
|---|---|---|
| engineering_flow_k1 | `064eaa84eba0ef66178a9d44fda5783d5bc25c556c22554e03a18e8d88b3a394` | `82baf732242cd9e5be31e60e3a6ff3c09d5290b6adffef04b0896998de322957` |
| engineering_aerodynamic_k1 | `7307e7fd7fc181aee2ed62a60008e46205a1120e21d1bba015629cb80cda6152` | `c1809bba152d97f02ce3833f843d63c5ac9fe58f0a2d1a91c2bc159179896fe7` |
| engineering_aerodynamic_k4 | `4ee2518cda911115b3be7e6c8f20900922ed9a807a6a94ab69238df51140d17d` | `b6564f9ba14fb55fc4a54e11c9db4496e39c4fb420e6c20d3ce59ffbbd5314d0` |

All fixture epochs are deliberately 1, distinct from the unchanged flow parent's epoch 0. Explicit metadata status is `FC_P026_HISTORY_ENGINEERING_FIXTURE_NOT_CANDIDATE`; role/history/channel identity and false optimizer/training/candidate/admission flags survive exact metadata equality checks on fresh reload. K1 uses six physical channels; K4 uses 18, both with two appended coordinates.

Receipt and reviewed producer establish exact fresh state comparisons and mapped tensor hashes for all three. K1 mapped hashes equal their respective parent hashes. K4 mapped tensor SHA `21d7c3ee5bfd55da854eca2c8d2db5cd8c5a5f63cea087bf3f45f3c687f045fb` also matches the prior production-size resource fixture. Parent immutability is checked during the run.

## Retained raw-evidence hashes

- Docker inspect stdout: `87aab65bd7effef61c3fadab45e87bcf46547cea7e717ae8ccc2d33072be97a2`.
- Docker logs stdout: `9e398c2d4212a90738ef6a9cd8a162cd3802085cd3c4228908cb27bcfe9e8a13`.
- Docker logs stderr: `d8f1a81f065db95ab2a20607b93a8389f77e155bfbd55fb8986c7be07b073f2a`.

These describe the retained container outputs; this report does not claim separate raw archives. Logs contain the matching successful receipt, with expected single-process and newly created output-directory notices.

## Verdict and limits

No blocker found for this engineering roundtrip. The reviewer inspected retained configuration/logs, receipt and output bytes without rerunning or loading models. Fresh tensor comparisons are evidence from the reviewed, successfully completed producer, not an additional reviewer reload. `legacy_profile_rejected=true` means the local fixture metadata validator rejects K4 as K1; it does not certify rejection by every production caller. Training/admission/control integrations still need their own identity gates and tests. These files must remain engineering fixtures, never relabeled accepted candidates.
