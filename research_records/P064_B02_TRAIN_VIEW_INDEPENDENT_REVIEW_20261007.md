# E090 b02 train-view handoff — independent narrow review

**ACCEPT for binding a separately approved D25 training preparation; no training authorization.**

Actual unit `fluid-control-p064-b02-train-view-20261007.service`, invocation `2ea9508cf6194847b6f8337a0c98ac74`, independently queried MainPID0/Result success/ExecMainStatus0. Exact 8 GiB MemoryMax, swap0, CPUQuota1, RuntimeMax180s. CUDA-hidden Warp warning is expected; no model/forward/optimizer was run.

Read the entire executed adapter `artifacts/p064_b02_controlled_train_conversion_source_20261007_immutable/build_b02_train_view.py` and actual approval/journal/manifest/verification. Rehashed:

- approval `docs/P064_B02_CONTROLLED_TRAIN_VIEW_APPROVAL_20261007.json`: `83730386e8e00d118881374752543d78e13896e7de0fe7c5f8886a3b7b9f78dd`;
- adapter: `7051ee42c2dde896f0cb7c8cfff490c10fd46f52c6ef62449a933d155a4f9863`;
- view `artifacts/p064_b02_controlled_train_dataset_view_20261007/manifest.json`: `c8201847a7bbb77a0a3e7c2d1f121e9aef2cd294358fcc2d4f9d8d88037d0c4a`;
- `verification.json`: `c5a24716fc285983fbd6042e5be8d7625b5630ccda26f39dc47b7fa645f62b80`;
- `train/b02_canonical_ppo_train.h5`: `96140954487b40f8e7a8dd37cdbb90221fce5656120dedb0301865a046f4377f`;
- original train normalization: `f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1`.

The view HDF and independently accepted E090 source have the same actual inode18905930, size341355532 and bytes SHA. No arrays were copied or refitted. Only root/train directories exist; no validation/test/frozen_test directory. Manifest states train-only,801frames,one trajectory, action scale0.75, endpoint-CFD force provenance and exact conversion result3faf153b2a9857e880634be3347f6596834b3609a5235007f668b141d10b5ade.

Executed official-reader-backed TandemRolloutDataset has701 H100 windows. Actual first0→100 and last700→800 were compared elementwise against HDF-derived normalization: current/target fields,101 current-through-endpoint actions,100 subsequent forces,mask and metadata all pass. Journal and verification agree. The reviewer checked these actual execution records/source and hashes; did not rerun the DataPipe or model. Source HDF hash remains unchanged; producer also checked original mode and mtime. Metadata/directories are read-only, but HDF remains original mode0664 via hardlink: do not claim filesystem-enforced immutable HDF; future consumers must verify SHA and read only.

Together with [conversion independent review](P064_B02_CONTROLLED_TRAIN_CONVERSION_TERMINAL_REVIEW_20261007.md) SHA561624b0fd98a6db0fbff5722f950444b19d0de1aca3eea71f83ba445eb28b90, this is sufficient data-interface evidence to bind D25 preparation. It does not demonstrate candidate accuracy, authorize training, reopen heldout data, or alter previous model/control failures.
