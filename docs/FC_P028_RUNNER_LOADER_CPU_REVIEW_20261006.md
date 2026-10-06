# P028 runner and updated-flow loading: CPU integration

2026-10-06. Engineering preparation, not actual training or model admission.

Root integrated reviewed staged code and ran the actual canonical files:
`test_train_p028_flow_rollout`, `test_p028_dual_loader`, `test_p026_dual_loader`,
`test_dual_fno`, `test_dual_fno_p015`, `test_dual_fno_p018_protocol`:
**90PASS in1.06s**. CUDA disabled, synthetic/mocked models and checkpoints only.
No scientific results CSV row is created for these tests.

- Runner SHA: `4ad0dfa41fcfeb93481e313d2f0ca24685ea3ea183a074a7b017284f8c5ee057`.
- Runner tests SHA: `ba971fd45c311b872ef6517fec9a5562dbadd65cc79cdf52f0d25a7be39bc35b`.
- Dual loader SHA: `538ee11626756ec7f7c7d251437b48db4f53b427b81ab733f7e6ffb126216627`.
- Loader tests SHA: `48157257f036568698c0d262b50d536694d098824450f3b4de59ded085c7a2d5`.

Independent reviewer verified the runner's scope/AdamW row protection and the
loader's standard-role metadata alignment. Root identified and had corrected
precision initialization before parent validation. Final receipt validation
rejects missing/nonhex tensor identities, wrong parent/mode/window counts and
nonmatching source maps. Actual receipts still need independent runtime review.

Only explicit P028 permits trained flow epoch1 plus exact frozen P026-K1 force
archives. Existing P026 retains frozen calibrated P009 flow epoch0. Tests cover
wrong roles/epochs/frozen flags, parents, protocol and metadata, plus legacy
regression. These mock tests do not replace actual official checkpoint roundtrip.
Full formal invocation and optional P027 updated-flow comparison still need
scope-aware caller bindings before complete training/evaluation execution.

Resource-source snapshot was actually created at
`artifacts/fcp028_resource_source_20261006_immutable`:418 files, manifest SHA
`0db54ffcc16c0b54ad34c62159ace3cac4eb0bca2afac0d157f888a19c860bc0`,
preparation spec `e6f47e7f80e7ddc65f74627ec4693e61432836925365c7a5224d71e00da1385e`,
receipt `e2d32d90df051e52a0ccec6e0724f7b9aaa5d58ca2cb656ae6043e1222c0011e`.
It deliberately keeps the already reviewed old dual loader for loading the K1
parent: this no-update probe does not save or load a P028 candidate. It is not
the eventual training/evaluation source snapshot. No HDF/model/GPU access occurred
during this source-only preparation. Execution approval remains separate.
