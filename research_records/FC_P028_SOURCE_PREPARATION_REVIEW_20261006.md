# FC-P028 source preparation independent review — 2026-10-06

Status: SOURCE PREPARATION VERIFIED; not execution approval, training completion,
scientific admission, or PPO authorization.

Reviewed the actual tree
`artifacts/fcp028_resource_source_20261006_immutable` using Python standard-library
file/JSON operations only. Independently recomputed every one of its 418 source
hashes; all match the flat manifest. All 414 inherited P027 files are byte-identical
to the preceding immutable tree. The original 411 formal files also match the K4
formal receipt's source map. Exactly four reviewed source/test files were added.

## Actual identities and permissions

- Source manifest: `0db54ffcc16c0b54ad34c62159ace3cac4eb0bca2afac0d157f888a19c860bc0`.
- Freeze receipt: `e2d32d90df051e52a0ccec6e0724f7b9aaa5d58ca2cb656ae6043e1222c0011e`.
- Preparation spec: `e6f47e7f80e7ddc65f74627ec4693e61432836925365c7a5224d71e00da1385e`.
- Objective: `28a9a462f72a7ed5965fc97d22de47ef7fd12740edd66c3f505e822eb1ffd41a`.
- Runner: `4ad0dfa41fcfeb93481e313d2f0ca24685ea3ea183a074a7b017284f8c5ee057`.
- Objective tests: `9b9eaff2803c2525efe7685636679c7485babf45206e213f0eb8608ebdd12907`.
- Runner tests: `ba971fd45c311b872ef6517fec9a5562dbadd65cc79cdf52f0d25a7be39bc35b`.

The tree contains exactly 418 source files plus three external-to-source-map
JSON records, with no extra files or symlinks. All 421 regular files have mode
0444; all nine directories, including the root, have mode 0555. The source map
and spec agree exactly; config points to this tree's `training_config.yaml` with
SHA `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`.

## Metadata-only data and parent checks

Spec status remains `PREPARATION_ONLY`, mode `resource-probe`, allocator fraction
0.06. The embedded protocol retains H10, original H100-window identity, 1368
starts / 171 eight-window updates for the separately approved future training
path, seed 20261003, AdamW learning rate 1e-5 and original order SHA
`177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`.
These training settings do not authorize an optimizer in resource mode.

All data metadata equals the pinned P027 preparation spec: base20, train8,
train16, unchanged manifest/normalization and 44 filename/hash entries. Those
44 entries independently match the existing audit's map after filename mapping;
the small audit JSON itself was SHA-checked against
`03153fa51e94db01c7abacfb80037d99bb6757ac7aacdc87f1c7266a002323b9`.
No HDF was opened or rehashed. Parent path/hash is exactly P027's K1 binding,
manifest `7adca21e3a75691b10f164c342ea91995cc38060e7416dd217b8bd8e5feeacc7`;
no model/archive was opened or rehashed in this review.

## Deliberately older loader and execution boundary

The resource source intentionally retains the reviewed P026 dual loader
`3343dba367dd6e45fdc914fc321e90b94efc2d8a00553c61b025ef7d776fc2a8`.
The current canonical P028-aware loader is different
(`538ee11626756ec7f7c7d251437b48db4f53b427b81ab733f7e6ffb126216627`).
This is not a stale-source substitution: the no-update resource probe loads only
the existing K1 parent and saves no candidate, so it requires P026 loading,
not P028 terminal loading. This closure does not prove future P028 candidate
evaluation compatibility; a separately reviewed source/approval is required for
that later stage. Neither tree was altered by this review.

No project/model import, Docker, GPU, HDF access, optimizer, checkpoint save,
cache advice or scientific evaluation was performed. Launcher/guard bindings,
actual runtime resource evidence and any cache maintenance remain separate
approvals. A no-optimizer probe also cannot alone establish AdamW training peak
capacity; the recorded two-moment accounting is only a lower bound.
