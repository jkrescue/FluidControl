# FC-P029 CPU preparation and source-freeze review

Status: CPU ENGINEERING REVIEW ACCEPTED; NOT EXECUTION APPROVAL OR SCIENTIFIC ADMISSION.

## Independently verified source artifact

The control/evaluation reviewer read and SHA-256 checked all 423 entries under
`artifacts/fcp029_training_source_20261006_immutable`. All 421 entries inherited
from the pinned P028 r2 manifest are unchanged; exactly two source files were
added: the P029 runner and objective. No model, HDF, or scientific data was read.

- Basis manifest: `72c0513e03c154b032f162d9c4c3cbd5ddfad7c478694f4e652aeb949759012c`.
- New manifest: `c6584b8fdbbb38cf7149ad2f086526169f44174a800ef49a32fcda08a678627a`.
- Pending spec: `bd35b02ce1a8ddfc3e8201cf20b3c9eeb59c70ade5dc4e9e2cfa54d8d7236307`.
- Freeze receipt: `c6141275134c388ee19333edea301900d5767d7d5b166c71c25cfd8dfc4bc558`.
- Preparation helper: `9bae9f03ab7bbcea36f996dd722a1417d4f9575afd4d3934784180dd4cecd719`.
- Configuration: `07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`.

Every file is mode 0444; every directory, including the root, is 0555. There
are no symlinks. The only files outside the 423-entry source map are
`source_manifest.json`, `input_spec_preparation_only.json`, and
`freeze_receipt.json`. Spec source-root/map and configuration path/hash match
the actual frozen files. The spec remains
`FC_P029_PREPARATION_ONLY_NOT_APPROVED`, mode `scales`, without future scales or
resource receipts. It cannot authorize execution.

The checked dependency set includes P011 identity helpers, P013 input packing,
P026 inventory/history helpers, P028 runner/objective, official training entry,
dual/calibrated loaders, both dataset modules, and the configuration. This is
the parent-loading/training closure: its inherited loader reads the original
K1 parent, not the future P029 candidate. Post-training P029 loading requires
the separately reviewed explicit-profile evaluation runtime.

## Implementation and CPU evidence

- Runner `train_p029_control_aware_flow.py`:
  `e7dd8b9a261b4a47dbfcec6ce4be792dce0294454460b9924e81c37fb87c4aa0`.
- Runner tests:
  `767ae546f828d1d61d04e6d1cb0e2633fe6b9064cfa0d7ab0d83ab301bf50316`.
- Objective `p029_control_aware_flow_objective.py`:
  `904fc903b35b6a754faf1243217b3a7f4a5f552afe90e1e8e3a33654d4524516`.
- Objective tests:
  `14c2021a9529071657dd7743a1a1d991cf553ecb8f538c6cf31fd2eef44db8a9`.

Implementation tests: 19 passed in 2.51 seconds. Independent runner review:
19 passed in 2.55 seconds. This reviewer independently ran the objective's
13 tests in 1.46 seconds. Root reports the combined canonical 32 tests passed.
All are small synthetic CPU fixtures, not model-accuracy or capacity results.

After canonical integration, Root additionally reports 68 new-suite tests
passed in 2.85 seconds and nine legacy suites with 133 passed / 1 skipped in
1.45 seconds. An attempted additional
`tests/test_p026_formal_history_callers.py` collection failed on the host because
PhysicsNeMo is not installed there; its actual dependency requires the official
container. No environment or test was modified to hide that limitation. The
133-pass rerun excludes that file: this is explicitly not a whole-repository
test PASS, nor evidence of an executed official-model/GPU test.

Tests cover exact mode/receipt/source bindings, positive fixed denominators,
full 1368-record mean recomputation, forward-only scales with no gradients or
tensor changes, the actual objective with eight-gradient averaging before
clipping/Adam, frozen aero isolation, preserved unused output rows/moments,
and explicit P029 checkpoint/manifest fields. Mock checkpoint APIs test schema
only; actual official save/reload remains a future execution requirement.

The objective's first force term uses true current state and contributes zero
flow gradient. The following nine terms propagate through preceding flow
predictions; the tenth predicted state is field-supervised only. Frozen aero
parameters do not imply a detached aero input. Equal parent-normalized losses
are not a claim of equal gradient magnitudes.

## Launcher review and corrected blockers

Launcher `run_p029_control_aware_flow.py` SHA
`354112951e20729c24027370de1eafb6ce70640fc9dd8ef380226c432a27e9d7`
and tests SHA
`30f8f87d0be9682e765af08cfa5c3c8eb53113ab245d68f62e3e5335af0d4717`
passed 11 independent CPU tests in 0.07 seconds. It reuses the SHA-pinned P028
container lifecycle, with read-only source/parent/train/proof mounts, original
data aliases, exclusive output, allocator 0.06, 12 GiB container, startup
free/available 30/50 GiB and continuous host/CUDA 20 GiB floors. Limits are
900 seconds for scales/probe and 14400 seconds for training; outer timeout
escapes through the reviewed cleanup path.

Two concrete preparation defects were corrected before execution: the freeze
helper previously attempted to create the already source-mapped configuration
twice; the launcher previously required the full 1368-order digest for the
one-window resource probe. The final helper writes configuration once, and
the final launcher checks singleton `[816]` for the probe while retaining the
original full-order digest for scales/training. Neither failed draft was used
to launch P029.

Next steps require separate Lead authorization: parent-only scales pass, a new
actual-size field-plus-aero backward resource probe with no optimizer, then
fixed-budget training only if capacity and the conditional scientific plan
permit. P028's resource evidence does not establish capacity for this new
backward graph. No P029 GPU operation, model training, admission, PPO, or CFD
result is claimed here; original evaluation thresholds remain unchanged.
