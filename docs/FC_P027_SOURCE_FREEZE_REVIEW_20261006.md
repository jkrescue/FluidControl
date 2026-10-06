# FC-P027 source-freeze independent review

Status: source-only verification passed; preparation only, not execution approval.

Root executed the reviewed freeze script with PYTHONOPTIMIZE unset. The reviewer
then independently read and rehashed the small source files at
`artifacts/fcp027_diagnostic_source_20261006_immutable` without importing project
modules, loading models, reading HDF payloads, or starting Docker/GPU work.

## Verified closure and permissions

All414 manifest-listed source/configuration files match their SHA256 values.
The411 inherited files also individually match both the existing P026 formal
source tree and the K4 formal receipt's source map. The only additions are the
reviewed P027 diagnostic, its focused test, and pinned `training_config.yaml`.
No inherited source was changed. Every file is0444, every directory0555, and no
symlink was found in the new tree.

The frozen spec has414 exact source path/hash entries, each rooted in the new
immutable tree and equal to the flat manifest. Its config points to that tree's
`training_config.yaml`, SHA
`07e55fd11df8030313338cef0344490c3453e515aae9c6b6122e997bad5085d9`.
The spec remains `PREPARATION_ONLY`. Manifest, spec and freeze receipt are
outside the hashed source closure, avoiding circular hashes.

## Exact artifact identities

- Source manifest: `ffdd7e623f9c8da0b39ccb167ca0b74c9013313fdf9cddc37266e884df578c1d`
- Preparation spec: `82c63e444e49b53f60e02097cdccd0daed13e5e8a6bf8242f046cdb8a09d3fa2`
- Freeze receipt: `7e44fecb4dd1480b94f0e6da89ae14c0d4b63c42e64844c87f28be7eb4da1308`
- K4 formal source-map receipt: `729f9ce1f307d5462307470af20491806f6cfe31b5c81fc74ec284a2461841d9`

The original eight direct dependency entries were separately compared against
the frozen versions, not merely replaced by a larger file count. All eight are
byte-identical:

| File | SHA256 |
|---|---|
| scripts/diagnose_p027_short_horizon_errors.py | f398c86f66cb2f30df9b1832b3081d851bac8fd0f9dad3a3d150cc9d25ce99d2 |
| scripts/p026_history_inference.py | fd568f6457b980046a0419be96562291e9f96736270d45f20dd2adb2ffc5878c |
| scripts/p026_state_history.py | 2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c |
| scripts/evaluate_tandem_fno.py | daef4a3a6eb1b9190f5cde728581b0c1b4b9656f56e865cf61a53deb1f37de88 |
| scripts/train_tandem_fno.py | 9e5bbebd338af02b1d73530c9cb74fc2f840455f56b7bf34ed2bb517be19a22a |
| src/fluid_control/dual_fno.py | 3343dba367dd6e45fdc914fc321e90b94efc2d8a00553c61b025ef7d776fc2a8 |
| src/fluid_control/canonical_joint_v1.py | 138ab2b49ebddcbed2a24486c995b85a3a5ae226ee936ff2ed5de318496ee8bd |
| src/fluid_control/tandem_datapipe.py | c939e4553dbef9e227b6a3a4d5f36242114a690b32ff907339b5be2a4ec693ae |

Root reports the exact integrated diagnostic/test passed22 canonical CPU tests
in0.57s. This source review does not claim another test run or actual official
model execution. Python/environment dependencies remain bound by the separately
reviewed official image; launcher/continuous resource guard and an explicitly
authorized execution spec still require their own review.

No model/data identities were recomputed here, no scientific gate was changed,
and no P027 result, candidate admission, PPO or real-CFD success is implied.
