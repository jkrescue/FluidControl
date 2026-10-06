# Fixed b01/b03 controlled development evaluation frames — preparation only

## Actual approval and launch (supersedes preparation-only execution restriction)

On2026-10-06 Lead separately approved the exact frozen profile in
`docs/DEVELOPMENT_PHASE_H5_CONVERSION_APPROVAL_20261006.json`, SHA256
`c82e4a865d0d6e0927faee026baa84ebca2fcabb0707daa00f988174349dd5a3`.
Root launched one unit `fluid-control-development-b01-b03-h5-conversion-20261006.service`
at10:27:34UTC, invocation `8ac389027a954b79bc1d766f89029a6b`.
The following preparation text is historical; the immutable source and original
pending file remain unchanged. No terminal result or prediction accuracy is
claimed by this launch record. A different reviewer audits the actual terminal.

## Preserved preparation

No conversion, model inference, training or CFD execution is authorized here.
The 800-cycle projected-PPO b01 and b03 trajectories are already opened
development evidence, not untouched tests or statistically independent phases.

Fixed selection: b01 base time130 and b03 base time144; each uses starts
0,100,...700 and the next five frames, exactly48 controlled frames per phase.
No zero-branch or b00 training frames enter this conversion. Across two phases
there are96 frames,16 six-frame HDFs and80 prediction endpoints. Fields remain
physical u/v/ROI-centered pressure on the existing grid/mask, with the existing
float32 VTK-time tolerance. Actual action-before/action-after endpoints and
four force coefficients come from each phase's complete recorded progress.

Metadata inventory found both phases have all801 expected U/p pairs and800
completed cycles. Selected U/p bytes: b01 40,814,201; b03 40,796,016. These are
saved fields, not newly solved CFD. Each selected source and constant/system
file is subsequently SHA-bound in SPEC_PENDING.json; originals remain read-only.

Implementation: one small phase-selection adapter plus a thin orchestration
profile. Import the exact reviewed R2 converter304fece8 unchanged; reuse its
inventory/copy/export/sample/owned-container cleanup. Reuse unchanged core
pack_and_verify_mini_hdf, Curator sample_frame and official Reader adapter.
Distinct phase output subdirectories and owned export names avoid collisions.
Retain root-owned exported scratch rather than failing unprivileged recursive
deletion. No chmod/chown, new resampler, normalization refit or large source tree.

Budget: CPU1,12GiB host cgroup,noSwap; each serial exporter4GiB/noSwap/CPU1.
MemAvailable50GiB startup/22GiB runtime preserves20GiB physical reserve. Hard
900s, systemd TimeoutStopSec120, KillMode=control-group; startup10GiB disk
headroom. Roughly3min expected from prior96-frame conversion, not guaranteed.
Use existing pinned OpenFOAM image only for foamToVTK, never a solver.

Pending new driver/selector paths still refer to the unique /tmp review stage.
Before any execution, freeze only those two reviewed source files into a new
exclusive project artifact and rebind their paths with unchanged SHA bytes;
base/helper sources stay at existing SHA-bound locations. The pending spec
remains execution_authorized=false. Root must review the actual serialized
final approval and launch exactly once with --execute. No retry is automatic.

Terminal evidence must include exact96/16/80 counts, per-HDF hashes and official
Reader equality, source posthash, common grid/mask, no model/optimizer/solver,
resource samples and owned-container absence. These HDFs only enable a later
separately approved identical K1/A/B H1–H5 development comparison, not acceptance.
