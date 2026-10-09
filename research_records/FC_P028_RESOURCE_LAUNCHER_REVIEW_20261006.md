# P028 no-update resource launcher: CPU verification

2026-10-06. Preparation only; no execution approval or resource success claimed.

`scripts/run_p028_resource_probe.py` SHA256:
`d06fa0c28287befbeb18f098f84f2fd4f2afdc1f7f234a89939f75813d909e97`.
Canonical test SHA256:
`f68c068231c57bb0e57ec8544691fc14d34805449e3098f15a11d3b2f8f6f94a`.

Root adapted the existing reviewed P027 resource launcher; independent evaluation
checked the diff and ran14 synthetic mock tests. Root made only test import paths
repository-relative and reran both P027/P028 suites:24PASS in0.06s. Tests use no
Docker/GPU/HDF/model operations. Exact-CID cleanup, signal recovery, inspection
and continuous host resource sampling remain unchanged.

Approved interface capability (not a run authorization): resource-probe mode
only, fixed official b40 image, GPU0, allocator0.06, container12GiB/no extra swap,
900-second budget, startup free30/available50GiB, runtime free/available/CUDA20GiB.
Sources, single K1 parent, exact training families and source manifests are
read-only mounts; validation/frozen siblings are absent. Only exclusive output
is writable; runner creates its payload child. Completion requires an explicit
no-update result and equal nonempty initial/terminal flow tensor identities.

Remaining before real probe: stable reviewed runner, immutable source/spec,
separate execution approval and actual resource clearance. Current observed
physical free memory was about26GiB, below the extra30GiB startup requirement.
Any targeted release of verified project-file clean cache needs a recorded,
bounded pass; no global cache clearing or data modification is authorized here.
Training remains separate: a no-optimizer probe does not measure AdamW peak
memory, and P028 candidate-loading compatibility is still under independent review.
