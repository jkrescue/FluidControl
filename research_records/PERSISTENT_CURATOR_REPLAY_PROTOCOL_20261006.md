# Persistent Curator sampling: CPU preparation only

No actual replay has been authorized or executed. Old canonical sampler and
trial driver remain unchanged. New module extracts the original numerical body;
the AST regression verifies identical statements after argument-name substitution.
Each call creates a fresh official VTKSource and Mesh. No mesh/locator/query cache,
new official API, model, CFD solver, action, or sampling-rule change is introduced.

## Proposed fixed replay (requires Root execution approval)

Use the existing 20 exports (mpc/zero, 148.0 through 148.9), in alternating role
order, then repeat the first frame once to detect stale persistent state. Each of
21 checks runs the original CLI and the new in-process function against the same
existing VTU. Compare both to the previously stored sample. Require exact keys,
dtype, shape and contiguous array bytes for state/mask/time/x/y, including signed
zero; NPZ container byte identity is not the criterion. Preserve fresh outputs
under exclusive artifacts/persistent_curator_equivalence_20261006 only.

Run in the existing pinned .venv-curator-py312, CUDA_VISIBLE_DEVICES empty,
OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1. Outer reviewed
supervision must impose CPUQuota=100%, MemoryMax=4G, MemorySwapMax=0 and
RuntimeMaxSec=300; startup physical MemAvailable >=50 GiB, continuously sampled
runtime MemAvailable >=22 GiB. Stop the entire owned unit/process group on
violation; do not stop any scientific job. Script boundary checks supplement,
not replace, the external monitor. No service command is executed by this prep.

The replay entry requires --module and --module-sha256; --execute is explicitly
required. It checks historical result and old sampler SHA, and exact environment.
Root should bind all three staged source hashes and retain actual unit properties,
memory observations and terminal status before interpreting result.json.

## Measurements and limits

Record one-time persistent import, old CLI wall time, new function wall time,
Linux self/children cumulative ru_maxrss, and available memory for every frame.
The first new call and later calls must be reported separately. External cgroup
peak supplies whole-unit memory evidence; ru_maxrss is not per-call memory.
Old CLI timing includes import/startup/IO; new calls exclude the separately timed
one-time import. Existing-file warm-cache replay is not a CFD real-time benchmark.
Any mismatch or resource failure rejects this engineering optimization without
changing tolerances. No speedup or scientific/control success is claimed yet.

## Completed preparation evidence

Three synthetic CPU tests passed in 1.11 s: fresh source plus pressure/mask/time
semantics, exact-array comparison sensitivity, and original numerical AST identity.
They import no real Curator/VTU/model and are not actual sampling equivalence proof.
