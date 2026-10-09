# K1 UMA GPU inference — preparation only

NVIDIA primary guidance: https://nvidia.custhelp.com/app/answers/detail/a_id/5728/kw/x79
states cudaMemGetInfo does not include reclaimable page cache on Spark UMA.
Physical MemAvailable is the admission/runtime authority here; CUDA free remains
recorded observationally. This does not guarantee allocation success or change
historical guards/results. No global drop_caches, cache advice or swap is used.

One existing K1 official dual model, one already sampled physical frame at148.0,
three independent identical five-action H2 inference calls (first/cold and two
warm). Old reviewed selector is reused only as bounded inference load; no action
is applied, no CFD/optimizer/PPO/Curator sampling/model save. Original K1 formal
failure remains bound and scientific_admission=false. This is capacity/latency
engineering, not a new controller objective or GPU-vs-CPU numerical acceptance.

Exact host official Curator/PhysicsNeMo environment and FNO/checkpoint module bytes
are checked. Existing reviewed K1 manifest/config/norm/sample/source hashes are
read from the SHA-pinned historical approval as identity evidence, not new
execution authorization. A separate new approval must bind this source SHA and
the exclusive output path with status K1_UMA_GPU_INFERENCE_APPROVED.

Root-reviewed launch scope must use the existing .venv-curator-py312 interpreter,
CUDA_VISIBLE_DEVICES=0, OMP/MKL/OPENBLAS_NUM_THREADS=1, MemoryMax=12G,
MemorySwapMax=0, CPUQuota=100%, RuntimeMaxSec=200, KillMode=control-group.
The script verifies the cgroup memory/no-swap limits. Parent samples physical
MemAvailable every0.5s before/during child execution, startup>=50GiB/runtime>=22GiB,
180s inner deadline. Missing measurement/failure/signal stops only its own child
process group. External systemd lifetime is the independent backstop if supervisor
dies. PyTorch CUDA allocator fraction is fixed .06; it is not a total-system or
non-PyTorch memory cap. Record CUDA free/peak allocated/reserved, synchronized
inference timing, exact model-tensor before/after equality and no gradients.

Before launch Root must verify no competing GPU process and bind the exact unit,
source, approval and output; no launch command has been executed by preparation.
Retain failure evidence if model load/allocator/available guard fails; no retry or
guard reduction is automatic. Three-call success cannot establish80D/U endurance.
