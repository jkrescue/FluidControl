# Actual K1 UMA GPU inference — terminal engineering review

Independent read-only inspection found successful terminal execution of user unit
`fluid-control-k1-uma-gpu-inference-20261006.service`, invocation
`74f94b14bb81495daed3568042bed798`, 05:58:03–05:58:11 UTC. MainPID0,
active/exited, Result=success, ExecMainCode1/Status0. No restart was issued.

Approval SHA4ba9ea08b0297c43dfd1c15a1cbb1d677bb30f24b688f0a88d8e2301fdf3d958
and executed source56c17443245c60f51b4e4c8f15b1851136a091c73441e5ecd76eebe0695f5e74
were independently rehashed. K1 manifest is exact7adca21e…acc7. Source checks the
official FNO/checkpoint module hashes and uses the existing role loader. Review
read small result/log/properties only; it did not independently reload models.

Three actual synchronized CUDA inference calls took1.150291011s, .139241173s,
and .137660911s. Full5×2×4 force predictions and five costs repeat exactly across
all three calls. Each selected+0.05 diagnostically; no action was applied. Reported
GPU allocation peak755,589,120bytes, reservation peak796,917,760bytes. Model-state
digest equality and all gradients absent were checked by the executed worker.
No optimizer, CFD, checkpoint or scientific admission was produced.

Eighteen host samples span8.512270s; minimum MemAvailable121,838,997,504bytes,
minimum MemFree1,855,971,328bytes. Inference CUDA-free observations1,914,490,880bytes
are below20GiB while real inference completed and physical available memory was
ample. This accords with [NVIDIA5728](https://nvidia.custhelp.com/app/answers/detail/a_id/5728/kw/x79)
on UMA reclaimable page cache; no cache flush
was performed. It does not guarantee arbitrary allocations or longer workloads.

Retained unit properties confirm MemoryMax12GiB, MemorySwapMax0, CPUQuota1CPU,
RuntimeMax200s, KillMode=control-group. Worker validated the same cgroup limits.
The terminal systemd MemoryPeak3,670,016bytes is inconsistent with total CUDA
allocation and must not be presented as whole-run/GPU peak. Torch allocator peaks
and physical observations are reported separately; non-PyTorch allocation is
not included in torch peaks. Sampling does not prove unobserved instantaneous
minima. Neither CUDA-vs-CPU exactness nor80D/U endurance was tested.

Evidence under artifacts/k1_uma_gpu_inference_probe_20261006:

- worker_result.json:7e3c373d895ae839ad360369988c49aa52a1ecc14d0f6d94b4793b83efb0559c
- supervisor_result.json:4341c84dd882ef0bd1c70d01ca3b5af876509c383b46963a205fd03854b9d9be
- inference.jsonl:95a3fba48d689e1e98efe1b6eb8ae98fc1eef38b8a0cb177f1bc15a08a1b2fac
- memory.jsonl:717e044ac5a08460e8ed8cb53970af223dd8fb61dce943c64550611532f8067e
- run.log:8057e222a15e0f38a5cb445e1425b066bd334754e6a1d3bcc83941c46d481e9b

`inference.jsonl` rows equal the worker result records exactly. K1's original
formal rejection remains unchanged. This establishes a small actual official-model
GPU inference path under the UMA-aware guard, not control benefit or admission.
