# FC-P031 first resource attempt: preflight refusal

Status: operational preflight failure; no H25 objective/resource result was produced. No retry or cache advice was executed by this reviewer.

Exact unit `fluid-control-fcp031-resource-probe-20261006.service`, invocation `42df9cff77e24312a23a431ef1c5870a`, was independently observed failed, MainPID=0, ExecMainStatus=1. Approval SHA: `3177f364230fe4900c7f1d3062d0a80483c8771861fe0efdd92093d54a42acca`.

Actual official-image container `e8f38212520f2dd9c4aae27fc2f37ef0c1962965060f3b6ab0d6a23251574f50` exited75, OOMKilled=false, at04:45:27.543693311UTC after starting04:45:24.165287585UTC. The inner GPU guard refused to launch its child: CUDA free30.674636840820312GiB was below required31.30140998840332GiB (allocator cap7.30140998840332 + floor20 + margin4). MemAvailable114.95566940307617GiB passed. There is no payload directory. Thus this is not a model, gradient, H25-memory-peak or scientific failure.

Two outer watcher samples have minimum host MemFree33,301,004,288bytes and MemAvailable123,795,468,288bytes. Host MemFree is not interchangeable with CUDA-reported free memory. No guard reduction is justified.

Retained artifacts under `artifacts/fcp031_h25_resource_probe_20261006`:

| Artifact | SHA256 |
|---|---|
| run.log | 8b39acce24ec3b30a8ec4b3bf773d9c23b34e5caf117709f4f43f60b278516f8 |
| resource_watch.jsonl | f25f14075c2f278b1c4a8dad7dd66a0bd824865ad22799c107d08403d7c34acc |
| evidence/container_created.json | 6d9b684593e272559ed8d47f995fe509d177895e3c1971471f489714d4db5ee3 |
| evidence/container_terminal.json | ba60948abbc77cd7bc9603248241149001680843282758dfd3abf274db8f5196 |

## Read-only bounded cache-recovery eligibility

The previously reviewed completed-validation helper remains at `/tmp/p029-cache-review/advise_p029_completed_validation16_once.py`, SHA `a6c96d35cd479afc4d6f50d8186fb355dce63f38bb76dec3d441f78fa4ef4d7e`, with exact-FD helper `/tmp/p028-runner-review.hf3bdM/advise_p028_checkpoint_cache_exact22_once.py`, SHA `ec76ca56e3bd3b090416ccccef68d9657daf36117dd86bb316f3bac651d825ac`. It is not the train44 helper. Historical receipt `artifacts/fcp029_cache_advice_20261006/completed_validation16_r1.jsonl`, SHA `acad2af0c12ea2b5178db97eeb5dc08a7231061ff75480b4e04cc3cbdf417264`, records16 complete in2.3894s.

This review checked only existing small manifests, source bytes, file metadata and live reader/process lists. Exact validation10+dynamic6 targets total3,198,373,880bytes; all are regular files with no symlink ancestors. At the check, `fuser` returned1 with empty stdout/stderr for all16, `docker ps -q` was empty and NVIDIA compute-PID output was empty. These observations must be repeated immediately before any separately approved action. No HDF payload was read or hashed and no cache hint was issued.

Minimal proposed preparation: retain the reviewed16 helper body and all exact manifest/completed-evaluation proof pins; change only the fixed exclusive receipt to `artifacts/fcp031_cache_advice_20261006/completed_validation16_r1.jsonl` and operational label. Preserve no-follow same-FD read-only hash/fstat/advice, per-file idle checks,120s deadline and dual20GiB floors. No global cache operation, scheduler, automatic retry or default repeated44 pass. The total file size is only an upper bound on potentially reclaimable clean resident pages, not proof that2GiB will be recovered. Fresh CUDA/host headroom must determine any subsequent separately approved attempt.
