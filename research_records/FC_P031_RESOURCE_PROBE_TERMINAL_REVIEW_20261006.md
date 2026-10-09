# FC-P031 first resource attempt: preflight refusal

## Update: separately approved R2 also failed, with an observed floor breach

R2 unit `fluid-control-fcp031-resource-probe-r2-20261006.service`, invocation `48a1f83e007a4bb59f1ea025af2e7dd7`, terminated failed/exit1, MainPID0. It used the same approval/source after Root's separately approved16-file cache pass. Actual container `b6964cb89fcc3b38ecf733a786ba31b8b24815ee81bb6bddf11ebb0abdac2050` started04:48:41.543211313UTC and ended04:49:05.391445413UTC, exit75/noOOM.

Preflight CUDA free33.690525GiB passed. Execution subsequently reached the H25 objective and the guarded flow-prediction callback raised `physical/CUDA memory floor`. The outer guard explicitly recorded CUDA free **19.816570281982422GiB**, below20. This is an actual recorded resource-floor breach; stopping on detection does **not** prove the floor was maintained. There is no completed forward/backward result, gradient audit, optimizer step, saved candidate or reload result.

Twelve host watcher samples show MemFree36,597,305,344→21,282,054,144bytes, a14.2634GiB drop; MemAvailable fell7.8881GiB. Before the final sample, free fell9.0331GiB while available fell2.7206GiB, consistent with substantial reclaimable cache growth during model/data reads and the exact44 hash scan. In the final2s both fell about5.2GiB, consistent with active allocation during objective execution. These are phase-consistent inferences, not exact attribution: the inner runner checked before appending its readings and lost its in-memory resource list on failure. The actual failed allocation phase/rollout index and complete H25/backward peak remain unknown.

A proposed≥40GiB startup would provide about25.7GiB after the same observed14.26GiB consumption, but cannot guarantee the unobserved remaining graph/backward peak. Preserve all20GiB floors, allocator and scientific arithmetic. Before any new approval, minimally persist the measured failing inner-guard row before raising and log phase boundaries; do not fabricate a successful result or silently retry.

R2 hashes: run.log `139498090154ec79d23ee83f8d4f3828ecdb5fac3e29a788d5218acbb9fb77a6`; resource_watch.jsonl `9985e98a774764dc0280e773a6497b816ccd2a7bdfc564d7090336847842574f`; terminal inspect `fc8a4f381004ff434c6aa50f041f65812e632ecc05aba026e3d0f99c71365b3e`.

Read-only exact22 completed-checkpoint eligibility: manifest `/tmp/p028-runner-review.hf3bdM/p028_checkpoint_cache_exact22_manifest.json`, SHA `c0ae36af0de1c1a582e615bc978eda2b54b62b5a0ef512446c8b9d5720dd1b42`, lists22 files totaling3,589,185,354bytes. All stored stat identities, no-symlink paths and small audit-reference hashes matched. `fuser` returned1/empty, Docker and GPU compute lists were empty at inspection. The size is an upper bound, not measured resident reclaimable memory; no checkpoint payload was read or advised. A separately approved new receipt would be required. No additional cache pass or retry was executed by this reviewer.

## Original first-attempt evidence

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
