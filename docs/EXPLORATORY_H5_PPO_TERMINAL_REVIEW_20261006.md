# FC-E050: actual exploratory HydroGym / SB3 PPO terminal review

Training completed; this is not a CFD result or formal scientific admission.
The actual user unit `fluid-control-exploratory-h5-ppo-20261006.service`,
invocation `21cb82da66214924b38f120eb30723e5`, ran06:51:48–06:53:11UTC and
ended active/exited, MainPID0, ExecMainStatus0. The supervisor independently
retained returncode0/error=null. Type=exec makes the1950s external runtime limit
effective; cgroup MemoryMax12GiB/MemorySwapMax0/CPUQuota100%, internal1800s.
No restart, CFD simulation, model-FNO optimization or formal gate alteration.

## Exact execution identity

- Approval/source_spec SHA: `8aa44f5f177d6c7d831a1efc55abf9d8db4b700d640cb640c5fcbb709cc44569`.
- Immutable39-source manifest: `730c9f315234a59c381fefa176d46a124eb310d41886176547496bce8b998856`.
- Executed trainer: `8806682ec66d0a1b8dde30b0f361bb77e115f9add85f207651d9d01d1d8d4100`;
  wrapper `cfda96c4d8bd6e1fa6b47bd41dd01b2308b0b9a7d264f10f54301431c50e0532`;
  supervisor `fb1f61f1fdcbc6d9d29192a09f77ad29f6aed3589737a45f7b5d9b116d24d482`.
- Base Git HEAD observed at terminal was76f13dc; the new scripts were not yet
  committed. A later source-capture commit must not be called execution HEAD.
  The read-only source snapshot and approval existed before actual launch.
- Result `artifacts/exploratory_h5_ppo_training_20261006/payload/result.json`:
  `138a7b192eef1a6454cefa47cda7803c9b362937641a645c00889ac5a5d7a0c4`.
- Supervisor result: `87d9f0be7dfb49565c0ea335691ad598f644f411b22297e6cce7fac4e8ab384c`;
  memory trace `6d730975e14f9278ded26d256ec2a3979f2a753eb2aaa82f426746f8ec09bf61`;
  worker log `5a6f08eeb0349f7425a59533a8e2b20fd0002029eda20ddb62757e66c6436a11`.

All five result artifact hashes were independently recomputed; the supervisor's
result hash matches. No checkpoint deserialization or HDF reread was needed for
this terminal audit. Original K1 manifest7adca21e and its P009 flow / P026K1
aerodynamic pair remain unchanged; official loading used high/TF32 checks,
followed by explicit highest/no-TF32 inference. Runtime package/source bindings
are in the copied source_spec, including installed SB3/Gymnasium and HydroGym.

## Actual learning and lifecycle

The raw transition log contains exactly4096 records:1024 from each of the four
fixed train-zero starts b00/b02/b04/b06, frame0. There are816 completed episodes,
all length5,816 time-limit truncations and no divergence termination. This is
four early train starts, not all1368 windows, validation or frozen performance.
The actual CPU lifecycle test separately proved terminal-observation bootstrap
using real HydroGym/DummyVecEnv/SB3 rollout code; see the runtime review.

Eight512-transition rollouts completed32 PPO epoch updates and64 recorded
optimizer steps in exact order1..64. FNO parameters were excluded from the PPO
optimizer and checked frozen/no-grad throughout; the executed before/after FNO
tensor digest check passed. Policy tensor SHA changed from
`6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf` to
`2bce6ffb402e01cc9079c5df4920d41b18cdeb5a758923d798c3141c721231a0`.
This is genuine policy learning, not a constant controller or MPC action choice.

Final SB3 telemetry: policy-gradient loss−0.0116701482, value loss0.0210340086,
approximateKL0.0106169721, explained variance0.7726419568, clip fraction0.048828125.
All logged numeric metrics are finite, including the final flushed update.
Mean five-step episode return−0.544470935;93.041992% of training actions were
rate-limited, applied omega range[-0.5,0.5]. Mean per-step drag-gate penalty
−0.107810925 dominates; mean rearCl fluctuation and bias penalties were zero.
These are surrogate learning diagnostics, not physical improvement or a basis
to change reward weights. A five-step reset never explores the long-duration
action distribution, including saturation at±0.75.

The final policy was saved once:
`ppo_final.zip` SHA `3af2b2863f7fffa3579832c10dd2e7053caf80fc2719ed72ad842858f3da9fe1`.
Paired identity-normalization artifact `vecnormalize.pkl` SHA
`54a08a438501aac0663e50da931f41aabb63cdeb8255aa80051e7af1b4eaaba2`.
Transition log SHA `8e9150fa8e75cc1851bf5b5ef881427ee2b909d153a41ef5d4819f4fe9340849`;
SB3 progress JSONL SHA `74697e520ccab5bc07e1ab6cf0693e3b16ced5b647d9d2b361cce7abd7c3d888`.

## Resources and interpretation

Worker time80.901766945s.167 physical-memory observations show minimum
MemAvailable119542509568bytes (above22GiB runtime and20GiB reserve). Low MemFree
was reclaimable cache, not exhaustion. CUDAfree was not a failure gate under the
explicit Spark UMA profile. No sampled resource breach or supervisor error.

The canonical62-sample reward window has only1..5 newly predicted samples per
episode, so mean-force feedback is diluted while action penalties are immediate.
Timeout value bootstrap extrapolates beyond short resets; short-model bias and
poor transfer remain live risks. Training success does not repair K1's retained
H100 formal FAIL, override the original physical10% condition, or establish any
drag reduction. The next separately approved experiment is direct deterministic
policy feedback on actual paired CFD, not substituting the previous MPC selector.
