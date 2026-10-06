# FC-E078 — exact initial-policy attribution control launched

This is a single predeclared comparison, not a new optimization or a claim of
physical success. At2026-10-06 15:30:14UTC the approved unit
`fluid-control-p064-initial-projected-ppo-long-cfd-20261006.service` launched
with invocation `dbc0e8f47f994e7280694e9ed6714c56`, PID2315536. An actual26/800
progress observation confirmed both paired branches completed20solver steps
per interval with clean termination. No conclusion is drawn from this prefix.

The hypothesis is that the trained B policy weights add physical benefit
relative to the exact original initialization under the same transformation.
The fixed b00 restart148→228,800cycles, primary(168,228], all six windows,
reflection projection, single amplitude/rate filter and original physical
criteria (2% drag,1.05 centered lift-RMS ratio,10% mean-bias) are unchanged.
Compare to the existing trained B result, not a rerun. There is no posthoc
difference threshold, seed search, automatic retry or policy replacement.
An indistinguishable benefit would weaken the learning-contribution claim;
superiority supports only this seed and observed phase, not general RL
superiority over simple controllers or a full factorial causal decomposition.

## Exact identities

- Approval: `docs/P064_INITIAL_PROJECTED_PPO_LONG_CFD_APPROVAL_20261006.json`, SHA
  `f4e35a92a227b29fcf216018f09b3d320b382ffc392d9aad7e73616dc32c3797`.
- Executed driver: `artifacts/p064_initial_policy_cfd_source_20261006_immutable/run_p064_initial_projected_ppo_long_cfd.py`, SHA
  `2777f8b8a2dc948c1714c64fa7b8bddaec7c9dc9b7edc764e572324cd398c277`.
- Reconstructed initial policy: `artifacts/p064_exact_initial_policy_cpu_20261006_r3/ppo_initial.zip`, SHA
  `8a99bc1ad855b6ca510950206253021accb186cab393d136dbc4935ac3cc0108`.
- Exact original initial tensor SHA:
  `6bc539885d8c63fc922eccaba0363593555cf1b85ece5e48783d79d2ea2fa1cf`.
- Reconstruction source SHA:
  `c6884bb1c12c72bb6954e627d16796da312f3ccdb00ed884742e6f8c18eb8de2`.
- Shared original VecNormalize SHA:
  `8c07ef15bd41a8981f2ec0d241c85092b643ca740fea9f44866eecceac1197ad`.
- Trained B policy reference SHA:
  `f764463983355779efff8d1b1994cfaf560ab7274d54b014d34a1f084b4b307e`.
- Output: `artifacts/p064_initial_projected_ppo_long_cfd_20261006`.

VecNormalize has norm_obs=False and norm_reward=False. The exact same artifact
is frozen with training=False and checked as an identity transform; this is
not learned nontrivial normalization. Only policy weights are changed. A new
empty optimizer object is constructed by SB3, but zero optimizer updates,
environment steps or training iterations are performed. No FNO or GPU is used
online. The policy is deterministic, followed by the same projection/filter.

## CPU reconstruction and preserved recoveries

R1 invocation `9deaab9fd4f94d0aadff97d71fda6ee6` passed the exact weight check
but failed the extra Vec snapshot check because SB3's detached object's
pickle hook expected a missing class_attributes field. No policy artifact was
written. R2 `801925ab812745fd99ea8e03f3e9c0a8` changed only that check and
completed, but its serialized synthetic spaces used float32 instead of the
original float64. It is retained and not used for CFD.

Root separately approved R3, which used the exact original Vec spaces.
Invocation `5f256594d5994101b1fe9410e2658b3a` completed15:27:15UTC, exit0.
Both initial construction and actual saved-policy CPU reload verified exact
tensor6bc539 and observation/action-space equality, with empty optimizer
state and zero updates. The reconstruction used4GiB/noSwap/CPU1/120seconds
and CUDA hidden. Original R1/R2 scripts/results remain on Spark as history;
only corrected R3 source is promoted to canonical, after execution.

## Reviewed execution contract

Root and independent reviewer accepted the minimal driver delta and11CPU
tests. Seven numerical helper ASTs and the entire800-cycle loop match the
existing B driver. Final persisted-spec preflight checked11input hashes,
9source hashes, actual initialized-policy proof and unchanged restart trees.
Actual policy/Vec space equality is retained before any solver starts.

The controller cgroup is8GiB/noSwap/CPUQuota400%, TasksMax512, Type=exec,
RuntimeMaxSec3750, TimeoutStopSec120, KillMode=control-group and OOMPolicy=stop.
The two owned solver containers retain8GiB/noSwap each. Existing MemAvailable
50GiB startup/22GiB runtime guards and3600second inner budget are unchanged.
No execution is authorized merely by this report or a reproduction command.

Independent terminal review is assigned to the reviewer, not the author of
this phase adapter. It will check raw six-window forces, exact paired-zero
comparability,800projection/filter actions, solver health and cleanup before
any outcome is recorded. The failed full surrogate gate and rejected H25
candidate remain separate; this comparison does not declare overall success.
