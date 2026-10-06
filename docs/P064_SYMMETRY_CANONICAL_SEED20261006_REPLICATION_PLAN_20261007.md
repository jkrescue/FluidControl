# P064 symmetry-canonical seed20261006 replication — prospective plan

Status: preparation only. No PPO training or CFD execution is authorized by this document.

## Hypothesis and one intended change

E083 showed that the symmetry-canonical train/deploy interface recovered the original b00
primary physical criteria for seed20261007, while the first 6.2 D/U mean-bias metric still
failed. This fixed replication tests whether the same interface is reproducible across the
two predeclared seeds `20261007` and `20261006`. Relative to the reviewed E082 runner, the
only scientific change is `PROTOCOL["seed"]: 20261007 -> 20261006`.

Unchanged: P064-B manifest `92766915...e7891`, frozen official FNO, adapter SHA
`a55b5699...38ae`, standard SB3 PPO `MlpPolicy`, 32768 steps, H5 episodes, four environments,
24 reset starts, 69 observations, reward and 62-force history, PPO hyperparameters,
precision, final-policy-only selection, and all data/runtime identities. This is not a seed
scan and no reward-selected checkpoint is allowed.

## Evaluation and decision

Training completion only establishes an engineering candidate. After independent terminal
audit, a separately approved paired b00 run must reuse E083's exact 800-cycle 148->228 CFD
protocol, zero branch, six windows, canonical adapter, and original primary criteria:
drag reduction >=2%, rear-Cl RMS ratio <=1.05, and absolute mean-bias ratio <=0.10. The
early window remains separately reported. Cross-seed support requires both fixed seeds'
primary windows to pass; a failure is retained and does not trigger another seed, tuning,
or checkpoint choice.

Expected wall time from actual prior runs is about 10 minutes for PPO and about 18 minutes
for the later CFD, so the two stages can fit within 45 minutes if separately approved and
started promptly. This estimate is operational, not a completion guarantee.

## Resource and launch contract

Training remains one GPU under the reviewed supervisor: `MemoryMax=12 GiB`, no swap,
`CPUQuota=100%`, `TasksMax=256`, `RuntimeMaxSec=1950`, `TimeoutStopSec=20`, control-group
kill and OOM stop. The inner checks require physical `MemAvailable >=50 GiB` at startup and
`>=22 GiB` at runtime, preserving the user's >=20 GiB reserve. CPU b01 CFD may coexist only
while both independent guards remain satisfied.

After Lead converts the reviewed pending JSON to the final approval and freezes the exact
runner under the stated immutable artifact, the precise launch form is:

```sh
approval=/workspace/fluid_control/docs/P064_B_SYMMETRY_CANONICAL_SEED20261006_PPO_APPROVAL_20261007.json
approval_sha=$(sha256sum "$approval" | awk '{print $1}')
systemd-run --user \
  --unit=fluid-control-p064-b-symmetry-canonical-seed20261006-ppo-20261007.service \
  --remain-after-exit --property=Type=exec \
  --property=WorkingDirectory=/workspace/fluid_control \
  --property=MemoryMax=12884901888 --property=MemorySwapMax=0 \
  --property=CPUQuota=100% --property=TasksMax=256 \
  --property=RuntimeMaxSec=1950 --property=TimeoutStopSec=20 \
  --property=KillMode=control-group --property=OOMPolicy=stop \
  /workspace/fluid_control/.venv-curator-py312/bin/python \
  /workspace/fluid_control/artifacts/p064_symmetry_canonical_seed20261006_source_20261007_immutable/scripts/supervise_p064_symmetry_canonical_ppo.py \
  --approval "$approval" --approval-sha256 "$approval_sha" --execute
```

The command is a reviewed template, not permission to create the final approval or run it.

