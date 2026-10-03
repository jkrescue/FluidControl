# CFD sample-budget audit (2026-10-04)

This ledger counts each real OpenFOAM trajectory once, using its physical
duration in `D/U`; rollout-window reuse during FNO training is not counted as
new CFD.  Frame totals are stored endpoint frames and therefore include both
ends of every trajectory.  Shared restart frames can appear in more than one
file, but contribute no additional simulated interval.  Wall time, hardware
energy, unused frozen data, and synthetic/FNO rollout steps are outside scope.

## Training-data provenance

| source | trajectories | intervals per trajectory | stored frames | real CFD `D/U` |
|---|---:|---:|---:|---:|
| FNO base `train20` | 20 | 800 at 0.1 | 16,020 | 1,600.0 |
| FNO dynamic `train8` | 8 | 200 at 0.1 | 1,608 | 160.0 |
| **Current FNO training total** | **28** | - | **17,628** | **1,760.0** |
| Direct-CFD PPO training | 16 episodes | 2,048 total at 0.1 | 2,064 endpoint frames | 204.8 |

The proposed `directppo_train16` release is a curation of those same 2,048
already executed PPO decisions, not a new acquisition.  Its **incremental CFD
cost is 0 D/U**.  If added to the FNO corpus, the resulting training-data
provenance is 44 trajectory files, 19,692 stored frames and 1,964.8 D/U of
real-CFD history; the last 204.8 D/U remains charged to the original direct-PPO
training, not relabelled as free data.

## Validation, commissioning, and controls (not training)

| source | real branches | stored/observed endpoint frames | real CFD `D/U` |
|---|---:|---:|---:|
| FNO development `validation10` | 10 x 80 D/U | 8,010 | 800.0 |
| Dynamic6 validation-only panel | 6 x 20 D/U | 1,206 | 120.0 |
| Direct-CFD transport probe | 2 x 1 D/U | 22 | 2.0 |
| b00 frozen-policy/zero pair | 2 x 80 D/U | 1,602 | 160.0 |
| b01 frozen-policy/zero pair | 2 x 80 D/U | 1,602 | 160.0 |
| b01 fixed b00-sequence replay | 1 x 80 D/U | 801 | 80.0 |
| **Validation/control subtotal** | - | **13,243** | **1,322.0** |

The b01 zero branch is counted once: the fixed-sequence comparison reused that
existing reference.  The failed `probe20_v1` produced no CFD transitions and
is therefore zero in this ledger.  The declared frozen10 split is neither
materialized nor used and is excluded.

## Claim boundary

The evidence supports a narrow **zero-marginal-CFD data-reuse** statement for
`directppo_train16`.  It does **not** yet support a CFD-sample-saving or
end-to-end acceleration claim: the 204.8 D/U used to learn the direct-CFD PPO
policy was genuinely solved, and the current FNO training data already cost
1,760 D/U.  There is no controlled equal-quality comparison showing that the
surrogate path reaches the same physical result with fewer CFD samples, nor a
validated surrogate-only PPO result.  Existing data may be reused without a
new solve, but it is not free in an end-to-end accounting.

Evidence is bound by the following immutable artifacts (SHA-256 prefixes and
suffixes shown): `train20` manifest `5213c7bb...ddd2`, `train8` manifest
`a0bd0e3b...3f35`, dynamic6 manifest `bfa49031...7dae`, direct-PPO result
`5a8d552b...3fb4`, and transport-probe result `f907c3fb...7441`.  The paired
and replay results are documented in `DIRECT_CFD_PPO_RESULTS_20261004.md` and
their underlying JSON receipts.
