# P064 B canonical-reward sequence replay

Status: `COMPLETE_NOT_ADMISSION` (2026-10-07).

This CPU-only diagnostic applied the frozen `canonical_joint_v1` arithmetic to
the already saved B H1--H5 prediction/truth arrays for the fixed b01/b05
minus/zero/plus branches.  It made no model forward pass, optimizer update, or
CFD call.

## Bound inputs and scope

- Saved action arrays: `artifacts/p064_b_short_action_sequences_20261007/result.json`
  SHA-256 `ce4dab24d07fef642be9d43faf9897519271eb0b1281c71e7b0488c06598e6df`.
- Frozen reward: `canonical_joint_v1.py` SHA-256
  `138ab2b49ebddcbed2a24486c995b85a3a5ae226ee936ff2ed5de318496ee8bd`.
- Frozen causal-history reader SHA-256
  `ec8720581e0f362c308a2bf82fb1f05585eaf876cf17956a6c8e7df0ca14c763`.
- Each mixed window used 62 real samples ending at the original q0.  Its last
  force was checked bitwise after float32 conversion against the bound original
  validation HDF q0; each future prefix used the stored float32 clock.
- B PPO training baselines only cover b00/b02/b04/b06.  For this b01/b05
  validation diagnostic, the unchanged formula used each existing same-phase
  validation-zero final 60 D/U source.  This is explicitly a phase-matched
  validation-baseline extension, not a claim of byte-identical B training reward.

## Result

The result is
`artifacts/p064_b_canonical_reward_sequences_20261007/result.json`, SHA-256
`af39c877bba7f990b35ea9802b58e966b91dbc78e9a0664292e798871060f5f1`.

For instantaneous endpoint-window canonical cost, the predicted minimizer set
matched truth for all b01 horizons.  At b05, H3 predicted `plus` while truth
selected `zero` (truth-cost regret `0.0043662956`); H1, H2, H4, and H5 matched.

For the separately reported five-step truncated return
`sum(gamma^j * -0.1 * canonical_cost_j)`, with frozen `gamma=0.99` and no value
bootstrap, b01 H5 predicted `zero` while truth selected `minus` (truth-return
regret `0.0006976581`).  At b05 H4 it predicted `plus` while truth selected
`zero` (regret `0.0017118183`); b05 H5 selected `plus` correctly.  Exact tie sets
are preserved rather than broken by role order.

## Decision boundary

The diagnostic shows useful but imperfect five-step action ranking.  It does
not authorize a drag-only selector, MPC, PPO, model training, or CFD.  Truth
reward substitution does not repair field/force prediction error, and two fixed
validation phases do not establish generalization or statistical significance.
The already accepted E114 B policy remains the default basic closed-loop result.

Execution unit:
`fluid-control-p064-b-canonical-reward-sequences-20261007.service`, invocation
`8668243edc3b4159a3b835997a4f8a5a`, terminal exit status 0 under CPU1,
2 GiB memory, no swap, and a 120-second limit.
