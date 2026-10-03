# Frozen final2048 PPO b01 independent-phase validation

The exact final policy and matching frozen `VecNormalize` artifacts from
`directppo2048_v1` are evaluated once at the independently reserved validation
phase `b01`. No checkpoint is selected from b01 and b01 is never used for
training or continuation.

- PPO and zero branches start from the identical audited t=130 restart.
- Both branches run 800 decisions at 0.1 D/U, covering 80 D/U through t=210.
- The unchanged paired physical statistics use only the final 60 D/U,
  t=150--210: total mean drag, rear `Cl'` RMS, and rear mean-lift bias.
- The unchanged joint thresholds are 2% drag reduction, rear `Cl'` RMS ratio
  at most 1.05, and absolute rear mean lift at most 0.10 of zero-control rear
  `Cl'` RMS.
- Policy inference is deterministic. Policy tensors and observation statistics
  must have identical before/after fingerprints; `VecNormalize.training=False`
  and `norm_reward=False`.
- The b01 zero-control baseline may be read only to journal the canonical reward;
  the frozen policy does not consume reward. Final physical metrics come from
  the newly run paired branches.

This is independent-phase validation evidence, not frozen-test evidence, a net
energy claim, or a final paper conclusion. `omega^2` and `delta omega^2` remain
control regularizers rather than measured rotary power.
