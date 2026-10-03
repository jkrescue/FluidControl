# Frozen direct-CFD PPO b00 paired evaluation predeclaration

After the fixed 2048-transition direct-CFD PPO run completes, its final policy
and matching `VecNormalize` statistics may be evaluated once on the training
phase `b00`. This is an initial physical check, not independent generalization.

- Two real OpenFOAM branches start from the identical audited b00 restart.
- The policy branch loads the final policy and its exact `VecNormalize` file,
  sets `training=False` and `norm_reward=False`, and uses deterministic actions.
- The zero branch applies zero rotation. Both branches run concurrently for
  80 D/U (800 decisions at 0.1 D/U) with the same solver and safety checks.
- No policy parameter or normalization statistic is updated during evaluation.
- The only physical comparison is the existing paired-CFD statistic over the
  predeclared final 60 D/U: total mean drag, rear `Cl'` RMS, and rear mean-lift
  bias. The unchanged joint thresholds are 2%, 1.05, and 0.10 respectively.
- Action amplitude and slew are audited. `omega^2` and `delta omega^2` remain
  regularization proxies, not torque, rotary power, or net-energy evidence.
- Every episode case is preserved. The launcher must run as a user systemd
  service so terminal lifetime cannot interrupt the 800-step pair.

A favorable b00 result can motivate a separately predeclared b01 validation
run. This b00 result alone is training-phase evidence and cannot support an
independent-generalization, frozen-test, or final paper conclusion. No frozen
split is accessed by this protocol.
