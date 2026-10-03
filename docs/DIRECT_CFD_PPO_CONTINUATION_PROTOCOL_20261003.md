# Optional direct-CFD PPO continuation protocol

If the fixed 2048-transition run is scientifically inconclusive, training may
continue from its exact final PPO archive and matching `VecNormalize` file. A
continuation is not a new random seed and does not relax any physical target.

The launcher verifies both artifact SHA-256 values against the completed prior
result. SB3 loads the saved policy, value function, optimizer state, learning
schedule state, random-generator state, and cumulative `num_timesteps`; the
saved observation/reward running statistics are loaded with normalization still
in training mode. A mismatch between the policy's `num_timesteps` and the
predeclared prior cumulative interaction count fails closed.

Reports distinguish prior, newly collected, and cumulative transitions. Each
new 256-transition block records SB3 optimizer diagnostics, cumulative
`_n_updates`, parameter-tensor SHA, and L2 parameter change relative to the
start of that continuation. Raw physical rewards remain separately journaled.

Continuation is justified only as added optimization/sample budget. It cannot
be selected by loosening the 2% drag, 1.05 rear-lift fluctuation, or 0.10
mean-lift-bias targets, and it cannot turn training-phase evidence into an
independent-generalization claim.
