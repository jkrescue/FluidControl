# PhysicsNeMo FNO 20-epoch independent CFD screen (2026-10-02)

The action-conditioned, official PhysicsNeMo 2.2.2 FNO was trained on the existing 24 train trajectories from genuine OpenFOAM v2512 tandem-cylinder CFD, with four validation and four **held-out** test trajectories in the independent-v2 split. The model consumes the 3-channel field, valid mask, and current/next rear-cylinder rotation; it was not replaced with a custom model. The Curator VTKSource + PhysicsNeMo Mesh recipe and PhysicsNeMo HDF5 DataPipe were retained. GPU 0 training ran for 20 epochs in an isolated container. The GPU guard sampled every five seconds, capped the PyTorch allocator to 20% of total unified memory, and observed a minimum 97.03 GiB system `MemAvailable`, above the required 20 GiB floor.

The test cases are `expanded_test_00/01/02/04` (not `03`); each horizon used 64 segments across these four cases. All 1-, 10- and 50-step rollouts were finite and stable. Errors below are physical-unit mean absolute errors; lower is better.

| Horizon | Field FNO | Field persistence | Field with zero action input | Field with sign-flipped action | Rear force FNO | Rear force persistence | Rear force with zero input |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.002460 | 0.008973 | 0.002717 | 0.003436 | 0.07055 | 0.09986 | 0.22143 |
| 10 | 0.021562 | 0.084783 | 0.023722 | 0.029936 | 0.11765 | 0.96768 | 0.41548 |
| 50 | 0.072846 | 0.167903 | 0.084686 | 0.111039 | 0.59210 | 1.98172 | 1.51861 |

The observed-action model beats persistence and the zero-action-input ablation on the conservative checks at all horizons; sign flipping degrades field and force estimates further. The [machine-readable readiness report](results/expanded_fno_20epoch_control_readiness.json) is `CANDIDATE_SURROGATE_SCREEN_PASS`. The [observed test report](results/expanded_fno_20epoch_heldout_evaluation.json), [zero-input report](results/expanded_fno_20epoch_heldout_evaluation_zero.json), [sign-flip report](results/expanded_fno_20epoch_heldout_evaluation_sign_flip.json), and [ablation summary](results/expanded_fno_20epoch_action_sensitivity.json) retain per-case and numerical details.

This is an **offline same-geometry surrogate screen**, not reinforcement-learning success or physical drag reduction. In particular, the 50-step rear-force MAE remains 0.592, which is material and requires care in control optimization. The original curated-v2 first frame also has a [documented 0.005-time-unit force alignment issue](FRAME0_FORCE_ALIGNMENT_20261002.md); the FNO's one-step force targets use following frames, while HydroGym's t=80 initial force is now read directly from the raw source CFD. The ongoing medium-mesh grid-pair check has not finished, so spatial numerical sensitivity is still open. A CPU-only HydroGym PPO pilot starts only after this gate and needs its own multi-start audit before a frozen policy is tried in real CFD.
