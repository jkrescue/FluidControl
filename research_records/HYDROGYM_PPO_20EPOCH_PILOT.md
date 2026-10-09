# Gated HydroGym PPO pilot on PhysicsNeMo FNO

This is the next **surrogate-only** experiment after the 20-epoch FNO independent-test and action-ablation gate. It does not claim real-CFD closed-loop performance. The policy uses the official HydroGym `FlowEnv` interface and Stable-Baselines3 PPO around the already trained, frozen PhysicsNeMo FNO. Neither a new PhysicsNeMo model nor a fake flow dataset is introduced.

The pilot trains for 512 timesteps from a real CFD t=80 frame in `expanded_train_00`. It compares the deterministic PPO policy with zero requested action on all four validation and all four test trajectories, at curated frames 0, 100, and 400, for 32 control intervals each. Frame 0 is the **same t=80 CFD restart in all trajectories** and is not treated as eight independent starts; the later frames probe differing physical action histories. The audit reports each start separately, including physical Cd/Cl, reward components, action magnitude, rate clipping and normalized-state bound. Reward changes are surrogate predictions only.

After `scripts/run_tandem_fno_20epoch_spark.sh` has completed its observed, zero-action and sign-flip held-out evaluations, create the gate without overwriting an existing report:

```bash
python3 scripts/assess_tandem_control_readiness.py \
  artifacts/tandem_fno_expanded_spark_20epoch/heldout_evaluation.json \
  artifacts/tandem_fno_expanded_spark_20epoch/heldout_evaluation_zero.json \
  --output artifacts/tandem_fno_expanded_spark_20epoch/control_readiness.json
```

Only if that report says `CANDIDATE_SURROGATE_SCREEN_PASS`, run:

```bash
bash scripts/run_tandem_hydrogym_ppo_pilot_spark.sh
```

The runner checks the pinned HydroGym checkout and CPU-only derived container, disk and memory budgets, refuses to overwrite outputs, and runs with no network or GPU access. The policy and full audit go under `artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark`. A passing FNO gate is necessary, not sufficient: a policy that worsens held-out surrogate reward, diverges, or fails real-CFD matched-baseline checks must not be described as successful control. Any frozen candidate subsequently needs a fresh t=80 real OpenFOAM feedback trajectory, matched zero-action baseline, longer horizon and numerical-sensitivity review before a physical result can be claimed.

For an unattended handoff from the existing `fluid-control-fno20` tmux job, use `bash scripts/run_tandem_postfno_pilot_spark.sh` in a separate monitored session. It waits for that job to exit, requires the pipeline's final action-ablation success marker, revalidates the training and all three test reports, writes the readiness report, and starts the CPU-only PPO pilot **only** when the screen passes. A failed or incomplete FNO pipeline cannot trigger PPO.
