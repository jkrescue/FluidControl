# Gated PPO-to-real-OpenFOAM feedback pilot

This is the short real-CFD execution stage after [the HydroGym surrogate PPO pilot](HYDROGYM_PPO_20EPOCH_PILOT.md). It uses **actual** wake-probe velocity and rear-cylinder force coefficients at every interval, not FNO-predicted observations. The FNO trained on genuine OpenFOAM trajectories is used only to train the HydroGym PPO policy; the frozen policy is then queried on real CFD observations. Policy inference occurs in a CPU-only, read-only, network-disabled container; OpenFOAM remains in its independently pinned solver container. No host Python environment is changed.

The runner refuses to start unless the 20-epoch independent FNO readiness report is `CANDIDATE_SURROGATE_SCREEN_PASS`, the PPO pilot has completed all multi-start surrogate evaluations, the checkpoint epoch is 20, and the pinned HydroGym checkout/image and source t=80 CFD restart are present. It requires both positive aggregate held-out validation/test reward and a positive surrogate reward specifically at the shared t=80 physical restart. The latter is important: a policy that improves on average across diverse starts can still be unsuitable for the only matched physical restart. It limits the real-CFD pilot to 2–32 intervals, `|omega|<=5`, and `|delta omega|<=0.5` per 0.1-time-unit interval. It verifies each 20-step OpenFOAM solve, maximum Courant number, continuity, finite observations and exact restart time. The policy's deterministic requested action and the rate-limited applied action are both recorded.

After the FNO gate and PPO pilot have **actually completed**, create a fresh isolated case and run a short pilot:

```bash
python3 cfd/tandem_cylinders/make_probe_feedback_case.py probe_feedback_ppo_pilot_01 --steps 3
python3 scripts/run_tandem_ppo_cfd_feedback.py probe_feedback_ppo_pilot_01 \
  --policy-run artifacts/hydrogym/tandem_ppo_multistart_8192_spark \
  --readiness artifacts/tandem_fno_expanded_spark_20epoch/control_readiness.json \
  --output artifacts/hydrogym/tandem_ppo_cfd_feedback_pilot_01
```

Both paths must be new: the scripts refuse to overwrite prior cases or outputs. The initial observation is read from the real monolithic t=80 baseline. For each subsequent interval, `scripts/infer_tandem_ppo_action.py` loads the frozen Stable-Baselines3 PPO policy, predicts from the 67-channel observation, and the host runner ramps the rear-cylinder `rotatingWallVelocity` boundary to the rate-limited action. The next observation comes from the new segmented OpenFOAM solve. The runner also reads the matched zero-action monolithic reference at the same time and records Cd, Cl, wake-probe differences, and an instantaneous objective using the same drag/lift/actuation/rate weights as the HydroGym surrogate. That comparison is **not** a performance claim over a 3-interval horizon.

## 2026-10-02 execution result

The complete 20-epoch-FNO → HydroGym PPO → real OpenFOAM feedback path has now run. The 8192-step PPO used eight real-CFD-derived training starts and was evaluated on 24 held-out validation/test starts. After counting the duplicated common t=80 restart only once per split, mean surrogate reward change was +0.02535 on validation (8/9 positive starts) and +0.01781 on test (7/9 positive starts).

A 3-interval integration run completed cleanly and had mean objective difference -0.00240 relative to the matched zero-action trajectory. This was only a software/solver integration result. A subsequent 32-interval run also completed cleanly, with maximum Courant number 0.244 and negligible continuity residual, but had mean objective difference **+0.00385** (worse; lower is better). Only 8/32 intervals improved: the first eight averaged -0.00217, whereas the last 24 averaged +0.00586. Drag increased by +0.01070 on average; reduced lift-squared did not compensate. Therefore the current controller has **not** demonstrated sustained real-CFD benefit.

The clean held-out audit also showed -0.00337 surrogate reward at the shared t=80 restart even though its multi-start average was positive. The runner is now fail-closed on that start-specific condition, so this known-unsuitable policy cannot be sent through another t=80 real-CFD run. The next controller must pass that gate before another expensive physical replay. A meaningful control claim additionally requires longer frozen-policy trajectories, matched zero and simple-action baselines, uncertainty/numerical sensitivity, and explicit time-averaged drag/lift/actuation accounting.
