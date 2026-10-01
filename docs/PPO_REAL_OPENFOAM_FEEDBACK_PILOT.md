# Gated PPO-to-real-OpenFOAM feedback pilot

This is the short real-CFD execution stage after [the HydroGym surrogate PPO pilot](HYDROGYM_PPO_20EPOCH_PILOT.md). It uses **actual** wake-probe velocity and rear-cylinder force coefficients at every interval, not FNO-predicted observations. The FNO trained on genuine OpenFOAM trajectories is used only to train the HydroGym PPO policy; the frozen policy is then queried on real CFD observations. Policy inference occurs in a CPU-only, read-only, network-disabled container; OpenFOAM remains in its independently pinned solver container. No host Python environment is changed.

The runner refuses to start unless the 20-epoch independent FNO readiness report is `CANDIDATE_SURROGATE_SCREEN_PASS`, the PPO pilot has completed all multi-start surrogate evaluations, the checkpoint epoch is 20, and the pinned HydroGym checkout/image and source t=80 CFD restart are present. It limits the real-CFD pilot to 2–32 intervals, `|omega|<=5`, and `|delta omega|<=0.5` per 0.1-time-unit interval. It verifies each 20-step OpenFOAM solve, maximum Courant number, continuity, finite observations and exact restart time. The policy's deterministic requested action and the rate-limited applied action are both recorded.

After the FNO gate and PPO pilot have **actually completed**, create a fresh isolated case and run a short pilot:

```bash
python3 cfd/tandem_cylinders/make_probe_feedback_case.py probe_feedback_ppo_pilot_01 --steps 3
python3 scripts/run_tandem_ppo_cfd_feedback.py probe_feedback_ppo_pilot_01 \
  --policy-run artifacts/hydrogym/tandem_ppo_pilot_20epoch_spark \
  --readiness artifacts/tandem_fno_expanded_spark_20epoch/control_readiness.json \
  --output artifacts/hydrogym/tandem_ppo_cfd_feedback_pilot_01
```

Both paths must be new: the scripts refuse to overwrite prior cases or outputs. The initial observation is read from the real monolithic t=80 baseline. For each subsequent interval, `scripts/infer_tandem_ppo_action.py` loads the frozen Stable-Baselines3 PPO policy, predicts from the 67-channel observation, and the host runner ramps the rear-cylinder `rotatingWallVelocity` boundary to the rate-limited action. The next observation comes from the new segmented OpenFOAM solve. The runner also reads the matched zero-action monolithic reference at the same time and records Cd, Cl, wake-probe differences, and an instantaneous objective using the same drag/lift/actuation/rate weights as the HydroGym surrogate. That comparison is **not** a performance claim over a 3-interval horizon.

This entry point is prepared and policy inference was smoke-tested using the earlier five-epoch software-smoke policy on a real t=80 observation. The actual 20-epoch PPO-to-CFD loop has **not** been run or validated yet. After a short successful integration run, a meaningful physical result still requires a longer frozen-policy CFD trajectory, matched zero and simple-action baselines, uncertainty/numerical sensitivity (including the ongoing medium-mesh grid-pair check), and explicit time-averaged drag/lift/actuation accounting.
