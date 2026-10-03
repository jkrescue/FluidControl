# FNO → HydroGym → PPO → OpenFOAM integration audit (2026-10-04)

## Decision

The code has a coherent **interface path**, but a newly trained dynamic FNO checkpoint is **not a drop-in replacement for an existing surrogate-trained PPO policy**. The current formal path intentionally binds the FNO checkpoint, configuration, data manifest, train-only normalization, validation gate, PPO run, and real-CFD replay by SHA. Replacing the checkpoint must therefore fail closed until the new model passes the same long-horizon and action-effect evidence gates and a new SB3 PPO policy is trained against that exact surrogate.

This does not affect the separately completed CFD-only PPO result. That policy is a frozen real-CFD baseline and must not be relabelled as FNO-trained or as evidence that the FNO contributes the measured physical benefit.

## Implemented contract and audit result

| Interface | Actual code contract | Audit result |
|---|---|---|
| FNO state/action | `TandemFNOStepper` supplies 6 input channels: normalized `(u,v,gauge_pressure)`, mask, normalized `omega_now`, and `omega_next`; it expects 7 outputs for three state increments plus four force channels. | Structurally compatible only when the new checkpoint uses the same resolved official PhysicsNeMo FNO architecture and four-force schema. Check the resolved config and model SHA, not the filename. |
| Action ramp | The surrogate advances one `0.1 D/U` interval from `omega_now` to `omega_next`. Real feedback applies the same endpoints as a linear OpenFOAM `rotatingWallVelocity` table, with `dt=0.005` (20 solver steps). Both enforce `|omega| <= 0.75` and `|delta omega| <= 0.1` per decision. | Contract is aligned. It is still necessary to demonstrate that the FNO predicts the *effect* of signed ramps, not merely that it accepts both endpoint inputs. |
| State normalization | Curated pressure is gauge pressure formed by subtracting the valid-domain spatial mean independently at each frame. State and force statistics are fitted on the explicit 20-case train manifest only. The adapter normalizes the initial state and autoregresses in that normalized space. | Correctly defined and SHA-bound. A new checkpoint must use the exact normalization used during its training; pressure-reference or statistics changes invalidate compatibility. |
| 69D observation | Policy order is exactly `32*(u,v), front Cd, front Cl, rear Cd, rear Cl, applied omega`. Surrogate probes are bilinearly sampled at `x=17`, `y=6..9`; real OpenFOAM reads the same 32 probe coordinates and exact-time forces. Pressure is internal to the FNO and is not a policy observation. | Ordering and dimensions agree. Keep the existing exact-time/probe-header checks. Do not add policy observation normalization silently. |
| Force feedback/reward | Four normalized force outputs are obtained by spatially averaging FNO output channels over the valid mask and then de-normalizing. Those predicted forces feed the causal canonical reward. Real replay instead reads both cylinders' OpenFOAM coefficients at the interval endpoint. | This decoder is a learned force head, not a force integration from predicted pressure/shear. Four-force/action-effect accuracy is therefore a hard scientific gate; field accuracy alone is insufficient. |
| Time and autoregression | Dataset frames, HydroGym solver steps, and control decisions are all `0.1 D/U`; OpenFOAM resolves each decision with 20 steps at `0.005`. The rollout trainer and runtime both update `q <- q + predicted_delta` and use consecutive action endpoints. | Algebra is aligned. H100 stability and force accuracy remain mandatory because PPO can exploit long-horizon surrogate errors even when H1/H50 is good. |
| Policy preprocessing | The historical canonical surrogate PPO uses SB3 `PPO` with `DummyVecEnv` and **no `VecNormalize`**. Its OpenFOAM inference loads the same PPO zip and passes raw 69D observations. The separate direct-CFD PPO does use a paired frozen `VecNormalize`; that policy/normalizer pair must remain inseparable. | Internally consistent, but the two tracks must not be mixed. If future surrogate PPO adds observation normalization, save, hash, freeze, and load its statistics during OpenFOAM replay. |
| Runtime/API | The pinned runtime is PhysicsNeMo `2.2.2`, SB3 `2.7.1`, Gymnasium `1.2.3`, HydroGym commit `4ab9854dea3d84e38a59c25e0f5835a00cf8225f`, and the recorded immutable container image ID. | Recheck all versions/image ID at every new run; a checkpoint alone is not sufficient provenance. |
| Lineage and replay | `validate_policy_and_gates` requires the PPO audit, readiness receipt, FNO gate, FNO checkpoint SHA, manifest SHA, normalization SHA, and final PPO checkpoint SHA to agree before the real OpenFOAM path can execute. | Correct fail-closed behavior. A new dynamic checkpoint necessarily invalidates the old receipts and old surrogate PPO lineage. |

## Concrete blockers before a new dynamic checkpoint can enter the loop

1. **Checkpoint promotion evidence:** exact model and resolved-config SHA, official PhysicsNeMo load check, 6-in/7-out schema, and immutable train-only normalization/manifest binding.
2. **Long-horizon dynamics:** finite output is not accuracy. Require the existing validation-only H1/H10/H50/H100 audit, especially H100 state and four-force error versus persistence, without reading frozen data for selection.
3. **Action-effect fidelity:** require the existing dynamic signed-ramp/low-action gate to show correct force response and ranking under the intended `0.75/0.1/0.1D/U` action contract. If this fails, no surrogate PPO training or closed-loop claim is allowed.
4. **Policy retraining:** train a new SB3 PPO against the exact accepted FNO SHA. Do not reuse either the old surrogate PPO or the validated CFD-only PPO as if environment dynamics were interchangeable.
5. **Frozen preprocessing and real replay:** bind the policy (and any future observation normalizer) by SHA, then run the predeclared paired OpenFOAM replay. Surrogate return is never evidence of physical benefit.

The smallest safe code changes are provenance checks, not objective changes: require a resolved-config SHA beside the model SHA; record the runtime package versions in the PPO audit; explicitly record whether observation normalization is absent or, if introduced later, its immutable SHA; and keep the existing action/observation/solver assertions. No reward, physical threshold, split, or b01 result should be changed.

## Terminology correction

“Official HydroGym/SB3 policy” is misleading. HydroGym supplies the `FlowEnv`/`PDEBase`/`TransientSolver` interface used here; the surrogate and OpenFOAM bridges are project-specific adapters. PPO is supplied by Stable-Baselines3, not by HydroGym.

Use: **“an SB3 PPO policy trained/evaluated through the pinned HydroGym `FlowEnv` API with a project-specific PhysicsNeMo FNO or OpenFOAM adapter.”** The official implementation claim applies separately to `physicsnemo.models.fno.FNO` and to the upstream HydroGym API, not to the complete project pipeline.

## Scope of claims

- The completed real-CFD PPO result demonstrates a CFD-only online simulation feedback loop with physical benefit on the audited starts. It does not establish an FNO-surrogate PPO benefit.
- A working API connection or finite FNO rollout does not establish action-effect fidelity, closed-loop validity, efficiency, or novelty.
- The b00 and b01 starts are separated by about `18 D/U`, roughly three nominal shedding periods. Calling b01 an untrained-time/held-out-start check is accurate; calling it an independent physical sample or broad phase-generalization proof is too strong.
- A useful post-hoc discriminator is a predeclared open-loop replay of the immutable b00 800-action endpoint sequence from the b01 restart, using the same linear ramps, `80 D/U` horizon, and final `60 D/U` statistics. Similar performance to b01 feedback would leave the feedback implementation valid but would not demonstrate feedback superiority. One such replay still would not prove generality across phases or disturbances.

## Code evidence

- `src/fluid_control/tandem_hydrogym.py`: FNO input/output construction, normalization, probe sampling, force decoder, and rate limiter.
- `src/fluid_control/full40_canonical_hydrogym.py`: 69D/full40 contract, canonical reward, `FlowEnv`, and surrogate-only labels.
- `src/fluid_control/tandem_datapipe.py` and `scripts/finalize_matched_start_full40.py`: train-only normalization and gauge-pressure definition.
- `scripts/train_tandem_fno_rollout.py`: autoregressive update and action-endpoint conditioning.
- `scripts/train_full40_hydrogym_ppo_canonical.py`: strict gates, SB3 PPO construction, and absence of `VecNormalize` on the surrogate track.
- `src/fluid_control/openfoam_observation.py` and `scripts/run_full40_canonical_ppo_openfoam_feedback.py`: exact 69D real observation, pinned real-CFD timing, ramp application, and lineage validation.
