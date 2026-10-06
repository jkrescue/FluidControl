# Fixed symmetry-canonical PPO adapter — prospective single-factor plan

Status: source/CPU feasibility only; no training, policy loading, FNO inference, or CFD is authorized.

## Prospective scientific protocol (still not authorized)

Use the same failed seed `20261007` as E079/E080—not a seed scan—and the same P064-B
candidate, 32768/H5/24-reset/69-observation/reward/optimizer/resource protocol. The
hypothesis is that matching the canonical coordinate interface during training and
deployment reduces seed sensitivity and improves the real-CFD primary window relative to
E080 (drag reduction `-0.6174007162%`, rear-Cl RMS ratio `1.00622`, bias ratio `1.6946`).
Only the final 32768-step policy is retained; neither training reward nor an intermediate
checkpoint selects the policy.

After independent terminal audit, a separately approved paired b00 CFD run would retain
800 cycles, `148 -> 228`, the unchanged zero branch, all six reporting windows, and the
original primary criteria: drag reduction at least 2%, rear-Cl RMS ratio at most 1.05,
and absolute mean rear-Cl / paired-zero RMS at most 0.10. Early-window failures remain
separately reported. The primary outcome is real CFD, not training reward. Existing FNO
precision failures are unchanged and this experiment is not formal surrogate admission.

Training resources remain 12 GiB/no-swap, MemAvailable 50 GiB startup / 22 GiB runtime,
and 1800 seconds. One execution attempt is proposed; a finite runtime fault may be
diagnosed, but no alternate seed, hyperparameter, model, or reward is selected. Any CFD
execution requires a new approval after the final training policy is independently audited.
Before execution, reviewed sources must be frozen under a repository artifact; no running
job may depend on `/tmp`.

## Single intervention

Retain the frozen P064-B FNO, official SB3 PPO `MlpPolicy`, 32768 steps, H5, 24 resets,
69 observations, reward, optimizer, seed policy, action bounds, and the one existing
physical rate/amplitude filter. Add one policy-independent Gym wrapper in both training
and deployment:

1. Apply the audited physical69 reflection `R` (probe order reversal, `u/Cd` even,
   `v/Cl/omega` odd).
2. For `d=o-R(o)`, choose the first index attaining `max(abs(d))`. Its sign defines
   orientation `s in {-1,+1}`. Present `C(o)=o` for `s=+1`, else `R(o)`.
3. The unchanged policy samples `a_c ~ pi(.|C(o))`. Map only at the environment boundary:
   `a_physical=s*a_c`, then let the existing physical filter run exactly once.
4. Deployment performs the same single policy evaluation and same mapping; remove the
   current policy-dependent two-evaluation projection from this prospective profile only.

For every non-fixed reflection orbit, `C(R(o))=C(o)` and `s(R(o))=-s(o)`. The action
map is invertible and preserves the existing `[-.75,.75]` support.

## Why PPO likelihoods remain consistent

Pinned Gymnasium 1.2.3 explicitly permits `ActionWrapper` to modify an action before
`env.step` (`gymnasium/core.py`, `ActionWrapper.step`). Pinned SB3 2.7.1 samples
`actions, values, log_probs`, sends a clipped copy to `env.step`, then stores the original
sampled `actions` with those `log_probs` in the rollout buffer
(`stable_baselines3/common/on_policy_algorithm.py`, rollout collection). Thus PPO learns
the canonical action distribution it sampled; the wrapper is a deterministic coordinate
map, not a second policy evaluation. A combined stateful wrapper is required because the
current observation's orientation—not the next observation's—maps the current action.

## Predeclared limitations

- At an exact reflection-fixed observation, no orientation distinguishes the orbit.
  The wrapper records `reflection_fixed=true` and uses `s=+1`; it cannot guarantee a
  nonzero stochastic action is odd there. A policy-independent map to zero could define a
  valid many-to-one latent-action environment (as clipping does), but would be a distinct,
  non-invertible intervention. It is not silently added to this single-factor proposal.
- The first-maximum rule makes ties deterministic, but orientation is discontinuous near
  the fixed set. Record pivot and `max|o-R(o)|`; do not add a tuned epsilon or hysteresis.
- The canonical physical objective is reflection invariant: total drag is even; rear-Cl
  fluctuation and squared/absolute mean-bias terms are sign invariant; action and rate
  penalties are squared. This must be checked against the unchanged reward implementation.
- Frozen P064-B is not proven reflection equivariant. The wrapper cannot remove asymmetric
  surrogate error or make the hidden 62-force history fully observed. Consequently a
  later physical difference could not be attributed solely to policy symmetry.
- This is not a formal surrogate admission and does not replace the already successful
  policies or results.

## CPU contract evidence required before a separately approved scientific run

The included tests use the pinned Gym interface and verify: reflection involution; orbit
canonical equality; opposite orientation and odd physical action on non-fixed orbits;
action invertibility/bounds; deterministic ties; explicit fixed-state limitation; and
current-state action mapping before next-state orientation update; actual DummyVecEnv
autoreset/truncated terminal-observation handling; and actual SB3 rollout-buffer latent
action/log-probability consistency before any optimizer update. Step audit info separates
the applied current orientation, canonical action received after SB3 clipping, physical
requested action, and next orientation. A later approved source
freeze must additionally bind the exact training and deployment adapter bytes and reject
the old two-policy projection in the new profile.
