# P064-B PPO seed=20261007 replication preparation

Status: preparation only; neither PPO training nor CFD execution is authorized.

## Single factor

Repeat the reviewed P064-B PPO training with seed `20261007`.  Relative to the
completed seed `20261006` run, the B dual-FNO manifest, frozen FNO weights,
fresh PPO initialization, 32768 steps, H5 episodes, 24-reset panel, 69-value
observation, reward, optimizer settings, runtime imports and resource contract
remain fixed.  The seed is passed through the existing single `PROTOCOL.seed`
to Python, NumPy, Torch CPU/CUDA and Stable-Baselines3.

The final 32768-step policy is the only policy eligible for the prospective
b00 paired 800-cycle CFD run.  There is no intermediate-checkpoint selection,
surrogate-reward selection, seed scan, or rerun of the prior trained and
initial policies.  The old successful policy and its artifacts remain intact.

## Minimal source change

The training runner differs by exactly one source line:
`PROTOCOL["seed"]` (`20261006` to `20261007`).  The reviewed supervisor is
seed-agnostic and can be reused byte-for-byte.  The reviewed projected-CFD
driver has no hard-coded seed; it already requires the terminal training
result protocol to equal the separately approved training protocol.

The two pending JSON files are deliberately unauthorized.  The CFD pending
file contains null future hashes and cannot become executable until the new
training result, final policy, VecNormalize file, actual unit/invocation and
independent terminal review are bound under a separate approval.

## Evidence and interpretation

The prior PPO training took about 586.9 s and its paired b00 CFD run about
1092.7 s, or about 28.0 minutes of observed sequential compute.  A 45-minute
end-to-end target is plausible if resource admission, independent review and
unit transitions are prompt, but it is not guaranteed; the existing outer
budgets remain larger and unchanged.

The replication will report the new physical b00 metrics beside the prior
trained-policy and exact-initial-policy evidence without rerunning either.
Two seeds can expose one instance of training variability, but cannot establish
a seed distribution or general robustness.
