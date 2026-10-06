# Exploratory H5 HydroGym / SB3 PPO — preparation, not execution approval

This is genuine surrogate-environment PPO training, not MPC substitution. It is
an explicitly exploratory branch; the existing canonical 100-step wrapper and
formal admission gates are untouched. No CFD execution is authorized here.

One fixed run: 4096 transitions, four fixed train-zero frame-0 starts (b00/b02/
b04/b06), seed 20261006, frozen official P026 K1 flow/aerodynamic pair. Each
environment uses actual HydroGym FlowEnv, existing canonical surrogate flow and
FNO stepper, 69 observations, dt=0.1, |omega|<=0.75, |delta omega|<=0.1, and the
unchanged canonical_joint_v1 cost with actual 62-sample causal force prehistory
and the existing phase-matched long-term train-zero baseline. No validation or
frozen trajectory is opened. The start field is read once per environment;
afterward all five predicted steps are autoregressive. Resets restore the same
real train state, physical start time, zero action, and measured force history.

SB3 2.7.1 PPO MlpPolicy: n_steps128, batch256, epochs4, lr3e-4, gamma.99,
GAE lambda.95, clip.2, entropy0, value coefficient.5, max grad norm.5. Four
environments give eight 512-transition rollouts, 32 epoch updates and 64 actual
optimizer steps. No sweep, reward-based checkpoint selection, intermediate
policy saves, or extra optimizer experiment. One final policy and identity
VecNormalize (norm_obs=false, norm_reward=false) are saved. Policy tensor hashes
before/after, actual optimizer step hooks, SB3 JSON loss/KL/value-loss records
(including final flush), every transition's canonical reward components/actions/
physical time and timeout information, and frozen-FNO tensor hashes are retained.

Exact step5 is a time-limit truncation, not a physical termination. Actual
DummyVecEnv supplies the pre-reset terminal observation; SB3 bootstraps using it.
A normalized-state bound violation is a true termination and does not bootstrap.
This does NOT validate the value estimate beyond H5. The 62-sample window contains
61..57 measured past samples and only1..5 predicted samples: mean-force feedback
is diluted while action/rate penalties are immediate. This may encourage low
action or exploitation of short-model bias. Loss/reward improvement is not a
claim of physical drag reduction. Four starts are not broad train coverage.

Official K1 loading preserves its original high/TF32 identity checks. Only after
validated loading, explicitly set highest/no-TF32, matching the existing small
engineering replay evidence (not a universal numerical-equivalence theorem).
All FNO parameters remain eval/frozen, with no gradients, optimizer membership,
or changed tensors. PPO parameters alone are optimized.

Resource contract: separate parent monitor samples physical MemAvailable every
0.5s, startup>=50GiB, runtime>=22GiB protecting the original20GiB reserve; CUDA
free is not an authority on Spark UMA. CUDA allocator fraction.06, cgroup
MemoryMax<=12GiB, MemorySwapMax=0, CPUQuota100%, inner1800s and external1950s/
TimeoutStopSec20. On failure/signal/deadline the owned worker process group is
stopped; no CFD containers exist in this task. Exclusive new output and exact
approval SHA are required. Runtime/install/source hashes and actual import paths
are bound, including SB3 timeout code and official FNO/checkpoint source.

Runtime overlay: new `.runtime/exploratory-h5-ppo-py312`, only three --no-deps
packages Farama-Notifications0.0.4, Gymnasium1.2.3, SB3 2.7.1; no base environment
mutation or Python3.11 package-path mixing. The proven Python3.12 official stack
and pinned HydroGym core remain in use. Source-only metadata preparation copies
four HDF hashes from the existing train manifest; actual validation/loading only
occurs under the separately approved protected worker.

Lead subsequently approved the first bounded4096 run through the separate
EXPLORATORY_H5_PPO_APPROVAL_20261006.json. That dated approval supersedes the
preparation-only status of this protocol; this document does not launch it.
No extra GPU smoke is a prerequisite. After training, direct PPO-to-CFD testing
requires a separate explicit exploratory policy binding and fixed physical test;
this result cannot masquerade as the canonical formal-ready PPO artifact.
