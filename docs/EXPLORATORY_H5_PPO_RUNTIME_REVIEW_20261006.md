# Isolated runtime and actual CPU lifecycle evidence — 2026-10-06

Lead authorized only the three-package isolated installation and CPU import/
lifecycle verification. No real FNO, HDF, optimizer, CFD or CUDA computation ran.
The original Python3.12 environment has no pip module; that first installer
attempt exited without installing anything. The existing host pip then used
`--python .venv-curator-py312/bin/python --no-deps --target
.runtime/exploratory-h5-ppo-py312` to install precisely the approved three wheels.
Full installer output is retained in staged `pip-install.log`. The overlay is
locally git-excluded, not committed; the base official environment is unchanged.

Installed distributions: Farama-Notifications0.0.4, Gymnasium1.2.3, SB3 2.7.1,
matching `docker/physicsnemo/Dockerfile.hydrogym`. Their METADATA SHA256s are
respectively bb07fdce31f3c0ebb716a97cec663d2d7b1c3918a415ff5120299b354a5babe9,
e1435ba6c78bc0f32b7f70e6c6c163669fa95f0a702ab97ffb00e9e597767139,
6952912c5ba48aeb0e1b05d5044ddbdf291086f60cfa7e5e7b175aa16587ff32.
Metadata evaluation of all non-extra requirements found no version conflicts.
Existing torch2.14.1/numpy2.5.3/cloudpickle3.1.2/pandas3.0.6/matplotlib3.11.2
were reused, not upgraded. The pending spec retains full requires metadata,
package versions and source file hashes, not merely these version labels.

Actual HydroGym core FlowEnv explicitly returns terminated=False and uses
iter>max_steps for truncation; the new separate wrapper corrects exactH5 using
iter==5. It does not alter Full40CanonicalRewardAudit's100-step requirement.

First CPU fixture attempt, invocation c8b61167c55d4726951ab4ca94749afd, failed
before environment stepping because the synthetic PDE omitted abstract render.
The fixture alone was corrected. Actual rerun
`fluid-control-exploratory-h5-ppo-lifecycle-cpu-r2-20261006.service`, invocation
10d916f24e9f4715b0059cdd32019db7, exited0, two tests PASS in0.012s. Boundaries:
4GiB/noSwap/1CPU/120s, CUDA_VISIBLE_DEVICES empty. This is synthetic engineering
data, not physical model or policy-training evidence.

Test1 exercises real HydroGym FlowEnv and real SB3 DummyVecEnv: steps1..4 do not
end; step5 has TimeLimit.truncated=true, terminal observation value5, returned
reset observation0, and restored physical time148/history62. Test2 calls the
actual SB3 OnPolicyAlgorithm.collect_rollouts with a synthetic policy and actual
RolloutBuffer (no optimizer): the fifth reward is -0.1+0.99*V(terminal5)=4.85,
not -0.1+0.99*V(reset0). This directly verifies the timeout-bootstrap path.

Pure contract suite:17CPU tests PASS0.01s; Ruff after formatting PASS. Source
formatting after the actual run changes no semantics; the actual lifecycle
test remains separately runnable under the same approved CPU-only scope.
