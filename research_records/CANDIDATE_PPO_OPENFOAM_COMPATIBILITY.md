# Candidate PPO to real-OpenFOAM compatibility

This CPU-only adapter connects the candidate-aware PPO evidence chain to the
existing canonical real-OpenFOAM feedback runner. It does not train a policy,
run CFD, authorize a failed surrogate, or change the controller.

`scripts/adapt_candidate_ppo_openfoam_readiness.py` requires the immutable
candidate approved preflight, candidate binding receipt, canonical PPO audit,
final SB3 policy and `vecnormalize.pkl`. It rejects mismatched candidate/FNO
identity, a non-identity VecNormalize contract, changed policy or Vec hashes,
frozen-test access claims, and an unapproved readiness status. It exports:

- `ppo_readiness.json`: the reviewed nested canonical readiness plus the
  candidate evidence hashes;
- `ppo_audit.json`: the canonical audit with only the final policy location
  translated from its container path to the byte-identical host path;
- `receipt.json`: hashes of both exported documents.

The VecNormalize check binds the artifact SHA to the candidate binding and to
the reviewed producer's exact `norm_obs=false, norm_reward=false` audit
contract. The adapter deliberately does not deserialize the pickle and must
not be described as an independent inspection of its internal flags.

The existing OpenFOAM runner must recompute its normal policy/FNO/data lineage
from the two exported documents and the exact final policy. A new execution
predeclaration must bind that resulting lineage and its own SHA. Copying a PASS
status by hand is not valid. The candidate's surrogate-development gates must
already pass; this adapter cannot override a scientific failure.

The observation and actuation contracts remain unchanged: 32 ordered wake
velocity probes, front Cd/Cl, rear Cd/Cl and applied rear-cylinder omega (69
channels); abs(omega) <= 0.75; delta omega <= 0.1 per 0.1 D/U control interval;
and a linear OpenFOAM boundary ramp between consecutive applied actions.

The 6.15 D/U surrogate reward window is initialized by the exact restart force
at t=0 and first becomes ready with 62 samples after action step 61. Real-CFD
policy inference is reward-free and memoryless, so this does not change its
actions. A future evidence-only enhancement may record the same rolling ledger
during CFD execution; it must not feed the policy or alter the physical gate.
