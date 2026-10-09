# P064 A/B same-protocol development evaluation preparation

Preparation only. The generator does not run inference or authorize it. A and B
use the e86ef8 evaluator's unchanged numerics, unchanged FC-E066 controlled b01/b03 panel
(16 starts, 80 endpoints), original normalization, core and official Reader.
Only the explicit candidate and production loader/source closure change from K1.
The post-load highest/no-TF32 protocol and all resource limits remain unchanged.

Before emitting a new exclusive PENDING file, the generator requires the actual
approved training unit/invocation to be successfully terminal, all 32 updates and
256 windows, the frozen flow tensor, fresh official reload, and matching arm.
The production 83ac4e loader validates manifest/protocol/checkpoint byte identities
without constructing a model; engineering fixtures remain disallowed. This is
not an independent model reload or scientific acceptance.

Bindings include the exact 434-member training source inventory, existing 192
runtime files, actual candidate manifest/result/approval, and fixed dev conversion.
PYTHONPATH is training-source/scripts then training-source/src. The existing
immutable e86ef8 worker remains unchanged. A separate 0d6f1ac worker changes only
the selector import to an explicit hash-bound module load: the 434-source training
tree does not contain the old K1 selector. Its original ec225 file is reused from
the previous immutable source. Exact byte reversal reproduces e86ef8. No pending
candidate SHA is invented. The first generator attempt stopped before writing
PENDING on this missing import source; no inference was attempted.

After A R2 actually succeeds, the proposed command is:

```sh
PYTHONDONTWRITEBYTECODE=1 /home/USER/env_isaaclab/bin/python /tmp/p064-dev-evaluation-review/prepare_p064_candidate_evaluation.py \
  --arm A \
  --training-unit fluid-control-fcp064-aero-arm-a-r2-20261006.service \
  --training-invocation 50de1d8b43ce42ac923752fad76ca4d9 \
  --training-approval /workspace/fluid_control/docs/FC_P064_ARM_A_TRAINING_R2_APPROVAL_20261006.json \
  --training-approval-sha256 773daa7329a46930b532586d884502d0c4170ea2c3a689ba73b2fb3d5da8b568 \
  --worker /workspace/fluid_control/artifacts/p064_candidate_development_source_20261006_immutable/evaluate_p064_candidate_development_h1_h5.py \
  --pending /workspace/fluid_control/artifacts/p064_arm_a_development_preparation_20261006/PENDING_SPEC.json
```

B uses the same function with its future actual approval/unit/invocation, never
the planned handle as evidence. Separate Lead execution authorization is required
after the final persisted pending configuration is reviewed. Tests are synthetic
contract tests only; they do not establish candidate correctness or GPU success.
