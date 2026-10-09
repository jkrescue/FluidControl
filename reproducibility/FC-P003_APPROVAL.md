# FC-P003 — distribute paired supervision across each training epoch

Lead approval: 2026-10-04. This continues the fixed Re100 tandem-cylinder,
rear-rotation, PhysicsNeMo → HydroGym/SB3 → real OpenFOAM closed-loop objective.
It does not replace the CFD-only PPO baseline or relax any acceptance criterion.

## Evidence and hypothesis

FC-P001 completed: both lambda0/lambda10 pass endpoint diagnostics but fail
development admission. Only the two zero-action force windows pass; all four
rotating branches fail. Their mean rear Cl-prime RMS errors are 0.182031 and
0.178966 respectively, versus approximately 0.0294 allowed per branch.
FC-P002 locates the failure in dynamic action responses with phase/sign
interaction, not a general zero-action rollout collapse.

The current trainer applies its 16 paired updates in the first 16 regular
batches, then continues regular-only updates. Hypothesis: distributing these
same 16 updates across an epoch preserves control-relevant supervision better.
End-of-epoch logs do NOT establish forgetting; this is a hypothesis to test.
Static paired trajectories may still be insufficient for dynamic control.

## Approved single change

Add an explicit scheduling option to the project trainer; keep frontloaded as
the compatibility default. Treatment: interleave exactly 16 paired batches over
N regular batches at floor(i*(N-1)/15), i=0..15, including first and last batch.
Require N>=16 and uniqueness. Pair order, count, lambda=10, regular minibatches,
data, normalization, model, action definition, optimization and evaluation stay
unchanged. Scheduling is project orchestration, not an invented NVIDIA API.

Parent: Main-e2 model SHA
8466bd47f2de188f5e741197832ec3bee1223f72f54d8956e584c586a2774240.
Do not continue from a paired candidate. Use seed20261003, LR1e-5, 2 epochs,
batch1, paired batch1, H100, zero teacher forcing and the existing official
PhysicsNeMo FNO2.2.2 image. Compare to the completed frontloaded lambda10 run;
do not attribute lambda0/lambda10 differences to schedule alone.

## Execution and ownership

Surrogate owns trainer option, new config, immutable launch and Main execution.
Evaluation owns independent schedule/protocol checks and actual task monitoring.
Physics/Data owns failure-map evidence, independent checks and Worker support.
Lead approves scientific decisions and next intervention.

Before full training: unit-test schedule counts/positions and unchanged default;
verify identical parent/data/normalization and regular-sampling contract; run a
bounded finite-gradient/resource probe in the isolated container. Record source
commit, resolved config, image, parent and manifests in an immutable launch
receipt. Abort if the intended one-factor comparison cannot be maintained.
Do not overwrite existing models or results.

Budget: one treatment training run of two epochs, followed by the unchanged
FC-P001 validation10/dynamic6/force-window posteval suite. Guard physical unified
MemAvailable >=20 GiB continuously; use measured existing training allocator
limits, not independent-VRAM assumptions. No duplicate training. Diagnose
actual failures; resume verified idempotent stages without repeating inference.

## Decision and next step

Use exactly the FC-P001 protocol and thresholds, reporting fields, forces,
action differences and every force-window branch. Never admit PPO from an
endpoint PASS alone. If admission passes, bind a new compatible PPO policy to
this checkpoint and proceed to paired real-CFD feedback validation. If it
fails, record the failure, use the action/phase/horizon map to choose the next
single-factor intervention, and continue toward the original closed-loop goal.
No GPU utilization target substitutes for scientific progress or acceptance.

The original final gate remains paired80D CFD, final60D statistics: total drag
reduction>=2%, rear Cl-prime RMS ratio<=1.05, normalized mean-lift bias<=0.10,
abs(omega)<=0.75 and per0.1D/U action change<=0.1.
