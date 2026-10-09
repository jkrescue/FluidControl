# FC-P028 — targeted official-FNO flow rollout repair

Lead decision, 2026-10-06. CPU implementation and synthetic tests approved;
real-data/model/GPU execution is not yet approved. Preserve all existing work.

## Evidence and hypothesis

FC-E039/P027 shows K1 rear-Cl pooled H1/AR RMSE0.038826/0.054240;
train8 is0.034615/0.080523. K4 does not materially repair this. Source inspection
of `scripts/train_fcp026_history.py` confirms P026 already used equal H1/H100
AR force training. Merely adding predicted-state exposure is not a new change.

Test whether improving the existing flow FNO's multi-step field prediction,
with the force FNO fixed, reduces the additional force error induced by predicted
states. This is not a claim that flow is the sole cause: H1 error remains and
some trajectories improve under AR. Differences of RMSE are not additive causal
error components.

## One controlled intervention

- Parent: existing P009 flow from the pinned P026 K1 dual candidate, and its
  unchanged K1 aerodynamic model. Use the official PhysicsNeMo FNO/checkpoint
  APIs already verified in the pinned b40 container; no new architecture/API.
- Data: existing44 training trajectories and original1368 start indices/order,
  unchanged train-only normalization, curated targets and recorded actions.
  Use first10 transitions at each original start; do not resample all H10 starts.
- Optimize only flow parameters. No force-model gradient/update; do not add a
  force loss or alter the deployed channel combination. Preserve unused flow
  force-output channels explicitly where the existing representation permits;
  document any shared parameter effect without calling it a force-model update.
- Loss: equal-weight ten-step normalized u/v/p state MSE inside the existing
  fluid mask, with the existing residual-state update and action alignment.
  No true future fields enter model inputs. Backpropagate through all ten steps;
  no detach/truncated gradient. Non-reentrant activation checkpointing is allowed
  only with CPU gradient-equivalence tests, including spectral complex weights.
- Fixed AdamW lr1e-5, betas(.9,.999), eps1e-8, weight decay1e-4; clip norm1
  after averaging eight windows. One pass1368 windows/171 updates, original seed.
  No learning-rate sweep, intermediate candidate selection or auto extension.
- Output terminal candidate once, official save/reload, unchanged frozen force
  identity, exact training window order/count, finite loss/gradient evidence.
  Project orchestration/loss remains labelled custom project code.

## Evaluation and decision

Engineering tests first check ten-step gradients, mask and action/target
alignment, eight-window averaging, frozen force identity and checkpoint roundtrip.
One necessary actual-size resource check may be separately approved; do not
turn probes into a scientific parameter search.

After actual training, rerun the exact P027 origin51/H10 recorded-action
diagnostic and the original complete formal fields/forces/action-response/window
evaluation. Report each family/phase, not just aggregate field loss. H1 force
predictions must be unchanged because its model and true input are unchanged.
Compare fields at identical starts/horizons and both force paths to the parent.
P027 H10 alone cannot admit the candidate or replace H100 requirements.

If field errors improve without force AR improvement, that falsifies the
sufficiency of this ordinary field-training repair for the current bottleneck.
If H10 improves but original formal fails, reject admission and retain evidence.
Only original formal acceptance permits compatible HydroGym/SB3 PPO training,
followed by paired real OpenFOAM closed-loop verification of original drag,
lift fluctuation, mean lift and action limits. No threshold relaxation.

## Resources, ownership and recovery

GPU0 on main Spark, official isolated b40 container, maximum4h training,
at least20GiB MemFree/MemAvailable/CUDAfree throughout. Initial allocator/container
limits must be chosen from the actual-size check, not guessed from force-only
training. No heavy concurrent jobs; worker optional only after separate scope.
Preserve failure artifacts; diagnose before bounded retry; no blind restart.
Lead owns approval/integration; implementation agent owns isolated source/tests;
independent evaluator reviews correctness and actual terminal evidence.

This plan authorizes CPU implementation only. Exact immutable code/config/data
and budget must be attached before launching real training. The accepted
surrogate + compatible policy + real-CFD closed loop remains the full goal.
