# FC-P029 — control-aware flow rollout repair plan

Status: CPU preparation authorized by Lead. This authorizes isolated helper and
synthetic CPU tests only; it does not authorize GPU use, real-data/model access,
training, validation access, formal evaluation, PPO, or a change to any gate.
The actual numerical experiment remains conditional on the ongoing original
FC-P028 formal evaluation ending in scientific FAIL and on separate resource
and execution review/approval.

## Evidence and bounded hypothesis

The matched train-only H10 comparison
`artifacts/fcp028_h10_comparison_20261006/result.json` (SHA256
`6146ea9276570981cc72c949e3e6fac46737c43aa83c561a37eaa0e54a4ab793`)
preserves the original 44 trajectories, origin 51, recorded actions, and K1
aerodynamic model. P028 improves mean-case normalized AR field RMSE from
0.038909 to 0.033544 (-13.79%), while rear-Cl AR MAE worsens from 0.038693 to
0.039507 (+2.10%). All four force-channel MAEs worsen. The change in field
RMSE and rear-Cl MAE is nearly uncorrelated across cases (about -0.065).
Constant-action cases worsen most (rear-Cl 0.048983 to 0.052924); zero and
dynamic cases improve slightly. Base20 worsens, while train8/train16 improve.

This evidence does **not** locate the error at the cylinder wall or prove that
flow error is the sole cause. It supports one narrower test: ordinary global
field MSE may improve states that are not the states most useful to the already
trained aerodynamic readout. Test whether a force-aware gradient through the
unchanged frozen aerodynamic FNO can shape the existing flow FNO toward states
that retain both field accuracy and force utility.

## Exact parent and matched control

FC-P029 must restart from the same flow parent used by FC-P028, not continue
from the FC-P028 terminal candidate:

- P009 flow model, epoch 0: `dc41fc91d42476e052970b39fc66aed22fa72aa8b6f218a341a3abb095f42e31`;
- P009 flow state: `4998e534d4b82b17393c217357ed18220fb8e739166a88147483bb9cc5fb771e`.

For clarity, the FC-P028 terminal epoch-1 flow is a result, not the P029 parent:
model `18c540460e5ea70e6c5437134646d9f21934d168dcead0df1b982842f6817d13`,
state `cd466499b68a3ddbcecd46b406ad3e0a61f28c3f31525dd0f741a677146ddf36`.

The aerodynamic model remains the exact frozen P026 K1 epoch-1 pair used by
P028: model
`e2f67dbde0ab28ccd7aa46b34ee3904178c7549cd1539f4a3ae40e2bd17e67b5`,
state `ab2fe103bca0a8c84156e2c9fd7ded5336f2d4236436b593fc414e564d7e92d3`.

The comparison holds fixed the existing 44 training trajectories, 1368 unique
window identities, sampler order
`177ebd9523cde918eb0c1fb8026286dac7e95f3d228ae0e349757a2a9a288f9f`,
eight-window gradient accumulation, 171 optimizer updates, seed 20261003,
AdamW learning rate `1e-5`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay
`1e-4`, and clip norm 1. There is one terminal checkpoint and no intermediate
selection, scheduler, restart, extension, or validation-driven decision.

## Single intervention and time alignment

Only the flow FNO is optimized. The official PhysicsNeMo FNO architecture,
normalization, residual-state update, masks, stored-action convention, dataset,
and K1 aerodynamic FNO remain unchanged. The loss is project-owned orchestration,
not a PhysicsNeMo API or an official PhysicsNeMo training recipe.

For a window starting at physical index `s`, use stored normalized states and
actions exactly as the deployed K1 path does:

1. Start with true normalized `q_s`.
2. For rollout step `j=0..9`, flow input is
   `[qhat_{s+j}, mask, omega_{s+j}, omega_{s+j+1}]`.
3. Apply the existing residual/mask update to obtain `qhat_{s+j+1}` and compare
   it with true `q_{s+j+1}` for the field term.
4. Preserve the existing P026/P027 aerodynamic timing: its K1 input at step `j`
   contains current `qhat_{s+j}`, the mask, stored `omega_{s+j}`, and stored
   target action `omega_{s+j+1}`; compare its four normalized force outputs with
   true force `F_{s+j+1}`. Thus the first force term is an H1 term from true
   `q_s` and has exactly zero gradient to the flow model. Only the later nine
   force terms transmit gradients through preceding flow predictions. The final
   `qhat_{s+10}` is supervised by the field term but is not consumed by a force
   term. Do not claim that all ten force endpoints train the flow.
5. No true future state or force enters a model input. All ten field and force
   targets are supervision only.

Let `L_field` be the existing equal-ten-step masked normalized u/v/p MSE and
`L_force` the equal-ten-step, equal-four-channel normalized force MSE. Before
training, run the unchanged parent over the same 1368 training windows once and
fix scalar denominators `S_field = mean(L_field_parent)` and
`S_force = mean(L_force_parent)`. No validation or frozen data contributes.
Require both denominators to be finite and strictly positive; zero, nonfinite,
or inconsistent recomputation fails closed. The sole new objective is

`L = 0.5 * L_field / S_field + 0.5 * L_force / S_force`.

The 50/50 weight is preregistered and is not swept. Raw losses, normalized
contributions, and both fixed denominators must be recorded at every update.
Equal normalized loss values do not imply equal gradient norms or directions.
Changing the joint scale can also interact with gradient clipping and AdamW's
epsilon. Therefore any result is evidence for this complete fixed objective,
not a clean attribution to force-gradient direction alone.

## Gradient and optimizer safeguards

- Set every aerodynamic parameter `requires_grad=False` and keep it in eval
  mode, but **do not** wrap its forward in `torch.no_grad()` and do not detach
  its input or output. Input gradients must reach the flow trajectory.
- Construct AdamW exclusively from allowed flow parameters. Assert that no
  aerodynamic parameter appears in optimizer groups, has a gradient, changes
  version/hash, or acquires optimizer state.
- The flow model's unused force-output rows are not part of deployed flow-state
  prediction. Preserve the same P028 row/bias restoration and optimizer-moment
  masking so AdamW weight decay cannot silently alter them. Verify their bytes
  after every update and after official save/reload.
- Backpropagate through all ten flow steps without detach or truncated BPTT.
  Check all trainable gradients, parameters, and Adam moments for finiteness
  before and after each update. A failure is terminal; it does not authorize a
  changed coefficient, learning rate, batch, or retry.

## Minimal independently testable code interface

A small project helper may expose a pure objective such as:

```text
control_aware_flow_rollout_objective(
    flow_model, frozen_aerodynamic_model,
    initial_state, target_states, mask,
    stored_actions, target_forces,
    field_scale, force_scale,
) -> (scalar_loss, diagnostics)
```

It must reuse the existing P028 input/residual/mask implementation and P026 K1
history/action packing rather than reproduce either convention. Synthetic CPU
tests must cover the ten target/action indices, full ten-step flow gradient,
nonzero input gradient through a frozen aerodynamic model, no aerodynamic
parameter gradient/update, exact unused-row preservation, mask behavior,
eight-window accumulation, finite/positive denominator rejection, and equality
with the P028 field term when the force contribution is inspected separately.
They must explicitly verify zero flow gradient from the first force endpoint,
nonzero flow gradient from later endpoints, and field-only supervision of the
terminal `qhat_{s+10}`.
Official save/reload must prove exact parent identities and allowed tensor scope.

Because the frozen aerodynamic forward now remains in the backward graph, P028
resource evidence is insufficient. Before any training approval, run one
separately approved no-update actual-size H10 forward/backward capacity check in
the pinned official image. It must construct no optimizer, save no model, use no
validation/frozen data, preserve all tensors, and report peak CUDA allocation,
host memory, elapsed time, finite input/flow gradients, and guard headroom.

## Decision and non-selection rule

The terminal candidate, if separately approved and produced, first reruns the
exact P027/P028 44-origin H10 comparison and then the original unchanged full
formal suite. Report every family, physical phase, constant/zero/dynamic action
group, four forces, and u/v/p field metrics. Validation cannot select the 50/50
weight, checkpoint, stopping point, or a retry.

Support requires meaningful force improvement, particularly the base constant
cases that regressed under P028, without losing the field repair, followed by
the pre-existing formal gates. If the force-aware flow does not improve that
failure map, materially damages fields, or fails the original formal gates,
reject it and close this bounded hypothesis. No outcome relaxes a threshold or
automatically authorizes PPO or real-CFD control. A gradient through the frozen
aerodynamic surrogate is not guaranteed to point toward a CFD-beneficial state;
the flow may exploit surrogate sensitivities. Even passing the existing formal
suite covers only its measured distribution. A separately reviewed compatible
PPO path and paired real-CFD closed-loop verification remain mandatory and
cannot be replaced by this loss or by formal surrogate acceptance.
