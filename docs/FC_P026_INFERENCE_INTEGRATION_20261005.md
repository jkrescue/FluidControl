# P026 inference integration contract

Lead decision: prepare a separate CPU-tested history inference module; do not
change or overwrite the accepted legacy CFD-only or P018 code path. This is
project glue around two official FNO instances, not a new model architecture.
No new checkpoint or control deployment is authorized by this document.

## Verified current incompatibility

`src/fluid_control/dual_fno.py` currently builds both models with the same
six-input-channel configuration and sends the same tensor to both. K4 cannot
be loaded into this path without changing its explicit model/input contract.
Independent read-only review identified corresponding six-channel construction
in `scripts/evaluate_tandem_fno.py`, `scripts/diagnose_fno_force_window.py`, and
`src/fluid_control/tandem_hydrogym.py`. These callers require integration before
full training and formal acceptance, not metric changes.

## Explicit interface

The new adapter takes `flow_inputs6` and `aerodynamic_inputs6_or18`. K4 must reject
a missing second input, rather than silently use the current-frame input.
Combine raw outputs exactly as before: first three channels from the frozen
flow model, last four from the aerodynamic model. Residual state updates, mask
application and force pooling remain outside and unchanged.

History is explicit caller-owned data, not mutable hidden state in a network.
Reuse reviewed `p026_state_history` packing/shift functions. Validate that the
current state, mask and current/selected-next actions in both model inputs match.
Reset/copy/restore operations must preserve independent buffers without aliases.

| Caller | Initial state history | Subsequent state history |
|---|---|---|
| Train or rolling validation segment | Actual same-trajectory past; only negative indices padded with trajectory frame0 | Frozen-flow predictions for AR; observed past for H1 |
| Force-window start0 | Explicit generated frame0 padding | Frozen-flow predictions |
| HydroGym initial reset | Explicit generated frame0 padding | Predicted state and actual applied/rate-limited command |
| Mid-episode restore | Saved full history buffer | Continue from restored buffer; never recreate initial padding |

The existing62-point force reward history is not flow-field history and must not
be substituted for it. No future measured states or forces enter the model.

## Later profile and evaluation requirements

Use new P026 K1/K4 profiles with separate flow/aerodynamic architectures, parent
identities, history length/layout, padding policy and action semantics. Missing
history fields must fail for P026; legacy P018 profiles remain unchanged.
Formal validation and force-window callers, immutable evaluation source maps,
receipts and later PPO environment creation must bind the explicit new profile.
Keep numerical metrics, thresholds, datasets, starts and horizon definitions.
Do not silently broaden this change to unrelated MPC/VTK/benchmark utilities.

CPU tests must cover legacy K1 equivalence, K4 legacy-call rejection, channel and
current-state consistency, causal reset/shift, independent environments and
mid-episode copy/restore. Official save/reload and actual formal-caller tests
remain separate required integration evidence before full training/deployment.
