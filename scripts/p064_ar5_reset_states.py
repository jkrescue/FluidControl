"""G training-only frozen-flow states; no model/optimizer/data construction."""

HORIZON = 100
RESET_EVERY = 5
RESET_INDICES = tuple(range(0, HORIZON, RESET_EVERY))


def training_states(flow_model, state, target_state, mask, omega, predict_fn, original):
    """Keep all 100 original calls, including discarded block-end updates.

    Index s is the current state for force target s+1. At block start s,
    use observed q_s, never q_(s+1). The original loss receives unchanged
    targets/actions and mixed H1/AR batch structure.
    """
    import torch

    if omega.shape[1] != HORIZON + 1 or target_state.shape[1] != HORIZON:
        raise ValueError("exact H100 states / 101 action endpoints required")
    if any(p.requires_grad for p in flow_model.parameters()):
        raise ValueError("flow model must be frozen")
    if any(x.requires_grad for x in (state, target_state, mask, omega)):
        raise ValueError("observed inputs must not require gradients")
    if any(not torch.isfinite(x).all() for x in (state, target_state, mask, omega)):
        raise ValueError("nonfinite observed inputs")
    truth = original.true_state_inputs(state, target_state)
    states = []
    calls = 0
    with torch.no_grad():
        for step in range(HORIZON):
            if step % RESET_EVERY == 0:
                current = truth[:, step]
            states.append(current)
            inputs = original.make_inputs(current, mask, omega[:, step], omega[:, step + 1])
            delta, _ = predict_fn(flow_model, inputs, mask)
            calls += 1
            current = (current + delta) * mask
    result = torch.stack(states, dim=1)
    if calls != HORIZON or result.shape != truth.shape or not torch.isfinite(result).all():
        raise ValueError("frozen-flow rollout contract differs")
    return result, dict(
        profile="training_ar_reset_every_5_true_current_states",
        reset_indices=list(RESET_INDICES),
        flow_forward_calls=calls,
        supervised_ar_points=HORIZON,
        discarded_block_end_updates=20,
        future_state_inputs=False,
        optimizer_steps=0,
    )
