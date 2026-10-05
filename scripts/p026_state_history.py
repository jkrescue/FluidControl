"""Project K1/K4 history adapter; original official-reader dataset is unchanged.

Stored prescribed action samples are retained in BOTH arms. They are not
certified exact nominal-time commands. This module does not use force inputs,
read future states for prediction, alter targets, or assemble rollout targets
into autoregressive history.
"""

import torch

LIFT = "spec_encoder.lift_network.0.conv.weight"
ACTION_SEMANTICS = "stored_prescribed_action_samples_not_exact_nominal_time_commands"


def history_indices(start, k):
    if k not in (1, 4) or not isinstance(start, int) or start < 0:
        raise ValueError("nonnegative start and K1/K4 required")
    raw = list(range(start - k + 1, start + 1))
    return [max(0, i) for i in raw], [i < 0 for i in raw]


class HistoryWindowAdapter:
    """Composition adapter, not a replacement official DatasetBase.

    `base` is one existing TandemRolloutDataset (not a global MultiDataset).
    Result is (original_sample, original_metadata, additional_history).
    Existing dataset indexing, normalization and reader lifetime remain owned
    by base; callers compose these adapters without changing source ordering.
    """

    def __init__(self, base, k):
        if k not in (1, 4):
            raise ValueError("K1/K4 only")
        self.base = base
        self.k = k

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample, metadata = self.base[index]
        file_index, start = self.base.index[index]
        if metadata["step"] != start or metadata["split"] != self.base.split:
            raise ValueError("base index/metadata mismatch")
        indices, padded = history_indices(start, self.k)
        mask = sample["mask"]
        states = {start: sample["state"]}
        actions = {start: sample["omega"][0]}
        reader = self.base._reader(file_index)
        for offset in sorted(set(indices) - {start}):
            # This is the existing official HDF5Reader; no alternate HDF loader.
            frame, _ = reader[offset]
            if not torch.equal(frame["mask"].float(), mask):
                raise ValueError("history mask differs in fixed geometry")
            states[offset] = (
                (frame["state"].float() - self.base.state_mean) / self.base.state_std
            ) * mask
            actions[offset] = frame["omega"].float().reshape(1) / self.base.action_scale
        state_history = torch.stack([states[i] for i in indices])
        action_history = torch.stack([actions[i].reshape(1) for i in indices])
        _validate_history(state_history, mask, action_history)
        history = dict(
            states=state_history,
            actions=action_history,
            metadata=dict(
                k=self.k,
                frame_indices=indices,
                padding_mask=padded,
                padded_frames=sum(padded),
                full_observed_history=not any(padded),
                padding="repeat_first_available_frame_and_action",
                action_semantics=ACTION_SEMANTICS,
                source_file=str(self.base.paths[file_index]),
                case=metadata["case"],
                start=start,
                split=metadata["split"],
            ),
        )
        return sample, metadata, history


def _validate_history(states, mask, actions):
    if states.ndim != 4 or states.shape[0] not in (1, 4) or states.shape[1] != 3:
        raise ValueError("Kx3xHxW state history required")
    k = states.shape[0]
    if mask.shape != (1, *states.shape[-2:]) or actions.shape not in ((k,), (k, 1)):
        raise ValueError("mask/action history shape differs")
    if any(not torch.isfinite(x).all() for x in (states, mask, actions)):
        raise FloatingPointError("nonfinite history")
    if any(
        x.device != states.device or x.dtype != states.dtype for x in (mask, actions)
    ):
        raise ValueError("history device/dtype differs")


def build_input(states, mask, actions, selected_next_action):
    """Chronological q history, mask, chronological applied actions + next command.

    No target state/force argument exists. Caller supplies already-normalized
    states/actions, and the next action selected before the simulated transition.
    FNO's two coordinate features are appended by official code, not here.
    """
    _validate_history(states, mask, actions)
    next_action = torch.as_tensor(
        selected_next_action, dtype=states.dtype, device=states.device
    )
    if next_action.numel() != 1 or not torch.isfinite(next_action).all():
        raise ValueError("one finite selected next action required")
    action_values = torch.cat((actions.reshape(-1), next_action.reshape(1)))
    planes = action_values[:, None, None].expand(-1, *states.shape[-2:])
    return torch.cat((states.reshape(-1, *states.shape[-2:]), mask, planes), dim=0)


def shift_history(states, actions, predicted_next_state, applied_next_action):
    """Roll forward with the model prediction and executed command, no detach."""
    if (
        states.ndim != 4
        or states.shape[:2] not in ((1, 3), (4, 3))
        or actions.shape not in ((states.shape[0],), (states.shape[0], 1))
    ):
        raise ValueError("history shape differs")
    if (
        predicted_next_state.shape != states.shape[1:]
        or predicted_next_state.device != states.device
        or predicted_next_state.dtype != states.dtype
    ):
        raise ValueError("predicted state shape/device/dtype differs")
    action = torch.as_tensor(
        applied_next_action, dtype=actions.dtype, device=actions.device
    )
    if (
        action.numel() != 1
        or not torch.isfinite(action).all()
        or not torch.isfinite(predicted_next_state).all()
    ):
        raise ValueError("finite predicted state/applied action required")
    return torch.cat((states[1:], predicted_next_state[None]), 0), torch.cat(
        (actions.reshape(-1)[1:], action.reshape(1)), 0
    ).reshape_as(actions)


def history_warmstart_state(old, new_template, k):
    """Map old6 physical +2 official coords; zero only added physical columns.

    K4: q_current9:12, mask12, omega_current/next16:18, coords18:20.
    No new official API or model class; caller uses official load_state_dict.
    """
    if k not in (1, 4) or old.keys() != new_template.keys():
        raise ValueError("K1/K4 identical state keys required")
    result = {}
    for name, value in old.items():
        target = new_template[name]
        if value.dtype != target.dtype:
            raise ValueError("state dtype differs")
        if name == LIFT:
            width = 8 if k == 1 else 20
            if (
                value.ndim != 4
                or value.shape[1] != 8
                or target.shape != (value.shape[0], width, *value.shape[2:])
            ):
                raise ValueError("official lifting shape differs")
            if k == 1:
                result[name] = value.detach().clone()
            else:
                mapped = torch.zeros_like(target)
                mapped[:, 9:12] = value[:, :3]
                mapped[:, 12:13] = value[:, 3:4]
                mapped[:, 16:18] = value[:, 4:6]
                mapped[:, 18:20] = value[:, 6:8]
                result[name] = mapped
        else:
            if value.shape != target.shape:
                raise ValueError("nonlifting state shape differs")
            result[name] = value.detach().clone()
    if LIFT not in result:
        raise ValueError("missing official lifting weight")
    return result
