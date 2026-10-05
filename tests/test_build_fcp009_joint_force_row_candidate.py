import numpy as np
import pytest

from build_fcp009_joint_force_row_candidate import (
    DOMAIN_MIX,
    KIND,
    candidate_metadata,
    fit_from_cache,
    native_h1_sanity,
    metadata_values,
)


def test_metadata_is_exact_train_only_epoch_zero_contract():
    assert candidate_metadata() == {
        "status": KIND,
        "candidate_checkpoint_epoch": 0,
        "parent_checkpoint_epoch": 2,
        "calibration_generation": 1,
        "parent_model_sha256": candidate_metadata()["parent_model_sha256"],
        "alpha": 0.0,
        "domain_mix": DOMAIN_MIX,
        "calibration_fit_performed": True,
        "optimizer_training_performed": False,
        "validation_accessed": False,
        "frozen_test_accessed": False,
        "ppo_executed": False,
    }


def test_metadata_values_supports_official_list_of_dicts_and_mapping():
    assert metadata_values([{"case": "a"}, {"case": "b"}], "case") == ["a", "b"]
    assert metadata_values({"step": [0, 1, 2, 3]}, "step") == [0, 1, 2, 3]


def test_fit_from_cache_is_fixed_joint_alpha_zero(monkeypatch):
    rng = np.random.default_rng(4)
    n = 24
    ar = rng.normal(size=(n, 128)).astype(np.float32)
    h1 = rng.normal(size=(n, 128)).astype(np.float32)
    y = rng.normal(size=(n, 4)).astype(np.float32)
    cache = {
        "features": ar,
        "matched_weight_h1_features": h1,
        "targets_normalized": y,
        "phases": np.asarray((["b00", "b02", "b04", "b06"] * 6)),
        "relative_steps": np.tile(np.arange(1, 7), 4),
    }
    monkeypatch.setattr(
        "build_fcp009_joint_force_row_candidate.validate_cache_schema",
        lambda value: (value["features"], value["matched_weight_h1_features"], value["targets_normalized"], value["phases"], value["relative_steps"]),
    )
    coefficients, fit = fit_from_cache(cache)
    assert coefficients.shape == (129, 4)
    assert fit["alpha"] == 0.0
    assert fit["domain_mix"] == DOMAIN_MIX
    assert fit["joint_design_rows"] == 48


def test_native_sanity_preserves_state_and_checks_pointwise_replay():
    torch = pytest.importorskip("torch")

    class Final(torch.nn.Module):
        def __init__(self, force_shift=0.0):
            super().__init__()
            self.linear = torch.nn.Linear(128, 7)
            with torch.no_grad():
                self.linear.weight.zero_()
                self.linear.bias.zero_()
                self.linear.bias[3:] = force_shift

    class Decoder(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.final_layer = Final(shift)

    class Model(torch.nn.Module):
        def __init__(self, shift):
            super().__init__()
            self.decoder_net = Decoder(shift)

        def forward(self, value):
            batch, _, height, width = value.shape
            hidden = torch.zeros(batch * height * width, 128)
            return self.decoder_net.final_layer.linear(hidden).reshape(batch, height, width, 7).permute(0, 3, 1, 2)

    parent, candidate = Model(0.0), Model(0.25)
    batch = {
        "state": torch.zeros(4, 3, 2, 2),
        "omega": torch.zeros(4, 2, 1),
        "mask": torch.ones(4, 1, 2, 2),
    }
    result = native_h1_sanity(
        parent,
        candidate,
        parent.decoder_net.final_layer.linear,
        candidate.decoder_net.final_layer.linear,
        batch,
        torch.device("cpu"),
    )
    assert result["parent_candidate_state_bitwise_equal"] is True
    assert result["candidate_pointwise_head_wiring_max_abs"] == 0.0
    assert np.asarray(result["candidate_force_normalized"]).shape == (4, 4)


def test_native_sanity_rejects_state_change():
    torch = pytest.importorskip("torch")
    class Model(torch.nn.Module):
        def __init__(self, state):
            super().__init__()
            self.decoder_net = torch.nn.Module()
            self.decoder_net.final_layer = torch.nn.Module()
            self.decoder_net.final_layer.linear = torch.nn.Linear(128, 7)
            with torch.no_grad():
                self.decoder_net.final_layer.linear.weight.zero_()
                self.decoder_net.final_layer.linear.bias.zero_()
                self.decoder_net.final_layer.linear.bias[:3] = state
        def forward(self, value):
            batch, _, height, width = value.shape
            hidden = torch.zeros(batch * height * width, 128)
            return self.decoder_net.final_layer.linear(hidden).reshape(batch, height, width, 7).permute(0, 3, 1, 2)
    parent, candidate = Model(0.0), Model(1.0)
    batch = {"state": torch.zeros(4, 3, 2, 2), "omega": torch.zeros(4, 2, 1), "mask": torch.ones(4, 1, 2, 2)}
    with pytest.raises(ValueError, match="state output"):
        native_h1_sanity(parent, candidate, parent.decoder_net.final_layer.linear, candidate.decoder_net.final_layer.linear, batch, torch.device("cpu"))
