import copy
import torch
import pytest
import verify_fcp026_checkpoint_cpu as p


@pytest.mark.parametrize("k,channels", [(1, 6), (4, 18)])
def test_fixture_explicit_profile_not_candidate(k, channels):
    meta = p.metadata("aerodynamic", k, "parent", "mapped")
    p.validate_fixture(meta, "aerodynamic", k, channels)
    assert meta["candidate"] is False and meta["training_performed"] is False
    for key, value in [
        ("status", "CANDIDATE"),
        ("checkpoint_epoch", 0),
        ("model_in_channels", 10),
        ("history_adapter_sha256", "wrong"),
        ("candidate", True),
    ]:
        broken = copy.deepcopy(meta)
        broken[key] = value
        with pytest.raises(ValueError):
            p.validate_fixture(broken, "aerodynamic", k, channels)


def test_legacy_rejected_before_k4_load():
    meta = p.metadata("aerodynamic", 4, "parent", "mapped")
    with pytest.raises(ValueError):
        p.validate_fixture(meta, "aerodynamic", 1, 6)
    with pytest.raises(ValueError):
        p.validate_fixture(meta, "aerodynamic", 4, 6)


def test_exact_all_tensor_comparison_real_complex():
    state = dict(
        real=torch.arange(4, dtype=torch.float32),
        complex=torch.tensor([1 + 2j], dtype=torch.complex64),
    )
    p.compare_states(copy.deepcopy(state), state)
    changed = copy.deepcopy(state)
    changed["complex"][0] += 1j
    with pytest.raises(ValueError):
        p.compare_states(changed, state)
    changed = copy.deepcopy(state)
    changed["real"] = changed["real"].double()
    with pytest.raises(ValueError):
        p.compare_states(changed, state)


def test_official_source_and_engineering_only_contract():
    import inspect

    assert (
        p.CHECKPOINT_SHA
        == "0d26a62251c3724a1ceebfa1daa1bb5ba9dcdc5e73a0ded3ccb955355af2f78e"
    )
    code = inspect.getsource(p.execute)
    assert "save_checkpoint(directory, models=model, epoch=1, metadata=info)" in code
    assert "torch.optim" not in code and ".backward(" not in code
    assert "output_created_by_this_run" in code
