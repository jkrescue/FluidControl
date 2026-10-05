"""Tiny software fixtures only: no real HDF reads or cache advice."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts" if (HERE.parent / "scripts/advise_p026_verified_train_cache_once.py").is_file() else HERE
SPEC = importlib.util.spec_from_file_location("p026_cache_amendment_tested", SCRIPTS / "advise_p026_verified_train_cache_once.py")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def test_old_default_is_preparation_only_r1():
    args = helper.parse_args([])
    assert args.receipt_name == "cache_advice_20261006_r1.jsonl"
    assert args.execute is False


@pytest.mark.parametrize("name", helper.RECEIPT_NAMES)
def test_four_explicit_names_stay_in_exact_k1_directory(name):
    args = helper.parse_args(["--receipt-name", name])
    target = helper.OUTPUT.with_name(args.receipt_name)
    assert target.parent == helper.ROOT / "artifacts/fcp026_history_training_k1_20261005"
    assert target.name == name
    assert not args.execute


@pytest.mark.parametrize("name", ["../cache_advice_20261006_r2.jsonl", "/tmp/escape", "r2", "cache_advice_20261006_r5.jsonl"])
def test_arbitrary_paths_or_names_rejected(name):
    with pytest.raises(SystemExit):
        helper.parse_args(["--receipt-name", name])


@pytest.mark.parametrize("name", helper.RECEIPT_NAMES)
def test_existing_selected_receipt_cannot_repeat_advice(tmp_path, monkeypatch, name):
    files = {f"data/curated/{family}/train/software_fixture_{i}.h5": "a" * 64
             for family, count in helper.FAMILIES.items() for i in range(count)}
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"train_hdf_sha256": files}))
    target = tmp_path / name
    target.write_text("historical receipt stays unchanged")
    monkeypatch.setattr(helper, "AUDIT", audit)
    monkeypatch.setattr(helper, "AUDIT_SHA", hashlib.sha256(audit.read_bytes()).hexdigest())
    monkeypatch.setattr(helper, "OUTPUT", tmp_path / helper.RECEIPT_NAMES[0])
    monkeypatch.setattr(helper, "parse_args", lambda: type("Args", (), {"execute": True, "receipt_name": name})())
    def forbidden(*args, **kwargs):
        raise AssertionError("data/cache/resource operation must not run")
    monkeypatch.setattr(helper, "open_confined", forbidden)
    monkeypatch.setattr(helper, "memory", forbidden)
    monkeypatch.setattr(helper.os, "posix_fadvise", forbidden)
    with pytest.raises(FileExistsError):
        helper.main()
    assert target.read_text() == "historical receipt stays unchanged"
