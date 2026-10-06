"""Synthetic path/parser fixtures only; never invokes cache advice."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).with_name('advise_p028_verified_train_cache_once.py')
if not path.exists():
    path = Path(__file__).resolve().parents[1] / 'scripts/advise_p028_verified_train_cache_once.py'
spec = importlib.util.spec_from_file_location('p028_cache_wrapper_test', path)
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_default_r1_and_only_r2_option():
    assert wrapper.arguments([]).receipt_name == wrapper.RECEIPT_NAMES[0]
    assert wrapper.arguments(['--receipt-name', wrapper.RECEIPT_NAMES[1]]).receipt_name == wrapper.RECEIPT_NAMES[1]
    for bad in ('cache_advice_20261006_r3.jsonl', '../cache_advice_20261006_r2.jsonl'):
        with pytest.raises(SystemExit):
            wrapper.arguments(['--receipt-name', bad])


def test_r1_original_exclusive_directory_r2_existing_directory(tmp_path, monkeypatch):
    output = tmp_path / 'receipts' / wrapper.RECEIPT_NAMES[0]
    monkeypatch.setattr(wrapper, 'OUTPUT', output)
    with pytest.raises(ValueError):
        wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[1])
    assert wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[0]) == output
    with output.open('x') as stream:
        stream.write('historical-r1')
    with pytest.raises(FileExistsError):
        wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[0])
    r2 = wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[1])
    with r2.open('x') as stream:
        stream.write('synthetic-r2')
    with pytest.raises(FileExistsError):
        wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[1])
    assert output.read_text() == 'historical-r1'


def test_symlink_directory_rejected(tmp_path, monkeypatch):
    real = tmp_path / 'real'
    real.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(wrapper, 'OUTPUT', alias / wrapper.RECEIPT_NAMES[0])
    with pytest.raises(ValueError):
        wrapper.prepare_receipt(wrapper.RECEIPT_NAMES[1])


def test_core_helper_pin_unchanged():
    assert wrapper.HELPER_SHA == '98efe4a3c7fd08268d82cac9be97b79f3eadfe69217d7f1943b250f28f43ae2c'
