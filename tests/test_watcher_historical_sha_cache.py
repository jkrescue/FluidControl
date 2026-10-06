import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

SOURCE = Path(__file__).with_name('watch_training_evaluation_state.py')
if not SOURCE.exists():
    SOURCE = Path(__file__).parents[1] / 'scripts/watch_training_evaluation_state.py'
spec = importlib.util.spec_from_file_location('watch_cache_fixture', SOURCE)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


@pytest.fixture
def files(tmp_path):
    folder = tmp_path/'artifacts/completed'
    folder.mkdir(parents=True)
    payload = folder/'model.pt'
    payload.write_bytes(b'old payload')
    receipt = folder/'receipt.json'
    receipt.write_text(json.dumps({'status':'TRAINING_COMPLETE','sha256':{'model.pt':hashlib.sha256(payload.read_bytes()).hexdigest()}}))
    return payload, receipt, tmp_path/'artifacts/monitor/historical_payload_sha_cache.json'


def test_second_verification_has_no_payload_reads(files, monkeypatch):
    payload, receipt, cache = files
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE') == (True, [])
    monkeypatch.setattr(watch.os,'read',lambda *_: pytest.fail('cached payload read'))
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE') == (True, [])
    assert json.loads(cache.read_text())['entries'][str(payload)]['last_verification'] == 'cached_prior_sha_stat_unchanged'


@pytest.mark.parametrize('change', ['content', 'replace', 'mtime_restore'])
def test_mutation_invalidates(files, change):
    payload, receipt, cache = files
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    old = payload.stat()
    if change == 'replace':
        replacement=payload.with_suffix('.tmp'); replacement.write_bytes(b'new payload'); replacement.replace(payload)
    else:
        payload.write_bytes(b'new payload')
    if change == 'mtime_restore':
        os.utime(payload, ns=(old.st_atime_ns, old.st_mtime_ns))
    assert not watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert json.loads(cache.read_text())['entries'][str(payload)]['last_verification'] == 'new_same_fd_sha256'


@pytest.mark.parametrize('contents', ['not json', '[]', '{"schema":1,"entries":[]}'])
def test_malformed_cache_rehashes(files, contents):
    payload, receipt, cache = files
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    cache.write_text(contents)
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert json.loads(cache.read_text())['entries'][str(payload)]['last_verification'] == 'new_same_fd_sha256'


def test_missing_and_symlink_rejected(files):
    payload, receipt, cache = files
    watch.verify_receipt(receipt,'TRAINING_COMPLETE')
    payload.unlink()
    assert not watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    with pytest.raises(OSError):
        watch.historical_payload_sha256(payload,receipt)
    assert str(payload) not in json.loads(cache.read_text())['entries']
    other=payload.with_suffix('.other'); other.write_bytes(b'old payload'); payload.symlink_to(other)
    assert not watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]


def test_change_during_hash_rejected(files, monkeypatch):
    payload, receipt, cache = files
    original=watch.os.read
    def changed(fd, count):
        data=original(fd,count)
        if data:
            payload.write_bytes(b'new payload')
        return data
    monkeypatch.setattr(watch.os,'read',changed)
    assert not watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert str(payload) not in json.loads(cache.read_text())['entries']


def test_non_completed_receipt_does_not_cache(files):
    payload, receipt, cache = files
    document=json.loads(receipt.read_text()); document['status']='LAUNCH_RECORDED'; receipt.write_text(json.dumps(document))
    assert watch.verify_receipt(receipt,'LAUNCH_RECORDED')[0]
    assert not cache.exists()


def test_entry_bound_and_atomic_output(files):
    payload, receipt, cache = files
    watch.verify_receipt(receipt,'TRAINING_COMPLETE')
    cache.write_text(json.dumps({'schema':1,'entries':{str(i):{} for i in range(1024)}}))
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert len(json.loads(cache.read_text())['entries']) == 1024
    assert list(cache.parent.glob('.historical_payload_sha_cache.json.*')) == []


def test_symlink_cache_never_used_or_written(files):
    payload, receipt, cache = files
    cache.parent.mkdir(parents=True)
    other=cache.parent/'elsewhere.json'; other.write_text('unchanged')
    cache.symlink_to(other)
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert other.read_text() == 'unchanged'


def test_oversized_cache_falls_back(files):
    payload, receipt, cache = files
    watch.verify_receipt(receipt,'TRAINING_COMPLETE')
    cache.write_text(' ' * 2_000_001)
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert json.loads(cache.read_text())['entries'][str(payload)]['last_verification'] == 'new_same_fd_sha256'


def test_execute_only_directories_use_opath(files, monkeypatch):
    payload, receipt, cache = files
    original=watch.os.open
    observed=[]
    def restricted(path, flags, *args, **kwargs):
        if flags & os.O_DIRECTORY:
            assert flags & os.O_NOFOLLOW
            if not flags & os.O_PATH:
                raise PermissionError('directory listing forbidden; traversal allowed')
            observed.append(path)
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(watch.os,'open',restricted)
    assert watch.verify_receipt(receipt,'TRAINING_COMPLETE')[0]
    assert observed
    assert json.loads(cache.read_text())['entries'][str(payload)]['last_verification'] == 'new_same_fd_sha256'
