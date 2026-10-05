import hashlib
import json
import pytest
from audit_fcp013_dual_candidate import expected_train_hdf_hashes


def test_filename_map_validated(tmp_path):
    value = {'case.h5': 'a'*64}
    assert expected_train_hdf_hashes(tmp_path, {'hdf_sha256': value}) == value
    for bad in ({'../case.h5':'a'*64}, {'case.h5':'no_hash'}, {}):
        with pytest.raises(ValueError):
            expected_train_hdf_hashes(tmp_path, {'hdf_sha256': bad})


def test_stem_split_requires_pinned_bytes_and_cases(tmp_path):
    path=tmp_path/'train.json'
    path.write_text(json.dumps({'split':'train','cases':['case'], 'hdf5_sha256':{'case':'b'*64}}))
    manifest={'split_manifests':{'train':{'path':'train.json','sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    assert expected_train_hdf_hashes(tmp_path,manifest)=={'case.h5':'b'*64}
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='pinned training split'):
        expected_train_hdf_hashes(tmp_path,manifest)
