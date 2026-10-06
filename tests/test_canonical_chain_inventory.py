import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).with_name('check_canonical_chain_inventory.py')
if not p.exists():p=Path(__file__).resolve().parents[1]/'scripts/check_canonical_chain_inventory.py'
s=importlib.util.spec_from_file_location('inventory_checker',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_escape(tmp_path):
    with pytest.raises(ValueError):m.inside(tmp_path,'../outside')
def test_not_authority(tmp_path):
    with pytest.raises(ValueError):m.validate(tmp_path,{'schema_version':1,'execution_authorized':True})
def test_wrong_hash(tmp_path):
    (tmp_path/'small').write_text('actual')
    i={'schema_version':1,'execution_authorized':False,'files':[{'path':'small','sha256':'0'*64,'payload':False}]}
    with pytest.raises(ValueError,match='SHA mismatch'):m.validate(tmp_path,i)
def test_missing_payload_even_default(tmp_path):
    i={'schema_version':1,'execution_authorized':False,'files':[{'path':'missing','sha256':'0'*64,'payload':True}]}
    with pytest.raises(ValueError,match='missing'):m.validate(tmp_path,i)
def test_no_launch_api():
    text=p.read_text();assert 'subprocess' not in text and 'import torch' not in text and '--execute' not in text
