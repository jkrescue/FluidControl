import hashlib
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_p064_i_dev_arrays as m

def test_only_four_identity_replacements():
    source=m.BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest()==m.BASE_SHA
    changed=m.adapted_source(source)
    for a,b in [("choices=['I']","choices=['D']"),("'I_rearCl'","'D_rearCl'"),("'I_totalCd'","'D_totalCd'"),("'B/I matched array'","'B/D matched array'")]:changed=changed.replace(a,b)
    assert changed==source

def test_unknown_source_rejected():
    with pytest.raises(AssertionError):m.adapted_source('unbound source')

def test_gc_success_and_wrong_or_incomplete_identity():
    unit='test.service';inv='one';sha='abc'
    start=dict(USER_UNIT=unit,USER_INVOCATION_ID=inv,MESSAGE_ID='39f53479d3a045ac8e11786248231fbf',JOB_RESULT='done',MESSAGE='Started abc',__REALTIME_TIMESTAMP='1')
    end=dict(USER_UNIT=unit,USER_INVOCATION_ID=inv,MESSAGE_ID='ae8f7b866b0347b9af31fe1c80b127c0',MESSAGE='Consumed',__REALTIME_TIMESTAMP='2')
    supervisor=dict(approval_sha256=sha,error=None,returncode=0,limits=dict(memory_max=12*2**30,memory_swap_max=0,path='/scope/'+unit))
    assert m.gc_terminal([start,end],unit,inv,sha,supervisor,False)['current_systemctl_limits_unavailable']
    for rows,identity,live in [([start],inv,False),([start,end],'wrong',False),([start,end],inv,True)]:
        with pytest.raises(AssertionError):m.gc_terminal(rows,unit,identity,sha,supervisor,live)
