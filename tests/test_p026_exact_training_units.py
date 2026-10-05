"""Unit-name fixtures stop at terminal query; no model/HDF/systemctl reads."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import audit_fcp026_candidate as audit


class ReachedTerminalQuery(Exception):
    pass


@pytest.mark.parametrize("k,unit,invocation,valid", [
    (1, "fluid-control-fcp026-history-k1-20261005.service", "a"*32, True),
    (4, "fluid-control-fcp026-history-k4-20261006.service", "eee5a6fbad40411cac2f05e00520b079", True),
    (4, "fluid-control-fcp026-history-k4-20261005.service", "a"*32, False),
    (1, "fluid-control-fcp026-history-k1-20261006.service", "a"*32, False),
    (4, "fluid-control-fcp026-history-k1-20261005.service", "a"*32, False),
    (4, "fluid-control-fcp026-history-k4-20261006.service.extra", "a"*32, False),
    (4, "fluid-control-fcp026-history-k4-20261006.service", "short", False),
])
def test_exact_arm_execution_unit(monkeypatch, k, unit, invocation, valid):
    monkeypatch.setitem(sys.modules, "audit_fcp013_dual_candidate", SimpleNamespace(validate_guard=None))
    monkeypatch.setitem(sys.modules, "audit_fcp015_candidate", SimpleNamespace(validate_resource_watch=None))
    approval = dict(status="FC_P026_APPROVED_MATCHED_HISTORY_TRAINING_NOT_ADMISSION",
                    history_k=k, image=audit.IMAGE)
    observation = dict(status=f"FC_P026_K{k}_TRAINING_RUNNING_NOT_ADMISSION", unit=unit,
        invocation=invocation, approval_sha256="a"*64, image=audit.IMAGE,
        observed_container_state="running", planned_windows=1368,
        planned_optimizer_updates=171, training_completed=False, scientific_admission=False)
    monkeypatch.setattr(audit, "checked", lambda *args: None)
    monkeypatch.setattr(audit, "read", lambda p: approval if p.name == "execution_approval.json" else observation)
    def terminal_query(actual):
        assert actual == unit
        raise ReachedTerminalQuery()
    monkeypatch.setattr(audit, "terminal_properties", terminal_query)
    expected = ReachedTerminalQuery if valid else ValueError
    with pytest.raises(expected):
        audit.validate_execution(Path("repo"), Path("root"), k, "a"*64,
            Path("observation.json"), "b"*64, unit, invocation)
