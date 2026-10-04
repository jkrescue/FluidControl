from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "scripts/audit_fc_p003_dataloader_order.py"
    spec = importlib.util.spec_from_file_location("fc_p003_order", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_approved_interleave_positions_are_unique_and_span_epoch():
    values = load_module().paired_indices(1368, 16, "interleaved")
    assert values == tuple(i * 1367 // 15 for i in range(16))
    assert values[0] == 0 and values[-1] == 1367
    assert len(set(values)) == 16
