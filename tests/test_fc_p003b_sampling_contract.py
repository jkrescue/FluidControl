from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fc_p003b_sampling", REPO / "scripts/audit_fc_p003b_real_sampling_contract.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SamplingContractTest(unittest.TestCase):
    def test_reference_receipt_is_cryptographically_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "result.json"
            receipt.write_text(json.dumps({
                "status": "FC_P003_OFFICIAL_DATALOADER_ORDER_COUNTERFACTUAL_PASS",
                "seed": MODULE.SEED,
                "regular_count": 1368,
                "regular_sequence_sha256": MODULE.FC_P003_REGULAR_SHA,
            }))
            with mock.patch.object(
                MODULE, "FC_P003_ORDER_RECEIPT_SHA", MODULE.file_sha(receipt)
            ):
                self.assertEqual(
                    MODULE.load_fc_p003_reference(receipt)["regular_count"], 1368
                )
                receipt.write_text("{}")
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    MODULE.load_fc_p003_reference(receipt)


if __name__ == "__main__":
    unittest.main()
