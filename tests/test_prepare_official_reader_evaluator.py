import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_official_reader_evaluator import OLD, CAP, SPLIT, RECORD, opt_in_source, restore_source

class Transform(unittest.TestCase):
    def test_input_only_and_explicit_use(self):
        source='def main():\n    if True:\n        torch.cuda.set_per_process_memory_fraction(\n'+CAP+'\n        )\n        network, dual_identity = load_dual_fno()\n    network.eval()\n    report = {}\n    for path in []:\n'+OLD+'\n        if True:\n            if True:\n                for offset in range(horizon):\n                    pass\n'
        source+='    parser.add_argument("--split", '+SPLIT+')\n'
        source+='    if True:\n        if True:\n            if True:\n                if True:\n                    if True:\n'+RECORD+'\n'
        changed=opt_in_source(source)
        self.assertIn('fields, coordinates = read_trajectory(path)',changed)
        self.assertEqual(restore_source(changed),source)
        self.assertLess(changed.index('load_dual_fno()'),changed.index('set_float32_matmul_precision("highest")'))
        self.assertIn('confirmation_inference_precision_override',changed)
        self.assertIn('for offset in range(horizon):\n                    confirmation_precision()',changed)
    def test_drift_rejected(self):
        with self.assertRaises(ValueError):opt_in_source('pass\n')
    def test_actual_canonical_source_only(self):
        p=Path(__file__).resolve().parents[1]/'scripts/evaluate_tandem_fno.py'
        if not p.exists():self.skipTest('canonical source unavailable in isolated stage')
        source=p.read_text();changed=opt_in_source(source);self.assertEqual(restore_source(changed),source)
        self.assertLess(changed.index('validate_dual_runtime_files(\n'),changed.index('set_float32_matmul_precision("highest")'))

if __name__=='__main__':unittest.main()
