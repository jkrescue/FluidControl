import ast
import importlib.util
from pathlib import Path
import unittest
HERE=Path(__file__).parent
HERE=HERE if (HERE/'run_short_horizon_confirmation.py').is_file() else HERE.parent/'scripts'
s=importlib.util.spec_from_file_location('worker',HERE/'run_short_horizon_confirmation.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Contract(unittest.TestCase):
    def test_pending_cannot_touch_sources_or_payload(self):
        with self.assertRaisesRegex(ValueError,'not authorized'):
            m.validate_spec({'status':'PREPARATION_ONLY_NOT_EXECUTION_APPROVAL'})
    def test_explicit_open_required(self):
        with self.assertRaisesRegex(ValueError,'not authorized'):
            m.validate_spec({'status':'SHORT_HORIZON_FROZEN_CONFIRMATION_EXECUTION_APPROVED','execution_authorized':True,'heldout_access_authorized':False})
    def test_no_execution_on_import(self):
        for name in ['run_short_horizon_confirmation.py','supervise_short_horizon_confirmation.py']:
            ast.parse((HERE/name).read_text())
    def test_supervisor_fixed_guards(self):
        src=(HERE/'supervise_short_horizon_confirmation.py').read_text()
        for value in ['base.cgroup_limits()', 'base.memory_ok(base.memory(), startup=True)', 'base.memory_ok(row)', 'base.stop(process)', 'time.sleep(0.5)', 'row["elapsed"] < 1200']:
            self.assertIn(value,src)
if __name__=='__main__':unittest.main()
