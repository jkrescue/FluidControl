import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'reproduce_canonical_closed_loop.py'
if not SOURCE.exists():
    SOURCE = HERE.parent / 'scripts/reproduce_canonical_closed_loop.py'
s = importlib.util.spec_from_file_location('launcher', SOURCE)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)

class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(output='artifacts/old', lead_statement='old', steps=800,
                         required_systemd={'MemoryMax': '8G'}, runtime_env={'CUDA_VISIBLE_DEVICES': ''})
        self.spec = copy.deepcopy(self.base)
        self.unit = 'fluid-control-canonical-reproduce-demo1.service'
        self.spec.update(output='artifacts/new', lead_statement='Lead approves this one reproduction',
                         reproduction=dict(base_approval_sha256=m.BASE_SHA, unit=self.unit, execution_authorized=True))

    def test_new_authority(self):
        m.compare_approval(self.base, self.spec, self.unit)

    def test_no_historical_reuse(self):
        self.spec['output'] = self.base['output']
        with self.assertRaises(ValueError): m.compare_approval(self.base, self.spec, self.unit)

    def test_no_science_or_resource_change(self):
        for key, value in [('steps', 124), ('required_systemd', {'MemoryMax': '99G'})]:
            spec = copy.deepcopy(self.spec); spec[key] = value
            with self.assertRaises(ValueError): m.compare_approval(self.base, spec, self.unit)

    def test_no_implicit_authorization(self):
        self.spec['reproduction']['execution_authorized'] = 'true'
        with self.assertRaises(ValueError): m.compare_approval(self.base, self.spec, self.unit)

    def test_pending_readonly_binding(self):
        self.spec['reproduction']['execution_authorized'] = False
        m.compare_approval(self.base, self.spec, self.unit)
        m.execution_gate(self.spec, False)
        with self.assertRaises(ValueError): m.execution_gate(self.spec, True)

    def test_explicit_execution_gate(self):
        m.execution_gate(self.spec, True)

    def test_unit_injection_rejected(self):
        with self.assertRaises(ValueError): m.compare_approval(self.base, self.spec, 'a;touch /tmp/x')

    def test_path_escape(self):
        with self.assertRaises(ValueError): m.inside(Path('/tmp/repo'), '../escape')

    def test_argv_preserves_resources_without_shell(self):
        argv = m.command(Path('/repo'), self.spec, Path('/repo/new.json'), 'a'*64, self.unit, Path('/repo/driver.py'))
        self.assertIn('--property=MemoryMax=8G', argv)
        self.assertIn('--setenv=CUDA_VISIBLE_DEVICES=', argv)
        self.assertIn('--property=RemainAfterExit=yes', argv)
        self.assertEqual(argv[-1], '--execute')

if __name__ == '__main__': unittest.main()
