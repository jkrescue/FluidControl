import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
source=Path(__file__).with_name('check_response_aux_terminal.py')
if not source.exists(): source=Path(__file__).resolve().parents[1]/'scripts/check_response_aux_terminal.py'
spec=importlib.util.spec_from_file_location('e',source)
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class AuxTests(unittest.TestCase):
    def fixture(self):
        r={'auxiliary_profile':'FC_P064_RESPONSE_AUX_K1_FRESH',
           'auxiliary_records':[dict(loss=.1,coefficient=1,backward_multiplier=8,aerodynamic_calls=1,sample_count=12,flow_calls=0) for _ in range(32)],
           'auxiliary_input_records':[dict(case=f'matched_start_acquisition_train_b{p:02d}_{role}',frames=[0,1]) for p in (0,2,4,6) for role in ('m075','zero','p075')]}
        p={'auxiliary_response':dict(phases=[0,2,4,6],roles=['m075','zero','p075'],coefficient=1,updates=32,extra_aerodynamic_samples=384,pre_accumulator_divide_compensation=8,extra_flow_forward_calls=0,validation_accessed=False,frozen_test_accessed=False)}
        return r,p
    def test_fixed_aux(self):
        self.assertEqual(e.check_aux(*self.fixture())['shared_q0_states'],4)
    def test_count(self):
        r,p=self.fixture();r['auxiliary_records'].pop()
        with self.assertRaises(AssertionError):e.check_aux(r,p)
    def test_nonfinite(self):
        r,p=self.fixture();r['auxiliary_records'][0]['loss']=float('nan')
        with self.assertRaises(AssertionError):e.check_aux(r,p)
    def test_gradient_multiplier(self):
        r,p=self.fixture();r['auxiliary_records'][0]['backward_multiplier']=1
        with self.assertRaises(AssertionError):e.check_aux(r,p)
    def test_no_new_split(self):
        r,p=self.fixture();p['auxiliary_response']['validation_accessed']=True
        with self.assertRaises(AssertionError):e.check_aux(r,p)
    def test_no_candidate_read_when_running(self):
        with self.assertRaises(ValueError):e.base.terminal({'InvocationID':'x','MainPID':'1'},'x')
    def test_same_basename_different_paths_cannot_replace_candidate(self):
        e.CAPTURE.clear()
        with patch.object(e,'original_read',return_value={'kind':'candidate'}):
            e.read(e.OUTPUT/'dual_model_manifest.json')
        with patch.object(e,'original_read',return_value={'kind':'parent'}):
            e.read(Path('/tmp/p064_response_aux_candidate_e_20261007/dual_model_manifest.json'))
            e.read(Path('/tmp/parent/dual_model_manifest.json'))
        self.assertEqual(e.CAPTURE['dual_model_manifest.json'],{'kind':'candidate'})

if __name__=='__main__':unittest.main()
