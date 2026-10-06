"""Synthetic files only, actual installed official reader, no model/GPU."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import sys
import h5py
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from official_evaluation_input import read_trajectory, FIELDS

class Equivalence(unittest.TestCase):
    def make(self, path):
        arrays={'state':np.arange(5*3*2*4,dtype=np.float32).reshape(5,3,2,4),
                'mask':np.ones((5,1,2,4),np.uint8),
                'omega':np.array([0,.01,-.02,.1,.2],np.float32)[:,None],
                'force':np.arange(20,dtype=np.float32).reshape(5,4),
                'time':(148+np.arange(5)*.1).astype(np.float64)[:,None],
                'x':np.linspace(8,25,4,dtype=np.float32),
                'y':np.linspace(4,11,2,dtype=np.float32)}
        with h5py.File(path,'w') as f:
            for k,v in arrays.items():f[k]=v
        return arrays
    def test_exact_all_fields_and_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.h5';arrays=self.make(p)
            before=hashlib.sha256(p.read_bytes()).hexdigest()
            actual,static=read_trajectory(p)
            for k in FIELDS:
                a=actual[k].numpy();self.assertEqual(a.dtype,arrays[k].dtype)
                self.assertEqual(a.shape,arrays[k].shape);np.testing.assert_array_equal(a,arrays[k])
            for k in ('x','y'):np.testing.assert_array_equal(static[k],arrays[k])
            self.assertEqual(before,hashlib.sha256(p.read_bytes()).hexdigest())
    def test_misaligned_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'synthetic.h5';self.make(p)
            with h5py.File(p,'a') as f:del f['force'];f['force']=np.zeros((4,4),np.float32)
            with self.assertRaises(ValueError):read_trajectory(p)

if __name__=='__main__':unittest.main()
