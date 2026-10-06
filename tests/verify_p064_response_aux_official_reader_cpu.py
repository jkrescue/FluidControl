"""Synthetic HDF only; actual official Dataset/HDF5Reader, no real data/model."""
import json
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import p064_response_aux as aux
from fluid_control.tandem_datapipe import TandemRolloutDataset


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    with tempfile.TemporaryDirectory(prefix='response-aux-reader-') as temp:
        root = Path(temp)
        (root/'train').mkdir()
        norm = dict(state_mean=[1,2,3],state_std=[2,3,4],all_force_channels=['front_cd','front_cl','rear_cd','rear_cl'],all_force_mean=[1,2,3,4],all_force_std=[2,3,4,5])
        (root/'normalization.json').write_text(json.dumps(norm))
        (root/'manifest.json').write_text(json.dumps(dict(max_abs_omega=.75)))
        rows=[]
        for phase in aux.PHASES:
            for role,sign in [('m075',-1),('m0375',-1),('zero',0),('p0375',1),('p075',1)]:
                case=f'matched_start_acquisition_train_b{phase:02d}_{role}'
                state=np.full((101,3,2,2),phase+1,dtype='float32')
                state[1:] += sign*.1
                mask=np.ones((101,1,2,2),dtype='float32')
                time=np.arange(101,dtype='float32')[:,None]*.1+100
                omega=np.full((101,1),sign*.1,dtype='float32');omega[0]=0
                force=np.full((101,4),phase+1,dtype='float32');force[1:]+=sign*.2
                vals=dict(state=state,mask=mask,time=time,omega=omega,force=force)
                path=root/'train'/f'{case}.h5'
                with h5py.File(path,'w') as f:
                    for key,val in vals.items():f.create_dataset(key,data=val)
                    f.attrs['case']=case;f.attrs['split']='train'
                rows.append(dict(case=case,hdf_sha256=aux.sha(path),frame_hashes={k:[aux.hashlib.sha256(v[i].tobytes()).hexdigest() for i in (0,1)] for k,v in vals.items()}))
        proof=root/'proof.json';proof.write_text(json.dumps(dict(rows=rows,all_clocks_and_action_endpoints_exact=True)))
        original=aux.RECEIPT_SHA
        aux.RECEIPT_SHA=aux.sha(proof) # Fixture-only synthetic proof; never alter production source.
        dataset=TandemRolloutDataset(root,'train',100,stride=1,num_workers=0,force_indices=(0,1,2,3))
        try:
            inputs,masks,targets,records=aux.load_pairs(dataset,proof,'cpu')
            assert inputs.shape==(12,6,2,2) and targets.shape==(12,4)
            assert [r['case'].split('_')[-1] for r in records]==list(aux.ROLES)*4
            assert torch.equal(inputs[0,:3],inputs[1,:3]) and torch.equal(inputs[2,:3],inputs[1,:3])
            torch.testing.assert_close(inputs[:3,4,0,0],torch.zeros(3))
            torch.testing.assert_close(inputs[:3,5,0,0],torch.tensor([-.1,0,.1])/.75)
            torch.testing.assert_close(targets[0]-targets[1],torch.full((4,),-.2)/torch.tensor(norm['all_force_std']))
        finally:
            dataset.close();aux.RECEIPT_SHA=original
    print('OFFICIAL_READER_SYNTHETIC_12_UNIQUE_K1_INPUTS_PASS_NO_REAL_DATA_OR_MODEL')


if __name__=='__main__':main()
