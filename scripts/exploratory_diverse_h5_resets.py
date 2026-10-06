"""Project-only coherent real-reset packets; original canonical flow is unchanged."""
from collections import deque
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

NORM_SHA = 'f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1'
SPLIT_SHA = '1eaa84f12ebd56e2fc9c4da3b51fa6392e94276fa44f6d2539db66c325296b89'
PHASES = ('00','02','04','06')
FAMILIES = ('m075','m0375','zero','p0375','p075')


def require(ok, reason):
    if not ok: raise ValueError(reason)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''): digest.update(block)
    return digest.hexdigest()


def starts(phase):
    require(phase in PHASES, 'fixed train phase only')
    prefix = f'matched_start_acquisition_train_b{phase}_'
    return [(prefix+'zero',0)] + [(prefix+family,62) for family in FAMILIES]


def scheduled_omega(points, t):
    a = np.asarray(points, dtype=np.float64)
    require(a.ndim==2 and a.shape[1]==2 and len(a)>=2 and np.isfinite(a).all()
            and np.all(np.diff(a[:,0])>0) and a[0,0]<=t<=a[-1,0], 'action schedule')
    return float(np.interp(t,a[:,0],a[:,1]))


def checked_arrays(frame, nominal_time, scheduled, times, forces, norm):
    state, mask = frame['state'].float(), frame['mask'].float()
    require(state.shape==(3,128,256) and mask.shape==(1,128,256), 'fixed curated grid')
    require(torch.isfinite(state).all() and torch.isfinite(mask).all()
            and torch.all((mask==0)|(mask==1)) and mask.sum()>0, 'finite state/binary mask')
    omega = frame['omega'].float().reshape(-1)
    timestamp = frame['time'].float().reshape(-1)
    require(omega.numel()==timestamp.numel()==1 and torch.isfinite(omega).all()
            and torch.isfinite(timestamp).all(), 'finite scalar time/action')
    require(float(omega[0])==float(np.float32(scheduled)) and abs(scheduled)<=.75, 'actual initial omega differs from source schedule')
    tolerance = max(2*float(np.spacing(np.float32(abs(nominal_time)))),1e-7)
    require(abs(float(timestamp[0])-nominal_time)<=tolerance, 'HDF time differs from nominal frame time')
    times, forces = np.asarray(times,np.float64),np.asarray(forces,np.float64)
    expected = np.round(nominal_time-.1*np.arange(61,-1,-1),8)
    require(times.shape==(62,) and forces.shape==(62,4) and np.isfinite(forces).all()
            and np.array_equal(times,expected) and np.all(np.diff(times)>0)
            and times[-1]<=nominal_time+1e-8, 'exact raw causal62 history required')
    mean = torch.tensor(norm['state_mean'],dtype=torch.float32)[:,None,None]
    std = torch.tensor(norm['state_std'],dtype=torch.float32)[:,None,None]
    require(mean.shape==std.shape==(3,1,1) and torch.isfinite(mean).all()
            and torch.isfinite(std).all() and torch.all(std>0), 'normalization')
    q = ((state-mean)/std*mask).detach().clone()
    curated_force = frame['force'].float().reshape(-1).numpy()
    require(curated_force.shape==(4,) and np.isfinite(curated_force).all(), 'four curated forces')
    return {'field':q,'mask':mask.clone(),'omega':float(omega[0]),
            'force':forces[-1].astype(np.float32),'time':nominal_time,
            'history_times':times.copy(),'history_forces':forces.copy(),
            'evidence':{'stored_hdf_time':float(timestamp[0]),'nominal_time':nominal_time,
                'time_tolerance':tolerance,'hdf_time_abs_difference':abs(float(timestamp[0])-nominal_time),
                'hdf_force_minus_raw_endpoint':(curated_force.astype(np.float64)-forces[-1]).tolist(),
                'omega_matches_source_schedule_fp32':True,'raw_history_interpolated':False}}


def load_packet(data_root, cases_root, case, frame_index, *, guard=lambda:None):
    """Actual loader: officialreader singleframe; rawpastforces only, no model."""
    from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
    from fluid_control.openfoam_force_history import actual_causal_prehistory
    import h5py
    data_root,cases_root=Path(data_root).resolve(),Path(cases_root).resolve()
    require((case,frame_index) in [s for p in PHASES for s in starts(p)], 'undeclared reset')
    manifest=json.loads((data_root/'manifest.json').read_text())
    split_path=data_root/'splits/train.json'
    require(sha(split_path)==SPLIT_SHA and sha(data_root/'normalization.json')==NORM_SHA
            and manifest['normalization_sha256']==NORM_SHA
            and manifest['split_manifests']['train']['sha256']==SPLIT_SHA, 'train manifests/norm')
    split=json.loads(split_path.read_text()); path=data_root/'train'/f'{case}.h5'
    require(case in split['cases'] and path.is_file() and not path.is_symlink(), 'train-only HDF')
    guard(); digest=sha(path); guard()
    require(digest==split['hdf5_sha256'][case], 'HDF byte identity')
    config_path=cases_root/case/'case_config.json'
    require(config_path.resolve().is_relative_to(cases_root) and not config_path.is_symlink(), 'confined case config')
    config=json.loads(config_path.read_text())
    require(config['case']==case and config['split']=='train' and config['field_write_interval']==.1
            and config['expected_field_frames']==801, 'source case timing/split')
    nominal=round(float(config['source_restart_time'])+.1*frame_index,8)
    with h5py.File(path,'r') as handle:
        require(handle.attrs['case']==case and handle.attrs['split']=='train'
                and json.loads(str(handle.attrs['config_json']))==config, 'curated/source metadata correspondence')
        x,y=np.asarray(handle['x'][:],np.float64),np.asarray(handle['y'][:],np.float64)
    require(x.shape==(256,) and y.shape==(128,) and np.all(np.diff(x)>0)
            and np.all(np.diff(y)>0), 'curated coordinate grid')
    reader=HDF5Reader(path,fields=['state','mask','omega','force','time'])
    try: current,_=reader[frame_index]
    finally: reader.close()
    reference=cases_root/config['source_restart_case'] if frame_index==0 else cases_root/case
    require(reference.resolve().is_relative_to(cases_root) and not reference.is_symlink(), 'confined raw case')
    raw_before={str(p.relative_to(cases_root)):sha(p) for body in ('forceFront','forceRear')
                for p in sorted((reference/'postProcessing'/body).glob('*/coefficient.dat'))}
    times,forces,sources=actual_causal_prehistory(reference,nominal,provenance_root=cases_root)
    require(raw_before=={r['path']:r['sha256'] for rows in sources.values() for r in rows}, 'raw source changed during history read')
    if frame_index==0:
        require(config['source_force_sha256']=={key:rows[0]['sha256'] if len(rows)==1 else None
                    for key,rows in sources.items()}, 'original zero source force identity')
    packet=checked_arrays(current,nominal,scheduled_omega(config['action_points'],nominal),times,forces,
                          json.loads((data_root/'normalization.json').read_text()))
    packet.update(case=case,frame=frame_index,path=path,x=x,y=y,
        provenance={'hdf_sha256':digest,'case_config_sha256':sha(config_path),
                    'normalization_sha256':NORM_SHA,'split_sha256':SPLIT_SHA,
                    'raw_sources':sources,'nominal_time':nominal})
    guard()
    return packet


def install_packet(flow, packet):
    """Explicit exploratory reset installation, never a relaxed canonical constructor."""
    import p026_history_inference as history
    require(flow.fno_history_runtime['profile']=='p026_k1'
            and flow.fno_history_runtime['history_length']==1, 'only auditedK1 currentstate')
    require((packet['case'],packet['frame']) in [s for p in PHASES for s in starts(p)], 'fixed reset packet')
    require(torch.equal(packet['mask'].to(flow.device),flow.mask), 'reset geometry must match anchor')
    require(np.array_equal(packet['x'],flow.x) and np.array_equal(packet['y'],flow.y), 'physical probe coordinate grid differs')
    q=packet['field'].to(flow.device).clone()
    require(torch.isfinite(q).all() and float(q.abs().max())<=flow.max_abs_normalized_state_guard,
            'real reset outside existing state support')
    omega=torch.tensor([packet['omega']],device=flow.device,dtype=q.dtype)/flow.MAX_CONTROL
    buffer=history.trajectory_history(q[None],omega,torch.tensor([0],device=flow.device),k=1)
    require(buffer.states.shape[1]==1 and not buffer.padding_mask.any(), 'K1 no invented history')
    flow.initial_state={'field':q,'omega':packet['omega'],'force':packet['force'].copy(),
                        'fno_history_runtime':dict(flow.fno_history_runtime),'fno_history':buffer}
    flow.initial_cfd_time=packet['time']; flow.frame=packet['frame']; flow.case_path=Path(packet['path'])
    flow.initial_force_source='verified_raw_causal_endpoint_exploratory_diverse_reset'
    flow.initial_state_bound=float(q.abs().max())
    flow._initial_reward_history=tuple((float(t),v.copy()) for t,v in zip(packet['history_times'],packet['history_forces'],strict=True))
    flow.initial_reward_history_sources=copy.deepcopy(packet['provenance'])
    flow.reset()


def packet_identity(packet):
    """JSON identity shared by bounded CPU verification and later approved training."""
    arrays={}
    for key in ('field','mask','force','history_times','history_forces','x','y'):
        value=packet[key]
        a=value.detach().cpu().numpy() if isinstance(value,torch.Tensor) else np.asarray(value)
        a=np.ascontiguousarray(a)
        arrays[key]={'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
    return {'case':packet['case'],'frame':packet['frame'],'time':packet['time'],
            'omega':packet['omega'],'arrays':arrays,'provenance':copy.deepcopy(packet['provenance']),
            'evidence':copy.deepcopy(packet['evidence'])}


def wrap_phase_cycle(original_h5_wrapper, packets):
    """Retain original H5 step audit; override reset only in this separate wrapper."""
    require(len(packets)==6 and [(p['case'],p['frame']) for p in packets]
            in [starts(phase) for phase in PHASES], 'fixed deterministic six-start phase panel')
    class DiverseResetAudit(type(original_h5_wrapper)):
        def __init__(self,raw):
            super().__init__(raw)
            self.reset_index=0
            self.reset_counts=[0]*6
            self.active_reset=None
        def reset(self,*,seed=None,options=None):
            require(options in (None,{}), 'no external reset-time override')
            packet=packets[self.reset_index%6]
            install_packet(self.env.flow,packet)
            self.env.initial_states=[self.env.flow.copy_state()]
            obs,info=self.env.reset(seed=seed)
            flow=self.env.flow
            require(flow.t==packet['time'] and len(flow._reward_history)==62
                    and flow.omega==packet['omega'], 'actual HydroGym reset restoration')
            self.reset_index+=1
            self.reset_counts[(self.reset_index-1)%6]+=1
            self.active_reset={'reset_panel_index':(self.reset_index-1)%6,
                               'reset_case':packet['case'],'reset_frame':packet['frame']}
            return obs,{**info,'exploratory_horizon':5,'scientific_admission':False,
                'reset_panel_index':(self.reset_index-1)%6,'reset_case':packet['case'],
                'reset_frame':packet['frame'],'reset_provenance':copy.deepcopy(packet['provenance'])}
        def step(self,action):
            require(self.active_reset is not None,'reset before step')
            obs,reward,terminated,truncated,info=super().step(action)
            return obs,reward,terminated,truncated,{**info,**self.active_reset}
    return DiverseResetAudit(original_h5_wrapper.env)
