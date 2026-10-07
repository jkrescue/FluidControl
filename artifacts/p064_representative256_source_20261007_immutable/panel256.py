"""Project fixed-point selector using unchanged official-reader datasets."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

ROOT=Path('/workspace/fluid_control')

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def array_sha(value):
    a=value.detach().cpu().contiguous().numpy();h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def select(schedule):
    counts=Counter();rows=[]
    for row in schedule:
        row=asdict(row) if not isinstance(row,dict) else dict(row)
        key=row['source']
        if key not in ('original44','controlled_b00'):raise ValueError('unknown source')
        k=counts[key];counts[key]+=1
        rows.append(dict(**row,occurrence_key=key,occurrence_k=k,lead=1+(37*k%100),weight_denominator=256))
    if len(rows)!=256 or counts!={'original44':192,'controlled_b00':64}:raise ValueError('fixed192+64')
    return rows

def resolve_original(children,global_index):
    index=global_index
    for family,data in enumerate(children):
        if index<len(data.index):
            file_index,start=data.index[index]
            return family,data,file_index,start
        index-=len(data.index)
    raise IndexError(global_index)

def build_panel(approval, *, keep_tensors=False):
    import torch
    from fluid_control.tandem_datapipe import TandemRolloutDataset
    from fluid_control import p064_controlled_aero_ab as schedule
    from p026_state_history import build_input
    roots=[Path(approval['training_views'][key]['root']) for key in ('base','train8','train16')]
    roots.append(Path(approval['b00_view']['root']))
    children=[TandemRolloutDataset(root,'train',100,stride=(20,2,2,1)[i],num_workers=1,force_indices=(0,1,2,3)) for i,root in enumerate(roots)]
    try:
        if [len(x.index) for x in children]!=[720,408,240,701]:raise ValueError('original index shape')
        for d in children[1:]:
            for key in ('state_mean','state_std','force_mean','force_std'):
                if not torch.equal(getattr(d,key),getattr(children[0],key)):raise ValueError('normalization mismatch')
            if d.action_scale!=children[0].action_scale:raise ValueError('action scale')
        argv=approval['argv'];order_path=Path(argv[argv.index('--parent-order')+1])
        rows=select(schedule.compile_schedule(json.loads(order_path.read_text()),'B'))
        tensors=[];physical=[]
        for row in rows:
            if row['source']=='controlled_b00':
                family,d,f,start=3,children[3],0,row['b00_start']
            else:family,d,f,start=resolve_original(children[:3],row['original_global_index'])
            current=start+row['lead']-1;reader=d._reader(f)
            now,_=reader[current];nxt,_=reader[current+1]
            mask=now['mask'].float()
            if not torch.equal(mask,nxt['mask'].float()):raise ValueError('mask time mismatch')
            state=((now['state'].float()-d.state_mean)/d.state_std)*mask
            omega0=now['omega'].float().reshape(1)/d.action_scale
            omega1=nxt['omega'].float().reshape(())/d.action_scale
            inputs=build_input(state[None],mask,omega0,omega1)[None]
            target=(nxt['force'].float()-d.force_mean)/d.force_std
            t0=float(now['time']);t1=float(nxt['time'])
            if abs((t1-t0)-.1)>2e-5:raise ValueError('one-step time')
            if not all(bool(torch.isfinite(x).all()) for x in (inputs,target,mask)):raise ValueError('nonfinite')
            path=d.paths[f]
            row.update(dataset_index=family,case=path.stem,hdf_path=str(path),window_start=start,current_index=current,target_index=current+1,time_current=t0,time_target=t1,omega_current=float(now['omega']),omega_next=float(nxt['omega']),input_sha256=array_sha(inputs),state_sha256=array_sha(state),mask_sha256=array_sha(mask),target_normalized_sha256=array_sha(target),target_physical_sha256=array_sha(nxt['force'].float()),action_pair_sha256=array_sha(torch.stack([now['omega'].float().reshape(()),nxt['omega'].float().reshape(())])),time_pair_sha256=array_sha(torch.stack([now['time'].float().reshape(()),nxt['time'].float().reshape(())])))
            physical.append((str(path.resolve()),current,current+1))
            if keep_tensors:tensors.append((inputs,mask,target))
        per_case={}
        for r in rows:
            c=per_case.setdefault(r['case'],dict(count=0,source=r['source'],family=r['dataset_index'],time_min=r['time_current'],time_max=r['time_target'],lead_min=r['lead'],lead_max=r['lead']))
            c['count']+=1;c['time_min']=min(c['time_min'],r['time_current']);c['time_max']=max(c['time_max'],r['time_target']);c['lead_min']=min(c['lead_min'],r['lead']);c['lead_max']=max(c['lead_max'],r['lead'])
        summary=dict(points=256,unique_physical_points=len(set(physical)),duplicate_rows=256-len(set(physical)),source_counts=dict(Counter(r['source'] for r in rows)),family_counts=dict(Counter(r['dataset_index'] for r in rows)),case_count=len(per_case),per_case=per_case,source_occurrence_key='ScheduleRow.source; two counters; zero-based',microbatch_sizes=[10]*25+[6],gradient_denominator=256,source_roots=[str(x) for x in roots])
        return rows,summary,tensors
    finally:
        for d in children:d.close()

def main():
    import argparse,sys
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    approval_path=ROOT/'docs/FC_P064_ARM_B_TRAINING_APPROVAL_20261006.json';s=json.loads(approval_path.read_text())
    src=Path(s['source_root']);sys.path[:0]=[str(src/'src'),str(src/'scripts')]
    rows,summary,_=build_panel(s)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(dict(status='FIXED256_CPU_PANEL_ONLY_NO_MODEL',source_sha256=sha(__file__),parent_approval_sha256=sha(approval_path),rows=rows,summary=summary),f,indent=2,allow_nan=False)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
