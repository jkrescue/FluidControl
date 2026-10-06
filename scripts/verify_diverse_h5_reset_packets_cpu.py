"""Bounded CPU-only24-reset verification; no FNO/optimizer/CFD/GPU execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path('/workspace/fluid_control')
OUTPUT=REPO/'artifacts/exploratory_diverse_h5_reset_packets_cpu_20261006'
DATA=REPO/'data/curated/tandem_cylinders_matched_start_full40_dev30_v1'
CASES=REPO/'cfd/tandem_cylinders/cases'
STATUS='DIVERSE_H5_REAL_RESET_PACKETS_VERIFIED_CPU_NOT_TRAINING'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory():
    row={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
         if line.startswith(('MemAvailable:','MemFree:'))}
    if row['MemAvailable']<22*2**30:raise RuntimeError('Available22GiB runtime reserve')
    return row


def write(path,value):
    with path.open('x') as out:json.dump(value,out,indent=2,allow_nan=False);out.write('\n')


def worker():
    import inspect
    import torch
    import exploratory_diverse_h5_resets as adapter
    from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
    torch.set_num_threads(2)
    adapter.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    adapter.require(sha(inspect.getfile(HDF5Reader))=='cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0','official reader source')
    before={str(p):sha(p) for p in (Path(__file__),Path(adapter.__file__),
        REPO/'src/fluid_control/openfoam_force_history.py',Path(inspect.getfile(HDF5Reader)))}
    rows=[]
    for phase in adapter.PHASES:
        for case,frame in adapter.starts(phase):
            packet=adapter.load_packet(DATA,CASES,case,frame,guard=memory)
            rows.append(adapter.packet_identity(packet))
    adapter.require(len(rows)==24 and sum(r['frame']==0 for r in rows)==4,'fixed24 packets')
    adapter.require(all(sha(p)==digest for p,digest in before.items()),'source changed')
    write(OUTPUT/'result.json',{'status':STATUS,'packets':rows,'packet_count':24,
        'source_sha256':before,'manifest_sha256':sha(DATA/'manifest.json'),
        'normalization_sha256':adapter.NORM_SHA,'split_sha256':adapter.SPLIT_SHA,
        'no_model_loaded':True,'no_optimizer':True,'no_cfd':True,'no_gpu':True,
        'scientific_admission':False})


def supervise():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CUDA must be hidden')
    if memory()['MemAvailable']<50*2**30:raise RuntimeError('Available50GiB startup reserve')
    cg=Path('/sys/fs/cgroup')/next(x.split('::',1)[1] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::')).lstrip('/')
    if not(0<int((cg/'memory.max').read_text())<=8*2**30 and int((cg/'memory.swap.max').read_text())==0):
        raise RuntimeError('bounded8GiB/noSwap required')
    if OUTPUT.exists() or OUTPUT.is_symlink():raise RuntimeError('exclusive output required')
    OUTPUT.mkdir()
    def term(*_):raise RuntimeError('termination requested')
    signal.signal(signal.SIGTERM,term);signal.signal(signal.SIGINT,term)
    started=time.monotonic();failure=None;child=None
    try:
        with (OUTPUT/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],
                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                row=memory();row['elapsed']=time.monotonic()-started
                with (OUTPUT/'memory.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if row['elapsed']>300:raise RuntimeError('300s deadline')
                time.sleep(.5)
            if child.returncode!=0:raise RuntimeError(f'worker exit{child.returncode}')
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        write(OUTPUT/'supervisor.json',{'returncode':None if child is None else child.returncode,
            'failure':failure,'elapsed':time.monotonic()-started,'source_sha256':sha(__file__),
            'result_sha256':sha(OUTPUT/'result.json') if (OUTPUT/'result.json').exists() else None})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args()
    if a.worker:worker()
    elif a.execute:supervise()
    else:p.error('separate explicit execution approval required')
