"""Bounded host supervisor for the approved CPU-only 27-frame diagnostic."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise ValueError(m)
def mem_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'): return int(line.split()[1])/2**20
    raise ValueError('MemAvailable missing')


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--spec-sha256',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    require(sha(a.spec)==a.spec_sha256,'spec SHA');s=json.loads(a.spec.read_text())
    argv=[sys.executable,s['worker']['path'],'--spec',str(a.spec.resolve()),'--spec-sha256',a.spec_sha256,'--output',s['planned_output']]
    require(s['resources']=={'cuda_visible_devices':'','cpu_quota':1,'memory_max_gib':2,'memory_swap_max':0,'runtime_seconds':120,'worker_timeout_seconds':100,'startup_mem_available_gib':50,'runtime_mem_available_gib':22},'resources')
    require(not Path(s['planned_output']).exists(),'output exists')
    if not a.execute:
        subprocess.run(argv,check=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':''});print('P064_COARSE_FORCE_SUPERVISOR_DRYRUN_PASS');return
    require(s['execution_authorized'] is True,'not authorized')
    unit=os.environ.get('P064_EXPECTED_UNIT','');inv=os.environ.get('INVOCATION_ID','')
    require(unit==s['planned_unit'] and len(inv)==32,'unit identity')
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryMax,MemorySwapMax,CPUQuotaPerSecUSec,RuntimeMaxUSec'],text=True).splitlines())
    require(props['MemoryMax']==str(2*2**30) and props['MemorySwapMax']=='0' and props['CPUQuotaPerSecUSec']=='1s' and props['RuntimeMaxUSec']=='2min','unit resources')
    require(mem_available_gib()>=50,'startup memory')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','P064_COARSE_FORCE_TOKEN':'EXECUTE_APPROVED_P064_COARSE_FORCE'}
    child=subprocess.Popen(argv+['--execute'],env=env,start_new_session=True);samples=[];start=time.monotonic()
    try:
        while child.poll() is None:
            available=mem_available_gib();samples.append({'elapsed_seconds':time.monotonic()-start,'mem_available_gib':available})
            if available<22 or time.monotonic()-start>100:
                os.killpg(child.pid,signal.SIGTERM);raise RuntimeError('runtime guard')
            time.sleep(.5)
        require(child.returncode==0,'worker failed')
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL);child.wait()
    out=Path(s['planned_output']);result=out/'result.json';require(result.is_file(),'result missing')
    receipt={'status':'P064_COARSE_FORCE_SUPERVISOR_COMPLETE','unit':unit,'invocation_id':inv,'spec_sha256':a.spec_sha256,'result_sha256':sha(result),'resource_samples':samples,'elapsed_seconds':time.monotonic()-start}
    tmp=out/'supervisor_receipt.json.tmp';tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(out/'supervisor_receipt.json')

if __name__=='__main__': main()
