"""Terminal-only raw B04 metadata/force audit; no field-value QC or conversion."""
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/workspace/fluid_control')
UNIT='fluid-control-p064-b04-long-excitation-cfd-20261007.service'
INV='e2fa5d4e5fa4465a90a2dfbbdd402fc8'
APPROVAL='f2bfcd8f6d1526af266946448ca452d3c271047466d77a9e439d843081338536'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def clock(values,start,count,dt):
    a=np.asarray(values,float);assert a.shape==(count,) and np.isfinite(a).all()
    assert np.allclose(a,start+dt*np.arange(count),rtol=0,atol=1e-8)

def check_components(total,pressure,viscous):
    a=np.asarray([total,pressure,viscous],float);assert a.shape==(3,4) and np.isfinite(a).all()
    difference=float(np.max(np.abs(a[0]-a[1]-a[2])));assert difference<=1e-10
    return difference

def main():
    props=dict(x.split('=',1) for x in subprocess.check_output(['systemctl','--user','show',UNIT,
      '-p','InvocationID','-p','MainPID','-p','Result','-p','ExecMainStatus',
      '-p','MemoryMax','-p','MemorySwapMax','-p','CPUQuotaPerSecUSec','-p','RuntimeMaxUSec'],text=True).splitlines())
    assert props['InvocationID']==INV and props['MainPID']=='0' and props['Result']=='success' and props['ExecMainStatus']=='0'
    assert props['MemoryMax']==str(8*2**30) and props['MemorySwapMax']=='0' and props['CPUQuotaPerSecUSec']=='1s' and props['RuntimeMaxUSec']=='21min'
    ap=ROOT/'docs/P064_B04_LONG_EXCITATION_RAW_CFD_APPROVAL_20261007.json';assert sha(ap)==APPROVAL
    s=read(ap);out=Path(s['planned_output']);r=read(out/'result.json');case=out/'case';source=Path(s['source_root'])
    assert r['status']=='P064_B04_LONG_EXCITATION_OPENFOAM_COMPLETE_PENDING_CURATOR' and r['spec_sha256']==APPROVAL
    assert not r['curator_executed'] and not r['training_executed']
    driver=Path('/tmp/p064-b04-long-review/run_p064_b04_long_excitation_cfd.py');assert sha(driver)==s['driver_sha256']==r['driver_sha256']
    assert sha(s['helper']['path'])==s['helper']['sha256'];helper=load('b04_contract',s['helper']['path']);contract=helper.build_contract(ROOT)
    assert contract['action']['points_sha256']==s['action_points_sha256']==r['action_points_sha256']
    assert sha(s['normalization']['path'])==s['normalization']['sha256']
    for p,h in s['source_inventory'].items():assert sha(source/p)==h
    fields=[]
    for p in case.iterdir():
        try:t=float(p.name)
        except ValueError:continue
        if 120<=t<=200:
            assert (p/'U').is_file() and (p/'p').is_file();fields.append(t)
    clock(sorted(fields),120,801,.1)
    for name in ['U_0','p','phi','phi_0','uniform/time','uniform/cumulativeContErr','uniform/functionObjects/functionObjectProperties']:
        assert (case/'120'/name).read_bytes()==(source/'120'/name).read_bytes()
    for p in (source/'constant').rglob('*'):
        if p.is_file():assert p.read_bytes()==(case/'constant'/p.relative_to(source/'constant')).read_bytes()
    d=load('b04_driver',driver);assert d.parse_table(case/'120/U')==contract['action']['points']
    u=(case/'120/U').read_text();old=(source/'120/U').read_text()
    strip=lambda t:re.sub(r'rearCylinder.*?(?=frontBack)','rearCylinder\n',t,flags=re.S)
    assert strip(u)==strip(old)
    control=(case/'system/controlDict').read_text()
    for text in ['startTime 120;','endTime 200;','deltaT 0.005;','writeInterval 0.1;','purgeWrite 0;']:assert text in control
    checker=s['solver_log_checker'];assert sha(checker['path'])==checker['sha256']
    log=out/'log.pimpleFoam';assert sha(log)==r['solver_log_sha256']
    qc=load('b04_logcheck',checker['path']).validate_solver_log(log,16000,200.)
    assert qc==read(out/'solver_qc.json') and sha(out/'solver_qc.json')==r['solver_qc_sha256']
    clock([float(x) for x in re.findall(r'^Time = ([0-9.eE+-]+)',log.read_text(),re.M)],120.005,16000,.005)
    coeff={};initial={};raw_sha={}
    for body in ['forceFront','forceRear']:
        files=list((case/'postProcessing'/body).glob('*/coefficient.dat'));assert len(files)==1
        p=files[0];a=np.loadtxt(p,ndmin=2);assert a.shape[1]==13 and np.isfinite(a).all();clock(a[:,0],120.005,16000,.005)
        coeff[body]=a;raw_sha[str(p)]=sha(p)
        v=s['initial_force_sources'][body];assert sha(v['path'])==v['sha256']
        old=np.loadtxt(v['path'],ndmin=2);selected=old[np.abs(old[:,0]-120)<1e-8];assert selected.shape==(1,13) and np.isfinite(selected).all()
        initial[body]=selected[0]
    component_path=ROOT/'src/fluid_control/p064_coarse_force_recoverability.py'
    assert sha(component_path)=='7d0485bc07aea692f1ca5d5d4897af6f0d70d218db9c7eefc54e8f2e6454c196'
    component=load('component_parser',component_path)
    max_sum=0.;max_total=0.;component_hashes={}
    for i in range(801):
        t=round(120+.1*i,10);p=case/f'{t:g}'/'uniform/functionObjects/functionObjectProperties'
        parsed=component.parse_force_components(p.read_text());total,pressure,viscous=component.component_vectors(parsed)
        max_sum=max(max_sum,check_components(total,pressure,viscous));component_hashes[str(p)]=sha(p)
        expected=[]
        for body in ['forceFront','forceRear']:
            row=initial[body] if i==0 else coeff[body][i*20-1]
            expected.extend([row[1],row[4]])
        difference=float(np.max(np.abs(total-expected)));assert difference<1e-8;max_total=max(max_total,difference)
    assert sha(out/'resources.jsonl')==r['resource_log_sha256']
    resources=[json.loads(x) for x in (out/'resources.jsonl').read_text().splitlines()]
    minimum=min(x['MemAvailable_GiB'] for x in resources);disk=min(x['disk_available_GiB'] for x in resources)
    assert minimum>=22 and disk>=20 and minimum==r['minimum_MemAvailable_GiB'] and disk==r['minimum_disk_available_GiB']
    cid=(out/'solver.cid').read_text().strip();assert re.fullmatch('[0-9a-f]{64}',cid)
    assert not subprocess.check_output(['docker','ps','-aq','--filter','id='+cid],text=True).strip()
    print(json.dumps({'status':'B04_RAW_METADATA_FORCE_REVIEW_NOT_CURATED','science_invocation':INV,
       'approval_sha256':APPROVAL,'result_sha256':sha(out/'result.json'),'solver_quality':qc,
       'frames':801,'frame_field_values_independently_checked':False,'source_bound_entries':len(s['source_inventory']),
       'initial_time_force_source':'bound original coefficient tables at t120, not zero padding',
       'force_rows_per_body':16000,'force_columns':13,'raw_force_sha256':raw_sha,
       'component_frames_verified':801,'maximum_pressure_plus_viscous_difference':max_sum,
       'maximum_component_total_vs_coefficient_difference':max_total,
       'component_inventory_sha256':hashlib.sha256(json.dumps(component_hashes,sort_keys=True).encode()).hexdigest(),
       'minimum_available_gib':minimum,'minimum_disk_gib':disk,'owned_container_absent':True,
       'curator_executed':False,'model_executed':False},indent=2))

if __name__=='__main__':main()
