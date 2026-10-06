"""Prepare non-authorizing source/actual-policy metadata; no payload deserialization."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluate_diverse_policy_h5_comparison import POLICIES
REPO=Path('/workspace/fluid_control')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    parent=REPO/'docs/EXPLORATORY_DIVERSE_H5_32768_PPO_R2_APPROVAL_20261006.json'
    if sha(parent)!='1cd1d5182fd7e7a3eed11060e7a6ffe9fad840c776f021f239f059525a515132':raise ValueError('parent approval')
    s=json.loads(parent.read_text());s['reference_trainer']=s['runner']
    for k in ('lead_statement','lead_approval','source_root','source_manifest_sha256','preparation_spec_sha256'):s.pop(k,None)
    s.update(status='PREPARATION_ONLY_NOT_EXECUTION_APPROVAL',execution_authorized=False,reviewed_by_lead=False,
             interpretation='Two fixed terminal policies evaluated deterministically on same24trainH5; no training/CFD/admission')
    here=Path(__file__).resolve().parent
    for name in ('evaluate_diverse_policy_h5_comparison.py','supervise_diverse_policy_h5_comparison.py',Path(__file__).name):
        p=here/name;s['source_files'][str(p)]=sha(p)
    s['runner']=str(here/'evaluate_diverse_policy_h5_comparison.py')
    s['pythonpath']=[str(here),*s['pythonpath']]
    s['supervision_output']=str(REPO/'artifacts/diverse_policy_h5_comparison_20261006')
    s['output']=str(Path(s['supervision_output'])/'payload')
    s['evaluation_policies']={}
    for label,root in [('4096','exploratory_diverse_h5_ppo_training_20261006'),('32768','exploratory_diverse_h5_32768_ppo_training_20261006_r2')]:
        s['evaluation_policies'][label]={}
        for key,file,digest in zip(('policy','vecnormalize','result'),('ppo_final.zip','vecnormalize.pkl','result.json'),POLICIES[label]):
            p=REPO/'artifacts'/root/'payload'/file
            s['evaluation_policies'][label][key]={'path':str(p),'sha256':digest}
    s['evaluation_protocol']={'cases':24,'horizon':5,'policies':['4096','32768'],'deterministic':True,
        'optimizer_steps':0,'deadline_seconds':240,'maximum_allocated_gib':6,'macro_equal_case_weight':True}
    return s
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(prepare(),f,indent=2,allow_nan=False)
