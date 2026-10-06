"""Source/metadata-only pending budget profile; never authorizes execution."""
import argparse
import hashlib
import json
from pathlib import Path
from train_exploratory_diverse_h5_32768_ppo import PROTOCOL

PARENT_SHA='760e1f9e81494bdd8c3742cd0ce77e168b0df2bf288bd04546412096721f41e2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare(parent):
    if sha(parent)!=PARENT_SHA:raise ValueError('exact prior diverse approval required')
    s=json.loads(Path(parent).read_text())
    old=dict(s['protocol']);old['timesteps']=32768
    if old!=PROTOCOL:raise ValueError('unintended protocol change')
    for key in ('lead_statement','lead_approval','approval_statement','execution_approval','source_root',
                'source_manifest_sha256','preparation_spec_sha256'):
        s.pop(key,None)
    s.update(status='PREPARATION_ONLY_NOT_EXECUTION_APPROVAL',execution_authorized=False,
             reviewed_by_lead=False,protocol=PROTOCOL,parent_budget_approval_sha256=PARENT_SHA,
             interpretation='Fresh initialization; only4096 to32768 budget changes; no execution authorized')
    here=Path(__file__).resolve().parent
    for name in ('train_exploratory_diverse_h5_32768_ppo.py','supervise_exploratory_diverse_h5_32768_ppo.py',Path(__file__).name):
        p=here/name;s['source_files'][str(p)]=sha(p)
    s['runner']=str(here/'train_exploratory_diverse_h5_32768_ppo.py')
    s['pythonpath']=[str(here),*s['pythonpath']]
    s['supervision_output']=str(Path(s['repo'])/'artifacts/exploratory_diverse_h5_32768_ppo_training_20261006') if 'repo' in s else '/workspace/fluid_control/artifacts/exploratory_diverse_h5_32768_ppo_training_20261006'
    s['output']=str(Path(s['supervision_output'])/'payload')
    return s

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    with a.output.open('x') as f:json.dump(prepare(a.parent),f,indent=2,allow_nan=False)
