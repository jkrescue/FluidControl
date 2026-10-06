"""Derive one K1/A/B development profile from the exact executed FC-E063 worker.

Only profile/panel/model identity changes. Rollout, sufficient statistics,
official reader, precision order and resource lifecycle are kept byte-identical.
"""
import argparse
import ast
import hashlib
from pathlib import Path

PARENT_SHA='94e583182b453d8200a91700f1e68a997e47a9033debf1a9e08ed37cf297841f'

RECORDS='''def conversion_records(inputs):
    r=json.loads(inputs['conversion_result'].read_text())
    require(r['status']=='DEVELOPMENT_PHASE_CONVERSION_COMPLETE_NOT_ADMISSION'
        and r['frames']==96 and r['trajectories']==16 and r['endpoints']==80
        and r['split']=='development_already_opened' and r['source_unchanged'] is True
        and r['owned_containers_cleaned'] is True and r['model_loaded'] is False,
        'development conversion receipt')
    require(sha(inputs['selection'])==r['selection_sha256'],'selection binding')
    selection=json.loads(inputs['selection'].read_text())
    expected=[(phase,j) for phase in ('b01','b03') for j in range(0,701,100)]
    records=[]
    for phase in ('b01','b03'):
        panel=selection['phases'][phase]
        require(panel['split']=='development_already_opened' and panel['frames']==48,'phase panel')
        for row in panel['records']:
            require(row['branch']=='mpc' and row['phase']==phase,'controlled phase only')
            records.append(dict(row,branch=phase))
    require([(x['branch'],x['start_index']) for x in records]==expected,'fixed16 panel')
    require([(x['phase'],x['start_index']) for x in r['hdf']]==expected,'HDF order')
    for row in r['hdf']:
        path=Path(row['path'])
        require(row['frames']==6 and row['official_reader_verified'] is True
            and path.resolve().is_relative_to(inputs['conversion_result'].parent)
            and not path.is_symlink() and sha(path)==row['sha256'],'verified HDF identity')
    require(len({(x['mask_sha256'],x['x_sha256'],x['y_sha256']) for x in r['hdf']})==1,'common grid')
    return records,r['hdf']
'''

LOADER='''def load_evaluation_pair(s,manifest,cfg,device,load_dual_fno,build_model,load_bound_k1):
    label=s['candidate_label']
    if label=='K1':
        return load_bound_k1(manifest,cfg,device,load_dual_fno=load_dual_fno,build_model=build_model)
    expected_kind='FC_P064_ARM_'+label+'_CONTROLLED_AERO_FORCE_FNO'
    payload=json.loads(manifest.read_text())
    require(payload['kind']==expected_kind,'explicit P064 arm kind')
    adapter,identity=load_dual_fno(manifest,cfg,device,build_model=build_model,
        expected_manifest_sha256=s['inputs']['manifest']['sha256'])
    require(identity.payload['kind']==expected_kind,'loaded arm identity')
    # Source-pinned P064 loader independently enforces fixed P009 flow,
    # K1 aero parent, arm schedule/training semantics and every checkpoint SHA.
    return adapter.flow_model,adapter.aerodynamic_model,identity

'''

def derive(source):
    if hashlib.sha256(source.encode()).hexdigest()!=PARENT_SHA:raise ValueError('parent SHA')
    replacements=[
        ("STATUS = 'PROJECTED_POLICY_H1_H5_INFERENCE_EXECUTION_APPROVED'","STATUS = 'P064_DEVELOPMENT_H1_H5_EVALUATION_EXECUTION_APPROVED'"),
        ("'branches':['mpc','zero']","'branches':['b01','b03']"),
        ("require(s['inputs']['manifest']['sha256'] == K1_SHA and\n            s['inputs']['normalization']['sha256'] == NORM_SHA, 'frozen K1/norm')",
         "require(s.get('candidate_label') in ('K1','A','B'), 'fixed candidate label')\n    require(s['inputs']['normalization']['sha256'] == NORM_SHA, 'fixed norm')\n    require(s['candidate_label'] != 'K1' or s['inputs']['manifest']['sha256'] == K1_SHA, 'original K1')"),
        ("flow,aero,identity = load_bound_k1(inputs['manifest'],OmegaConf.load(inputs['config']),\n        torch.device('cuda:0'),load_dual_fno=load_dual_fno,build_model=build_model)",
         "flow,aero,identity = load_evaluation_pair(s,inputs['manifest'],OmegaConf.load(inputs['config']),\n        torch.device('cuda:0'),load_dual_fno,build_model,load_bound_k1)"),
        ("'status':'PROJECTED_POLICY_H1_H5_REPLAY_COMPLETE_NOT_ADMISSION'","'status':'P064_DEVELOPMENT_H1_H5_COMPLETE_NOT_ADMISSION', 'candidate_label':s['candidate_label'], 'split':'development_already_opened'"),
        ("'summary':core.sufficient_statistics(metric_rows),'branches':{","'summary':core.sufficient_statistics(metric_rows),'phases':{"),
        ("enumerate(('mpc','zero'))","enumerate(('b01','b03'))"),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('unique source anchor')
        source=source.replace(old,new)
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='conversion_records')
    lines=source.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[RECORDS]
    source=''.join(lines)
    source=source.replace('def tensor_digest(models):',LOADER+'def tensor_digest(models):',1)
    ast.parse(source)
    return source

def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();text=derive(a.parent.read_text())
    with a.output.open('x') as out:out.write(text)

if __name__=='__main__':main()
