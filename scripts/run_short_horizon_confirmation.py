"""Opt-in sealed confirmation worker; requires separately issued final approval.

Project wrapper around existing evaluator. No training or checkpoint selection.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import math
from pathlib import Path
import sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)

def validate_spec(spec):
    require(spec['status']=='SHORT_HORIZON_FROZEN_CONFIRMATION_EXECUTION_APPROVED' and spec['execution_authorized'] is True and spec['heldout_access_authorized'] is True,'not authorized to open heldout')
    require(spec['horizons']==[1,2,3,4,5] and spec['batch_size']==4 and spec['segment_stride']==25,'fixed evaluation')
    require(spec['optimizer_steps']==0 and spec['deadline_seconds']==1200,'scope')
    require(len(spec['cases'])==10 and len(set(spec['cases']))==10,'ten unique cases')
    expected={f'matched_start_acquisition_frozen_test_b{phase}_{action}' for phase in ('03','07') for action in ('m0375','m075','p0375','p075','zero')}
    require(set(spec['cases'])==expected,'fixed sealed panel')
    for path,digest in spec['source_files'].items():require(sha(path)==digest,'source identity')
    for path,digest in spec['runtime_files'].items():require(sha(path)==digest,'runtime identity')
    for key in ('freeze_receipt','split_manifest','seal','normalization','model_manifest','training_config','evaluation_config','evaluation_data_manifest','normalization_data_manifest'):
        item=spec['inputs'][key];require(sha(item['path'])==item['sha256'],key+' identity')
    frozen=json.loads(Path(spec['inputs']['freeze_receipt']['path']).read_text())
    require(frozen['k1_manifest_sha256']==spec['inputs']['model_manifest']['sha256'] and frozen['train_normalization_sha256']==spec['inputs']['normalization']['sha256'],'frozen candidate binding')
    manifest_path=Path(spec['inputs']['model_manifest']['path'])
    manifest=json.loads(manifest_path.read_text())
    for role in ('flow','aerodynamic'):
        record=manifest[role]
        for kind in ('model','state'):
            path=manifest_path.parent/record['checkpoint_relative_directory']/record[kind+'_file']
            require(path.resolve().is_relative_to(manifest_path.parent.resolve()) and not path.is_symlink(),'checkpoint confinement')
            require(sha(path)==record[kind+'_sha256'],'checkpoint bytes before sealed access')

def run(args):
    require(sha(args.approval)==args.approval_sha256,'approval hash')
    spec=json.loads(args.approval.read_text());validate_spec(spec)
    require(os.environ.get('SHORT_HORIZON_SUPERVISED')==args.approval_sha256,'supervisor required')
    require(spec['source_files'].get(str(Path(__file__).resolve()))==sha(__file__),'worker identity')
    out=Path(spec['output']);out.mkdir(exist_ok=False)
    import prepare_official_reader_evaluator as generator
    import official_evaluation_input as adapter
    for module in (generator,adapter):
        origin=str(Path(module.__file__).resolve())
        require(spec['source_files'].get(origin)==sha(origin),'actual imported input/generator origin')
    opt_in_source=generator.opt_in_source
    source=Path(spec['original_evaluator']).read_text()
    derived=opt_in_source(source)
    require(hashlib.sha256(derived.encode()).hexdigest()==spec['derived_evaluator_sha256'],'derived identity')
    # No import or payload access occurs before the exact final authorization.
    module_spec=importlib.util.spec_from_loader('confirmation_evaluator',loader=None)
    evaluator=importlib.util.module_from_spec(module_spec)
    exec(compile(derived,str(spec['original_evaluator'])+'[official-reader+precision]','exec'),evaluator.__dict__)
    original_preflight=evaluator.preflight_evaluation_paths
    manifest=json.loads(Path(spec['inputs']['split_manifest']['path']).read_text())
    root=Path(spec['frozen_root'])
    # Explicit derived CLI extension; all opens remain mode r, no mount or copy.
    def sealed_preflight(data,split,horizons,batch_size,stride=None):
        require(split=='frozen_test' and Path(data)==root.parent,'sealed split identity')
        require(horizons==[1,2,3,4,5] and batch_size==4 and stride==25,'fixed args')
        paths=[root/(case+'.h5') for case in sorted(spec['cases'])]
        require(set(manifest['cases'])==set(spec['cases']),'manifest exact set')
        require(set(p.name for p in root.glob('*.h5'))==set(p.name for p in paths),'no extra/missing cases')
        for p in paths:
            require(not p.is_symlink() and sha(p)==manifest['hdf5_sha256'][p.stem],'sealed HDF identity')
        checked=original_preflight(root.parent,'frozen_test',horizons,batch_size,stride)
        require(checked==paths,'preflight exact ordering')
        return paths
    evaluator.preflight_evaluation_paths=sealed_preflight
    sys.argv=[spec['original_evaluator'],'--data',str(root.parent),'--split','frozen_test',
      '--normalization-data',str(Path(spec['inputs']['normalization']['path']).parent),
      '--config',spec['inputs']['evaluation_config']['path'],
      '--dual-fno-manifest',spec['inputs']['model_manifest']['path'],
      '--expected-dual-fno-manifest-sha256',spec['inputs']['model_manifest']['sha256'],
      '--dual-training-config',spec['inputs']['training_config']['path'],
      '--fno-history-profile','p026_k1','--horizons','1','2','3','4','5',
      '--segment-stride','25','--evaluation-batch-size','4','--action-mode','observed',
      '--visualizations-per-horizon','0','--visualization-dir',str(out/'unused_visualizations'),
      '--output',str(out/'evaluation.json'),'--segment-metrics-output',str(out/'segments.json')]
    evaluator.main()
    evaluation=json.loads((out/'evaluation.json').read_text())
    segments=json.loads((out/'segments.json').read_text())['segments']
    require({r['case'] for r in evaluation['cases']}==set(spec['cases']) and len(evaluation['cases'])==10,'exact output cases')
    failed=0
    for case in evaluation['cases']:
        require(set(case['horizons'])=={'1','2','3','4','5'},'exact output horizons')
        for value in case['horizons'].values():
            failed+=value['failed_segments']
            require(value['segments']+value['failed_segments']==32,'segment accounting')
    def nonfinite(value):
        if isinstance(value,float):return int(not math.isfinite(value))
        if isinstance(value,dict):return sum(nonfinite(v) for v in value.values())
        if isinstance(value,list):return sum(nonfinite(v) for v in value)
        return 0
    invalid=nonfinite(evaluation)+nonfinite(segments)
    expected={(case,h,start) for case in spec['cases'] for h in range(1,6) for start in range(0,776,25)}
    require({(r['case'],r['horizon'],r['start']) for r in segments}==expected,'exact per-start coverage')
    coverage={'cases':10,'horizons_per_case':5,'expected_segments':1600,'reported_segments':len(segments),'failed_segments':failed,'nonfinite_numeric_leaves':invalid}
    with (out/'coverage.json').open('x') as f:json.dump(coverage,f,indent=2)
    require(failed==0 and invalid==0 and len(segments)==1600,'failed/nonfinite/incomplete confirmation; preserve outputs')
    with (out/'result.json').open('x') as f:
        json.dump({'status':'SHORT_HORIZON_CONFIRMATION_COMPLETE_NOT_ADMISSION','approval_sha256':args.approval_sha256,'original_split':'frozen_test','cli_split':'frozen_test','official_reader_used':True,'precision_override':'post-load highest/noTF32','optimizer_steps':0,'scientific_admission':False,'evaluation_sha256':sha(out/'evaluation.json'),'segments_sha256':sha(out/'segments.json')},f,indent=2)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--approval',type=Path,required=True);p.add_argument('--approval-sha256',required=True);p.add_argument('--execute',action='store_true');p.add_argument('--output')
    a=p.parse_args();require(a.execute,'explicit execution required');run(a)
