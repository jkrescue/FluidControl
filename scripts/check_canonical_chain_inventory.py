"""Read-only inventory checker. Never imports scientific code or executes a stage."""
import argparse, hashlib, json
from pathlib import Path

def require(ok, message):
    if not ok: raise ValueError(message)
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inside(repo, value):
    p=Path(value);p=p if p.is_absolute() else repo/p
    require(p.resolve().is_relative_to(repo.resolve()), 'path outside repository: '+str(p))
    return p
def validate(repo, inventory, payloads=False):
    require(inventory['schema_version']==1 and inventory['execution_authorized'] is False,'inventory is not execution approval')
    checked=0;skipped=0
    for e in inventory['files']:
        path=inside(repo,e['path']);require(path.is_file(),'missing: '+str(path))
        if e['payload'] and not payloads:skipped+=1;continue
        require(sha(path)==e['sha256'],'SHA mismatch: '+str(path));checked+=1
    for name,e in inventory['training_families'].items():
        root=inside(repo,e['root']);names=sorted(p.name for p in (root/'train').glob('*.h5'))
        require(names==e['train_files'] and len(names)==e['expected_count'],'family membership: '+name)
        require(not any((root/x).exists() for x in ('validation','frozen_test','test')),'non-train directory in view')
    b=inventory['stages']['B_training'];source_root=inside(repo,b['source_root'])
    sources=json.loads((source_root/'source_manifest.json').read_text())
    require(len(sources)==434,'B source closure count')
    for path,digest in sources.items():require(sha(inside(repo,str(source_root/path)))==digest,'B closure SHA: '+path);checked+=1
    for name,binding in inventory['nested_bindings'].items():
        spec=json.loads(inside(repo,binding['file']).read_text());entries=spec[binding['field']]
        if name=='cfd_restart_tree':
            entries={str(Path(spec['source_restart'])/folder/path):digest for folder,items in entries.items() for path,digest in items.items()}
        if isinstance(entries,list):pairs=[(x['path'],x['sha256']) for x in entries]
        else:pairs=list(entries.items())
        for path,digest in pairs:
            require(sha(inside(repo,path))==digest,'nested SHA: '+path);checked+=1
    return dict(status='READ_ONLY_INVENTORY_PASS_NOT_EXECUTION',hash_checks=checked,payloads_existence_only=skipped,
                scope='No model/HDF decoding, imports, Docker or unit launch; installed package versions not independently checked')
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--verify-payload-bytes',action='store_true',help='Also stream large HDF/checkpoint bytes; no decoding')
    a=ap.parse_args();i=json.loads(a.inventory.read_text());print(json.dumps(validate(a.repo,i,a.verify_payload_bytes),indent=2))
if __name__=='__main__':main()
