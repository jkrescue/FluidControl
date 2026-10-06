import ast
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import convert_development_phase_frames as m

def test_pending_rejected_before_io():
    with pytest.raises(ValueError,match='not approved'):
        m.validate({'status':'PENDING','execution_authorized':False})

def test_exact_phase_contract():
    spec=dict(status=m.STATUS,execution_authorized=True,phases=['b01','b03'],
        phase_inputs={},frames=96,trajectories=16,split='development_already_opened',
        resources=m.RESOURCES,runtime_versions=m.VERSIONS,python='.venv-curator-py312/bin/python')
    for phase in spec['phases']:
        root=f'artifacts/exploratory_projected_32768_ppo_{phase}_long_cfd_20261006'
        spec['phase_inputs'][phase]=dict(source_root=root,progress={'path':root+'/progress.json'},
            result={'path':root+'/result.json'},selected_source_inventory={'case_mpc/constant/example':{}})
    m.validate(spec)
    spec['split']='train'
    with pytest.raises(ValueError,match='scope'):m.validate(spec)

def test_helper_reuse_and_phase_output_isolation():
    source=Path(m.__file__).read_text();tree=ast.parse(source)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    execute=ast.get_source_segment(source,functions['execute'])
    for name in ['selected_inventory','copy_selected','export_branch','sample_branch','pack_and_verify_mini_hdf']:
        assert name in execute
    assert "child=output/phase" in execute and "'development-'+phase" in execute
    assert 'rmtree' not in source and 'chmod' not in source and '.backward' not in source
    assert execute.index("monitor.check()") < execute.index("output/'result.json'")
