import copy
from pathlib import Path
import pytest
import run_two_segment_current_frame as m


def row():
    return {"Name":"/"+m.NAME,"Image":"sha256:fixture","Config":{"Labels":{"fluid-control.engineering":m.LABEL.split("=",1)[1]}},
            "Mounts":[{"Type":"bind","Source":"/fresh/case","Destination":"/case","RW":True}],
            "HostConfig":{"Memory":8*2**30,"MemorySwap":8*2**30,"NanoCpus":2*10**9,
                          "ReadonlyRootfs":True,"NetworkMode":"none","Runtime":"runc","DeviceRequests":[]}}


def test_exact_owned_cpu_case():
    r=row();m.owned(r,Path("/fresh/case"),"sha256:fixture");m.isolation(r)

@pytest.mark.parametrize("fault",["mount","image","label","gpu","memory"])
def test_wrong_container_rejected(fault):
    r=copy.deepcopy(row())
    if fault=="mount":r["Mounts"][0]["Source"]="/original"
    if fault=="image":r["Image"]="other"
    if fault=="label":r["Config"]["Labels"]={}
    if fault=="gpu":r["HostConfig"]["DeviceRequests"]=[{}]
    if fault=="memory":r["HostConfig"]["Memory"]=0
    with pytest.raises(ValueError):
        m.owned(r,Path("/fresh/case"),"sha256:fixture");m.isolation(r)

def test_pending_cannot_execute():
    with pytest.raises(ValueError,match="separate approval"):
        m.execute({"status":"PREPARATION_ONLY"})

def test_copy_tree_rejects_symlinks(tmp_path):
    (tmp_path/"bad").symlink_to("/missing")
    with pytest.raises(ValueError,match="symlink"):m.tree(tmp_path)


@pytest.mark.parametrize("free,available,startup,valid",[(34,50,True,True),(3,50,True,True),
    (34,49.99,True,False),(22,22,False,True),(0,22,False,True),(30,21.99,False,False)])
def test_memory_contract(free,available,startup,valid):
    memory={"MemFree":free*2**30,"MemAvailable":available*2**30}
    if valid:m.validate_memory(memory,startup)
    else:
        with pytest.raises(ValueError):m.validate_memory(memory,startup)


def test_actual_canonical_transport_import_no_execution():
    import importlib.util
    import os
    repo=Path(os.environ.get("FC_REPO",str(Path(__file__).resolve().parents[1])))
    path=repo/"scripts/run_full40_canonical_ppo_openfoam_feedback.py"
    assert m.sha(path)==m.PINS["scripts/run_full40_canonical_ppo_openfoam_feedback.py"]
    spec=importlib.util.spec_from_file_location("actual_transport_import_only",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.configure_interval) and callable(module.check_segment)
    assert module.CONTROL_INTERVAL==.1 and module.SOLVER_DT==.005
    import fluid_control.canonical_joint_v1 as joint
    import fluid_control.openfoam_observation as observations
    assert m.sha(joint.__file__)==m.PINS["src/fluid_control/canonical_joint_v1.py"]
    assert m.sha(observations.__file__)==m.PINS["src/fluid_control/openfoam_observation.py"]


def test_solver_and_export_use_pinned_entrypoint():
    from two_segment_current_frame import export_arguments
    assert m.openfoam_exec("cid",["pimpleFoam","-case","/case"]) == [
        "docker","exec","cid","/openfoam/run","pimpleFoam","-case","/case"]
    assert m.openfoam_exec("cid",export_arguments("/case",148.1,"VTK_one")) == [
        "docker","exec","cid","/openfoam/run","foamToVTK","-case","/case",
        "-time","148.1","-fields","(U p)","-no-boundary","-name","VTK_one"]
    import ast
    tree=ast.parse(Path(m.__file__).read_text())
    functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
    for name in ("solve","export"):
        assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
                   and n.func.id=="openfoam_exec" for n in ast.walk(functions[name]))
