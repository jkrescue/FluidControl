"""Bounded CPU engineering transport. Never invokes a model, policy or admission gate."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from two_segment_current_frame import run_two_segments, export_arguments

IMAGE="opencfd/openfoam-default@sha256:33fb575aa9980d2bc42fd58c75ae698c489293ba30c991380fe3f899c622f319"
NAME="fluid-control-two-segment-frame-20261006"
LABEL="fluid-control.engineering=two-segment-frame-20261006"
PINS={
 "src/fluid_control/canonical_joint_v1.py":"138ab2b49ebddcbed2a24486c995b85a3a5ae226ee936ff2ed5de318496ee8bd",
 "src/fluid_control/openfoam_observation.py":"799d569c3f986a5e0e5cda95a3f07cd3b3993814ff0867763166549ef312293e",
 "scripts/run_full40_canonical_ppo_openfoam_feedback.py":"3dbb20106f45ed709e0c91731817540903db961555930177f81f0394104594cc",
 "scripts/sample_tandem_vtk_frame.py":"bbd04828a4de3c9f54b1c2deb8227c4e8616173ee945d905fed41d905af9cfa6",
 "src/fluid_control/online_current_frame.py":"2945d38773100ca05e66d4be268839fad7f2d78fca016ab867fdd5c7b742eaa4",
 "scripts/p026_state_history.py":"2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c",
 "cfd/tandem_cylinders/analyze_baseline.py":"fb2fb651ab498c41ea632581a255cb3e7e4f76d87ccb1daf712ffb03cb0bf7cb",
 "cfd/tandem_cylinders/make_expanded_control_dataset.py":"1c48ee0e5761ac90d5f04b4cb4dfeb6580b3f0ed2a9863c2dc7f860792020ec2",
 "cfd/tandem_cylinders/make_probe_feedback_case.py":"9960e38cbfc9afc6ec2a56ae01ce3da3974167b101a3274e33faa770728ebbbd",
 "scripts/run_tandem_phase_feedback_pair.py":"866b4dce33c7401eb447e897641d734828e04b10b38ccde8a0b724e8b5e65d3d",
}
VERSIONS={"torch":"2.14.1","numpy":"2.5.3","h5py":"3.16.0","pyvista":"0.49.0",
          "physicsnemo-curator":"0.1.0","nvidia-physicsnemo":"2.2.2"}
RUNTIME_PINS={"physicsnemo/datapipes/readers/hdf5.py":"cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0",
              "physicsnemo_curator/domains/mesh/sources/vtk.py":"a717a49d51c5695d306776d5df202f97e6ec918a0dcbd7c0e02437a182280b50"}


def memory_snapshot():
    return {l.split()[0].rstrip(":"):int(l.split()[1])*1024
            for l in Path("/proc/meminfo").read_text().splitlines()
            if l.startswith(("MemFree:","MemAvailable:"))}


def validate_memory(memory,startup=False):
    require(set(memory)=={"MemFree","MemAvailable"},"memory fields")
    # CPU-only: reclaimable cache is available capacity, not committed memory.
    # MemFree remains logged; this does not change any GPU/CUDA admission rule.
    available=50 if startup else 22
    require(memory["MemAvailable"]>=available*2**30,
            "startup memory reserve" if startup else "runtime memory buffer")


def validate_runtime(repo):
    require(Path(sys.prefix).resolve()==(repo/".venv-curator-py312").resolve(),"exact Curator environment")
    require({k:importlib.metadata.version(k) for k in VERSIONS}==VERSIONS,"runtime versions")
    root=Path(sys.prefix)/"lib/python3.12/site-packages"
    require(all(sha(root/p)==h for p,h in RUNTIME_PINS.items()),"official runtime source pins")


def require(ok,msg):
    if not ok: raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def save(p,obj):
    with p.open("x") as f: json.dump(obj,f,indent=2,allow_nan=False);f.write("\n")


def tree(root):
    require(root.is_dir() and not root.is_symlink(),"source directory")
    result={}
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink(),"source symlink")
        if p.is_dir(): continue
        require(p.is_file(),"source nonregular")
        result[str(p.relative_to(root))]=sha(p)
    return result


def inspect(cid):
    return json.loads(subprocess.check_output(["docker","inspect",cid],text=True,timeout=10))[0]


def owned(row,case,image):
    require(row["Name"]=="/"+NAME and row["Image"]==image,"container identity")
    require(row["Config"].get("Labels",{}).get("fluid-control.engineering")==LABEL.split("=",1)[1],"owned label")
    mounts=[(r["Source"],r["Destination"],r["RW"]) for r in row["Mounts"] if r["Type"]=="bind"]
    require(mounts==[(str(case),"/case",True)],"exclusive case mount")


def isolation(row):
    h=row["HostConfig"]
    require(h["Memory"]==8*2**30 and h["MemorySwap"]==8*2**30 and h["NanoCpus"]==2*10**9,"container resource limits")
    require(h["ReadonlyRootfs"] and h["NetworkMode"]=="none" and h["Runtime"]=="runc" and not h.get("DeviceRequests"),"CPU isolation")


def openfoam_exec(cid, arguments):
    """Pinned image entrypoint sources /openfoam/profile.rc before exec."""
    return ["docker","exec",cid,"/openfoam/run",*arguments]


def execute(spec):
    require(spec["status"]=="TWO_SEGMENT_CPU_EXECUTION_APPROVED","separate approval required")
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="","CUDA hidden")
    repo=Path(spec["repo"]); output=Path(spec["output"])
    validate_runtime(repo)
    validate_memory(memory_snapshot(),startup=True)
    require(not output.exists() and all(not p.is_symlink() for p in (output,*output.parents)),"exclusive output path")
    require(output.parent.is_dir(),"existing output parent")
    require(spec["source_code_sha256"]==PINS,"exact reviewed source map")
    require(all(sha(repo/p)==h for p,h in PINS.items()),"source SHA")
    require(sha(Path(__file__))==spec["driver_sha256"] and sha(Path(__file__).with_name("two_segment_current_frame.py"))==spec["sequencer_sha256"],"driver identity")
    source=repo/"cfd/tandem_cylinders/cases/tandem_backward_dt005"
    baseline={part:tree(source/part) for part in ("constant","system","148")}
    require(baseline==spec["source_restart_tree_sha256"],"approved restart/config tree")
    validate_memory(memory_snapshot(),startup=True)
    image=subprocess.check_output(["docker","image","inspect",IMAGE,"--format","{{.Id}}"],text=True,timeout=10).strip()
    require(image==spec["image_id"],"actual pinned image ID")
    names=subprocess.check_output(["docker","ps","-a","--format","{{.Names}}"],text=True,timeout=10).splitlines()
    require(NAME not in names,"owned container name already exists")
    output.mkdir();case=output/"case";case.mkdir()
    for part in baseline: shutil.copytree(source/part,case/part)
    require({p:tree(case/p) for p in baseline}==baseline,"copy exact bytes")
    sys.path[:0]=[str(repo/"src"),str(repo/"scripts")]
    modspec=importlib.util.spec_from_file_location("reviewed_cfd",repo/"scripts/run_full40_canonical_ppo_openfoam_feedback.py")
    transport=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(transport)
    transport.substitute(case/"system/controlDict","writeInterval",.1)
    import numpy as np
    from fluid_control.online_current_frame import normalize_current,build_current_input
    norm=repo/"data/curated/tandem_cylinders_matched_start_full40_dev30_v1/normalization.json"
    require(sha(norm)==spec["normalization_sha256"],"normalization identity")
    reference=Path(spec["reference_sample"])
    require(sha(reference)=="b9bdf87db1b4744633fb3c175050449952a6661ec95676a055d7f62111f80bc9","reference mask packet")
    with np.load(reference,allow_pickle=False) as f: mask=f["mask"].copy()
    started=time.monotonic();cid=None
    def guard():
        memory=memory_snapshot()
        with (output/"memory.jsonl").open("a") as f:f.write(json.dumps({"elapsed":time.monotonic()-started,**memory})+"\n")
        validate_memory(memory)
        require(time.monotonic()-started<300,"overall deadline")
    def run(cmd,log):
        with log.open("x") as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:guard();time.sleep(.25)
                require(proc.returncode==0,f"command failed: {cmd[0]}")
            finally:
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=5)
    def stop_signal(*_):raise RuntimeError("termination requested")
    old={s:signal.signal(s,stop_signal) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        validate_memory(memory_snapshot(),startup=True)
        guard()
        cmd=["docker","create","--name",NAME,"--label",LABEL,"--runtime","runc","--network","none","--read-only","--memory","8g","--memory-swap","8g","--cpus","2","--pids-limit","128","--cap-drop","ALL","--security-opt","no-new-privileges","--tmpfs","/tmp:rw,nosuid,nodev,size=512m","--user",f"{os.getuid()}:{os.getgid()}","--mount",f"type=bind,src={case},dst=/case","--workdir","/case",IMAGE,"sh","-c","while :; do sleep 3600; done"]
        save(output/"create_command.json",cmd)
        cid=subprocess.check_output(cmd,text=True,timeout=20).strip()
        row=inspect(cid);owned(row,case,image);isolation(row);save(output/"container_created.json",row)
        subprocess.run(["docker","start",cid],check=True,timeout=10,stdout=subprocess.DEVNULL)
        def solve(c,step):
            log=output/f"solver_{step}.log";run(openfoam_exec(cid,["pimpleFoam","-case","/case"]),log);return log
        def export(c,t,destination):
            name=f"VTK_endpoint_{str(t).replace('.','_')}"
            require(not (case/name).exists(),"stale case export")
            run(openfoam_exec(cid,export_arguments("/case",t,name)),output/f"export_{t}.log")
            shutil.copytree(case/name,destination)
        def sample(root,path):
            run([str(repo/".venv-curator-py312/bin/python"),str(repo/"scripts/sample_tandem_vtk_frame.py"),str(root),str(path)],path.with_suffix(".log"))
        def normalize(path,t):
            with np.load(path,allow_pickle=False) as f: packet={k:f[k].copy() for k in f.files}
            current=normalize_current(packet,expected_time=t,expected_mask=mask,normalization_bytes=norm.read_bytes(),expected_normalization_sha256=spec["normalization_sha256"])
            packed=build_current_input(current,applied_omega_now=0.,constrained_omega_next=0.)
            return {"time":current["time"],"sample_sha256":sha(path),"input_shape":list(packed.shape),"input_sha256":hashlib.sha256(packed.numpy().tobytes()).hexdigest()}
        result=run_two_segments(case=case,output=output/"segments",latest_time=transport.latest_time,
            configure_interval=transport.configure_interval,solve=solve,check_segment=transport.check_segment,
            export_frame=export,sample_frame=sample,normalize_frame=normalize,
            source_identity=lambda:{p:tree(source/p) for p in baseline},guard=guard)
        save(output/"result.json",result)
    finally:
        for s in old:signal.signal(s,signal.SIG_IGN)
        # Creation can finish daemon-side after a client timeout; recover exact owned name.
        if cid is None:
            for _ in range(6):
                names=subprocess.check_output(["docker","ps","-a","--format","{{.Names}}"],text=True,timeout=10).splitlines()
                if NAME in names:
                    row=inspect(NAME);owned(row,case,image);cid=row["Id"];break
                time.sleep(1)
        errors=[]
        if cid:
            row=inspect(cid);owned(row,case,image)
            try:subprocess.run(["docker","stop","--timeout","5",cid],check=True,timeout=10,stdout=subprocess.DEVNULL)
            except subprocess.SubprocessError as e:errors.append(repr(e))
            try:save(output/"container_terminal.json",inspect(cid))
            except Exception as e:errors.append(repr(e))
            try:subprocess.run(["docker","rm","-f",cid],check=True,timeout=10,stdout=subprocess.DEVNULL)
            except subprocess.SubprocessError as e:errors.append(repr(e))
            ids=subprocess.check_output(["docker","ps","-aq","--no-trunc"],text=True,timeout=10).split()
            if cid in ids:errors.append("owned CID remains")
        save(output/"cleanup.json",{"cid":cid,"errors":errors})
        for s,h in old.items():signal.signal(s,h)
        require(not errors,"cleanup failed")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--spec",type=Path,required=True);p.add_argument("--spec-sha256",required=True);p.add_argument("--execute",action="store_true")
    a=p.parse_args();require(sha(a.spec)==a.spec_sha256,"spec SHA")
    if a.execute:execute(json.loads(a.spec.read_text()))
    else:print("PREPARATION_ONLY_NOT_EXECUTED")
