"""One existing-frame engineering comparison. Project code; never a control gate."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

CASE = "matched_start_acquisition_train_b00_zero"
VTU = f"cfd/tandem_cylinders/cases/{CASE}/VTK_curator/{CASE}_29600/internal.vtu"
DATA = "data/curated/tandem_cylinders_matched_start_full40_dev30_v1"
PINS = {
    VTU: "1e5021431a45ed2465e3fd932da57418e186a0da4b27548c18f7177c86fa2f2b",
    f"{DATA}/train/{CASE}.h5": "243caa79ac320b421adc1bf0c2cc830a32482dc758c3c0f9ce71d26171a61a01",
    f"{DATA}/normalization.json": "f1b4607e2eace8f8d3c2c9aa5dcfa642ed43f470ab62fe3e5c051cce0a292bc1",
    "scripts/sample_tandem_vtk_frame.py": "bbd04828a4de3c9f54b1c2deb8227c4e8616173ee945d905fed41d905af9cfa6",
    "scripts/p026_state_history.py": "2b5b37dc79211bb5c4985a1d44d25262080f503011766206c536938f49a6b64c",
    "src/fluid_control/online_current_frame.py": "2945d38773100ca05e66d4be268839fad7f2d78fca016ab867fdd5c7b742eaa4",
}
VERSIONS = {"torch":"2.14.1", "numpy":"2.5.3", "h5py":"3.16.0",
            "pyvista":"0.49.0", "physicsnemo-curator":"0.1.0", "nvidia-physicsnemo":"2.2.2"}
RUNTIME_PINS = {
    "physicsnemo/datapipes/readers/hdf5.py": "cafa65d615555e1e4b1d6cb58895983682aae826957765c105142b71e934caa0",
    "physicsnemo_curator/domains/mesh/sources/vtk.py": "a717a49d51c5695d306776d5df202f97e6ec918a0dcbd7c0e02437a182280b50",
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok: raise ValueError(message)


def differences(left, right):
    import numpy as np
    a,b=np.asarray(left),np.asarray(right)
    require(a.shape==b.shape and a.size>0, "comparison shape")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "comparison finite")
    d=a.astype(np.float64)-b.astype(np.float64)
    return {"exact_equal": bool(np.array_equal(a,b)), "max_abs":float(np.abs(d).max()),
            "rmse":float(np.sqrt(np.mean(d*d))), "values":int(d.size)}


def execute(repo, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="", "CUDA must be hidden")
    require(Path(sys.prefix).resolve()==(repo/".venv-curator-py312").resolve(), "exact Curator environment")
    versions={key:importlib.metadata.version(key) for key in VERSIONS}
    require(versions==VERSIONS, "runtime package versions")
    runtime_root=Path(sys.prefix)/"lib/python3.12/site-packages"
    require(all(sha(runtime_root/p)==h for p,h in RUNTIME_PINS.items()), "official source pins")
    require(not output.exists() and not output.is_symlink(), "exclusive output")
    require(output.parent.is_dir(), "output parent must exist")
    require(all(not p.is_symlink() for p in (output,*output.parents)), "output symlink")
    actual={p:sha(repo/p) for p in PINS}
    require(actual==PINS, "input/source hashes")
    output.mkdir()
    isolated=output/"single_vtu"/"frame"
    isolated.mkdir(parents=True)
    copied=isolated/"internal.vtu"
    shutil.copyfile(repo/VTU,copied)
    require(sha(copied)==PINS[VTU], "copied VTU bytes")
    copied.chmod(0o444); isolated.chmod(0o555); isolated.parent.chmod(0o555)
    env={**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"2",
         "PYTHONDONTWRITEBYTECODE":"1"}
    cmd=[sys.executable,str(repo/"scripts/sample_tandem_vtk_frame.py"),
         str(isolated.parent),str(output/"sample.npz")]
    with (output/"sampler.log").open("x") as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    import numpy as np
    import torch
    from physicsnemo.datapipes.readers.hdf5 import HDF5Reader
    import inspect
    require(sha(inspect.getfile(HDF5Reader))==RUNTIME_PINS["physicsnemo/datapipes/readers/hdf5.py"], "imported reader SHA")
    sys.path[:0]=[str(repo/"src"),str(repo/"scripts")]
    import fluid_control.online_current_frame as adapter
    require(sha(adapter.__file__)==PINS["src/fluid_control/online_current_frame.py"], "imported adapter SHA")
    reader=HDF5Reader(repo/DATA/"train"/f"{CASE}.h5", fields=["state","mask","omega","time"])
    try: frame,metadata=reader[0]
    finally: reader.close()
    with np.load(output/"sample.npz",allow_pickle=False) as archive:
        packet={k:archive[k].copy() for k in archive.files}
    require(float(frame["time"].reshape(-1)[0])==148.0, "HDF frame0 time")
    expected_mask=frame["mask"].cpu().numpy().astype(np.uint8)
    raw=(repo/DATA/"normalization.json").read_bytes()
    current=adapter.normalize_current(packet,expected_time=148.0,expected_mask=expected_mask,
        normalization_bytes=raw,expected_normalization_sha256=PINS[f"{DATA}/normalization.json"])
    stats=json.loads(raw)
    mean=torch.tensor(stats["state_mean"],dtype=torch.float32)[:,None,None]
    std=torch.tensor(stats["state_std"],dtype=torch.float32)[:,None,None]
    expected=(frame["state"].float()-mean)/std
    expected*=frame["mask"].float()
    now=float(frame["omega"].reshape(-1)[0]); require(now==0.0,"zero-case current action")
    packed=adapter.build_current_input(current,applied_omega_now=now,constrained_omega_next=now)
    action=(torch.tensor(now,dtype=torch.float32)/.75).expand(1,128,256)
    expected_input=torch.cat([expected,frame["mask"].float(),action,action])[None]
    valid=expected_mask[0].astype(bool)
    result={"status":"CURRENT_FRAME_CPU_COMPARISON_COMPLETE_NOT_ADMISSION",
        "case":CASE,"frame_index":0,"time":148.0,"source_sha256":actual,
        "runtime_versions":versions,"runtime_source_sha256":RUNTIME_PINS,
        "sampler_command":cmd,"sample_sha256":sha(output/"sample.npz"),
        "mask_equal":bool(np.array_equal(packet["mask"],expected_mask)),
        "grid_equal":True,"time_equal":bool(float(packet["time"][0])==148.0),
        "physical_valid":differences(packet["state"][:,valid],frame["state"].numpy()[:,valid]),
        "normalized_valid":differences(current["state"][0].numpy()[:,valid],expected.numpy()[:,valid]),
        "packed_input":differences(packed.numpy(),expected_input.numpy()),
        "model_loaded":False,"optimizer_steps":0,"control_actions_executed":0,
        "scientific_admission":False}
    require(all(sha(repo/p)==h for p,h in PINS.items()), "post-run input/source hashes")
    with (output/"result.json").open("x") as f: json.dump(result,f,indent=2,allow_nan=False); f.write("\n")
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({"status":"PREPARATION_ONLY", "execution_authorized":False,"input_pins":PINS}))
        return
    print(json.dumps(execute(a.repo,a.output),indent=2))


if __name__=="__main__": main()
