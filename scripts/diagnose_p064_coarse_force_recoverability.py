"""Execute the preregistered 27-frame coarse-force diagnostic; no fitting."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path

import h5py
import numpy as np
from physicsnemo.datapipes.readers.hdf5 import HDF5Reader


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(value, message):
    if not value:
        raise ValueError(message)


def mem_available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    raise ValueError("MemAvailable missing")


def load_module(path: Path, expected: str):
    require(digest(path) == expected, "helper SHA")
    spec = importlib.util.spec_from_file_location("p064_coarse_force", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_geometry_sources(spec):
    sources = spec["geometry_sources"]
    for item in sources.values():
        require(digest(Path(item["path"])) == item["sha256"], "geometry source SHA")
    control = Path(sources["control_dict"]["path"]).read_text()
    require(re.search(r"rhoInf\s+1\s*;", control) is not None, "rhoInf")
    require(len(re.findall(r"magUInf\s+1\s*;", control)) == 2, "magUInf")
    require(len(re.findall(r"Aref\s+0\.1\s*;", control)) == 2, "Aref")
    require(len(re.findall(r"dragDir\s+\(1\s+0\s+0\)\s*;", control)) == 2, "drag direction")
    require(len(re.findall(r"liftDir\s+\(0\s+1\s+0\)\s*;", control)) == 2, "lift direction")
    require(re.search(r"CofR\s+\(10\s+7\.5\s+0\)\s*;", control), "front center")
    require(re.search(r"CofR\s+\(15\s+7\.5\s+0\)\s*;", control), "rear center")
    require(re.search(r"writePrecision\s+9\s*;", control), "writePrecision")
    p_text = Path(sources["pressure_header"]["path"]).read_text()
    require(re.search(r"dimensions\s+\[0\s+2\s+-2\s+0\s+0\s+0\s+0\]\s*;", p_text), "p is not kinematic")
    points = Path(sources["points"]["path"]).read_text()
    vectors = re.findall(r"\(\s*[-+0-9.eE]+\s+[-+0-9.eE]+\s+([-+0-9.eE]+)\s*\)", points)
    z = np.asarray([float(x) for x in vectors], dtype=np.float64)
    require(len(z) > 0 and z.min() == 0.0 and z.max() == 0.1, "mesh span")
    return {"rho": 1.0, "u_inf": 1.0, "span": 0.1, "area_ref": 0.1,
            "pressure_dimensions": "[0 2 -2 0 0 0 0]", "write_precision": 9}


def metrics(values):
    array = np.asarray(values, dtype=np.float64)
    require(array.ndim == 2 and array.shape[1] == 4 and np.isfinite(array).all(), "metric rows")
    return {"mae": np.mean(np.abs(array), axis=0).tolist(),
            "rmse": np.sqrt(np.mean(array**2, axis=0)).tolist(),
            "bias": np.mean(array, axis=0).tolist()}


def array_digest(values) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def preflight(spec, helper, args):
    require(spec["status"] in {"P064_COARSE_ROI_FORCE_RECOVERABILITY_CPU_PENDING",
                               "P064_COARSE_ROI_FORCE_RECOVERABILITY_CPU_EXECUTION_APPROVED"}, "status")
    require(digest(Path(spec["worker"]["path"])) == spec["worker"]["sha256"], "worker SHA")
    require(digest(Path(spec["supervisor"]["path"])) == spec["supervisor"]["sha256"], "supervisor SHA")
    require(digest(Path(spec["plan"]["path"])) == spec["plan"]["sha256"], "plan SHA")
    require(Path(spec["planned_output"]).resolve() == args.output.resolve(), "output binding")
    require(not args.output.exists(), "output exists")
    require(spec["resources"] == {"cuda_visible_devices":"","cpu_quota":1,"memory_max_gib":2,
        "memory_swap_max":0,"runtime_seconds":120,"startup_mem_available_gib":50,
        "worker_timeout_seconds":100,"runtime_mem_available_gib":22}, "resources")
    require(spec["geometry"] == {"front_center":[10.0,7.5],"rear_center":[15.0,7.5],
        "radius":0.5,"rho":1.0,"u_inf":1.0,"span":0.1,"area_ref":0.1,
        "pressure_is_kinematic":True,"pressure_dimensions":"[0 2 -2 0 0 0 0]",
        "probe_radius":"R + 2*max(dx,dy)","integration_radius":"R","angles":128,
        "force_sign":"-integral(p*n*span*R*dtheta)",
        "normalization":"coefficient denominator 0.5*rho*u_inf^2*area_ref"}, "geometry contract")
    require(tuple(spec["selection"]["cases"]) == helper.CASES
            and tuple(spec["selection"]["frames"]) == helper.FRAME_INDICES, "selection binding")
    split = read_json(Path(spec["train_split"]["path"]))
    require(digest(Path(spec["train_split"]["path"])) == spec["train_split"]["sha256"], "split SHA")
    validate_geometry_sources(spec)
    primary = spec["geometry_sources"]
    static = {"system/controlDict":primary["control_dict"]["sha256"],
              "constant/transportProperties":primary["transport"]["sha256"],
              "constant/polyMesh/boundary":primary["boundary"]["sha256"],
              "constant/polyMesh/points":primary["points"]["sha256"]}
    for case in helper.CASES:
        cfg = spec["cases"][case]
        root = Path(cfg["case_root"])
        require(root.name == case, "case root identity")
        require(split["hdf5_sha256"][case] == cfg["hdf_sha256"], "HDF manifest binding")
        require(Path(cfg["hdf_path"]).name == case+".h5", "HDF path identity")
        require(cfg["expected_times"] == [148.0+10.0*i for i in range(9)], "fixed times")
        require(set(cfg["properties"]) == {str(x) for x in helper.FRAME_INDICES}, "property frames")
        for rel, expected in static.items():
            require(digest(root/rel) == expected, "case static geometry differs")
        for index, expected_time in zip(helper.FRAME_INDICES, cfg["expected_times"]):
            item = cfg["properties"][str(index)]
            path = Path(item["path"])
            require(path == root/f"{expected_time:g}"/"uniform/functionObjects/functionObjectProperties", "property path/time")
            require(digest(path) == item["sha256"], "property SHA")
    return split


def execute(spec, helper, spec_sha256):
    split = read_json(Path(spec["train_split"]["path"]))
    require(digest(Path(spec["train_split"]["path"])) == spec["train_split"]["sha256"], "split SHA")
    geometry_evidence = validate_geometry_sources(spec)
    rows = []
    for case in helper.CASES:
        cfg = spec["cases"][case]
        require(split["hdf5_sha256"][case] == cfg["hdf_sha256"], "HDF manifest binding")
        hdf = Path(cfg["hdf_path"])
        with h5py.File(hdf, "r") as handle:
            require(str(handle.attrs["case"]) == case and str(handle.attrs["split"]) == "train", "HDF identity")
            require(handle["state"].shape == (801,3,128,256), "state shape")
            x, y = handle["x"][:], handle["y"][:]
        selected = helper.read_selected_frames(HDF5Reader, hdf)
        require([x[0] for x in selected] == list(helper.FRAME_INDICES), "selected indices")
        for index, sample, _ in selected:
            require(mem_available_gib() >= spec["resources"]["runtime_mem_available_gib"],
                    "runtime memory before selected frame")
            expected_time = 148.0 + 0.1*index
            require(expected_time == cfg["expected_times"][helper.FRAME_INDICES.index(index)], "derived time")
            prop = cfg["properties"][str(index)]
            path = Path(prop["path"])
            require(digest(path) == prop["sha256"], "property SHA")
            parsed = helper.parse_force_components(path.read_text())
            state = sample["state"].detach().cpu().numpy()
            mask = sample["mask"].detach().cpu().numpy()[0]
            force = sample["force"].detach().cpu().numpy()
            time = float(sample["time"].detach().cpu().reshape(-1)[0])
            check = helper.validate_alignment(time, expected_time, force, parsed)
            total, pressure, viscous = helper.component_vectors(parsed)
            proxy = np.concatenate([
                helper.coarse_pressure_coefficients(state[2], mask, x, y,
                    helper.ForceGeometry(10,7.5,span=.1,area_ref=.1)),
                helper.coarse_pressure_coefficients(state[2], mask, x, y,
                    helper.ForceGeometry(15,7.5,span=.1,area_ref=.1)),
            ])
            denom = np.abs(pressure)+np.abs(viscous)
            require(np.all(denom > 0), "zero component denominator")
            rows.append({"case":case,"frame":index,"expected_time":expected_time,"hdf_time":time,
                "inputs":{"hdf_path":str(hdf),"property_path":str(path),
                    "content_sha256":{"state":array_digest(state),"force":array_digest(force),
                        "time":array_digest(sample["time"].detach().cpu().numpy()),
                        "mask":array_digest(mask),"x":array_digest(x),"y":array_digest(y)}},
                "omega":float(sample["omega"].detach().cpu().reshape(-1)[0]),
                "total":total.tolist(),"pressure":pressure.tolist(),"viscous":viscous.tolist(),
                "pressure_proxy":proxy.tolist(),"pressure_proxy_error":(proxy-pressure).tolist(),
                "viscous_fraction":(np.abs(viscous)/denom).tolist(),
                "cancellation_ratio":(np.abs(total)/denom).tolist(),"alignment":check})
    require(len(rows) == 27, "27 fixed rows")
    proxy_errors = [x["pressure_proxy_error"] for x in rows]
    by = {(x["case"],x["frame"]):x for x in rows}
    response_errors=[]
    for action in (helper.CASES[0],helper.CASES[2]):
        for frame in helper.FRAME_INDICES:
            active, zero = by[(action,frame)], by[(helper.CASES[1],frame)]
            predicted=np.asarray(active["pressure_proxy"])-np.asarray(zero["pressure_proxy"])
            target=np.asarray(active["pressure"])-np.asarray(zero["pressure"])
            response_errors.append((predicted-target).tolist())
    return {"status":"P064_COARSE_ROI_FORCE_RECOVERABILITY_CPU_COMPLETE_NOT_ADMISSION",
            "identity":{"spec_sha256":spec_sha256,"driver_sha256":digest(Path(__file__)),
                "helper_sha256":spec["helper"]["sha256"],"plan_sha256":spec["plan"]["sha256"],
                "train_split_path":spec["train_split"]["path"],
                "train_split_sha256":spec["train_split"]["sha256"],
                "geometry_sources":spec["geometry_sources"]},
            "selection":{"cases":list(helper.CASES),"frames":list(helper.FRAME_INDICES),"rows":27},
            "geometry_evidence":geometry_evidence,"rows":rows,
            "pressure_proxy_metrics":metrics(proxy_errors),
            "action_minus_zero_pressure_proxy_metrics":metrics(response_errors),
            "model_loaded":False,"optimizer_steps":0,"cfd_executed":False,"hdf_payload_rehashed":False,
            "interpretation":"fixed offset-ring proxy; not exact wall traction or representation causality"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--spec",type=Path,required=True)
    parser.add_argument("--spec-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    require(digest(args.spec)==args.spec_sha256,"spec SHA")
    spec=read_json(args.spec)
    helper=load_module(Path(spec["helper"]["path"]),spec["helper"]["sha256"])
    preflight(spec, helper, args)
    if not args.execute:
        print("P064_COARSE_FORCE_PREPARATION_ONLY_NO_DATA_READ")
        return
    require(spec["execution_authorized"] is True and os.environ.get("P064_COARSE_FORCE_TOKEN")=="EXECUTE_APPROVED_P064_COARSE_FORCE", "not authorized")
    require(not args.output.exists(),"output exists")
    result=execute(spec,helper,args.spec_sha256)
    args.output.mkdir(parents=True)
    temporary=args.output/"result.json.tmp"
    temporary.write_text(json.dumps(result,indent=2)+"\n")
    temporary.replace(args.output/"result.json")
    print(json.dumps({"status":result["status"],"result_sha256":digest(args.output/"result.json")}))


if __name__=="__main__":
    main()
