"""Preparation-only project sequencer for a separately owned CPU CFD container.

No PPO or FNO model. Container creation, exact source copying/identity proof,
and resource/cleanup supervision belong to a separately approved launch layer.
This module deliberately has no execution CLI until those bindings are reviewed.
"""
from pathlib import Path

ENDPOINTS = ((148.0,148.1),(148.1,148.2))


def run_two_segments(*, case, output, latest_time, configure_interval, solve,
                     check_segment, export_frame, sample_frame, normalize_frame,
                     source_identity, guard):
    """Callbacks must bind reviewed existing transports; synthetic tests only so far.

    solve(case,step) returns completed log; export_frame(case,t,destination)
    must export ONLY t; sample_frame(root,npz) is the unchanged Curator sampler.
    normalize_frame(npz,t) calls canonical adapter and returns a JSON report.
    """
    case,output=Path(case),Path(output)
    if output.exists() or output.is_symlink(): raise FileExistsError(output)
    if not case.is_dir(): raise FileNotFoundError(case)
    baseline=source_identity()
    output.mkdir()
    rows=[]
    for step,(start,end) in enumerate(ENDPOINTS,1):
        guard()
        if abs(latest_time(case)-start)>2e-6: raise ValueError("stale/missing restart")
        configure_interval(case,start,end,0.0,0.0)
        log=solve(case,step)
        health=check_segment(case,log,end)
        if abs(latest_time(case)-end)>2e-6: raise ValueError("CFD endpoint mismatch")
        vtk=output/f"vtk_step_{step}"
        if vtk.exists() or vtk.is_symlink(): raise FileExistsError(vtk)
        export_frame(case,end,vtk)
        files=list(vtk.glob("*/internal.vtu"))
        if len(files)!=1 or not files[0].is_file() or files[0].is_symlink():
            raise ValueError("exactly one fresh endpoint VTU required")
        sample=output/f"frame_{step}.npz"
        if sample.exists(): raise FileExistsError(sample)
        sample_frame(vtk,sample)
        packet=normalize_frame(sample,end)
        if abs(packet["time"]-end)>1e-5: raise ValueError("stale sampled endpoint")
        if source_identity()!=baseline: raise ValueError("original restart changed")
        rows.append({"step":step,"start":start,"end":end,"applied_now":0.0,
                     "applied_next":0.0,"health":health,"frame":packet})
        guard()
    return {"status":"TWO_SEGMENT_ENGINEERING_ONLY_NOT_CONTROL_ADMISSION",
            "rows":rows,"model_loaded":False,"policy_loaded":False,
            "source_unchanged":True,"scientific_admission":False}


def export_arguments(case_in_container, endpoint, vtk_name):
    """Existing foamToVTK export, exact time rather than ambiguous latestTime."""
    if endpoint not in (148.1,148.2): raise ValueError("fixed endpoint only")
    return ["foamToVTK","-case",case_in_container,"-time",f"{endpoint:g}",
            "-fields","(U p)","-no-boundary","-name",vtk_name]
