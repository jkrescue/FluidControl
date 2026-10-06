from pathlib import Path
import pytest
from two_segment_current_frame import run_two_segments, export_arguments


def callbacks(tmp_path,wrong_time=False, mutate=False):
    case=tmp_path/"case";case.mkdir()
    state={"time":148.0,"identity":"original"};calls=[]
    def configure(c,start,end,a,b):
        assert state["time"]==start and a==b==0
        calls.append(("configure",start,end))
    def solve(c,step):
        state["time"]=148.0+.1*step
        if mutate: state["identity"]="changed"
        return c/f"log{step}"
    def export(c,t,root):
        (root/"frame").mkdir(parents=True)
        (root/"frame"/"internal.vtu").touch()
        calls.append(("export",t))
    return dict(case=case,output=tmp_path/"out",latest_time=lambda c:state["time"],
                configure_interval=configure,solve=solve,check_segment=lambda *a:{"safe":True},
                export_frame=export,sample_frame=lambda root,p:p.touch(),
                normalize_frame=lambda p,t:{"time":148.0 if wrong_time else t},
                source_identity=lambda:state["identity"],guard=lambda:None),calls


def test_two_successive_fixed_zero_steps(tmp_path):
    args,calls=callbacks(tmp_path);r=run_two_segments(**args)
    assert [row["end"] for row in r["rows"]]==[148.1,148.2]
    assert calls==[("configure",148.0,148.1),("export",148.1),
                   ("configure",148.1,148.2),("export",148.2)]
    assert not r["scientific_admission"] and not r["model_loaded"]

@pytest.mark.parametrize("fault",["time","source","output"])
def test_fail_closed(tmp_path,fault):
    args,_=callbacks(tmp_path,wrong_time=fault=="time",mutate=fault=="source")
    if fault=="output": args["output"].mkdir()
    with pytest.raises((ValueError,FileExistsError)): run_two_segments(**args)

def test_export_exact_time_not_latest():
    args=export_arguments("/case",148.1,"VTK_step1")
    assert "-latestTime" not in args and args[args.index("-time")+1]=="148.1"
    with pytest.raises(ValueError): export_arguments("/case",149,"bad")
