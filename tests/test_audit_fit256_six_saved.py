import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).parents[1]/'scripts'/'audit_fit256_six_saved.py';s=importlib.util.spec_from_file_location('audit',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_weighted_error_and_total_cd_cancellation():
    y=np.zeros((100,4));pred=np.tile([1.,2.,-1.,4.],(100,1));v=m.calc(pred,y,[2,3,2,5])
    assert v['balanced']==10.75
    assert v['total_cd']==dict(mae=0.,rmse=0.)
    assert v['rear_cl']==dict(mae=20.,rmse=20.)
    assert v['stats']['bias_mse']==400 and v['stats']['rms_error_mse']==0
def test_tail_is_last62_and_centered():
    y=np.zeros((100,4));pred=y.copy();pred[:38,3]=100;pred[38:,3]=np.tile([-1,1],31)
    v=m.calc(pred,y,[1,1,1,2]);assert v['stats']['bias_mse']==0
    assert v['stats']['rms_error_mse']==4 and v['stats']['centered_residual_mse']==4
