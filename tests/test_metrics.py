
from env.metrics import mse, mae
import numpy as np

def test_mse_mae_basic():
    y = np.array([0.,1.,2.]); r = np.array([0.,1.,2.])
    assert mse(y,r)==0.0 and mae(y,r)==0.0
    y2 = np.array([1.,2.,3.])
    assert abs(mse(y2,r) - 1.0) < 1e-12
    assert abs(mae(y2,r) - 1.0) < 1e-12
