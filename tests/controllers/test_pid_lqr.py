
import numpy as np
from env.sim import run

def test_pid_lqr_short_run():
    res = run(controller="pid_lqr", disturbance=False, animation=False,
              step_type="position_step", archive_results=False, config_variant=None)
    assert res.X.shape[0] == res.t.shape[0]
    assert np.max(np.abs(res.U)) <= 10.0 + 1e-6
    assert np.isfinite(res.metrics["mse_theta"])
