
import numpy as np
from env.sim import run

def test_ts_fuzzy_short_run():
    res = run(controller="ts_fuzzy", disturbance=False, animation=False,
              step_type="position_step", archive_results=False, config_variant=None)
    assert res.X.shape[0] == res.t.shape[0]
    assert np.isfinite(res.metrics["mse_theta"])
