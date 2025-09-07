
import pytest
from env.sim import run

def test_lmpc_smoke():
    pytest.importorskip("cvxpy")
    res = run(controller="lmpc", disturbance=False, animation=False,
              step_type="position_step", archive_results=False, config_variant=None)
    assert res.X.shape[0] == res.t.shape[0]
