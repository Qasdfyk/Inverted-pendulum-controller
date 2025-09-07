
import pytest
from env.sim import run

def test_mpc_altcost_smoke():
    pytest.importorskip("cvxpy")
    res = run(controller="mpc_altcost", disturbance=False, animation=False,
              step_type="position_step", archive_results=False, config_variant=None)
    assert res.X.shape[0] == res.t.shape[0]
