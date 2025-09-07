
import pytest
from env.sim import run

@pytest.mark.parametrize("name", ["pid_lqr","pd_pd","ts_fuzzy"])
def test_run_each(name):
    res = run(controller=name, disturbance=False, animation=False,
              step_type="position_step", archive_results=False, config_variant=None)
    assert res.t.size > 5
