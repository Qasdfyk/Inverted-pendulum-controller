# Make tests robust regardless of where pytest is launched from,
# and keep tests headless (no GUI popups).

import os, sys
from pathlib import Path

# --- Ensure project root is on sys.path ---
ROOT = Path(__file__).resolve().parents[1]  # .../cartpole
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Headless plotting so tests don't block on plt.show() ---
os.environ.setdefault("MPLBACKEND", "Agg")      # use non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None                 # no-op during tests

# --- Optional: mark runs as headless for any custom checks ---
os.environ["CARTPOLE_HEADLESS"] = "1"

# If you still need configs in fixtures:
from utils.cfg import load_configs

# (Your existing fixtures can stay below; example kept for reference)
import pytest

@pytest.fixture
def short_env_cfg():
    cfg = load_configs("pid_lqr", None)
    cfg["sim"]["t_end"] = 2.0
    cfg["sim"]["dt_log"] = 0.02
    cfg["sim"]["x_ref"] = 0.05
    return cfg

@pytest.fixture(params=["pid_lqr","pd_pd","ts_fuzzy"])
def controller_name(request):
    return request.param
