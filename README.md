
# Cart-Pole Controllers (PID+LQR, PD/PD, LMPC, MPC, T–S Fuzzy)

Unified environment for testing multiple controllers on the same nonlinear cart–pole plant.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy scipy matplotlib pyyaml
# Optional for MPC
pip install cvxpy osqp ecos
# Run:
python main.py --controller pid_lqr --disturbance off --animation off --step position_step --archive on
python main.py --controller pd_pd   --disturbance on  --animation off --step position_step --archive off
python main.py --controller ts_fuzzy --disturbance off --animation off --step position_step --archive off
# Tests:
pip install pytest
pytest -q
```
