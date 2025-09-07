
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
python main.py --controller lmpc --disturbance off --animation off --step position_step --archive off
python main.py --controller mpc_no --disturbance off --animation off --step position_step --archive off
python main.py --controller ts_fuzzy --disturbance off --animation off --step position_step --archive off
# Run tuners:
python tools\tune_pd_pd.py --trials 80 --seed 12345 --out pd_pd_tuned
python tools\tune_pid_lqr.py --trials 60 --seed 123
python tools\tune_lmpc.py --trials 40 --seed 42
# Run tuned version:
python main.py --controller pd_pd --config-variant pd_pd_tuned --disturbance off --animation off --step position_step --archive off
python main.py --controller pid_lqr --config-variant pid_lqr_tuned --disturbance off --animation off --step position_step
python main.py --controller lmpc --config-variant lmpc_tuned --disturbance off --animation off --step position_step
python main.py --controller mpc_no --config-variant mpc_no_tuned --disturbance off --animation off --step position_step
# Tests:
pip install pytest
pytest -q
```
