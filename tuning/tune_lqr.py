import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from lqr import LQRController
from lqr_i import LQRIController
from mpc_utils import PLANT, SIM

sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory_lqr(params):
    # params: [q_th, q_thd, q_x, q_xd, R]
    *q, r = params
    return LQRController(PLANT, SIM["dt"], {"Q": list(q), "R": r})

def factory_lqr_i(params):
    # params: [q_th, q_thd, q_x, q_xd, q_z, R]
    *q, r = params
    return LQRIController(PLANT, SIM["dt"], {"Q": list(q), "R": r})

if __name__ == "__main__":
    # Same bounds as the PID-LQR tuning (tune_pd_lqr.py), plus the integrator weight
    q_bounds = [
        (0.1, 500.0),   # Q_theta
        (0.1, 100.0),   # Q_thd
        (0.1, 500.0),   # Q_x
        (0.1, 100.0),   # Q_xd
    ]
    r_bounds = [(0.001, 10.0)]  # R

    variant = sys.argv[1] if len(sys.argv) > 1 else "lqr"
    if variant == "lqr":
        best = run_tuning(factory_lqr, q_bounds + r_bounds, max_iter=20, pop_size=15)
        print("\n--- Optimized Params for LQR ---")
    else:
        best = run_tuning(factory_lqr_i, q_bounds + [(0.01, 500.0)] + r_bounds, max_iter=20, pop_size=15)
        print("\n--- Optimized Params for LQR-I ---")
    print(f"Q={[round(float(v), 3) for v in best[:-1]]}, R={best[-1]:.4f}")
