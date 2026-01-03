import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from mpc import MPCController
from mpc_utils import PLANT, SIM

sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory(params):
    # params: [q_th, q_thd, q_x, q_xd, R]
    q_th, q_thd, q_x, q_xd, r_val = params
    
    Q = np.diag([q_th, q_thd, q_x, q_xd])
    
    return MPCController(PLANT, SIM["dt"], N=10, Nu=5, umin=-SIM["u_sat"], umax=SIM["u_sat"], Q=Q, R=r_val)

if __name__ == "__main__":
    bounds = [
        (1.0, 200.0),   # Q_theta
        (0.1, 50.0),    # Q_thd
        (1.0, 200.0),   # Q_x
        (0.1, 50.0),    # Q_xd
        (0.0001, 1.0)   # R
    ]
    
    best_params = run_tuning(factory, bounds, max_iter=15, pop_size=10)
    
    print("\n--- Optimized Params for MPC ---")
    print(f"Q=[{best_params[0]:.2f}, {best_params[1]:.2f}, {best_params[2]:.2f}, {best_params[3]:.2f}], R={best_params[4]:.5f}")
