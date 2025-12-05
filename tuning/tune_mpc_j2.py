import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from mpc_J2 import MPCControllerJ2
from mpc_utils import PLANT, SIM

sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory(params):
    # params: [q_theta, q_x, q_thd, q_xd, r, r_abs]
    q_theta, q_x, q_thd, q_xd, r, r_abs = params
    
    return MPCControllerJ2(
        PLANT, SIM["dt"], N=15, Nu=5, umin=-SIM["u_sat"], umax=SIM["u_sat"],
        q_theta=q_theta, q_x=q_x, q_thd=q_thd, q_xd=q_xd,
        r=r, r_abs=r_abs
    )

if __name__ == "__main__":
    bounds = [
        (1.0, 200.0),   # q_theta
        (1.0, 200.0),   # q_x
        (0.1, 50.0),    # q_thd
        (0.1, 50.0),    # q_xd
        (0.0001, 1.0),  # r
        (0.0, 0.1)      # r_abs (keep small)
    ]
    
    best_params = run_tuning(factory, bounds, max_iter=15, pop_size=10)
    
    print("\n--- Optimized Params for MPC J2 ---")
    print(f"q_theta={best_params[0]:.2f}, q_x={best_params[1]:.2f}, q_thd={best_params[2]:.2f}, q_xd={best_params[3]:.2f}")
    print(f"r={best_params[4]:.5f}, r_abs={best_params[5]:.5f}")
