import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from pd_lqr import PDLQRController
from mpc_utils import PLANT, SIM

sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory(params):
    # params: [kp_pid, kd_pid, q_th, q_thd, q_x, q_xd, R]
    kp_pid, kd_pid, q_th, q_thd, q_x, q_xd, r_lqr = params
    
    pid_gains = {"Kp": kp_pid, "Ki": 0.0, "Kd": kd_pid}
    lqr_gains = {"Q": [q_th, q_thd, q_x, q_xd], "R": r_lqr}
    
    return PDLQRController(PLANT, SIM["dt"], pid_gains, lqr_gains)

if __name__ == "__main__":
    # Bounds:
    # PID (Cart): Kp, Kd
    # LQR: Q_diag (4 elems), R (1 elem)
    # Q usually positive. R positive.
    
    bounds = [
        (-20.0, 20.0),  # Kp_pid (cart)
        (-20.0, 20.0),  # Kd_pid (cart)
        (0.1, 500.0),   # Q_theta
        (0.1, 100.0),   # Q_thd
        (0.1, 500.0),   # Q_x
        (0.1, 100.0),   # Q_xd
        (0.001, 10.0)   # R
    ]
    
    best_params = run_tuning(factory, bounds, max_iter=20, pop_size=15)
    
    print("\n--- Optimized Params for PD-LQR ---")
    print(f"PID: Kp={best_params[0]:.4f}, Kd={best_params[1]:.4f}")
    print(f"LQR: Q=[{best_params[2]:.2f}, {best_params[3]:.2f}, {best_params[4]:.2f}, {best_params[5]:.2f}], R={best_params[6]:.4f}")
