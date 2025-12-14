import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../controllers')))

from controllers.mpc import MPCController
from controllers.mpc_utils import PLANT, SIM, simulate_mpc, print_summary, mse, settling_time

def test_mpc_lqr_params():
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    
    # LQR Paremeters
    Q_diag = [69.44, 76.98, 17.70, 14.17]
    R_val = 8.0280
    
    print(f"Testing MPC with LQR params: Q={Q_diag}, R={R_val}")
    
    ctrl = MPCController(
        pars=plant, dt=dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
        Q=np.diag(Q_diag), R=R_val
    )
    
    X, U, _, ctrl_time, wall_time = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=None)
    
    t = np.linspace(0.0, T, len(U) + 1)
    
    metrics = {
        "mse_theta": mse(X[:, 0], np.zeros_like(X[:, 0])),
        "mse_x": mse(X[:, 2], np.ones_like(X[:, 2]) * x_ref[2]),
        "settling_time_x": settling_time(t, X[:, 2], np.ones_like(X[:, 2]) * x_ref[2], 0.01, 0.5)
    }
    
    print("MPC with LQR Params Results:")
    print(metrics)

if __name__ == "__main__":
    test_mpc_lqr_params()
