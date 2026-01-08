
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add parent directory to path to allow importing controllers
# Add parent directory and controllers directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../controllers')))

from controllers.pd_pd import PDPDController
from controllers.pd_lqr import PDLQRController
from controllers.mpc_utils import (PLANT, SIM, Wind, simulate_mpc, 
                                   mse, mae, iae, ise, settling_time, 
                                   steady_state_error, disturbance_robustness)

def run_ki_sweep(controller_type="pd_pd", ki_values=[0.0], with_wind=False):
    print(f"\n--- Running Sweep for {controller_type.upper()} | Wind: {with_wind} ---")
    
    dt, T = SIM["dt"], SIM["T"]
    x0 = SIM["x0"] # [0, 0, 0, 0] usually, check mpc_utils logic or override if needed.
    # We want a step response usually. SIM["x_ref"] is [0, 0, 0.1, 0] often.
    x_ref = SIM["x_ref"]
    
    plant = PLANT
    
    if with_wind:
        # Stronger wind to test Ki
        wind = Wind(T, seed=42, Ts=0.01, power=5e-2, smooth=10) # Stronger wind
    else:
        wind = None

    results = []

    # Baseline gains from the files
    if controller_type == "pd_pd":
        # ang_pid = {"Kp": -95.0, "Ki": 0.0, "Kd": -14.0}
        # cart_pid = {"Kp": -16.0, "Ki": 0.0, "Kd": -14.0}
        base_ang = {"Kp": -95.0, "Ki": 0.0, "Kd": -14.0}
        base_cart = {"Kp": -16.0, "Ki": 0.0, "Kd": -14.0}
    elif controller_type == "pd_lqr":
        # pid_gains = {"Kp": -1.5, "Ki": 0.0, "Kd": -5.0}
        # lqr_gains = {"Q": [1.0, 1.0, 500.0, 250.0], "R": 1.0}
        base_pid = {"Kp": -1.5, "Ki": 0.0, "Kd": -5.0} # This is for cart
        base_lqr = {"Q": [1.0, 1.0, 500.0, 250.0], "R": 1.0}
    
    for Ki in ki_values:
        print(f"Testing Ki = {Ki:.4f}...", end="")
        
        # Construct controller
        if controller_type == "pd_pd":
            # Vary Cart Ki
            cart_pid = base_cart.copy()
            cart_pid["Ki"] = Ki
            ctrl = PDPDController(
                pars=plant, dt=dt,
                ang_pid=base_ang, cart_pid=cart_pid,
                u_limit=80.0, integ_limit=10.0 # Allow some room for integrator
            )
        elif controller_type == "pd_lqr":
            # Vary PID Ki (which is cart PID in PD-LQR)
            pid_gains = base_pid.copy()
            pid_gains["Ki"] = Ki
            ctrl = PDLQRController(
                pars=plant, dt=dt,
                pid_gains=pid_gains, lqr_gains=base_lqr,
                u_limit=80.0, integ_limit=10.0
            )
            
        # Run Simulation
        X, U, Fw_tr, _, _ = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)
        
        # Calculate Metrics
        steps = len(U)
        t = np.linspace(0.0, T, steps + 1)
        tf = t[:-1]
        
        target_pos = x_ref[2]
        x_ref_tr = np.ones_like(t) * target_pos
        
        # Calc metrics
        mse_x = mse(X[:, 2], x_ref_tr)
        iae_x = iae(t, X[:, 2], x_ref_tr)
        
        # Steady State Error
        ess_mean, ess_rms = steady_state_error(t, X[:, 2], x_ref_tr, window_frac=0.2)
        
        metric_row = {
            "Ki": Ki,
            "MSE_x": mse_x,
            "IAE_x": iae_x,
            "Ess_Mean": ess_mean, 
            "Ess_RMS": ess_rms
        }
        results.append(metric_row)
        print(" Done")

    df = pd.DataFrame(results)
    print("\nResults:")
    print(df.to_string(float_format="%.5f"))
    
    # Save to file for inspection
    filename = f"ki_results_{controller_type}_{'wind' if with_wind else 'nowind'}.csv"
    df.to_csv(filename, float_format="%.6f", index=False)
    print(f"Saved results to {filename}")
    
    return df

def main():
    # 1. PD-PD Sweep
    print("==========================================")
    print("Checking PD-PD (Adding Ki to Cart Controller)")
    print("==========================================")
    
    ki_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    # Without Wind - check stability/overshoot (indirectly via MSE/IAE)
    run_ki_sweep("pd_pd", ki_vals, with_wind=False)
    
    # With Wind - check disturbance rejection (steady state error)
    run_ki_sweep("pd_pd", ki_vals, with_wind=True)
    
    # 2. PD-LQR Sweep
    print("\n==========================================")
    print("Checking PD-LQR (Adding Ki to Cart PID term)")
    print("==========================================")
    
    ki_vals_lqr = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    run_ki_sweep("pd_lqr", ki_vals_lqr, with_wind=False)
    run_ki_sweep("pd_lqr", ki_vals_lqr, with_wind=True)

if __name__ == "__main__":
    main()
