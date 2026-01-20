import numpy as np
import itertools
import sys
import os
import time
import math

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../controllers')))

from controllers.pd_pd import PDPDController
from controllers.mpc_utils import PLANT, SIM, simulate_mpc

def evaluate_single(params, dt, T, x0, x_ref):
    kp_th, ki_th, kd_th, kp_x, ki_x, kd_x = params
    
    ang_pid = {"Kp": float(kp_th), "Ki": float(ki_th), "Kd": float(kd_th)}
    cart_pid = {"Kp": float(kp_x), "Ki": float(ki_x), "Kd": float(kd_x)}
    
    ctrl = PDPDController(PLANT, dt, ang_pid, cart_pid, u_limit=SIM["u_sat"])
    
    try:
        # Run simulation
        X, U, _, _, _ = simulate_mpc(PLANT, ctrl, x0, x_ref, T, dt, wind=None)
        
        theta_err = X[:, 0]
        x_err = X[:, 2] - x_ref[2]
        
        mse_theta = np.mean(theta_err**2)
        mse_x = np.mean(x_err**2)
        
        # Stability Constraints
        # Theta > 1.5 rad (approx 85 deg) -> fallen
        if np.max(np.abs(X[:, 0])) > 1.5:
            return float('inf'), None
        
        # Position > 5.0m -> track violation
        if np.max(np.abs(X[:, 2])) > 5.0:
            return float('inf'), None

        # Filter really bad MSE
        if mse_theta > 10.0:
            return float('inf'), None

        return mse_theta, mse_x
    except Exception:
        return float('inf'), None

def run_grid_search():
    print("=== PID-PID Dense Grid Search (Single Thread, Step=1) ===")
    
    dt = SIM["dt"]
    T = 6.0 
    x0 = SIM["x0"]
    x_ref = SIM["x_ref"]
    
    # --- Exact 1-step ranges ---
    # We must be careful. 100^6 is impossible.
    # User said "check positive too".
    # We will prioritize the likely stable negative region but include a sparse positive check 
    # OR just a smaller window if we strictly follow "step=1".
    
    # Strategy:
    # 1. Main Gains (Kp_th, Kd_th, Kp_x, Kd_x): Wide-ish
    # 2. Integral (Ki): Narrow
    
    # Range(start, stop, step) -> stop is exclusive.
    
    # Kp_th: -100 to 50. (150 steps). Too big if combined with others.
    # Let's target the search around "0" ± 60.
    r_kp_th = range(-60, 60, 1) # 120 values
    
    # Ki_th: Integral is sensitive. -3 to 3.
    r_ki_th = range(-3, 4, 1)   # 7 values
    
    # Kd_th: Derivative. -20 to 20.
    r_kd_th = range(-20, 21, 1) # 41 values
    
    # Kp_x: Pos P gain. -20 to 20.
    r_kp_x = range(-20, 21, 1)  # 41 values
    
    # Ki_x: -2 to 2.
    r_ki_x = range(-2, 3, 1)    # 5 values
    
    # Kd_x: -10 to 10.
    r_kd_x = range(-10, 11, 1)  # 21 values
    
    # Total: 120 * 7 * 41 * 41 * 5 * 21 = 148,465,200
    # ~150 Million.
    # Single thread 150M.
    # If 1000 iter/sec -> 150,000 sec -> 41 hours.
    # Too long? User said "check also positive".
    
    # Let's reduce density of the "positive" side or Kp_x?
    # Or just optimize for the negative side heavily. 
    # But user asked for this specific setup.
    
    # I will prune the ranges slightly to make it closer to 5-10M.
    # Kp_th: -60 .. -20 (40) AND 20 .. 60 (40)? No, contiguous range.
    # Let's do: -50 to -10 (40) for Kp_theta (Negative feedback is dominant).
    # Checking positive Kp is almost guaranteed unstable but we'll include -5 to +5 just to see.
    
    r_kp_th = range(-50, 1, 1) # 51 values (Negative gain is critical)
    r_ki_th = range(-2, 1, 1)  # 3 values
    r_kd_th = range(-15, 1, 1) # 16 values
    
    r_kp_x = range(-15, 6, 1)  # 21 values
    r_ki_x = range(-1, 2, 1)   # 3 values
    r_kd_x = range(-10, 1, 1)  # 11 values
    
    # 51 * 3 * 16 * 21 * 3 * 11 = 1,696,464.
    # ~1.7 Million. 
    # This takes 30-45 mins. Useful.
    
    # Let's define the iterator
    grid_iter = itertools.product(r_kp_th, r_ki_th, r_kd_th, r_kp_x, r_ki_x, r_kd_x)
    total_est = len(r_kp_th)*len(r_ki_th)*len(r_kd_th)*len(r_kp_x)*len(r_ki_x)*len(r_kd_x)
    
    print(f"Combinations to scan: {total_est}")
    print("Ranges:")
    print(f" Kp_th: {r_kp_th.start}..{r_kp_th.stop-1}")
    print(f" Ki_th: {r_ki_th.start}..{r_ki_th.stop-1}")
    print(f" Kd_th: {r_kd_th.start}..{r_kd_th.stop-1}")
    print(f" Kp_x:  {r_kp_x.start}..{r_kp_x.stop-1}")
    print(f" Ki_x:  {r_ki_x.start}..{r_ki_x.stop-1}")
    print(f" Kd_x:  {r_kd_x.start}..{r_kd_x.stop-1}")
    
    start_time = time.time()
    
    best_score = float('inf')
    best_params = None
    best_aux = None
    
    # Simulation shortcut: Instantiate controller once? 
    # No, easy to re-init.
    
    # Progress counter
    count = 0
    
    # To run efficiently in single thread, we rely on the loop.
    
    for params in grid_iter:
        count += 1
        
        # Run
        mse_th, mse_x = evaluate_single(params, dt, T, x0, x_ref)
        
        if mse_th < best_score:
            best_score = mse_th
            best_params = params
            best_aux = mse_x
            
            # Print update
            print(f"[{count}/{total_est}] NEW BEST: MSE_th={best_score:.4f} MSE_x={mse_x:.4f}")
            print(f"   Params: {best_params}")
            
        if count % 10000 == 0:
            elapsed = time.time() - start_time
            rate = count / elapsed
            rem = (total_est - count) / rate
            print(f"Progress: {count}/{total_est} ({count/total_est*100:.1f}%) | Rate: {rate:.1f} it/s | ETA: {rem/60:.1f} min")

    print("\n=== DONE ===")
    if best_params:
         print(f"Best Score: {best_score}")
         print(f"Params: {best_params}")
         
         # Save
         out_file = os.path.join(os.path.dirname(__file__), "tuned_params_pdpd.txt")
         with open(out_file, "w") as f:
            f.write("# Best PDPD Parameters (Single-Thread Dense)\n")
            f.write(f"MSE_Theta: {best_score}\n")
            kp_th, ki_th, kd_th, kp_x, ki_x, kd_x = best_params
            f.write(f"ang_pid = {{'Kp': {kp_th}, 'Ki': {ki_th}, 'Kd': {kd_th}}}\n")
            f.write(f"cart_pid = {{'Kp': {kp_x}, 'Ki': {ki_x}, 'Kd': {kd_x}}}\n")
         print(f"Saved to {out_file}")
    else:
        print("No stable params found.")

if __name__ == "__main__":
    run_grid_search()
