
import sys
import os
import numpy as np
from scipy.optimize import differential_evolution

# Add paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant
from mpc_utils import PLANT, SIM, simulate_mpc, Wind
from mpc_utils import mse # for cost

# Objective function logic
def calculate_cost(X, U, x_ref):
    # Custom cost for wind rejection
    # Heavy penalty on Theta devitation
    # Moderate penalty on Position
    # Minimal penalty on Control
    
    th = X[:, 0]
    x  = X[:, 2]
    # ref is 0 for th, x_ref[2] for x
    
    cost_th = np.mean(th**2)
    cost_x  = np.mean((x - x_ref[2])**2)
    
    # We want to minimize THETA variance primarily
    return 100.0 * cost_th + 1.0 * cost_x

def factory(params):
    # params: [gain_scale, base_th, base_thd, base_x, base_xd]
    gain_scale, base_th, base_thd, base_x, base_xd = params
    
    base = starter_ts_params16(u_sat=SIM["u_sat"], 
                               base_th=base_th, 
                               base_thd=base_thd, 
                               base_x=base_x, 
                               base_xd=base_xd)
    base.gain_scale = gain_scale
    K_lqr = lqr_from_plant(PLANT)
    return TSFuzzyController(PLANT, base, K_lqr, SIM["dt"], du_max=800.0, ramp_T=1.0)

def run_optimization():
    print("Starting Wind Optimization for Fuzzy LQR...")
    
    # Bounds
    # gain_scale: 0.1 to 2.0
    # base_th: 10 to 200
    # base_thd: 1 to 100 (High damping allowed)
    # base_x: 0 to 50
    # base_xd: 0 to 20
    bounds = [
        (0.1, 1.5),     # gain_scale
        (10.0, 200.0),  # base_th
        (1.0, 100.0),   # base_thd
        (0.0, 50.0),    # base_x
        (0.0, 20.0)     # base_xd
    ]
    
    # Wind Config
    dt = SIM["dt"]
    T = SIM["T"]
    
    # Pre-generate wind to be fair (deterministic)
    wind_obj = Wind(T, seed=42, Ts=0.05, power=5e-3, smooth=10)
    
    def objective(params):
        try:
            ctrl = factory(params)
            X, U, _, _, _ = simulate_mpc(PLANT, ctrl, SIM["x0"], SIM["x_ref"], T, dt, wind=wind_obj)
            
            # Instability checks
            if np.any(np.isnan(X)): return 1e9
            if np.max(np.abs(X[:, 0])) > 1.5: return 1e9 # Fell over (~85 deg)
            if np.max(np.abs(X[:, 2])) > 10.0: return 1e9 # Ran away
            
            return calculate_cost(X, U, SIM["x_ref"])
        except Exception:
            return 1e9

    result = differential_evolution(objective, bounds, maxiter=10, popsize=6, disp=True, strategy='best1bin')
    
    print("\n--- Optimized Params ---")
    print(f"gain_scale= {result.x[0]:.4f}")
    print(f"base_th=    {result.x[1]:.4f}")
    print(f"base_thd=   {result.x[2]:.4f}")
    print(f"base_x=     {result.x[3]:.4f}")
    print(f"base_xd=    {result.x[4]:.4f}")
    print(f"Cost:       {result.fun:.6f}")

if __name__ == "__main__":
    run_optimization()
