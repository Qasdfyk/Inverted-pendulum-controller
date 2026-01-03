import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers.fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant
from controllers.mpc_utils import PLANT, SIM

sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory(params):
    # params: [gain_scale, base_th, base_thd, base_x, base_xd]
    gain_scale, base_th, base_thd, base_x, base_xd = params
    
    # We will reuse the starter params structure but override scale and base rules
    base = starter_ts_params16(u_sat=SIM["u_sat"], 
                               base_th=base_th, 
                               base_thd=base_thd, 
                               base_x=base_x, 
                               base_xd=base_xd)
    base.gain_scale = gain_scale
    
    # Also we need K_lqr
    K_lqr = lqr_from_plant(PLANT)
    
    return TSFuzzyController(PLANT, base, K_lqr, SIM["dt"], du_max=800.0, ramp_T=1.0)

if __name__ == "__main__":
    # Bounds for:
    # 0. gain_scale: 0.0 to 1.0 (master volume)
    # 1. base_th:    5.0 to 100.0 (penalty for theta error)
    # 2. base_thd:   0.0 to 50.0  (penalty for theta velocity)
    # 3. base_x:     0.0 to 50.0  (penalty for x error)
    # 4. base_xd:    0.0 to 20.0  (penalty for x velocity)
    
    bounds = [
        (0.01, 1.0),    # gain_scale
        (5.0, 100.0),   # base_th
        (0.1, 50.0),    # base_thd
        (0.1, 50.0),    # base_x
        (0.1, 20.0)     # base_xd
    ]
    
    best_params = run_tuning(factory, bounds, max_iter=20, pop_size=10)
    
    print("\n--- Optimized Params for Fuzzy ---")
    print(f"gain_scale={best_params[0]:.4f}")
    print(f"base_th=   {best_params[1]:.4f}")
    print(f"base_thd=  {best_params[2]:.4f}")
    print(f"base_x=    {best_params[3]:.4f}")
    print(f"base_xd=   {best_params[4]:.4f}")
