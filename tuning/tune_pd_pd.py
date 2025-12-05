import sys
import os
import numpy as np

# Add parent path to import controllers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from pd_pd import PDPDController
from mpc_utils import PLANT, SIM

# Add local path for optimizer
sys.path.append(os.path.dirname(__file__))
from optimizer import run_tuning

def factory(params):
    # Unpack params
    # params: [kp_ang, kd_ang, kp_cart, kd_cart]
    kp_ang, kd_ang, kp_cart, kd_cart = params
    
    # Check signs? Usually gains are negative for this system if error is (ref - x)
    # But let the optimizer find signs or bounds. 
    # Usually: u = Kp*e + Kd*de. If e = ref - x.
    # We will let the bounds decide sign.
    
    ang_pid = {"Kp": kp_ang, "Ki": 0.0, "Kd": kd_ang}
    cart_pid = {"Kp": kp_cart, "Ki": 0.0, "Kd": kd_cart}
    
    return PDPDController(PLANT, SIM["dt"], ang_pid, cart_pid)

if __name__ == "__main__":
    # Bounds: [Kp_ang, Kd_ang, Kp_cart, Kd_cart]
    # Angle gains usually higher. Cart gains lower.
    # Angle PD stabilizes upright. Cart PD tracks position.
    
    # Typical values seen: Kp_ang ~ -40, Kd_ang ~ -8. 
    # Let's search in loose range.
    bounds = [
        (-100.0, 100.0), # Kp_ang
        (-50.0, 50.0),   # Kd_ang
        (-50.0, 50.0),   # Kp_cart
        (-50.0, 50.0)    # Kd_cart
    ]
    
    # Note: Differential Evolution handles global search well.
    best_params = run_tuning(factory, bounds, max_iter=20, pop_size=15)
    
    print("\n--- Optimized Params for PD-PD ---")
    print(f"Angle: Kp={best_params[0]:.4f}, Kd={best_params[1]:.4f}")
    print(f"Cart:  Kp={best_params[2]:.4f}, Kd={best_params[3]:.4f}")
