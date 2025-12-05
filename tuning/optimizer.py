import numpy as np
from scipy.optimize import differential_evolution
import sys
import os

# Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from mpc_utils import PLANT, SIM, simulate_mpc
from objectives import calculate_cost

def run_tuning(controller_factory, bounds, max_iter=10, pop_size=10):
    """
    Generic tuning driver.
    controller_factory: function(params) -> controller_instance
    bounds: list of (min, max) for each param
    """

    dt = SIM["dt"]
    T = SIM["T"] # Use shorter T for tuning speed if needed, but 10s is fine
    x0 = SIM["x0"]
    x_ref = SIM["x_ref"]
    plant = PLANT
    
    print(f"Starting optimization with bounds: {bounds}")
    
    def objective(params):
        try:
            ctrl = controller_factory(params)
            # Run simulation
            X, U, _, _, _ = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=None)
            
            # Check for instability or NaNs
            if np.any(np.isnan(X)) or np.max(np.abs(X[:, 0])) > 10.0 or np.max(np.abs(X[:, 2])) > 5.0:
                return 1e9 # Penalty for instability

            cost = calculate_cost(X, U, x_ref)
            return cost
        except Exception as e:
            return 1e9 # Penalty for crash

    result = differential_evolution(objective, bounds, maxiter=max_iter, popsize=pop_size, disp=True, strategy='best1bin')
    
    print("\noptimization Complete!")
    print(f"Best Cost: {result.fun}")
    print(f"Best Params: {result.x}")
    return result.x
