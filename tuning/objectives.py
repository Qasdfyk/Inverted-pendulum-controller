import numpy as np
import sys
import os

# Import mpc_utils from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'controllers')))
from mpc_utils import mse

def calculate_cost(X, U, x_ref):
    """
    Fair cost function for all controllers.
    J = w1 * MSE(theta) + w2 * MSE(x) + w3 * RMS(u)
    """
    th = X[:, 0]
    x = X[:, 2]
    
    # Target is upright (0) and at x_ref
    th_ref = np.zeros_like(th)
    x_ref_arr = np.ones_like(x) * x_ref[2]

    mse_th = mse(th, th_ref)
    mse_x = mse(x, x_ref_arr)
    
    u_rms = np.sqrt(np.mean(U**2)) if len(U) > 0 else 0.0
    
    # Weights for the "Fair" function
    # Penalize angle error heavily (stability)
    # Penalize position error moderately (tracking)
    # Penalize control effort lightly (efficiency)
    w_th = 4.0
    w_x = 1.0
    w_u = 0.01

    J = w_th * mse_th + w_x * mse_x + w_u * u_rms
    return J
