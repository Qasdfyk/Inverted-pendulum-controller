
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import env from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from controllers.mpc_utils import PLANT, rk4_step

# Define a locally damped dynamics function to show "real" physical behavior (convergence)
def f_damped(x, u, pars):
    # Unpack state
    th, thd, pos, posd = x
    M, m, l, g = pars["M"], pars["m"], pars["l"], pars["g"]
    b = 0.5 # Viscous friction coefficient
    
    s, c = np.sin(th), np.cos(th)

    denom_x  = (M + m) - m * c * c
    denom_th = (m * l * c * c) - (M + m) * l
    
    # Dynamics with damping
    # We want friction to oppose velocity.
    # thdd = (Torque_gravity + Torque_friction) / Inertia_term
    # Denominator 'denom_th' is NEGATIVE (approx -(M)*l).
    # To get deceleration (negative thdd) when thd > 0:
    # We need (Term / Negative) < 0  => Term > 0.
    # So friction term must be POSITIVE for thd > 0.
    
    friction_term = + b * thd * (M+m)  # Changed sign to positive to act as damping given negative denominator
    
    thdd = (u * c - (M + m) * g * s + m * l * (c * s) * (thd ** 2) + friction_term) / denom_th
    xdd  = (u + m * l * s * (thd ** 2) - m * g * c * s) / denom_x
    
    return np.array([thd, thdd, posd, xdd], dtype=float)

def generate_open_loop_plot():
    # Parameters
    dt = 0.05
    T = 6.0 
    steps = int(T / dt)
    
    # Initial condition: slightly unstable (small angle)
    # Start at 0.1 rad. It should fall towards PI (3.14)
    x0 = np.array([0.1, 0.0, 0.0, 0.0]) 
    x = x0.copy()
    
    # Simulation
    time = np.linspace(0, T, steps+1)
    theta_hist = [x[0]]
    
    for _ in range(steps):
        u = 0.0
        x = rk4_step(f_damped, x, u, PLANT, dt)
        theta_hist.append(x[0])
        
    theta_hist = np.array(theta_hist)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time, theta_hist, lw=2, label=r'$\theta(t)$', color='#0072BD')
    
    # Mark PI 
    ax.axhline(np.pi, color='r', linestyle='--', alpha=0.7, label=r'$\pi$ (stabilny dół)')
    
    # Annotate behavior
    ax.text(0.2, 0.2, 'Start (0.1 rad)', fontsize=9)
    
    ax.set_title("Odpowiedź układu w pętli otwartej (bez sterowania)")
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Kąt wychylenia [rad]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    out_path = os.path.join("latex", "img", "open_loop.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    generate_open_loop_plot()
