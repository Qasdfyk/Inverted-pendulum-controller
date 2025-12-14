import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Update font sizes for better visibility in LaTeX
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'lines.linewidth': 3
})

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../controllers')))

from controllers.pd_pd import PDPDController
from controllers.pd_lqr import PDLQRController
from controllers.mpc import MPCController
from controllers.mpc_J2 import MPCControllerJ2
from controllers.mpc_utils import PLANT, SIM, simulate_mpc, Wind, mse, mae, iae, ise, control_energy_l2, control_energy_l1, settling_time, overshoot, steady_state_error
from controllers.fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant

SAVE_DIR = os.path.join(os.path.dirname(__file__), '../latex/images/experiments')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'experiments_results.json')
os.makedirs(SAVE_DIR, exist_ok=True)

def calculate_metrics(t, x, u, x_ref, dt):
    # x state: [theta, theta_dot, pos, pos_dot]
    # x_ref: [0, 0, x_ref, 0]
    
    metrics = {}
    
    # Extract signals
    theta = x[:, 0]
    pos = x[:, 2]
    
    ref_theta = np.zeros_like(theta)
    ref_pos = np.ones_like(pos) * x_ref[2]
    
    # Calculate metrics
    metrics['mse_theta'] = mse(theta, ref_theta)
    metrics['mae_theta'] = mae(theta, ref_theta)
    metrics['mse_x'] = mse(pos, ref_pos)
    metrics['mae_x'] = mae(pos, ref_pos)
    
    metrics['iae_theta'] = iae(t, theta, ref_theta)
    metrics['ise_theta'] = ise(t, theta, ref_theta)
    metrics['iae_x'] = iae(t, pos, ref_pos)
    metrics['ise_x'] = ise(t, pos, ref_pos)
    
    # tf should be passed for control metrics as u has N steps, t has N+1
    tf = t[:-1]
    metrics['energy_l2'] = control_energy_l2(tf, u)
    metrics['energy_l1'] = control_energy_l1(tf, u)
    
    # Settling time params from controller files: eps=0.01, hold_time=0.5
    metrics['ts_theta'] = settling_time(t, theta, ref_theta, eps=0.01, hold_time=0.5)
    metrics['ts_x'] = settling_time(t, pos, ref_pos, eps=0.01, hold_time=0.5)
    
    metrics['os_theta'] = overshoot(theta, 0.0) # Target is 0, so Overshoot relative to 0 is weird if we start at 0.785.
    # Usually overshoot is for step response. Here we have initial condition response. 
    # Max deviation in OPPOSITE direction? Or just max amplitude?
    # Let's simple record max absolute value after initial period?
    # For now, let's skip conventional overshoot for regulation from IC.
    
    return metrics

def run_single_experiment(ctrl, name, wind_enabled=False):
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]
    
    # Use exact x0 from SIM as requested by user
    # x0 is [0.05, 0, 0, 0] approx 2.8 deg
    
    if wind_enabled:
        # Use parameters matching the controllers/mpc_utils defaults/comments
        # power=1e-3, smooth=5, seed=23341
        wind = Wind(T, seed=23341, power=1e-3, smooth=5)
        scenario_name = "Wind"
        suffix = "wind"
    else:
        wind = None
        scenario_name = "Nominal"
        suffix = "nom"
        
    X, U, Fw, t_ctrl, t_sim = simulate_mpc(PLANT, ctrl, x0, x_ref, T, dt, wind=wind)
    
    t = np.linspace(0.0, T, len(U) + 1)
    tf = t[:-1] # For controls
    
    metrics = calculate_metrics(t, X, U, x_ref, dt)
    metrics['ctrl_time'] = t_ctrl
    metrics['sim_time'] = t_sim
    
    # Plotting for this single run
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # Angle
    axes[0].plot(t, X[:, 0], label=r'$\theta$')
    axes[0].plot(t, np.zeros_like(t), 'k--', alpha=0.5)
    axes[0].set_ylabel(r'$\theta$ [rad]')
    axes[0].set_title(f'{name} - {scenario_name}')
    axes[0].grid(True)
    
    # Position
    axes[1].plot(t, X[:, 2], label=r'$x$')
    axes[1].plot(t, np.ones_like(t)*x_ref[2], 'k--', alpha=0.5, label=r'$x_{ref}$')
    axes[1].set_ylabel(r'$x$ [m]')
    axes[1].grid(True)
    axes[1].legend()
    
    # Control
    axes[2].plot(tf, U, label=r'$u$')
    axes[2].set_ylabel(r'$u$ [N]')
    axes[2].set_xlabel('Time [s]')
    axes[2].grid(True)
    
    plt.tight_layout()
    filename = f"{name.lower().replace(' ', '_')}_{suffix}.png"
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close(fig)
    
    return t, X, U, metrics

def main():
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # Initialize Optimized Controllers
    controllers = {}
    
    # 1. PD-PD
    controllers['PD-PD'] = PDPDController(PLANT, dt,
                                          ang_pid={"Kp": -95.0, "Ki": 0.0, "Kd": -14.0},
                                          cart_pid={"Kp": -16.0, "Ki": 0.0, "Kd": -14.0})
    
    # 2. PD-LQR
    controllers['PD-LQR'] = PDLQRController(PLANT, dt,
                                            pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3},
                                            lqr_gains={"Q": [69.44, 76.98, 17.70, 14.17], "R": 8.0280})
    
    # 3. MPC
    controllers['MPC'] = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                       Q=np.diag([158.39, 36.80, 43.41, 19.71]), R=0.08592)
    
    # 4. MPC-J2
    controllers['MPC-J2'] = MPCControllerJ2(PLANT, dt, N=15, Nu=5, umin=-u_sat, umax=u_sat,
                                            q_theta=80.0, q_x=120.0, q_thd=5.0, q_xd=5.0,
                                            r=0.0001, r_abs=0.0) # Optimized
    
    # 5. Fuzzy-LQR
    K_lqr = lqr_from_plant(PLANT)
    p_opt = starter_ts_params16(u_sat)
    controllers['Fuzzy-LQR'] = TSFuzzyController(PLANT, p_opt, K_lqr, dt, du_max=800.0, ramp_T=1.0)

    results = {}
    
    # Store trajectories for combined plot
    trajectories_nom = {}
    trajectories_wind = {}
    
    print("Running Nominal Experiments...")
    for name, ctrl in controllers.items():
        print(f"  Simulating {name}...")
        t, X, U, metrics = run_single_experiment(ctrl, name, wind_enabled=False)
        results[f"{name}_Nominal"] = metrics
        trajectories_nom[name] = (t, X, U)

    print("Running Wind Experiments...")
    for name, ctrl in controllers.items():
        print(f"  Simulating {name}...")
        t, X, U, metrics = run_single_experiment(ctrl, name, wind_enabled=True)
        results[f"{name}_Wind"] = metrics
        trajectories_wind[name] = (t, X, U)

    # Save Results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {RESULTS_FILE}")

    # --- Combined Plot (Nominal) ---
    fig_nom, ax_nom = plt.subplots(figsize=(12, 8))
    for name, (t, X, U) in trajectories_nom.items():
        ax_nom.plot(t, X[:, 0], label=name)
    
    ax_nom.plot(t, np.zeros_like(t), 'k--', alpha=0.3, linewidth=2)
    ax_nom.set_title("Porównanie odpowiedzi skokowej (Warunki nominalne)")
    ax_nom.set_ylabel(r'Kąt $\theta$ [rad]')
    ax_nom.set_xlabel('Czas [s]')
    ax_nom.grid(True)
    ax_nom.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "combined_nominal.png"))
    plt.close(fig_nom)
    
    # --- Combined Plot (Wind) ---
    fig_wind, ax_wind = plt.subplots(figsize=(12, 8))
    for name, (t, X, U) in trajectories_wind.items():
        ax_wind.plot(t, X[:, 0], label=name)
        
    ax_wind.plot(t, np.zeros_like(t), 'k--', alpha=0.3, linewidth=2)
    ax_wind.set_title("Porównanie odpowiedzi przy zakłóceniach (Wiatr)")
    ax_wind.set_ylabel(r'Kąt $\theta$ [rad]')
    ax_wind.set_xlabel('Czas [s]')
    ax_wind.grid(True)
    ax_wind.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "combined_wind.png"))
    plt.close(fig_wind)
    
    print("Combined plots saved.")

if __name__ == "__main__":
    main()
