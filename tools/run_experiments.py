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
from controllers.lmpc import LinearMPCController

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
    # Use the actual target position from x_ref passed to the function
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
    metrics['ts_theta'] = settling_time(t, theta, ref_theta, eps=0.001, hold_time=0.5)
    metrics['ts_x'] = settling_time(t, pos, ref_pos, eps=0.002, hold_time=0.5)
    
    metrics['os_theta'] = overshoot(theta, 0.0) 
    metrics['max_theta'] = np.max(np.abs(theta))
    metrics['max_x'] = np.max(np.abs(pos))
    
    return metrics

def run_single_experiment(ctrl, name, wind_enabled=False):
    dt, T = SIM["dt"], SIM["T"]
    x0 = SIM["x0"]
    
    # --- ZMIANA: Ustawienie x_ref na 0.1 ---
    # x_ref state: [theta, theta_dot, pos, pos_dot]
    x_ref = np.array([0.0, 0.0, 0.1, 0.0])
    
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
    # Angle
    axes[0].plot(t*10, X[:, 0], label=r'$\theta$')
    axes[0].plot(t*10, np.zeros_like(t), 'k--', alpha=0.5, label='Wartość zadana')
    axes[0].set_ylabel(r'Kąt $\theta$ [rad]')
    axes[0].set_title(f'{name} - {scenario_name}')
    axes[0].grid(True)
    
    # Position
    axes[1].plot(t*10, X[:, 2], label=r'$x$')
    # --- ZMIANA: Rysowanie linii referencyjnej x_ref[2] (0.1) zamiast x0[2] ---
    axes[1].plot(t*10, np.ones_like(t)*x_ref[2], 'k--', alpha=0.5, label='Wartość zadana')
    axes[1].set_ylabel(r'Pozycja $x$ [m]')
    axes[1].grid(True)
    axes[1].legend()
    
    # Control
    axes[2].plot(tf*10, U, label=r'$u$')
    axes[2].set_ylabel(r'Siła sterująca $u$ [N]')
    axes[2].set_xlabel('k')
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
    controllers['PID-PID'] = PDPDController(PLANT, dt,
                                            ang_pid = {"Kp": -40.0, "Ki": -1.0, "Kd": -8.0},
                                            cart_pid = {"Kp": -1.0, "Ki": -0.1, "Kd": -3.0})
    
    # 2. PD-LQR
    controllers['PID-LQR'] = PDLQRController(PLANT, dt,
                                            pid_gains = {"Kp": -7.0, "Ki": 0.1, "Kd": -3.0},
                                            lqr_gains = {"Q": [200.0, 3.0, 35.0, 40.0], "R": 1.0})
    
    # 3. MPC
    controllers['MPC'] = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                     Q=np.diag([158.39, 40.80, 43.41, 19.71]), R=0.08592)
    
    # 4. MPC-J2 Variants
    controllers['MPC-J2'] = MPCControllerJ2(
        pars=PLANT, dt=dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
        Q=np.diag([158.39, 40.80, 43.41, 19.71]), R=0.001, r_abs=1
    )
    
    # 5. Fuzzy-LQR
    K_lqr = lqr_from_plant(PLANT)
    p_opt = starter_ts_params16(u_sat)
    controllers['Fuzzy-LQR'] = TSFuzzyController(PLANT, p_opt, K_lqr, dt, du_max=800.0, ramp_T=1.0)

    # 6. LMPC
    controllers['LMPC'] = LinearMPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                              Q=np.diag([15.0, 1.0, 15.0, 1.0]), R=0.1)

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

    # Define Controller Groups
    group_classical = ['PID-PID', 'PID-LQR']
    group_advanced = ['MPC', 'MPC-J2', 'LMPC', 'Fuzzy-LQR']
    group_mpc_study = ['MPC-J2']
    group_all = ['PID-PID', 'PID-LQR', 'MPC', 'MPC-J2', 'Fuzzy-LQR', 'LMPC']
    group_all_lmpc = ['PID-PID', 'PID-LQR', 'MPC', 'MPC-J2', 'Fuzzy-LQR', 'LMPC']

    # Helper function for grouped plots
    def plot_group(group_names, trajectories, title_suffix, filename_suffix, scenario_title, signal_type='theta'):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plot_ref = False
        t_ref = None
        ref_val = None
        ylabel = ""

        for name in group_names:
            if name in trajectories:
                t, X, U = trajectories[name]
                tf = t[:-1] # For control signal
                
                if signal_type == 'theta':
                    ax.plot(t*10, X[:, 0], label=name, linewidth=3)
                    ylabel = r'$\theta$ [rad]'
                    # Reference
                    t_ref = t*10
                    ref_val = np.zeros_like(t_ref)
                    plot_ref = True
                elif signal_type == 'x':
                    ax.plot(t*10, X[:, 2], label=name, linewidth=3)
                    ylabel = r'x [m]'
                    # Reference
                    t_ref = t*10
                    # --- ZMIANA: Wartość zadana na wykresie zbiorczym to 0.1 ---
                    ref_val = np.ones_like(t_ref) * 0.1 
                    plot_ref = True
                elif signal_type == 'u':
                    ax.plot(tf*10, U, label=name, linewidth=3)
                    ylabel = r'u [N]'
                    plot_ref = False
        
        if plot_ref and t_ref is not None:
             ax.plot(t_ref, ref_val, 'k--', alpha=0.3, linewidth=2, label='Wartość zadana')

        #ax.set_title(f"Porównanie: {scenario_title}")
        ax.set_ylabel(ylabel)
        ax.set_xlabel('k')
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"combined_{filename_suffix}.png"))
        plt.close(fig)

    # --- Generate Plots ---
    # Nominal
    # Theta
    plot_group(group_classical, trajectories_nom, "Klasyczne", "nominal_classical", "Kąt (Nominal)", 'theta')
    plot_group(group_advanced, trajectories_nom, "Zaawansowane", "nominal_advanced", "Kąt (Nominal)", 'theta')
    plot_group(group_all, trajectories_nom, "Wszystkie", "nominal_classical_all", "Kąt (Nominal)", 'theta')
    # Position
    plot_group(group_classical, trajectories_nom, "Klasyczne", "nominal_pos_classical", "Pozycja (Nominal)", 'x')
    plot_group(group_advanced, trajectories_nom, "Zaawansowane", "nominal_pos_advanced", "Pozycja (Nominal)", 'x')
    plot_group(group_all, trajectories_nom, "Wszystkie", "nominal_pos_advanced_all", "Pozycja (Nominal)", 'x')
    # Control
    plot_group(group_classical, trajectories_nom, "Klasyczne", "nominal_control_classical", "Sterowanie (Nominal)", 'u')
    plot_group(group_advanced, trajectories_nom, "Zaawansowane", "nominal_control_advanced", "Sterowanie (Nominal)", 'u')
    plot_group(group_all, trajectories_nom, "Wszystkie", "nominal_control_advanced_all", "Sterowanie (Nominal)", 'u')
    # Wind
    # Theta
    plot_group(group_classical, trajectories_wind, "Klasyczne", "wind_classical", "Kąt (Wiatr)", 'theta')
    plot_group(group_advanced, trajectories_wind, "Zaawansowane", "wind_advanced", "Kąt (Wiatr)", 'theta')
    plot_group(group_all, trajectories_wind, "Wszyskie", "wind_advanced_all", "Kąt (Wiatr)", 'theta')
    # Position
    plot_group(group_classical, trajectories_wind, "Klasyczne", "wind_pos_classical", "Pozycja (Wiatr)", 'x')
    plot_group(group_advanced, trajectories_wind, "Zaawansowane", "wind_pos_advanced", "Pozycja (Wiatr)", 'x')
    plot_group(group_all, trajectories_wind, "Wszystkie", "wind_pos_advanced_all", "Pozycja (Wiatr)", 'x')
    # Control
    plot_group(group_classical, trajectories_wind, "Klasyczne", "wind_control_classical", "Sterowanie (Wiatr)", 'u')
    plot_group(group_advanced, trajectories_wind, "Zaawansowane", "wind_control_advanced", "Sterowanie (Wiatr)", 'u')
    plot_group(group_advanced, trajectories_wind, "Wszystkie", "wind_control_advanced_all", "Sterowanie (Wiatr)", 'u')
    
    # --- New LMPC Combined Plots ---
    # Nominal
    plot_group(group_all_lmpc, trajectories_nom, "All + LMPC", "nominal_all_lmpc_theta", "Kąt (Nominal) - All+LMPC", 'theta')
    plot_group(group_all_lmpc, trajectories_nom, "All + LMPC", "nominal_all_lmpc_pos", "Pozycja (Nominal) - All+LMPC", 'x')
    plot_group(group_all_lmpc, trajectories_nom, "All + LMPC", "nominal_all_lmpc_u", "Sterowanie (Nominal) - All+LMPC", 'u')
    # Wind
    plot_group(group_all_lmpc, trajectories_wind, "All + LMPC", "wind_all_lmpc_theta", "Kąt (Wiatr) - All+LMPC", 'theta')
    plot_group(group_all_lmpc, trajectories_wind, "All + LMPC", "wind_all_lmpc_pos", "Pozycja (Wiatr) - All+LMPC", 'x')
    plot_group(group_all_lmpc, trajectories_wind, "All + LMPC", "wind_all_lmpc_u", "Sterowanie (Wiatr) - All+LMPC", 'u')
    # --- MPC Study Plots ---
    # Nominal
    plot_group(group_mpc_study, trajectories_nom, "MPC-J2 Study", "nominal_mpc_j2_study", "MPC-J2 r_abs (Nominal)", 'u')
    plot_group(group_mpc_study, trajectories_nom, "MPC-J2 Study", "nominal_mpc_j2_study_theta", "MPC-J2 r_abs (Nominal)", 'theta')
    # Wind
    plot_group(group_mpc_study, trajectories_wind, "MPC-J2 Study", "wind_mpc_j2_study", "MPC-J2 r_abs (Wiatr)", 'u')
    plot_group(group_mpc_study, trajectories_wind, "MPC-J2 Study", "wind_mpc_j2_study_theta", "MPC-J2 r_abs (Wiatr)", 'theta')
    
    # --- Generate Transposed Tables ---
    
    def print_transposed_table(scenario_suffix, title, caption, label, metrics_map):
        """
        metrics_map: dict of {InputKey: DisplayName}
        """
        # Define desired order
        # User requested: pd-pd pd-lqr lmpc mpc mpc-j2 fuzzy
        desired_order = ['PID-PID', 'PID-LQR', 'LMPC', 'MPC', 'MPC-J2', 'Fuzzy-LQR']
        
        # Filter controllers to only include those in desired_order that actually exist
        ordered_keys = [k for k in desired_order if k in controllers]
        
        print(f"\n\n=== LATEX TABLE: {title} ===")
        print(r"\begin{table}[h!]")
        print(r"    \centering")
        print(f"    \\caption{{{caption}}}")
        print(f"    \\label{{{label}}}")
        
        # Determine columns string: |l|c|c|...|
        cols_str = "|l|" + "c|" * len(ordered_keys)
        print(f"    \\begin{{tabular}}{{{cols_str}}}")
        print(r"        \hline")
        
        # Header Row
        headers = ["Wskaźnik"] + ordered_keys
        print(f"        \\textbf{{{headers[0]}}} & " + " & ".join([f"\\textbf{{{h}}}" for h in headers[1:]]) + r" \\ \hline")
        
        # Data Rows
        for key, display_name in metrics_map.items():
            if key == "SEPARATOR":
                print(r"        \hline")
                # print(r"        \multicolumn{" + str(len(ordered_keys)+1) + r"}{|c|}{\textit{" + display_name + r"}} \\ \hline")
                continue

            row_str = f"        {display_name}"
            for name in ordered_keys:
                m = results[f"{name}_{scenario_suffix}"]
                val = m.get(key, 0.0)
                if val is None: val = 999.9 # Handle NaNs if any
                
                # Formatting
                # Formatting - consistent 5 decimal places for "elegant" look
                val_str = f"{val:.5f}"
                      
                row_str += f" & {val_str}"
            print(row_str + r" \\ \hline")
            
        print(r"    \end{tabular}")
        print(r"\end{table}")

    # Nominal Table Metrics
    metrics_nom = {
        'mse_theta': r'$MSE_\theta$',
        'iae_theta': r'$IAE_\theta$',
        'ts_theta': r'$T_{s, \theta}$ [s]',
        'SEPARATOR': 'Pozycja',
        'mse_x': r'$MSE_x$',
        'ts_x': r'$T_{s, x}$ [s]',
        'energy_l2': r'$E_{u}$'
    }
    print_transposed_table("Nominal", "NOMINAL", "Wskaźniki jakości (Kąt i Pozycja) - warunki nominalne", "tab:results_nominal", metrics_nom)

    # Wind Table Metrics
    metrics_wind = {
        'mse_theta': r'$MSE_\theta$',
        'iae_theta': r'$IAE_\theta$',
        'max_theta': r'$Max |\theta|$ [rad]',
        'SEPARATOR': 'Pozycja',
        'mse_x': r'$MSE_x$',
        'max_x': r'$Max |x|$ [m]',
        'energy_l2': r'$E_{u}$'
    }
    print_transposed_table("Wind", "WIND", "Wskaźniki jakości (Kąt i Pozycja) - zakłócenia wiatrem", "tab:results_wind", metrics_wind)
    
    print("\nCombined plots and tables generated.")

if __name__ == "__main__":
    main()