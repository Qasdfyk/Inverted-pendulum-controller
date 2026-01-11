"""
Robustness Experiment: Testing controllers with perturbed model parameters.

This script runs experiments where controllers are tuned for the nominal plant,
but the actual plant has modified parameters (e.g., +10% pendulum mass).
This tests the robustness of controllers to model uncertainty.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Update font sizes for better visibility in LaTeX (same as run_experiments.py)
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
from controllers.lmpc import LinearMPCController

from controllers.pd_pd import PDPDController
from controllers.pd_lqr import PDLQRController
from controllers.mpc import MPCController
from controllers.mpc_J2 import MPCControllerJ2
from controllers.mpc_utils import PLANT, SIM, f_nonlinear, rk4_step_wind, mse, mae, iae, ise, control_energy_l2, control_energy_l1, settling_time
from controllers.fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant
from time import perf_counter
from typing import Optional, Callable

# Output directories
SAVE_DIR = os.path.join(os.path.dirname(__file__), '../latex/images_odpornosc')
RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'robustness_results.json')
os.makedirs(SAVE_DIR, exist_ok=True)

# Define perturbed plant parameters (10% more pendulum mass)
MASS_PERTURBATION = 0.10  # 10% increase
PLANT_PERTURBED = {
    "M": PLANT["M"],                          # Cart mass unchanged
    "m": PLANT["m"] * (1 + MASS_PERTURBATION),  # Pendulum mass +10%
    "l": PLANT["l"],                          # Length unchanged
    "g": PLANT["g"]                           # Gravity unchanged
}

print(f"Nominal plant: M={PLANT['M']}, m={PLANT['m']}, l={PLANT['l']}")
print(f"Perturbed plant: M={PLANT_PERTURBED['M']}, m={PLANT_PERTURBED['m']:.4f}, l={PLANT_PERTURBED['l']}")
print(f"Mass perturbation: +{MASS_PERTURBATION*100:.0f}%")


def simulate_with_perturbed_plant(pars_real, controller, x0, x_ref, T, dt,
                                   u0=0.0,
                                   wind: Optional[Callable[[float], float]] = None):
    """
    Simulate using perturbed plant parameters.
    Controllers compute control based on their internal (nominal) model,
    but the actual plant evolves with pars_real (perturbed).
    """
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy()
    u_prev = float(u0)
    traj = [x.copy()]
    forces = []
    Fw_tr = []
    t = 0.0

    ctrl_time_total = 0.0
    sim_t0 = perf_counter()

    for _ in range(steps):
        # Controller computes control (using its internal nominal model)
        t0c = perf_counter()
        u_seq = controller.compute_control(x, x_ref, u_prev)
        ctrl_time_total += perf_counter() - t0c

        u_apply = float(u_seq[0])
        forces.append(u_apply)

        F_cur = float(wind(t)) if wind else 0.0
        Fw_tr.append(F_cur)

        # Plant evolves with REAL (perturbed) parameters
        x = rk4_step_wind(f_nonlinear, x, u_apply, pars_real, dt, t, wind)
        traj.append(x.copy())

        u_prev = u_apply
        t += dt

    sim_time_wall = perf_counter() - sim_t0

    return np.vstack(traj), np.asarray(forces), np.asarray(Fw_tr), ctrl_time_total, sim_time_wall


def calculate_metrics(t, x, u, x_ref, dt):
    """Calculate quality metrics for the experiment."""
    metrics = {}
    
    theta = x[:, 0]
    pos = x[:, 2]
    
    ref_theta = np.zeros_like(theta)
    ref_pos = np.ones_like(pos) * x_ref[2]
    
    metrics['mse_theta'] = mse(theta, ref_theta)
    metrics['mae_theta'] = mae(theta, ref_theta)
    metrics['mse_x'] = mse(pos, ref_pos)
    metrics['mae_x'] = mae(pos, ref_pos)
    
    metrics['iae_theta'] = iae(t, theta, ref_theta)
    metrics['ise_theta'] = ise(t, theta, ref_theta)
    metrics['iae_x'] = iae(t, pos, ref_pos)
    metrics['ise_x'] = ise(t, pos, ref_pos)
    
    tf = t[:-1]
    metrics['energy_l2'] = control_energy_l2(tf, u)
    metrics['energy_l1'] = control_energy_l1(tf, u)
    
    metrics['ts_theta'] = settling_time(t, theta, ref_theta, eps=0.01, hold_time=0.5)
    metrics['ts_x'] = settling_time(t, pos, ref_pos, eps=0.01, hold_time=0.5)
    
    metrics['max_theta'] = np.max(np.abs(theta))
    metrics['max_x'] = np.max(np.abs(pos))
    
    # Check if pendulum fell (theta > pi/4 rad ~ 45 deg)
    metrics['fell'] = bool(np.max(np.abs(theta)) > np.pi / 4)
    
    return metrics


def run_robustness_experiment(ctrl, name, plant_real):
    """Run experiment with controller designed for nominal plant but simulated on perturbed plant."""
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]
    
    X, U, Fw, t_ctrl, t_sim = simulate_with_perturbed_plant(
        plant_real, ctrl, x0, x_ref, T, dt, wind=None
    )
    
    t = np.linspace(0.0, T, len(U) + 1)
    metrics = calculate_metrics(t, X, U, x_ref, dt)
    metrics['ctrl_time'] = t_ctrl
    metrics['sim_time'] = t_sim
    
    return t, X, U, metrics


def plot_combined(trajectories, signal_type, filename, title):
    """
    Plot combined trajectories for all controllers.
    signal_type: 'theta', 'x', or 'u'
    Uses the same style as run_experiments.py for consistency.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    plot_ref = False
    t_ref = None
    ref_val = None
    
    for name, (t, X, U) in trajectories.items():
        tf = t[:-1]
        
        if signal_type == 'theta':
            ax.plot(t, X[:, 0], label=name, linewidth=3)
            ylabel = r'$\theta$ [rad]'
            t_ref = t
            ref_val = np.zeros_like(t_ref)
            plot_ref = True
        elif signal_type == 'x':
            ax.plot(t, X[:, 2], label=name, linewidth=3)
            ylabel = r'x [m]'
            t_ref = t
            ref_val = np.zeros_like(t_ref)
            plot_ref = True
        elif signal_type == 'u':
            ax.plot(tf, U, label=name, linewidth=3)
            ylabel = r'u [N]'
            plot_ref = False
    
    # Add reference line (same style as run_experiments.py)
    if plot_ref and t_ref is not None:
        ax.plot(t_ref, ref_val, 'k--', alpha=0.3, linewidth=2, label='Wartość zadana')
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Czas [s]')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def run_sensitivity_analysis(controllers):
    """
    Run sensitivity analysis across a range of mass perturbations.
    Returns dict: {controller_name: {perturbation: iae_theta}}
    """
    # Define perturbation range: -75% to +200%
    perturbations = [-0.75, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]
    
    sensitivity_results = {name: {} for name in controllers.keys()}
    
    print("\n" + "="*60)
    print("Running Sensitivity Analysis (Mass Perturbation Sweep)")
    print("="*60)
    
    for pert in perturbations:
        # Create perturbed plant
        plant_pert = {
            "M": PLANT["M"],
            "m": PLANT["m"] * (1 + pert),
            "l": PLANT["l"],
            "g": PLANT["g"]
        }
        
        print(f"\n  Perturbation: {pert*100:+.0f}% (m = {plant_pert['m']:.4f} kg)")
        
        for name, ctrl in controllers.items():
            # Run simulation
            X, U, _, _, _ = simulate_with_perturbed_plant(
                plant_pert, ctrl, x0, x_ref, T, dt, wind=None
            )
            
            t = np.linspace(0.0, T, len(U) + 1)
            metrics = calculate_metrics(t, X, U, x_ref, dt)
            
            # Store IAE theta (or mark as failed if pendulum fell)
            if metrics['fell']:
                sensitivity_results[name][pert] = np.nan
                print(f"    {name}: FELL")
            else:
                sensitivity_results[name][pert] = metrics['iae_theta']
                print(f"    {name}: IAE_th = {metrics['iae_theta']:.5f}")
    
    return perturbations, sensitivity_results


def plot_sensitivity_analysis(perturbations, sensitivity_results, filename):
    """
    Plot sensitivity analysis: IAE vs mass perturbation for all controllers.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert perturbations to percentages for X axis
    pert_percent = [p * 100 for p in perturbations]
    
    markers = ['o', 's', '^', 'D', 'v']
    
    for (name, results), marker in zip(sensitivity_results.items(), markers):
        iae_values = [results.get(p, np.nan) for p in perturbations]
        ax.plot(pert_percent, iae_values, label=name, linewidth=3, 
                marker=marker, markersize=10, markeredgewidth=2)
    
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=2)
    ax.set_xlabel('Zmiana masy wahadla [%]')
    ax.set_ylabel(r'$IAE_\theta$')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename))
    plt.close(fig)
    print(f"  Saved: {filename}")


def print_results_table(results, controllers):
    """Print LaTeX table for robustness results."""
    print("\n\n=== LATEX TABLE: Robustness Results ===")
    print(r"\begin{table}[h!]")
    print(r"    \centering")
    print(r"    \caption{Wskazniki jakosci - odpornosc na zmiane parametrow modelu (+10\% masy wahadla)}")
    print(r"    \label{tab:results_robustness}")
    
    cols_str = "|l|" + "c|" * len(controllers)
    print(f"    \\begin{{tabular}}{{{cols_str}}}")
    print(r"        \hline")
    
    headers = ["Wskaznik"] + list(controllers.keys())
    print(f"        \\textbf{{{headers[0]}}} & " + " & ".join([f"\\textbf{{{h}}}" for h in headers[1:]]) + r" \\ \hline")
    
    metrics_to_show = [
        ('mse_theta', r'$MSE_\theta$'),
        ('iae_theta', r'$IAE_\theta$'),
        ('ts_theta', r'$T_{s, \theta}$ [s]'),
        ('SEPARATOR', ''),
        ('mse_x', r'$MSE_x$'),
        ('ts_x', r'$T_{s, x}$ [s]'),
        ('SEPARATOR', ''),
        ('energy_l2', r'$E_{u}$'),
    ]
    
    for key, display_name in metrics_to_show:
        if key == "SEPARATOR":
            print(r"        \hline")
            continue
        
        row_str = f"        {display_name}"
        for name in controllers.keys():
            result_key = f"{name}_Robustness"
            if result_key in results:
                m = results[result_key]
                val = m.get(key, 0.0)
                if val is None or np.isnan(val):
                    val_str = "---"
                else:
                    val_str = f"{val:.5f}"
            else:
                val_str = "N/A"
            row_str += f" & {val_str}"
        print(row_str + r" \\ \hline")
    
    print(r"    \end{tabular}")
    print(r"\end{table}")


def main():
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # Initialize controllers with NOMINAL plant parameters (as in run_experiments.py)
    controllers = {}
    
    # 1. PD-PD
    # 1. PD-PD
    controllers['PD-PD'] = PDPDController(PLANT, dt,
                                            ang_pid = {"Kp": -40.0, "Ki": -1.0, "Kd": -8.0},
                                            cart_pid = {"Kp": -1.0, "Ki": -0.1, "Kd": -3.0})
    
    # 2. PD-LQR
    controllers['PD-LQR'] = PDLQRController(PLANT, dt,
                                                pid_gains = {"Kp": -1.5, "Ki": 0.1, "Kd": -1.0},
                                                lqr_gains = {"Q": [1.0, 1.0, 1.0, 1.0], "R": 1.0})

    # 3. LMPC
    controllers['LMPC'] = LinearMPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                              Q=np.diag([15.0, 1.0, 10.0, 1.0]), R=0.1)
    
    # 4. MPC
    controllers['MPC'] = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                       Q=np.diag([158.39, 40.80, 43.41, 19.71]), R=0.08592)
    
    # 5. MPC-J2
    controllers['MPC-J2'] = MPCControllerJ2(
        pars=PLANT, dt=dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
        Q=np.diag([158.39, 40.80, 43.41, 19.71]), R=0.08592, r_abs=0.001
    )
    
    # 6. Fuzzy-LQR
    K_lqr = lqr_from_plant(PLANT)
    p_opt = starter_ts_params16(u_sat)
    controllers['Fuzzy-LQR'] = TSFuzzyController(PLANT, p_opt, K_lqr, dt, du_max=800.0, ramp_T=1.0)

    results = {}
    trajectories = {}
    
    print("\n" + "="*60)
    print("Running Robustness Experiments (Perturbed Plant)")
    print("="*60)
    
    for name, ctrl in controllers.items():
        print(f"\n  Simulating {name}...")
        t, X, U, metrics = run_robustness_experiment(ctrl, name, PLANT_PERTURBED)
        results[f"{name}_Robustness"] = metrics
        trajectories[name] = (t, X, U)
        
        if metrics['fell']:
            print(f"    WARNING: Pendulum fell! Max theta = {metrics['max_theta']:.4f} rad")
        else:
            print(f"    OK: Max theta = {metrics['max_theta']:.4f} rad, MSE_theta = {metrics['mse_theta']:.6f}")
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Generate combined plots
    print("\nGenerating combined plots...")
    plot_combined(trajectories, 'theta', 'robustness_theta.png', 
                  r'Przebieg kąta $\theta$ - odporność na zmianę parametrów')
    plot_combined(trajectories, 'x', 'robustness_x.png', 
                  r'Przebieg pozycji $x$ - odporność na zmianę parametrów')
    plot_combined(trajectories, 'u', 'robustness_u.png', 
                  r'Sygnał sterujący $u$ - odporność na zmianę parametrów')
    
    # Print LaTeX table
    print_results_table(results, controllers)
    
    # Run sensitivity analysis (sweep over perturbations)
    perturbations, sensitivity_results = run_sensitivity_analysis(controllers)
    
    # Plot sensitivity analysis
    print("\nGenerating sensitivity plot...")
    plot_sensitivity_analysis(perturbations, sensitivity_results, 'robustness_sensitivity.png')
    
    # Save sensitivity results
    sensitivity_file = os.path.join(os.path.dirname(__file__), 'robustness_sensitivity.json')
    # Convert to serializable format
    sens_save = {name: {str(k): v for k, v in res.items()} 
                 for name, res in sensitivity_results.items()}
    with open(sensitivity_file, 'w') as f:
        json.dump(sens_save, f, indent=4)
    print(f"Sensitivity results saved to {sensitivity_file}")
    
    # Print comparison summary
    print("\n" + "="*60)
    print("SUMMARY: Robustness Analysis (+10% perturbation)")
    print("="*60)
    print(f"Plant perturbation: +{MASS_PERTURBATION*100:.0f}% pendulum mass")
    print(f"Nominal m = {PLANT['m']:.4f} kg -> Perturbed m = {PLANT_PERTURBED['m']:.4f} kg")
    print()
    
    for name in controllers.keys():
        key = f"{name}_Robustness"
        if key in results:
            m = results[key]
            status = "FELL" if m.get('fell', False) else "STABLE"
            print(f"{name:15s}: {status:7s} | MSE_th={m['mse_theta']:.6f} | Ts_th={m['ts_theta']:.2f}s | E_u={m['energy_l2']:.2f}")
    
    print("\nAll robustness experiments completed!")


if __name__ == "__main__":
    main()

