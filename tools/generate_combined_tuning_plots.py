import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Update font sizes for better visibility in LaTeX
# Taken from run_experiments.py
plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 15,
    'lines.linewidth': 3
})

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../controllers')))

from controllers.pd_pd import PDPDController
from controllers.pd_lqr import PDLQRController
from controllers.mpc import MPCController
from controllers.mpc_J2 import MPCControllerJ2
from controllers.mpc_utils import PLANT, SIM, simulate_mpc
from controllers.fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant, TSParams16, linearize_upright
from controllers.lmpc import LinearMPCController

SAVE_DIR = os.path.join(os.path.dirname(__file__), '../latex/images/tuning/combined')
os.makedirs(SAVE_DIR, exist_ok=True)

def run_simulation(ctrl, name):
    dt, T = SIM["dt"], SIM["T"]
    x0 = SIM["x0"]
    x_ref = SIM["x_ref"]
    
    # Run simulation
    X, U, _, _, _ = simulate_mpc(PLANT, ctrl, x0, x_ref, T, dt, wind=None)
    
    t = np.linspace(0.0, T, len(U) + 1)
    return t, X, U

def plot_combined_tuning(controller_name, configs, filename_suffix):
    """
    configs: list of dicts with keys: 'label', 'ctrl', 'color', 'linestyle'
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True) # Increased width slightly
    
    # Iterate through configurations
    for config in configs:
        t, X, U = run_simulation(config['ctrl'], config['label'])
        tf = t[:-1]
        
        lbl = config['label']
        col = config.get('color', None)
        ls = config.get('linestyle', '-')
        
        # We plot against t*10 to represent 'k' steps if dt=0.1s, or just scaled time?
        # In run_experiments.py it does: axes[0].plot(t*10, ...) and axes[2].set_xlabel('k')
        # This implies k = t * 10 (since dt is likely 0.02 or 0.1? No, usually dt=0.02 -> 50Hz. t*10 would be 0.2*k? 
        # Actually SIM["dt"] is key.
        # If dt=0.02, t=1.0s -> k=50. t*10 would be 10. That doesn't map to k directly.
        # However, the user explicitly said "take from run_experiments".
        # In run_experiments.py: `axes[0].plot(t*10, ...)` and `axes[2].set_xlabel('k')`.
        # I will strictly follow the user's instruction to copy that behavior.
        
        x_axis_data = t * 10
        x_axis_data_u = tf * 10
        
        # Theta
        axes[0].plot(x_axis_data, X[:, 0], label=lbl, color=col, linestyle=ls)
        
        # Position
        axes[1].plot(x_axis_data, X[:, 2], label=lbl, color=col, linestyle=ls)
        
        # Control
        # Control
        axes[2].step(x_axis_data, np.append(U, U[-1]), where='post', label=lbl, color=col, linestyle=ls)

    # Reference lines (using last t)
    # Theta Ref
    axes[0].plot(x_axis_data, np.zeros_like(x_axis_data), 'k--', alpha=0.3, label='Cel')
    axes[0].set_ylabel(r'$\theta$ $[\mathrm{rad}]$')
    axes[0].grid(True)
    # axes[0].legend(loc='upper right') # Legend might clutter, maybe put on bottom or combine?
    # run_experiments puts legend on axes[1] or 0.
    # Let's put legend on the first plot but carefully.
    axes[0].legend(loc='upper right', framealpha=0.9)
    
    # Position Ref
    axes[1].plot(x_axis_data, np.ones_like(x_axis_data) * SIM["x_ref"][2], 'k--', alpha=0.3, label='Cel')
    axes[1].set_ylabel(r'$\mathrm{x}$ $[\mathrm{m}]$')
    axes[1].grid(True)
    
    # Control
    axes[2].set_ylabel(r' $\mathrm{u}$ $[\mathrm{N}]$')
    axes[2].set_xlabel('k')
    axes[2].grid(True)
    
    plt.tight_layout()
    # SAVE_DIR is already defined globally
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, f"combined_{filename_suffix}.png")
    plt.savefig(path)
    print(f"Saved {path}")
    plt.close(fig)

def main():
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    print("Generating combined tuning plots...")

    # ==========================
    # PDPD
    # ==========================
    print("  PDPD...")
    pid_bad = PDPDController(PLANT, dt,
                                ang_pid={"Kp": -60.0, "Ki": -2.0, "Kd": -9.0},
                                cart_pid={"Kp": -2.0, "Ki": -1.0, "Kd": -1.0})
    
    pid_grid = PDPDController(PLANT, dt,
                              ang_pid={"Kp": -95.0, "Ki": 0.0, "Kd": -14.0},
                              cart_pid={"Kp": -16.0, "Ki": 0.0, "Kd": -14.0})
    
    pid_opt = PDPDController(PLANT, dt,
                             ang_pid={"Kp": -40.0, "Ki": -1.0, "Kd": -8.0},
                             cart_pid={"Kp": -1.0, "Ki": -0.1, "Kd": -3.0})

    configs_pid = [
        {'label': 'Wstępne ($K_{p,\\theta}=-60$)', 'ctrl': pid_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Siatka ($K_{p,\\theta}=-95$)', 'ctrl': pid_grid, 'color': 'blue', 'linestyle': '-'},
        {'label': 'Optymalne ($K_{p,\\theta}=-40$)', 'ctrl': pid_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("PID-PID", configs_pid, "pdpd")

    # ==========================
    # PID-LQR
    # ==========================
    print("  PID-LQR...")
    lqr_bad = PDLQRController(PLANT, dt,
                              pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3.0},
                              lqr_gains={"Q": [1.0, 1.0, 1.0, 1.0], "R": 1.0})
    
    lqr_man = PDLQRController(PLANT, dt,
                              pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3.0},
                              lqr_gains={"Q": [25.0, 1.0, 4.0, 1.0], "R": 10.0})

    lqr_opt = PDLQRController(PLANT, dt,
                              pid_gains={"Kp": -7.0, "Ki": 0.1, "Kd": -3.0},
                              lqr_gains={"Q": [200.0, 3.0, 35.0, 40.0], "R": 1.0})

    configs_lqr = [
        {'label': 'Jednostkowe ($Q=I, R=1$)', 'ctrl': lqr_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Bryson ($Q_{\\theta}=25, R=10$)', 'ctrl': lqr_man, 'color': 'blue', 'linestyle': '-'},
        {'label': 'Optymalne ($Q_{\\theta}=200, R=1$)', 'ctrl': lqr_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("PID-LQR", configs_lqr, "pdlqr")

    # ==========================
    # MPC
    # ==========================
    print("  MPC...")
    mpc_bad = MPCController(PLANT, dt, N=7, Nu=3, umin=-u_sat, umax=u_sat,
                            Q=np.diag([10.0, 1.0, 10.0, 1.0]), R=0.1)
    
    mpc_man = MPCController(PLANT, dt, N=10, Nu=3, umin=-u_sat, umax=u_sat,
                            Q=np.diag([50.0, 10.0, 50.0, 10.0]), R=0.1)
    
    mpc_opt = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                          Q=np.diag([158.39, 40.80, 43.41, 19.71]), R=0.08592)

    configs_mpc = [
        {'label': 'Krótki Horyzont ($N=5$)', 'ctrl': mpc_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Ręczne ($N=10$)', 'ctrl': mpc_man, 'color': 'blue', 'linestyle': '-'},
        {'label': 'Optymalne ($N=12$)', 'ctrl': mpc_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("MPC", configs_mpc, "mpc")

    # ==========================
    # 4. MPC-J2 (Energy)
    # ==========================
    print("  4. MPC-J2...")
    Q_mpc_opt = np.diag([158.39, 40.80, 43.41, 19.71])
    mpc_j2_bad = MPCControllerJ2(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                 Q=Q_mpc_opt, R=0.001, r_abs=10.0)
    
    mpc_j2_man = MPCControllerJ2(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                 Q=Q_mpc_opt, R=0.001, r_abs=5.0)

    mpc_j2_opt = MPCControllerJ2(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                 Q=Q_mpc_opt, R=0.001, r_abs=1.0)
    
    configs_mpc_j2 = [
        {'label': 'Wysoka kara ($R_{abs}=10$)', 'ctrl': mpc_j2_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Średnia kara ($R_{abs}=5$)', 'ctrl': mpc_j2_man, 'color': 'blue', 'linestyle': '-'},
        {'label': 'Optymalne ($R_{abs}=1$)', 'ctrl': mpc_j2_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("MPC-J2", configs_mpc_j2, "mpcj2")

    # ==========================
    # 5. LMPC
    # ==========================
    print("  5. LMPC...")
    lmpc_bad = LinearMPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                   Q=np.diag([1.0, 1.0, 1.0, 1.0]), R=0.1)
    
    lmpc_opt = LinearMPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                                   Q=np.diag([15.0, 1.0, 15.0, 1.0]), R=0.1)
    
    configs_lmpc = [
        {'label': 'Wagi jednostkowe ($Q=I$)', 'ctrl': lmpc_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Optymalne ($Q_{\\theta}=15$)', 'ctrl': lmpc_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("LMPC", configs_lmpc, "lmpc")


    # ==========================
    # 6. Fuzzy-LQR
    # ==========================
    print("  6. Fuzzy-LQR...")
    K_lqr = lqr_from_plant(PLANT)
    
    bad_params = starter_ts_params16(u_sat)
    bad_params.th_small = (-0.02, 0.0, 0.02)
    bad_params.gain_scale = 1.0
    fuzzy_bad = TSFuzzyController(PLANT, bad_params, K_lqr, dt, du_max=800.0)

    manual_params = starter_ts_params16(u_sat, base_th=20.0, base_thd=5.0, base_x=10.0, base_xd=2.0)
    manual_params.th_small = (-0.2, 0.0, 0.2)
    manual_params.gain_scale = 1.0
    fuzzy_manual = TSFuzzyController(PLANT, manual_params, K_lqr, dt, du_max=800.0)

    opt_params = starter_ts_params16(u_sat)
    fuzzy_opt = TSFuzzyController(PLANT, opt_params, K_lqr, dt, du_max=800.0)

    configs_fuzzy = [
        {'label': 'Wąski zakres ($[-0.02, 0.02]$)', 'ctrl': fuzzy_bad, 'color': 'red', 'linestyle': '-'},
        {'label': 'Słabe wzmocnienia ($F_{\\theta}=20$)', 'ctrl': fuzzy_manual, 'color': 'blue', 'linestyle': '-'},
        {'label': 'Optymalne ($F_{\\theta}=40, G=0.36$)', 'ctrl': fuzzy_opt, 'color': 'green', 'linestyle': '-'}
    ]
    plot_combined_tuning("Fuzzy-LQR", configs_fuzzy, "fuzzy")


if __name__ == "__main__":
    main()
