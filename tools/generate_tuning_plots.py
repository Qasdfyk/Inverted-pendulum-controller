import numpy as np
import matplotlib.pyplot as plt
import sys
import os
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
from controllers.mpc_utils import PLANT, SIM, simulate_mpc
from controllers.fuzzy_lqr import TSFuzzyController, starter_ts_params16, lqr_from_plant

SAVE_DIR = os.path.join(os.path.dirname(__file__), '../latex/images/tuning')
os.makedirs(SAVE_DIR, exist_ok=True)

def run_simulation(ctrl, name, filename_suffix, title_suffix):
    dt, T = SIM["dt"], SIM["T"]
    # Extend T for manual tuning plots to show settling if slow
    if "manual" in filename_suffix and "mpc" not in filename_suffix:
         T = 8.0 
    
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    
    X, U, _, _, _ = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=None)
    
    t = np.linspace(0.0, T, len(U) + 1)
    tf = t[:-1]
    
    # Calculate simple metrics for verification log
    max_theta = np.max(np.abs(X[:, 0]))
    settling_idx = np.where(np.abs(X[:, 0]) > 0.05)[0]
    settling_time = t[settling_idx[-1]] if len(settling_idx) > 0 and len(settling_idx) < len(t)-1 else (T if len(settling_idx) > 0 else 0.0)
    
    print(f"[{name} - {title_suffix}] Max Theta: {max_theta:.4f} rad, Settling (~0.05): {settling_time:.2f} s")

    fig = plt.figure(figsize=(14, 12)) 
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, X[:, 0]) 
    ax1.plot(t, np.zeros_like(t), 'k--', alpha=0.5, label='Wartość zadana')
    ax1.grid(True)
    ax1.set_ylabel(r'Kąt $\theta$ [rad]')
    ax1.set_title("") 
    
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, X[:, 2]) 
    ax2.plot(t, np.ones_like(t) * x_ref[2], 'k--', alpha=0.5, label='Wartość zadana')
    ax2.grid(True)
    ax2.set_ylabel(r'Pozycja $x$ [m]')
    ax2.set_title("") 
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(tf, U) 
    ax3.grid(True)
    ax3.set_ylabel(r'Siła sterująca $u$ [N]')
    ax3.set_xlabel('Czas [s]')
    ax3.set_title("") 
    
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, f"{filename_suffix}.png")
    # if os.path.exists(save_path):
    #     try:
    #         os.remove(save_path) 
    #     except OSError:
    #         pass 
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

def generate_pd_pd():
    print("\n--- PD-PD ---")
    dt = SIM["dt"]
    
    # 1. Bad (Unstable/Weak)
    # Gains too low to fight gravity effectively at larger angles
    ctrl_bad = PDPDController(PLANT, dt, 
                              ang_pid = {"Kp": -40.0, "Ki": 0.0, "Kd": -8.0},
                                cart_pid = {"Kp": -1.0, "Ki": 0.0, "Kd": -3.0})
    run_simulation(ctrl_bad, "PD-PD", "pdpd_1_manual", "Bad Gains")
    
    # 2. Manual (Stable but oscillatory)
    # Intuitive tuning: "Need more D for theta"
    ctrl_man = PDPDController(PLANT, dt,
                                ang_pid = {"Kp": -40.0, "Ki": -2.0, "Kd": -8.0},
                                cart_pid = {"Kp": -1.0, "Ki": -2.0, "Kd": -3.0})
    run_simulation(ctrl_man, "PD-PD", "pdpd_2_integral_bad", "Manual Tuning")
    
    # 3. Optimized (Current Best)
    ctrl_opt = PDPDController(PLANT, dt,
                              ang_pid={"Kp": -95.0, "Ki": 0.0, "Kd": -14.0},
                              cart_pid={"Kp": -16.0, "Ki": 0.0, "Kd": -14.0})
    run_simulation(ctrl_opt, "PD-PD", "pdpd_3_opt", "Optimized Gains")

def generate_pd_lqr():
    print("\n--- PD-LQR ---")
    dt = SIM["dt"]
    
    # 1. Bad (Q=I, R=1)
    # Classic "Naive LQR". Physics units mismatch makes this terrible.
    # 1 rad error cost == 1 m error cost == 1 N^2 force cost.
    ctrl_bad = PDLQRController(PLANT, dt,
                               pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3}, 
                               lqr_gains={"Q": [1.0, 1.0, 1.0, 1.0], "R": 1.0})
    run_simulation(ctrl_bad, "PD-LQR", "pdlqr_1_bad", "Q=I, R=1")
    
    # 2. Manual (Bryson-like/Heuristic)
    # Better, but manually picked Q=[25, 1, 4, 1], R=10
    ctrl_man = PDLQRController(PLANT, dt,
                               pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3},
                               lqr_gains={"Q": [25.0, 1.0, 4.0, 1.0], "R": 10.0}) 
    run_simulation(ctrl_man, "PD-LQR", "pdlqr_2_manual", "Manual (Bryson)")
    
    # 3. Optimized
    ctrl_opt = PDLQRController(PLANT, dt,
                                    pid_gains = {"Kp": -1.5, "Ki": 0.0, "Kd": -5.0},
                                    lqr_gains = {"Q": [1.0, 1.0, 500.0, 250.0], "R": 1.0})
    run_simulation(ctrl_opt, "PD-LQR", "pdlqr_3_opt", "Optimized")

def generate_mpc():
    print("\n--- MPC ---")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # 1. Bad (Short Horizon)
    # N=5. Too short to see the "turn around" needed.
    ctrl_bad = MPCController(PLANT, dt, N=5, Nu=2, umin=-u_sat, umax=u_sat,
                             Q=np.diag([10.0, 1.0, 10.0, 1.0]), R=0.1)
    run_simulation(ctrl_bad, "MPC", "mpc_1_bad", "Short Horizon N=5")
    
    # 2. Manual (Medium Horizon, Hand weights)
    # N=10. Stable but maybe not aggressive enough or slightly loose.
    ctrl_man = MPCController(PLANT, dt, N=10, Nu=3, umin=-u_sat, umax=u_sat,
                             Q=np.diag([50.0, 10.0, 50.0, 10.0]), R=0.1)
    run_simulation(ctrl_man, "MPC", "mpc_2_manual", "Manual Tuning (N=10)")
    
    # 3. Optimized (N=12)
    # Found by automated sweep
    ctrl_opt = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                             Q=np.diag([158.39, 36.80, 43.41, 19.71]), R=0.08592)
    run_simulation(ctrl_opt, "MPC", "mpc_3_opt", "Optimized (N=12)")

def generate_mpc_j2():
    print("\n--- MPC-J2 ---")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # Using same Q weights as run_experiments.py (q_theta=40, q_x=40), varying only R_abs
    # 1. Bad (High R_abs -> Passive, refuses to act)
    ctrl_bad = MPCControllerJ2(PLANT, dt, N=15, Nu=7, umin=-u_sat, umax=u_sat,
                               q_theta=40.0, q_x=40.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=1.0) 
    run_simulation(ctrl_bad, "MPC-J2", "mpcJ2_1_bad", "High Energy Cost")
    
    # 2. Manual (R_abs=1.0 - destabilizes when cart drifts too far)
    ctrl_man = MPCControllerJ2(PLANT, dt, N=15, Nu=7, umin=-u_sat, umax=u_sat,
                               q_theta=40.0, q_x=40.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=0.1)
    run_simulation(ctrl_man, "MPC-J2", "mpcJ2_2_manual", "Manual R_abs")
    
    # 3. Optimized (R_abs=0, same as run_experiments.py)
    ctrl_opt = MPCControllerJ2(PLANT, dt, N=15, Nu=7, umin=-u_sat, umax=u_sat,
                               q_theta=40.0, q_x=40.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=0.0) 
    run_simulation(ctrl_opt, "MPC-J2", "mpcJ2_3_opt", "Optimized")

def generate_fuzzy():
    print("\n--- Fuzzy-LQR ---")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    K_lqr = lqr_from_plant(PLANT)
    
    # 1. Bad (Chatter - Narrow MF)
    # Defining very narrow "small" region causes rapid switching between LQR and Aggressive rules.
    p_bad = starter_ts_params16(u_sat)
    p_bad.th_small = (-0.02, 0.0, 0.02) # Extremely narrow
    ctrl_bad = TSFuzzyController(PLANT, p_bad, K_lqr, dt, du_max=800.0)
    run_simulation(ctrl_bad, "Fuzzy-LQR", "fuzzy_1_bad", "Narrow MFs (Chatter)")
    
    # 2. Manual
    # Broader MFs, but maybe rules are too aggressive or too weak.
    # Let's use wider MFs but weak manual rules (defaults from code comments if available, or just guesses).
    # Using defaults from before optimization:
    # base_th=5.0 is actually very weak compared to opt 167.
    p_man = starter_ts_params16(u_sat, base_th=20.0, base_thd=5.0, base_x=10.0, base_xd=2.0)
    p_man.th_small = (-0.2, 0.0, 0.2) # Reasonable width
    ctrl_man = TSFuzzyController(PLANT, p_man, K_lqr, dt, du_max=800.0)
    run_simulation(ctrl_man, "Fuzzy-LQR", "fuzzy_2_manual", "Manual Tuning")
    
    # 3. Optimized
    # Best params found by differential evolution
    p_opt = starter_ts_params16(u_sat) # Function defaults are now the Optimized ones
    ctrl_opt = TSFuzzyController(PLANT, p_opt, K_lqr, dt, du_max=800.0, ramp_T=1.0)
    run_simulation(ctrl_opt, "Fuzzy-LQR", "fuzzy_3_opt", "Optimized")
    
if __name__ == "__main__":
    generate_pd_pd()
    generate_pd_lqr()
    generate_mpc()
    generate_mpc_j2()
    generate_fuzzy()

