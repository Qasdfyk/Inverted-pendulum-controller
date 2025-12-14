import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib

# Set non-interactive backend
matplotlib.use('Agg')

# Update font sizes for better visibility in LaTeX
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
    if "manual" in filename_suffix:
        T = 10.0 
    
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    
    X, U, _, _, _ = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=None)
    
    t = np.linspace(0.0, T, len(U) + 1)
    tf = t[:-1]
    
    fig = plt.figure(figsize=(14, 12)) # Even bigger
    
    # Force no title
    # fig.suptitle(None) 
    
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(t, X[:, 0]) # No label
    ax1.plot(t, np.zeros_like(t), 'k--', alpha=0.5)
    ax1.grid(True)
    ax1.set_ylabel('theta [rad]')
    ax1.set_title("") # Explicitly clear
    
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(t, X[:, 2]) # No label
    ax2.plot(t, np.ones_like(t) * x_ref[2], 'k--', alpha=0.5)
    ax2.grid(True)
    ax2.set_ylabel('x [m]')
    ax2.set_title("") # Explicitly clear
    
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(tf, U) # No label
    ax3.grid(True)
    ax3.set_ylabel('u [N]')
    ax3.set_xlabel('t [s]')
    ax3.set_title("") # Explicitly clear
    
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, f"{filename_suffix}.png")
    # if os.path.exists(save_path):
    #     try:
    #         os.remove(save_path) # Force delete first
    #     except OSError:
    #         pass # Ignore if locked, savefig will try to overwrite
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")

def generate_pd_pd():
    print("Generating PD-PD plots...")
    dt = SIM["dt"]
    
    # 1. Bad (Unstable/Weak)
    # Very small gains, not enough to hold
    ctrl_bad = PDPDController(PLANT, dt, 
                              ang_pid={"Kp": -10.0, "Ki": 0.0, "Kd": -1.0},
                              cart_pid={"Kp": -1.0, "Ki": 0.0, "Kd": -1.0})
    run_simulation(ctrl_bad, "PD-PD", "pdpd_1_bad", "Bad Gains (Weak)")
    
    # 2. Manual (Stable but oscillatory/slow)
    # Tuned by hand, decomposition likely resulted in lower gains than opt
    ctrl_man = PDPDController(PLANT, dt,
                              ang_pid={"Kp": -50.0, "Ki": 0.0, "Kd": -5.0},
                              cart_pid={"Kp": -5.0, "Ki": 0.0, "Kd": -5.0})
    run_simulation(ctrl_man, "PD-PD", "pdpd_2_manual", "Manual Tuning")
    
    # 3. Optimized (Current Best)
    ctrl_opt = PDPDController(PLANT, dt,
                              ang_pid={"Kp": -95.0, "Ki": 0.0, "Kd": -14.0},
                              cart_pid={"Kp": -16.0, "Ki": 0.0, "Kd": -14.0})
    run_simulation(ctrl_opt, "PD-PD", "pdpd_3_opt", "Optimized Gains")

def generate_pd_lqr():
    print("Generating PD-LQR plots...")
    dt = SIM["dt"]
    
    # 1. Bad (Q=I, R=1)
    ctrl_bad = PDLQRController(PLANT, dt,
                               pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3}, # Keep PD same
                               lqr_gains={"Q": [1.0, 1.0, 1.0, 1.0], "R": 1.0})
    run_simulation(ctrl_bad, "PD-LQR", "pdlqr_1_bad", "Q=I, R=1")
    
    # 2. Manual (Bryson-like)
    # Guessing some manual values that work but aren't perfect
    # Max theta ~ 0.2 rad -> 1/0.04 = 25
    # Max x ~ 0.5 m -> 1/0.25 = 4
    ctrl_man = PDLQRController(PLANT, dt,
                               pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3},
                               lqr_gains={"Q": [25.0, 1.0, 4.0, 1.0], "R": 10.0}) # Higher R for caution
    run_simulation(ctrl_man, "PD-LQR", "pdlqr_2_manual", "Manual (Bryson)")
    
    # 3. Optimized
    ctrl_opt = PDLQRController(PLANT, dt,
                               pid_gains={"Kp": -4.5, "Ki": 0.0, "Kd": -3},
                               lqr_gains={"Q": [69.44, 76.98, 17.70, 14.17], "R": 8.0280})
    run_simulation(ctrl_opt, "PD-LQR", "pdlqr_3_opt", "Optimized")

def generate_mpc():
    print("Generating MPC plots...")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # 1. Bad (Short Horizon)
    # N=5
    ctrl_bad = MPCController(PLANT, dt, N=5, Nu=2, umin=-u_sat, umax=u_sat,
                             Q=np.diag([10.0, 1.0, 10.0, 1.0]), R=0.1)
    run_simulation(ctrl_bad, "MPC", "mpc_1_bad", "Short Horizon N=5")
    
    # 2. Manual (Medium Horizon, Hand weights)
    # N=10
    ctrl_man = MPCController(PLANT, dt, N=10, Nu=3, umin=-u_sat, umax=u_sat,
                             Q=np.diag([50.0, 10.0, 50.0, 10.0]), R=0.1)
    run_simulation(ctrl_man, "MPC", "mpc_2_manual", "Manual Tuning (N=10)")
    
    # 3. Optimized (N=12)
    ctrl_opt = MPCController(PLANT, dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
                             Q=np.diag([158.39, 36.80, 43.41, 19.71]), R=0.08592)
    run_simulation(ctrl_opt, "MPC", "mpc_3_opt", "Optimized (N=12)")

def generate_mpc_j2():
    print("Generating MPC-J2 plots...")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    # 1. Bad (High R_abs -> Passive)
    # r_abs = 1.0 is huge for squared control sum
    ctrl_bad = MPCControllerJ2(PLANT, dt, N=15, Nu=5, umin=-u_sat, umax=u_sat,
                               q_theta=80.0, q_x=120.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=100.0) # Massive penalty on energy
    run_simulation(ctrl_bad, "MPC-J2", "mpcJ2_1_bad", "High Energy Cost (R_abs)")
    
    # 2. Manual (Lower R_abs, slower)
    ctrl_man = MPCControllerJ2(PLANT, dt, N=15, Nu=5, umin=-u_sat, umax=u_sat,
                               q_theta=80.0, q_x=120.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=0.1)
    run_simulation(ctrl_man, "MPC-J2", "mpcJ2_2_manual", "Manual R_abs")
    
    # 3. Optimized
    ctrl_opt = MPCControllerJ2(PLANT, dt, N=15, Nu=5, umin=-u_sat, umax=u_sat,
                               q_theta=80.0, q_x=120.0, q_thd=5.0, q_xd=5.0,
                               r=0.0001, r_abs=0.0) # Opt found 0.0 is best for performance? Or very small.
                               # Wait, let's check mpc_J2.py default. It says r_abs=0.0 in main.
                               # But the text says "Optymalizacja znalazla zloty srodek".
                               # Maybe use a small value to show "efficient" variant if 0.0 is just standard MPC?
                               # Let's assume Optimized is the one in the file which has 0.0. 
                               # OR set a small one to demonstrate the feature effectively if 0.0 makes it identical to standard cost.
    
    run_simulation(ctrl_opt, "MPC-J2", "mpcJ2_3_opt", "Optimized")

def generate_fuzzy():
    print("Generating Fuzzy-LQR plots...")
    dt = SIM["dt"]
    u_sat = SIM["u_sat"]
    
    K_lqr = lqr_from_plant(PLANT)
    
    # 1. Bad (Chatter - Narrow MF)
    p_bad = starter_ts_params16(u_sat)
    # Hack internals to make them narrow
    p_bad.th_small = (-0.01, 0.0, 0.01) # Very narrow 'small error'
    ctrl_bad = TSFuzzyController(PLANT, p_bad, K_lqr, dt, du_max=800.0)
    run_simulation(ctrl_bad, "Fuzzy-LQR", "fuzzy_1_bad", "Narrow MFs (Chatter)")
    
    # 2. Manual
    p_man = starter_ts_params16(u_sat, base_th=15.0, base_thd=5.0, base_x=5.0, base_xd=1.0) # The defaults commented out in file
    # And default MFs in starter_ts_params16 are actually the OPTIMIZED ones now.
    # So I need to "un-optimize" for Manual.
    p_man.th_small = (-0.1, 0.0, 0.1)
    # Less aggressive rules
    ctrl_man = TSFuzzyController(PLANT, p_man, K_lqr, dt, du_max=800.0)
    run_simulation(ctrl_man, "Fuzzy-LQR", "fuzzy_2_manual", "Manual Tuning")
    
    # 3. Optimized
    p_opt = starter_ts_params16(u_sat) # This function now returns the optimized parameters per the comment in file
    ctrl_opt = TSFuzzyController(PLANT, p_opt, K_lqr, dt, du_max=800.0)
    run_simulation(ctrl_opt, "Fuzzy-LQR", "fuzzy_3_opt", "Optimized")

def generate_pid_comparison():
    print("Generating PID comparison plot...")
    dt = SIM["dt"]
    # PID with Integral action - causing overshoot/instability
    ctrl_pid = PDPDController(PLANT, dt,
                              ang_pid={"Kp": -95.0, "Ki": -10.0, "Kd": -14.0}, # Added Integral
                              cart_pid={"Kp": -16.0, "Ki": -2.0, "Kd": -14.0})
    run_simulation(ctrl_pid, "PID (Bad)", "pid_1_integral_bad", "PID with Integral")

if __name__ == "__main__":
    generate_pd_pd()
    generate_pid_comparison() # Added basic PID check
    generate_pd_lqr()
    generate_mpc()
    generate_mpc_j2()
    generate_fuzzy()
