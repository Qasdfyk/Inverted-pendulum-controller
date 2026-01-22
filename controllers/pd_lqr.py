from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import env from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ZMIANA: Importujemy solwer dla czasu dyskretnego i funkcję do dyskretyzacji
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

from mpc_utils import (PLANT, SIM, Wind, f_nonlinear, rk4_step, simulate_mpc, 
                       print_summary, animate_cartpole, linearize_upright,
                       mse, mae, iae, ise, control_energy_l2, control_energy_l1,
                       settling_time, overshoot, steady_state_error, disturbance_robustness)

class PDLQRController:
    def __init__(self, pars: dict, dt: float, 
                 pid_gains: dict, lqr_gains: dict,
                 u_limit: float = 80.0, integ_limit: float = 5.0):
        self.pars = pars
        self.dt = dt
        self.pid = pid_gains
        self.lqr = lqr_gains
        self.u_limit = u_limit
        self.integ_limit = integ_limit
        
        self.z = 0.0 # integrator for PID

        # --- OBLICZANIE WZMOCNIEŃ LQR DLA CZASU DYSKRETNEGO (DLQR) ---
        M, m, l, g = pars["M"], pars["m"], pars["l"], pars["g"]
        
        # 1. Pobranie modelu ciągłego (Linearizacja)
        A, B = linearize_upright(M, m, l, g)
        
        # 2. Dyskretyzacja modelu (Zero-Order Hold)
        # Promotor wymagał przejścia z t na k.
        # Tworzymy macierze Ad, Bd dla okresu próbkowania dt
        C_dummy = np.zeros((1, A.shape[0]))
        D_dummy = np.zeros((1, 1))
        sys_d = cont2discrete((A, B, C_dummy, D_dummy), dt, method='zoh')
        Ad = sys_d[0]
        Bd = sys_d[1]
        
        # 3. Przygotowanie macierzy wag
        Q_diag = lqr_gains.get("Q", [1, 1, 1, 1])
        R_val = lqr_gains.get("R", 1.0)
        Q = np.diag(Q_diag)
        R = np.array([[R_val]], dtype=float)
        
        # 4. Rozwiązanie Dyskretnego Równania Riccatiego (DARE)
        # To odpowiada wzorowi P = A^T P A - ... w pracy
        P = solve_discrete_are(Ad, Bd, Q, R)
        
        # 5. Obliczenie wzmocnienia K dla DLQR
        # Wzór: K = (R + B^T P B)^-1 * (B^T P A)
        BtP = Bd.T @ P
        self.Klqr = np.linalg.solve(R + BtP @ Bd, BtP @ Ad).ravel()
        
        print(f"DLQR Gain K (Discrete): {self.Klqr}")

    def compute_control(self, x0, x_ref, u_prev):
        # x0 = [th, thd, x, xd]
        th, thd, pos, posd = x0
        
        target_pos = x_ref[2]

        # --- CZĘŚĆ PID (Pozycja wózka) ---
        e_x = target_pos - pos
        # Derivative on Measurement (użycie prędkości zamiast różnicy błędów)
        de_x = -posd 
        
        Kp = float(self.pid.get("Kp", 0.0))
        Ki = float(self.pid.get("Ki", 0.0))
        Kd = float(self.pid.get("Kd", 0.0))
        
        if Ki != 0.0:
            self.z += e_x * self.dt
            self.z = float(np.clip(self.z, -self.integ_limit, self.integ_limit))
            
        u_pid = Kp * e_x + Ki * self.z + Kd * de_x

        # --- CZĘŚĆ LQR (Pełny stan) ---
        # Uchyb stanu od punktu pracy
        state_dev = np.array([th, thd, pos - target_pos, posd])
        
        # Prawo sterowania: u = -K * x
        u_lqr = - float(self.Klqr @ state_dev)
        
        # Suma regulatorów
        u = u_pid + u_lqr
        u = float(np.clip(u, -self.u_limit, self.u_limit))
        
        return np.array([u])

if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    wind = None
    # wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)

    # Parametry (Tuning)
    pid_gains = {"Kp": -7.0, "Ki": 0.1, "Kd": -3.0}
    lqr_gains = {"Q": [200.0, 3.0, 35.0, 40.0], "R": 1}
    
    ctrl = PDLQRController(
        pars=plant, dt=dt, 
        pid_gains=pid_gains, lqr_gains=lqr_gains,
        u_limit=80.0, integ_limit=5.0
    )

    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)

    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    # --- WYKRESY ---
    controller_name = "pd_lqr"
    disturbance = wind is not None
    step_type = "position_step"
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(f"{controller_name}   |   wind={'on' if disturbance else 'off'}   |   step={step_type}",
                 fontsize=12, y=0.98)
    
    # Wykresy stanu (ciągłe linie są OK dla fizyki)
    ax1 = fig.add_subplot(3, 1, 1); ax1.plot(t, X[:, 0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3, 1, 2); ax2.plot(t, X[:, 2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    
    # ZMIANA: Wykres sterowania jako 'step' (schodki)
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.step(tf, U, where='post') # Schodkowy charakter sterowania cyfrowego
    ax3.grid(True)
    ax3.set_ylabel('u [N]')
    ax3.set_xlabel('t [s]')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---- metrics ----
    # (Reszta kodu bez zmian)
    th_ref_tr = np.zeros_like(t)
    x_ref_tr  = np.ones_like(t) * x_ref[2]

    eps_theta = 0.01; eps_x = 0.01; hold_time = 0.5
    ess_window_frac = 0.10; ess_window_min = 0.5
    is_step = True

    metrics = {
        "mse_theta": mse(X[:, 0], th_ref_tr),
        "mae_theta": mae(X[:, 0], th_ref_tr),
        "mse_x":     mse(X[:, 2], x_ref_tr),
        "mae_x":     mae(X[:, 2], x_ref_tr),
        "iae_theta": iae(t, X[:, 0], th_ref_tr),
        "ise_theta": ise(t, X[:, 0], th_ref_tr),
        "iae_x":     iae(t, X[:, 2], x_ref_tr),
        "ise_x":     ise(t, X[:, 2], x_ref_tr),
        "e_u_l2":    control_energy_l2(tf, U),
        "e_u_l1":    control_energy_l1(tf, U),
        "sim_time_wall": sim_time_wall,
        "ctrl_time_total": ctrl_time_total,
    }
    
    metrics["t_s_theta"] = settling_time(t, X[:,0], th_ref_tr, eps=eps_theta, hold_time=hold_time) if is_step else float('nan')
    metrics["t_s_x"]     = settling_time(t, X[:,2], x_ref_tr,  eps=eps_x,     hold_time=hold_time) if is_step else float('nan')
    metrics["overshoot_theta"] = overshoot(X[:,0], th_ref_tr[-1]) if is_step else float('nan')
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_tr[-1])  if is_step else float('nan')

    ess_th_mean, ess_th_rms = steady_state_error(t, X[:,0], th_ref_tr, window_frac=ess_window_frac, window_min=ess_window_min)
    ess_x_mean,  ess_x_rms  = steady_state_error(t, X[:,2], x_ref_tr,  window_frac=ess_window_frac, window_min=ess_window_min)
    metrics["ess_theta"] = ess_th_mean
    metrics["ess_x"]     = ess_x_mean

    if disturbance:
        e_th_tf = X[1:, 0] - th_ref_tr[1:]
        e_x_tf  = X[1:, 2] - x_ref_tr[1:]
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, e_th_tf, Fw_tr, window_frac=0.5, window_min=1.0)
        snr_x,  rms_e_x,  _     = disturbance_robustness(tf, e_x_tf,  Fw_tr, window_frac=0.5, window_min=1.0)
        metrics["snr_theta"] = snr_th
        metrics["snr_x"]     = snr_x
    else:
        metrics["snr_theta"] = float('nan')
        metrics["snr_x"]     = float('nan')
    
    print_summary(metrics)

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)