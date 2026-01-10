from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Tuple
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

from mpc_utils import (
    PLANT, SIM, Wind, f_nonlinear, rk4_step_wind, simulate_mpc,
    print_summary, animate_cartpole, linearize_upright,
    mse, mae, iae, ise, control_energy_l2, control_energy_l1,
    settling_time, overshoot, steady_state_error, disturbance_robustness
)

EPS = 1e-12

# =========================
# LQR Helper (Linearization)
# =========================
# linearize_upright imported from mpc_utils

def lqr_from_plant(plant: dict) -> np.ndarray:
    A, B = linearize_upright(plant["M"], plant["m"], plant["l"], plant["g"])
    Q = np.diag([1.0, 1.0, 1.0, 1.0])
    R = np.array([[1.0]], dtype=float)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P).ravel()
    return K

# =========================
# TS-Fuzzy Logic Core
# =========================
def tri_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

@dataclass
class TSParams16:
    th_small: Sequence[float]
    thd_small: Sequence[float]
    x_small: Sequence[float]
    xd_small: Sequence[float]
    F_rules: np.ndarray
    u_sat: float
    sign: int
    flip_u: bool
    gain_scale: float

#def starter_ts_params16(u_sat: float, base_th=15.0, base_thd=5.0, base_x=5.0, base_xd=1.0) -> TSParams16:
# def starter_ts_params16(u_sat: float, base_th=5.2, base_thd=4.5, base_x=20.9, base_xd=18.6) -> TSParams16:
# Optimized Params for Disturbance Rejection (Automated Optimization)
# Best Cost: 0.1007 | Wind: Enabled
def starter_ts_params16(u_sat: float, base_th=100.0, base_thd=5.27, base_x=19.82, base_xd=19.25):
    th_small  = (-0.15, 0.0, 0.15)
    thd_small = (-1.0, 0.0, 1.0)
    x_small   = (-0.3, 0.0, 0.3)
    xd_small  = (-0.5, 0.0, 0.5)

    F_rules = np.zeros((16, 4), dtype=float)

    for idx in range(16):
        bits = [(idx >> b) & 1 for b in (3, 2, 1, 0)]
        largeness = sum(bits)
        
        # Optimized gains (Positive Feedback logic verified by optimizer)
        F_rules[idx, 0] = base_th   + 20.0 * largeness
        F_rules[idx, 1] = base_thd  +  5.0 * largeness
        F_rules[idx, 2] = base_x    +  5.0 * largeness
        F_rules[idx, 3] = base_xd   +  2.0 * largeness

    return TSParams16(
        th_small=th_small, thd_small=thd_small, x_small=x_small, xd_small=xd_small,
        F_rules=F_rules, u_sat=u_sat, sign=-1, flip_u=False, gain_scale=0.36, 
    )

def ts_weights16(th: float, thd: float, x: float, xd: float, p: TSParams16) -> np.ndarray:
    ms_th   = tri_mf(th,  *p.th_small);  ml_th  = 1.0 - ms_th
    ms_thd  = tri_mf(thd, *p.thd_small); ml_thd = 1.0 - ms_thd
    ms_x    = tri_mf(x,   *p.x_small);   ml_x   = 1.0 - ms_x
    ms_xd   = tri_mf(xd,  *p.xd_small);  ml_xd  = 1.0 - ms_xd

    w = np.zeros(16, dtype=float)
    idx = 0
    for a in (ms_th, ml_th):
        for b in (ms_thd, ml_thd):
            for c in (ms_x, ml_x):
                for d in (ms_xd, ml_xd):
                    w[idx] = a * b * c * d
                    idx += 1
    return w / (np.sum(w) + EPS)

class TSFuzzyController:
    def __init__(self, pars: dict, ts_params: TSParams16, K_lqr: np.ndarray, dt: float, du_max: float = 1000.0, ramp_T: float = 2.0):
        self.pars = pars
        self.p = ts_params
        self.K = np.asarray(K_lqr, dtype=float).ravel()
        self.dt = dt
        self.du_max = du_max
        self.ramp_T = ramp_T
        self.u_prev = 0.0
        self.step_counter = 0

    def compute_control(self, x0, x_ref, u_prev):
        # Compatibility wrapper for simulate_mpc
        
        t = self.step_counter * self.dt
        self.step_counter += 1
        
        # update internal u_prev if passed (simulate_mpc passes it)
        self.u_prev = u_prev 
        
        u = self.step(t, x0, x_ref)
        
        # Return as sequence for simulate_mpc (which expects u_seq array)
        return np.array([u])

    def _u_ts(self, th: float, thd: float, ex: float, exd: float) -> float:
        mu = ts_weights16(th, thd, ex, exd, self.p)
        z = np.array([th, thd, ex, exd], dtype=float)
        u_rules = - self.p.sign * (self.p.F_rules @ z)
        u = float(mu @ u_rules) * self.p.gain_scale
        if self.p.flip_u: u = -u
        return u

    def step(self, t: float, state: np.ndarray, ref_state: np.ndarray) -> float:
        th, thd, x, xd = state
        xref = float(ref_state[2])
        target = np.array([0.0, 0.0, xref, 0.0], dtype=float)
        err = state - target
        
        u_lqr = -float(np.dot(self.K, err))

        ex  = x  - xref
        exd = xd - 0.0
        u_ts = self._u_ts(th, thd, ex, exd)

        u = u_lqr + u_ts

        if self.ramp_T > 0:
            alpha = min(1.0, t / self.ramp_T)
            u *= (0.25 + 0.75 * alpha)

        du = np.clip(u - self.u_prev, -self.du_max * self.dt, self.du_max * self.dt)
        u_limited = self.u_prev + du
        
        self.u_prev = float(np.clip(u_limited, -self.p.u_sat, self.p.u_sat))
        return self.u_prev

# =========================
# Simulation Logic
# =========================

# =========================
# Custom Plotting
# =========================
def plot_result(t: np.ndarray, tf: np.ndarray, X: np.ndarray, U: np.ndarray, controller_name: str, disturbance: bool):
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(f"{controller_name} | wind={'on' if disturbance else 'off'}", fontsize=12, y=0.98)
    
    ax1 = fig.add_subplot(3, 1, 1); ax1.plot(t, X[:, 0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3, 1, 2); ax2.plot(t, X[:, 2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    ax3 = fig.add_subplot(3, 1, 3); ax3.plot(tf, U);      ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    plant = PLANT
    dt = SIM["dt"]
    T = SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]

    # --- Wind Settings ---
    #wind = Wind(T, seed=42, Ts=0.05, power=5e-3, smooth=10) # Uncomment for wind
    wind = None 

    K_lqr = lqr_from_plant(plant)
    print(f"LQR Gain: {K_lqr}")

    base_ts = starter_ts_params16(u_sat=SIM["u_sat"])
    picked_ts = base_ts
    
    ctrl = TSFuzzyController(plant, picked_ts, K_lqr, dt, du_max=800.0, ramp_T=1.0)
    
    print(f"Running Simulation (dt={dt}s)...")
    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(
        plant, ctrl, x0, x_ref, T, dt, wind=wind
    )

    # --- Metrics and Plots ---
    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    th_ref_tr = np.zeros_like(t)
    x_ref_tr = np.ones_like(t) * x_ref[2]

    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr), 
        "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x": mse(X[:,2], x_ref_tr), 
        "mae_x": mae(X[:,2], x_ref_tr),
        "iae_theta": iae(t, X[:,0], th_ref_tr), 
        "ise_theta": ise(t, X[:,0], th_ref_tr),
        "iae_x": iae(t, X[:,2], x_ref_tr), 
        "ise_x": ise(t, X[:,2], x_ref_tr),
        "e_u_l2": control_energy_l2(tf, U), 
        "e_u_l1": control_energy_l1(tf, U),
        "sim_time_wall": sim_time_wall, 
        "ctrl_time_total": ctrl_time_total,
        "t_s_theta": settling_time(t, X[:,0], th_ref_tr, 0.01, 0.5),
        "t_s_x": settling_time(t, X[:,2], x_ref_tr, 0.01, 0.5),
        "overshoot_theta": overshoot(X[:,0], th_ref_tr[-1]),
        "overshoot_x": overshoot(X[:,2], x_ref_tr[-1]),
    }
    metrics["ess_theta"], _ = steady_state_error(t, X[:,0], th_ref_tr)
    metrics["ess_x"], _     = steady_state_error(t, X[:,2], x_ref_tr)

    if wind:
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, X[1:,0]-th_ref_tr[1:], Fw_tr)
        snr_x, rms_e_x, _       = disturbance_robustness(tf, X[1:,2]-x_ref_tr[1:], Fw_tr)
        metrics.update({"snr_theta": snr_th, "snr_x": snr_x})
    
    print("\nSimulation Result:")
    print_summary(metrics)
    
    plot_result(t, tf, X, U, "TS-Fuzzy", wind is not None)

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)