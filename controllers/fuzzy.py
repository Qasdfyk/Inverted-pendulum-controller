from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are
from time import perf_counter

# Import z mpc_utils (musi być w folderze obok!)
from mpc_utils import (
    PLANT, SIM, Wind, f_nonlinear, rk4_step_wind, 
    print_summary, animate_cartpole,
    mse, mae, iae, ise, control_energy_l2, control_energy_l1,
    settling_time, overshoot, steady_state_error, disturbance_robustness
)

EPS = 1e-12

# =========================
# LQR (Stabilizacja podstawowa)
# =========================
def linearize_upright(M: float, m: float, l: float, g: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.array([
        [0.0,                 1.0, 0.0, 0.0],
        [(M + m)*g/(M*l),     0.0, 0.0, 0.0],
        [0.0,                 0.0, 0.0, 1.0],
        [-m*g/M,              0.0, 0.0, 0.0]
    ], dtype=float)
    B = np.array([
        [0.0], [-1.0/(M*l)], [0.0], [1.0/M]
    ], dtype=float)
    return A, B

def lqr_from_plant(plant: dict) -> np.ndarray:
    A, B = linearize_upright(plant["M"], plant["m"], plant["l"], plant["g"])
    Q = np.diag([1.0, 1.0, 500.0, 250.0])
    R = np.array([[1.0]], dtype=float)
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P).ravel()
    return K

# =========================
# TS-fuzzy Logic
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

def starter_ts_params16(u_sat: float) -> TSParams16:
    # Zakresy zbiorów rozmytych
    th_small  = (-0.20, 0.0, 0.20)
    thd_small = (-1.5, 0.0, 1.5)
    x_small   = (-0.4, 0.0, 0.4)
    xd_small  = (-0.8, 0.0, 0.8)

    F_rules = np.zeros((16, 4), dtype=float)
    
    # !!! KLUCZOWA ZMIANA DLA dt=0.1 !!!
    # Zmniejszone wartości bazowe. Przy 10Hz nie można szarpać.
    # Stare wartości: 70, 12, 10, 3 -> Nowe: dużo mniejsze
    base_th, base_thd, base_x, base_xd = 15.0, 5.0, 5.0, 1.0
    
    for idx in range(16):
        bits = [(idx >> b) & 1 for b in (3, 2, 1, 0)]
        largeness = sum(bits)
        # Również przyrosty (mnożniki) zmniejszone
        F_rules[idx, 0] = base_th   + 10.0 * largeness
        F_rules[idx, 1] = base_thd  +  3.0 * largeness
        F_rules[idx, 2] = base_x    +  2.0 * largeness
        F_rules[idx, 3] = base_xd   +  0.5 * largeness

    return TSParams16(
        th_small=th_small, thd_small=thd_small, x_small=x_small, xd_small=xd_small,
        F_rules=F_rules, u_sat=u_sat, sign=+1, flip_u=False, gain_scale=0.2, # Startujemy nisko
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
        
        # LQR base
        u_lqr = -float(np.dot(self.K, err))

        # Fuzzy correction
        ex  = x  - xref
        exd = xd - 0.0
        u_ts = self._u_ts(th, thd, ex, exd)

        u = u_lqr + u_ts

        # Ramp up
        alpha = min(1.0, t / self.ramp_T)
        u *= (0.25 + 0.75 * alpha)

        # Rate limiter
        du = np.clip(u - self.u_prev, -self.du_max * self.dt, self.du_max * self.dt)
        u_limited = self.u_prev + du
        
        # Saturation
        self.u_prev = float(np.clip(u_limited, -self.p.u_sat, self.p.u_sat))
        return self.u_prev

# =========================
# Simulation & Autopick
# =========================
def evaluate_run(X: np.ndarray, U: np.ndarray, x_ref: float) -> float:
    th = X[:, 0]; x = X[:, 2]
    xr = np.ones_like(x) * x_ref
    u_rms = float(np.sqrt(np.mean(U**2))) if len(U) else 0.0
    return 4.0*mse(th, np.zeros_like(th)) + 1.5*mse(x, xr) + 0.01*u_rms

def simulate_fuzzy_local(pars: dict, controller: TSFuzzyController, x0: np.ndarray, x_ref: np.ndarray, T: float, dt: float, wind: Optional[Callable[[float], float]] = None, early_stop: Optional[Tuple[float, float]] = None) -> tuple:
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]; forces = []; Fw_tr = []; t = 0.0
    ok = True
    ctrl_time_total = 0.0; sim_t0 = perf_counter()

    for _ in range(steps):
        t0c = perf_counter()
        u = controller.step(t, x, x_ref)
        ctrl_time_total += perf_counter() - t0c
        
        forces.append(u)
        F_cur = float(wind(t)) if wind else 0.0
        Fw_tr.append(F_cur)
        
        x = rk4_step_wind(f_nonlinear, x, u, pars, dt, t, wind)
        traj.append(x.copy()); t += dt

        if early_stop is not None:
            if abs(x[0]) > early_stop[0] or abs(x[2]) > early_stop[1]:
                ok = False; break

    sim_time_wall = perf_counter() - sim_t0
    return np.vstack(traj), np.asarray(forces), ok, np.asarray(Fw_tr), ctrl_time_total, sim_time_wall

def autopick_variant(pars: dict, base: TSParams16, dt: float, K_lqr: np.ndarray) -> TSParams16:
    best_p = None; best_s = np.inf
    # Skala bardzo mała - bo dt jest duże!
    scales = [0.05, 0.1, 0.2, 0.3] 
    
    for sgn in (+1, -1):
        for sc in scales:
            cand = TSParams16(th_small=base.th_small, thd_small=base.thd_small, x_small=base.x_small, xd_small=base.xd_small, F_rules=base.F_rules.copy(), u_sat=base.u_sat, sign=sgn, flip_u=False, gain_scale=sc)
            ctrl = TSFuzzyController(pars, cand, K_lqr, dt, du_max=1000.0, ramp_T=1.5)
            Xs, Us, ok, _, _, _ = simulate_fuzzy_local(pars, ctrl, SIM["x0"], SIM["x_ref"], T=4.0, dt=dt, wind=None, early_stop=(1.0, 2.0))
            
            if ok:
                score = evaluate_run(Xs, Us, SIM["x_ref"][2])
                if score < best_s: best_s = score; best_p = cand
    
    return best_p if best_p is not None else base

def plot_result(t: np.ndarray, X: np.ndarray, U: np.ndarray, title: str):
    fig = plt.figure(figsize=(9,7)); fig.suptitle(title, fontsize=12, y=0.98)
    ax1 = fig.add_subplot(3,1,1); ax1.plot(t, X[:,0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3,1,2); ax2.plot(t, X[:,2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    tf = t[:-1] if len(U) == (len(t)-1) else np.linspace(0.0, t[-1], len(U))
    ax3 = fig.add_subplot(3,1,3); ax3.plot(tf, U); ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

# =========================
# Main
# =========================
if __name__ == "__main__":
    plant = PLANT
    dt = SIM["dt"] # Powinno być 0.1 z mpc_utils
    T = SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]

    wind = None
    # wind = Wind(T, seed=23341, Ts=0.02, power=2e-3, smooth=7)

    K_lqr = lqr_from_plant(plant)
    print(f"Running Fuzzy with dt={dt}s (From mpc_utils)")

    base_ts = starter_ts_params16(u_sat=SIM["u_sat"])
    picked_ts = autopick_variant(plant, base_ts, dt, K_lqr)
    
    print(f"Autopick result: scale={picked_ts.gain_scale:.2f}")

    ctrl = TSFuzzyController(plant, picked_ts, K_lqr, dt, du_max=1000.0, ramp_T=2.0)
    
    X, U, ok, Fw_tr, ctrl_time_total, sim_time_wall = simulate_fuzzy_local(plant, ctrl, x0, x_ref, T, dt, wind=wind, early_stop=None)

    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    plot_result(t, X, U, f"ts_fuzzy_art3ish | wind={'on' if wind else 'off'} | ok={ok}")

    # Metrics
    th_ref_tr = np.zeros_like(t); x_ref_tr = np.ones_like(t) * x_ref[2]
    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr), "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x": mse(X[:,2], x_ref_tr), "mae_x": mae(X[:,2], x_ref_tr),
        "iae_theta": iae(t, X[:,0], th_ref_tr), "ise_theta": ise(t, X[:,0], th_ref_tr),
        "iae_x": iae(t, X[:,2], x_ref_tr), "ise_x": ise(t, X[:,2], x_ref_tr),
        "e_u_l2": control_energy_l2(tf, U), "e_u_l1": control_energy_l1(tf, U),
        "sim_time_wall": sim_time_wall, "ctrl_time_total": ctrl_time_total,
    }
    metrics["t_s_theta"] = settling_time(t, X[:,0], th_ref_tr, 0.01, 0.5)
    metrics["t_s_x"]     = settling_time(t, X[:,2], x_ref_tr, 0.01, 0.5)
    metrics["overshoot_theta"] = overshoot(X[:,0], th_ref_tr[-1])
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_tr[-1])
    metrics["ess_theta"], metrics["ess_theta_rms"] = steady_state_error(t, X[:,0], th_ref_tr)
    metrics["ess_x"], metrics["ess_x_rms"] = steady_state_error(t, X[:,2], x_ref_tr)

    if wind:
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, X[1:,0]-th_ref_tr[1:], Fw_tr)
        snr_x, rms_e_x, _ = disturbance_robustness(tf, X[1:,2]-x_ref_tr[1:], Fw_tr)
        metrics["snr_theta"] = snr_th; metrics["rms_e_theta"] = rms_e_th; metrics["rms_Fw"] = rms_F
        metrics["snr_x"] = snr_x; metrics["rms_e_x"] = rms_e_x
    else:
        metrics["snr_theta"]=float('nan'); metrics["snr_x"]=float('nan')
        metrics["rms_e_theta"]=float('nan'); metrics["rms_e_x"]=float('nan'); metrics["rms_Fw"]=float('nan')

    print_summary(metrics)

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)