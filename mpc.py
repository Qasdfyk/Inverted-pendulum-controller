# mpc_cartpole_physical.py
# --------------------------------------------------------
# Nonlinear MPC for the cart-pole (art2 model) + animation + metrics.
# Parameters fixed to your env config.
# Wind disturbance (Fw) affects the PLANT ONLY (MPC predicts with Fw = 0).
# --------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Optional
from time import perf_counter  # timing

# =========================
# Fixed physical & sim params
# =========================
PLANT = {"M": 2.4, "m": 0.23, "l": 0.36, "g": 9.81}
SIM = {
    "T": 10.0,
    "dt": 0.1,         # MPC + integration step
    "x0": np.array([0.03, 0.0, 0.0, 0.0]),  # [theta, theta_dot, x, x_dot]
    "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),    # track 0.1 m cart step (position_step)
    "u_sat": 100.0
}

# =========================
# Disturbance (wind) — ENABLE / DISABLE
#   - Disable wind: wind = None
#   - Enable  wind: wind = Wind(T, seed=23341, Ts=0.02, power=2e-3, smooth=7)
# =========================
class Wind:
    def __init__(self, t_end: float, seed=23341, Ts=0.01, power=1e-3, smooth=5):
        rng = np.random.default_rng(seed)
        self.tgrid = np.arange(0.0, t_end + Ts, Ts)
        sigma = np.sqrt(power / Ts)
        w = rng.normal(0.0, sigma, size=self.tgrid.shape)
        if smooth and smooth > 1:
            kernel = np.ones(smooth) / smooth
            self.Fw = np.convolve(w, kernel, mode='same')
        else:
            self.Fw = w

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.tgrid, self.Fw))


# =========================
# Dynamics
# =========================
def f_nonlinear(x: np.ndarray, u: float, pars: dict, Fw: float = 0.0) -> np.ndarray:
    th, thd, pos, posd = x
    M, m, l, g = pars["M"], pars["m"], pars["l"], pars["g"]
    s, c = np.sin(th), np.cos(th)

    denom_x  = (M + m) - m * c * c
    denom_th = (m * l * c * c) - (M + m) * l

    thdd = (u * c - (M + m) * g * s + m * l * (c * s) * (thd ** 2) - (M / m) * Fw * c) / denom_th
    xdd  = (u + m * l * s * (thd ** 2) - m * g * c * s + Fw * (s * s)) / denom_x
    return np.array([thd, thdd, posd, xdd], dtype=float)

def rk4_step(f, x, u, pars, dt):
    k1 = f(x, u, pars)
    k2 = f(x + 0.5 * dt * k1, u, pars)
    k3 = f(x + 0.5 * dt * k2, u, pars)
    k4 = f(x + dt * k3, u, pars)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def rk4_step_wind(f, x, u, pars, dt, t,
                  Fw_fn: Optional[Callable[[float], float]] = None):
    """RK4 with time-varying disturbance Fw(t) for PLANT ONLY."""
    F1 = Fw_fn(t) if Fw_fn else 0.0
    k1 = f(x, u, pars, F1)

    F2 = Fw_fn(t + 0.5 * dt) if Fw_fn else 0.0
    k2 = f(x + 0.5 * dt * k1, u, pars, F2)

    F3 = Fw_fn(t + 0.5 * dt) if Fw_fn else 0.0
    k3 = f(x + 0.5 * dt * k2, u, pars, F3)

    F4 = Fw_fn(t + dt) if Fw_fn else 0.0
    k4 = f(x + dt * k3, u, pars, F4)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# =========================
# Nonlinear MPC (no preview; predicts with Fw = 0)
# =========================
class MPCController:
    def __init__(self,
                 pars: dict,
                 dt: float,
                 N: int,
                 Nu: int,
                 umin: float,
                 umax: float,
                 Q: np.ndarray,
                 R: float):
        self.pars = pars
        self.dt = dt
        self.N = N
        self.Nu = Nu
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.R = R

    def _rollout(self, x0, u_seq):
        """Predict with zero disturbance."""
        x = np.array(x0, dtype=float)
        traj = []
        for ui in u_seq:
            x = rk4_step(f_nonlinear, x, float(ui), self.pars, self.dt)
            traj.append(x.copy())
        return np.asarray(traj)

    def _cost(self, du, x0, x_ref, u_prev):
        du = np.asarray(du, dtype=float)
        u_seq = np.zeros(self.N)
        if self.Nu > 0:
            u_cum = u_prev + np.cumsum(du)
            upto = min(self.Nu, self.N)
            u_seq[:upto] = u_cum[:upto]
            if self.N > self.Nu:
                u_seq[self.Nu:] = u_cum[upto-1]
        else:
            u_seq[:] = u_prev

        preds = self._rollout(x0, u_seq)
        err = preds - x_ref.reshape(1, -1)
        cost_y = float(np.sum([e.T @ self.Q @ e for e in err]))
        cost_u = float(self.R * np.sum(du * du))
        return cost_y + cost_u

    def compute_control(self, x0, x_ref, u_prev):
        du_min = self.umin - u_prev
        du_max = self.umax - u_prev
        bounds = [(du_min, du_max)] * self.Nu
        du0 = np.zeros(self.Nu)

        res = minimize(
            self._cost, du0,
            args=(x0, x_ref, u_prev),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-4, 'disp': False}
        )
        du_opt = res.x if res.success else du0

        u_seq = np.zeros(self.N)
        u_cum = u_prev + np.cumsum(du_opt)
        upto = min(self.Nu, self.N)
        u_seq[:upto] = u_cum[:upto]
        if self.N > self.Nu:
            u_seq[self.Nu:] = u_cum[upto-1]
        np.clip(u_seq, self.umin, self.umax, out=u_seq)
        return u_seq


# =========================
# Animation
# =========================
def animate_cartpole(t, X, params=None, speed=1.0):
    th = X[:, 0]; x = X[:, 2]
    p = params or {}; l = p.get("l", 0.36)
    cart_w, cart_h = 0.35, 0.18; wheel_r = 0.05
    pole_len = l * 1.5; pad = 0.8

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(True, alpha=0.3); ax.set_title("Cart-Pole")
    xmin, xmax = float(np.min(x) - pad), float(np.max(x) + pad)
    ax.set_xlim(xmin, xmax); ax.set_ylim(-(wheel_r + 0.25), pole_len + 0.45)
    ax.plot([xmin, xmax], [0, 0], color='k', lw=1, alpha=0.6)

    cart = plt.Rectangle((x[0]-cart_w/2, wheel_r), cart_w, cart_h, ec='k', fc='#e6f3ff', lw=1.5)
    wheel1 = plt.Circle((x[0]-cart_w/3, wheel_r), wheel_r, ec='k', fc='#333333')
    wheel2 = plt.Circle((x[0]+cart_w/3, wheel_r), wheel_r, ec='k', fc='#333333')
    ax.add_patch(cart); ax.add_patch(wheel1); ax.add_patch(wheel2)
    pole_line, = ax.plot([], [], lw=3, solid_capstyle='round', alpha=0.9)
    trail_line, = ax.plot([], [], lw=1, alpha=0.4)

    def pole_end(i):
        cx = x[i]; cy = wheel_r + cart_h
        px = cx + pole_len * np.sin(th[i]); py = cy + pole_len * np.cos(th[i])
        return cx, cy, px, py

    def update(i):
        cx, cy, px, py = pole_end(i)
        cart.set_x(cx - cart_w/2)
        wheel1.center = (cx - cart_w/3, wheel_r)
        wheel2.center = (cx + cart_w/3, wheel_r)
        pole_line.set_data([cx, px], [cy, py])
        j0 = max(0, i - 150)
        trail_line.set_data(x[j0:i+1], (wheel_r + 0.01) * np.ones(i + 1 - j0))
        ax.set_xlim(cx - pad, cx + pad)
        return cart, wheel1, wheel2, pole_line, trail_line

    interval_ms = max(10, int(1000 * (t[1] - t[0]) / max(speed, 1e-6)))
    ani = FuncAnimation(fig, update, frames=len(t), interval=interval_ms, blit=False)
    setattr(fig, "_cartpole_ani", ani)
    plt.show()


# =========================
# Metrics (spójne z env/sim.py)
# =========================
def mse(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean((y - yref)**2))

def mae(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean(np.abs(y - yref)))

def iae(t, y, yref):
    t = np.asarray(t); e = np.abs(np.asarray(y) - np.asarray(yref))
    return float(np.trapezoid(e, t))

def ise(t, y, yref):
    t = np.asarray(t); e = np.asarray(y) - np.asarray(yref)
    return float(np.trapezoid(e*e, t))

def control_energy_l2(t, u):
    t = np.asarray(t); u = np.asarray(u)
    return float(np.trapezoid(u*u, t))

def control_energy_l1(t, u):
    t = np.asarray(t); u = np.asarray(u)
    return float(np.trapezoid(np.abs(u), t))

def settling_time(t, y, yref, eps, hold_time):
    """
    Najwcześniejszy czas, od którego |y - yref| <= eps utrzymuje się
    nieprzerwanie >= hold_time. Jeśli brak — zwraca NaN.
    """
    t = np.asarray(t); e = np.abs(np.asarray(y) - np.asarray(yref))
    inside = e <= eps
    N = len(t)
    if N < 2: return float('nan')
    dt = np.diff(t).mean()
    win = max(1, int(np.round(hold_time / dt)))
    for i in range(N - win + 1):
        if np.all(inside[i:i+win]):
            return float(t[i])
    return float('nan')

def overshoot(y, yref_final):
    """
    Maksymalne przeregulowanie względem końcowej wartości referencyjnej.
    Zwraca % (0..), lub NaN gdy ref_final ~ 0.
    """
    y = np.asarray(y); r = float(yref_final)
    if np.isclose(r, 0.0, atol=1e-12):
        return float('nan')
    peak = float(np.max(y)) if r >= y[0] else float(np.min(y))
    return float(100.0 * (peak - r) / abs(r))

def steady_state_error(t, y, yref, window_frac=0.1, window_min=0.5):
    """
    Średni błąd na końcówce (ostatnie window_frac czasu, min window_min sek).
    Zwraca (mean, rms).
    """
    t = np.asarray(t); y = np.asarray(y); yref = np.asarray(yref)
    T = t[-1] - t[0] if len(t) > 1 else 0.0
    w = max(window_min, window_frac * T)
    t0 = t[-1] - w
    idx = t >= t0
    e = y[idx] - yref[idx]
    if e.size == 0:
        return float('nan'), float('nan')
    return float(np.mean(e)), float(np.sqrt(np.mean(e**2)))

def disturbance_robustness(t, e, Fw, window_frac=0.5, window_min=1.0):
    """
    SNR = RMS(e)/RMS(Fw) w stanie ustalonym (ostatnie okno). Mniejsze = lepiej.
    """
    t = np.asarray(t); e = np.asarray(e); Fw = np.asarray(Fw)
    T = t[-1] - t[0] if len(t) > 1 else 0.0
    w = max(window_min, window_frac * T)
    t0 = t[-1] - w
    idx = t >= t0
    if not np.any(idx):
        return float('nan'), float('nan'), float('nan')
    e_win = e[idx]; F_win = Fw[idx]
    rms_e = float(np.sqrt(np.mean(e_win**2)))
    rms_F = float(np.sqrt(np.mean(F_win**2))) if np.any(np.abs(F_win) > 0) else 0.0
    snr = float(rms_e / (rms_F + 1e-12)) if rms_F > 0 else float('nan')
    return snr, rms_e, rms_F

def print_summary(metrics: dict):
    parts = [
        f"MSE(th)={metrics.get('mse_theta', float('nan')):.6f}",
        f"MAE(th)={metrics.get('mae_theta', float('nan')):.6f}",
        f"MSE(x)={metrics.get('mse_x', float('nan')):.6f}",
        f"MAE(x)={metrics.get('mae_x', float('nan')):.6f}",
        f"IAE(th)={metrics.get('iae_theta', float('nan')):.4f}",
        f"ISE(th)={metrics.get('ise_theta', float('nan')):.4f}",
        f"IAE(x)={metrics.get('iae_x', float('nan')):.4f}",
        f"ISE(x)={metrics.get('ise_x', float('nan')):.4f}",
        f"E_u(L2)={metrics.get('e_u_l2', float('nan')):.4f}",
        f"E_u(L1)={metrics.get('e_u_l1', float('nan')):.4f}",
        f"t_s(th)={metrics.get('t_s_theta', float('nan')):.3f}s",
        f"t_s(x)={metrics.get('t_s_x', float('nan')):.3f}s",
        f"OS(th)={metrics.get('overshoot_theta', float('nan')):.2f}%",
        f"OS(x)={metrics.get('overshoot_x', float('nan')):.2f}%",
        f"ess(th)={metrics.get('ess_theta', float('nan')):.5f}",
        f"ess(x)={metrics.get('ess_x', float('nan')):.5f}",
        f"SNR_th={metrics.get('snr_theta', float('nan')):.3g}",
        f"SNR_x={metrics.get('snr_x', float('nan')):.3g}",
        f"T_sim={metrics.get('sim_time_wall', float('nan')):.3f}s",
        f"T_ctrl={metrics.get('ctrl_time_total', float('nan')):.3f}s",
    ]
    print("  ".join(parts))


# =========================
# Simulation loop
# =========================
def simulate_mpc(pars, controller: MPCController, x0, x_ref, T, dt,
                 u0=0.0,
                 wind: Optional[Callable[[float], float]] = None):
    """
    Sim with optional wind on PLANT (MPC predicts with Fw=0).
    Zwraca: X, U, Fw_tr, ctrl_time_total, sim_time_wall
    """
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy(); u_prev = float(u0)
    traj = [x.copy()]
    forces = []      # sterowanie u(t_k)
    Fw_tr = []       # Fw(t_k) w chwili sterowania
    t = 0.0

    ctrl_time_total = 0.0
    sim_t0 = perf_counter()

    for _ in range(steps):
        # --- MPC compute time
        t0c = perf_counter()
        u_seq = controller.compute_control(x, x_ref, u_prev)
        ctrl_time_total += perf_counter() - t0c

        u_apply = float(u_seq[0])
        forces.append(u_apply)

        F_cur = float(wind(t)) if wind else 0.0
        Fw_tr.append(F_cur)

        # Plant evolves with actual wind
        x = rk4_step_wind(f_nonlinear, x, u_apply, pars, dt, t, wind)
        traj.append(x.copy())

        u_prev = u_apply
        t += dt

    sim_time_wall = perf_counter() - sim_t0

    return np.vstack(traj), np.asarray(forces), np.asarray(Fw_tr), ctrl_time_total, sim_time_wall


# =========================
# Main
# =========================
if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT

    # ---- Wind toggle -------------------------------------------------------
    # Enable wind:
    # wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)
    # Disable wind:
    wind = None
    #wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)
    # -----------------------------------------------------------------------

    ctrl = MPCController(
        pars=plant,
        dt=dt,
        N=10,
        Nu=8,
        umin=-u_sat,
        umax=u_sat,
        Q=np.diag([60.0, 3.0, 15.0, 1.0]),
        R=1e-3
    )

    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)

    # siatki czasu: stany w węzłach t_k = 0..T (len = steps+1),
    # sterowanie/Fw w węzłach k=0..steps-1 (len = steps)
    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]  # czas dla U i Fw_tr

    # ---- plotting (jak w env) ----
    controller_name = "mpc_no"
    disturbance = wind is not None
    step_type = "position_step"
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(f"{controller_name}  |  wind={'on' if disturbance else 'off'}  |  step={step_type}",
                 fontsize=12, y=0.98)
    ax1 = fig.add_subplot(3, 1, 1); ax1.plot(t, X[:, 0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3, 1, 2); ax2.plot(t, X[:, 2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    ax3 = fig.add_subplot(3, 1, 3); ax3.plot(tf, U);      ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---- refs ----
    th_ref_tr = np.zeros_like(t)
    x_ref_tr  = np.ones_like(t) * x_ref[2]

    # ---- metryki (spójne z env/sim.py) ----
    eps_theta = 0.01  # rad
    eps_x     = 0.01  # m
    hold_time = 0.5   # s
    ess_window_frac = 0.10
    ess_window_min  = 0.5
    is_step = True  # ten plik robi position_step

    # błędy
    e_th = X[:,0] - th_ref_tr
    e_x  = X[:,2] - x_ref_tr

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

    metrics["overshoot_theta"] = overshoot(X[:,0], th_ref_tr[-1]) if is_step else float('nan')  # NaN gdy ref=0
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_tr[-1])  if is_step else float('nan')

    ess_th_mean, ess_th_rms = steady_state_error(t, X[:,0], th_ref_tr, window_frac=ess_window_frac, window_min=ess_window_min)
    ess_x_mean,  ess_x_rms  = steady_state_error(t, X[:,2], x_ref_tr,  window_frac=ess_window_frac, window_min=ess_window_min)
    metrics["ess_theta"] = ess_th_mean
    metrics["ess_x"]     = ess_x_mean
    metrics["ess_theta_rms"] = ess_th_rms
    metrics["ess_x_rms"]     = ess_x_rms

    if disturbance:
        # e na siatce tf (pomijamy pierwszy punkt stanu)
        e_th_tf = X[1:, 0] - th_ref_tr[1:]
        e_x_tf  = X[1:, 2] - x_ref_tr[1:]
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, e_th_tf, Fw_tr, window_frac=0.5, window_min=1.0)
        snr_x,  rms_e_x,  _     = disturbance_robustness(tf, e_x_tf,  Fw_tr, window_frac=0.5, window_min=1.0)
        metrics["snr_theta"] = snr_th
        metrics["snr_x"]     = snr_x
        metrics["rms_e_theta"] = rms_e_th
        metrics["rms_e_x"]     = rms_e_x
        metrics["rms_Fw"]      = rms_F
    else:
        metrics["snr_theta"] = float('nan')
        metrics["snr_x"]     = float('nan')
        metrics["rms_e_theta"] = float('nan')
        metrics["rms_e_x"]     = float('nan')
        metrics["rms_Fw"]      = float('nan')

    print_summary(metrics)

    # ---- optional animation ----
    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
