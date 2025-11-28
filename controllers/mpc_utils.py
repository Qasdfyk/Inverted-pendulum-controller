import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable, Optional
from time import perf_counter

# =========================
# Fixed physical & sim params
# =========================
PLANT = {"M": 2.4, "m": 0.23, "l": 0.36, "g": 9.81}
SIM = {
    "T": 10.0,
    "dt": 0.1,
    "x0": np.array([0.03, 0.0, 0.0, 0.0]),
    "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),
    "u_sat": 100.0
}

# =========================
# Disturbance (Wind)
# =========================
class Wind:
    def __init__(self, t_end: float, seed=23341, Ts=0.1, power=1e-3, smooth=5):
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
# Metrics (FIXED np.trapezoid -> np.trapz)
# =========================
def mse(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean((y - yref)**2))

def mae(y, yref):
    y, yref = np.asarray(y), np.asarray(yref)
    return float(np.mean(np.abs(y - yref)))

def iae(t, y, yref):
    t = np.asarray(t); e = np.abs(np.asarray(y) - np.asarray(yref))
    
    return float(np.trapz(e, t)) 

def ise(t, y, yref):
    t = np.asarray(t); e = np.asarray(y) - np.asarray(yref)
    
    return float(np.trapz(e*e, t))

def control_energy_l2(t, u):
    t = np.asarray(t); u = np.asarray(u)
    
    return float(np.trapz(u*u, t))

def control_energy_l1(t, u):
    t = np.asarray(t); u = np.asarray(u)
    
    return float(np.trapz(np.abs(u), t))

def settling_time(t, y, yref, eps, hold_time):
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
    y = np.asarray(y); r = float(yref_final)
    if np.isclose(r, 0.0, atol=1e-12):
        return float('nan')
    peak = float(np.max(y)) if r >= y[0] else float(np.min(y))
    return float(100.0 * (peak - r) / abs(r))

def steady_state_error(t, y, yref, window_frac=0.1, window_min=0.5):
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
def simulate_mpc(pars, controller, x0, x_ref, T, dt,
                 u0=0.0,
                 wind: Optional[Callable[[float], float]] = None):
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