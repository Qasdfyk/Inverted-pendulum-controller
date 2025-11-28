# controllers/ts_fuzzy_art3ish.py
# --------------------------------------------------------
# TS-fuzzy (16 reguł, 4 premise) + LQR (art3-ish) + AUTOPICK
# + animacja + metryki (IAE/ISE, energia sterowania L2/L1, czasy,
#   overshoot, czas ustalania, błąd ustalony, SNR odporności)
# --------------------------------------------------------

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import solve_continuous_are
from time import perf_counter

# =========================
# Parametry fizyczne i sym
# =========================
PLANT = {"M": 2.4, "m": 0.23, "l": 0.36, "g": 9.81}
SIM = {
    "T": 10.0,
    "dt": 0.05,
    "x0": np.array([0.03, 0.0, 0.0, 0.0]),        # [theta, theta_dot, x, x_dot]
    "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),      # chcemy przestawić wózek na 0.10 m
    "u_sat": 100.0,
}
EPS = 1e-12

# =========================
# Zakłócenie (wiatr)
# =========================
class Wind:
    def __init__(self, t_end: float, seed=23341, Ts=0.02, power=2e-3, smooth=7):
        rng = np.random.default_rng(seed)
        self.tgrid = np.arange(0.0, t_end + Ts, Ts)
        sigma = np.sqrt(power / Ts)
        w = rng.normal(0.0, sigma, size=self.tgrid.shape)
        if smooth and smooth > 1:
            ker = np.ones(smooth) / smooth
            self.Fw = np.convolve(w, ker, mode="same")
        else:
            self.Fw = w

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.tgrid, self.Fw))

# =========================
# Dynamika + RK4
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

def rk4_step_wind(
    x: np.ndarray,
    u: float,
    pars: dict,
    dt: float,
    t: float,
    wind: Optional[Callable[[float], float]] = None,
) -> np.ndarray:
    F1 = wind(t) if wind else 0.0
    k1 = f_nonlinear(x, u, pars, F1)
    F2 = wind(t + 0.5 * dt) if wind else 0.0
    k2 = f_nonlinear(x + 0.5 * dt * k1, u, pars, F2)
    F3 = wind(t + 0.5 * dt) if wind else 0.0
    k3 = f_nonlinear(x + 0.5 * dt * k2, u, pars, F3)
    F4 = wind(t + dt) if wind else 0.0
    k4 = f_nonlinear(x + dt * k3, u, pars, F4)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# =========================
# LQR
# =========================
def linearize_upright(M: float, m: float, l: float, g: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearizacja wokół theta = 0, stan: [th, thd, x, xd]"""
    A = np.array([
        [0.0,                 1.0, 0.0, 0.0],
        [(M + m)*g/(M*l),     0.0, 0.0, 0.0],
        [0.0,                 0.0, 0.0, 1.0],
        [-m*g/M,              0.0, 0.0, 0.0]
    ], dtype=float)
    B = np.array([
        [0.0],
        [-1.0/(M*l)],
        [0.0],
        [1.0/M]
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
# TS-fuzzy 16 reguł (4 premise)
# =========================
def tri_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

@dataclass
class TSParams16:
    th_small: Sequence[float]
    thd_small: Sequence[float]
    x_small: Sequence[float]
    xd_small: Sequence[float]
    F_rules: np.ndarray         # (16,4) -> [theta, theta_dot, x, x_dot]
    u_sat: float
    sign: int
    flip_u: bool
    gain_scale: float

def starter_ts_params16(u_sat: float) -> TSParams16:
    th_small  = (-0.20, 0.0, 0.20)
    thd_small = (-1.5, 0.0, 1.5)
    x_small   = (-0.4, 0.0, 0.4)
    xd_small  = (-0.8, 0.0, 0.8)

    F_rules = np.zeros((16, 4), dtype=float)
    base_th, base_thd, base_x, base_xd = 70.0, 12.0, 10.0, 3.0
    for idx in range(16):
        bits = [(idx >> b) & 1 for b in (3, 2, 1, 0)]  # th, thd, x, xd
        largeness = sum(bits)
        F_rules[idx, 0] = base_th   + 35.0 * largeness
        F_rules[idx, 1] = base_thd  +  6.0 * largeness
        F_rules[idx, 2] = base_x    +  4.0 * largeness
        F_rules[idx, 3] = base_xd   +  1.5 * largeness

    return TSParams16(
        th_small=th_small,
        thd_small=thd_small,
        x_small=x_small,
        xd_small=xd_small,
        F_rules=F_rules,
        u_sat=u_sat,
        sign=+1,
        flip_u=False,
        gain_scale=0.45,
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
    def __init__(self,
                 pars: dict,
                 ts_params: TSParams16,
                 K_lqr: np.ndarray,
                 dt: float,
                 du_max: float = 800.0,
                 ramp_T: float = 2.0):
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
        if self.p.flip_u:
            u = -u
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

        # miękki rozruch
        alpha = min(1.0, t / self.ramp_T)
        u *= (0.25 + 0.75 * alpha)

        # ograniczenie du/dt
        du = np.clip(u - self.u_prev, -self.du_max * self.dt, self.du_max * self.dt)
        u_limited = self.u_prev + du

        # saturacja
        self.u_prev = float(np.clip(u_limited, -self.p.u_sat, self.p.u_sat))
        return self.u_prev

# =========================
# METRYKI (spójne z innymi plikami)
# =========================
def mse(y,yref): return float(np.mean((np.asarray(y)-np.asarray(yref))**2))
def mae(y,yref): return float(np.mean(np.abs(np.asarray(y)-np.asarray(yref))))
def iae(t,y,yref): t=np.asarray(t); e=np.abs(np.asarray(y)-np.asarray(yref)); return float(np.trapezoid(e,t))
def ise(t,y,yref): t=np.asarray(t); e=np.asarray(y)-np.asarray(yref); return float(np.trapezoid(e*e,t))
def control_energy_l2(t,u): t=np.asarray(t); u=np.asarray(u); return float(np.trapezoid(u*u,t))
def control_energy_l1(t,u): t=np.asarray(t); u=np.asarray(u); return float(np.trapezoid(np.abs(u),t))
def settling_time(t,y,yref,eps,hold_time):
    t=np.asarray(t); e=np.abs(np.asarray(y)-np.asarray(yref))
    N=len(t)
    if N<2: return float('nan')
    inside=(e<=eps); dt=np.diff(t).mean(); win=max(1,int(np.round(hold_time/dt)))
    for i in range(N-win+1):
        if np.all(inside[i:i+win]): return float(t[i])
    return float('nan')
def overshoot(y,yref_final):
    y=np.asarray(y); r=float(yref_final)
    if np.isclose(r,0.0,atol=1e-12): return float('nan')
    peak=float(np.max(y)) if r>=y[0] else float(np.min(y))
    return float(100.0*(peak-r)/abs(r))
def steady_state_error(t,y,yref,window_frac=0.1,window_min=0.5):
    t=np.asarray(t); y=np.asarray(y); yref=np.asarray(yref)
    T=t[-1]-t[0] if len(t)>1 else 0.0; w=max(window_min,window_frac*T); t0=t[-1]-w
    idx=t>=t0; e=y[idx]-yref[idx]
    if e.size==0: return float('nan'), float('nan')
    return float(np.mean(e)), float(np.sqrt(np.mean(e**2)))
def disturbance_robustness(t,e,Fw,window_frac=0.5,window_min=1.0):
    t=np.asarray(t); e=np.asarray(e); Fw=np.asarray(Fw)
    T=t[-1]-t[0] if len(t)>1 else 0.0; w=max(window_min,window_frac*T); t0=t[-1]-w
    idx=t>=t0
    if not np.any(idx): return float('nan'), float('nan'), float('nan')
    e_win=e[idx]; F_win=Fw[idx]
    rms_e=float(np.sqrt(np.mean(e_win**2)))
    rms_F=float(np.sqrt(np.mean(F_win**2))) if np.any(np.abs(F_win)>0) else 0.0
    snr=float(rms_e/(rms_F+1e-12)) if rms_F>0 else float('nan')
    return snr, rms_e, rms_F
def print_summary(m: dict):
    parts=[
        f"MSE(th)={m.get('mse_theta',float('nan')):.6f}",
        f"MAE(th)={m.get('mae_theta',float('nan')):.6f}",
        f"MSE(x)={m.get('mse_x',float('nan')):.6f}",
        f"MAE(x)={m.get('mae_x',float('nan')):.6f}",
        f"IAE(th)={m.get('iae_theta',float('nan')):.4f}",
        f"ISE(th)={m.get('ise_theta',float('nan')):.4f}",
        f"IAE(x)={m.get('iae_x',float('nan')):.4f}",
        f"ISE(x)={m.get('ise_x',float('nan')):.4f}",
        f"E_u(L2)={m.get('e_u_l2',float('nan')):.4f}",
        f"E_u(L1)={m.get('e_u_l1',float('nan')):.4f}",
        f"t_s(th)={m.get('t_s_theta',float('nan')):.3f}s",
        f"t_s(x)={m.get('t_s_x',float('nan')):.3f}s",
        f"OS(th)={m.get('overshoot_theta',float('nan')):.2f}%",
        f"OS(x)={m.get('overshoot_x',float('nan')):.2f}%",
        f"ess(th)={m.get('ess_theta',float('nan')):.5f}",
        f"ess(x)={m.get('ess_x',float('nan')):.5f}",
        f"SNR_th={m.get('snr_theta',float('nan')):.3g}",
        f"SNR_x={m.get('snr_x',float('nan')):.3g}",
        f"T_sim={m.get('sim_time_wall',float('nan')):.3f}s",
        f"T_ctrl={m.get('ctrl_time_total',float('nan')):.3f}s",
    ]
    print("  ".join(parts))

# =========================
# Symulacja (z logiem Fw i czasów)
# =========================
def evaluate_run(X: np.ndarray, U: np.ndarray, x_ref: float) -> float:
    th = X[:, 0]; x = X[:, 2]
    th_ref = np.zeros_like(th); xr = np.ones_like(x) * x_ref
    u_rms = float(np.sqrt(np.mean(U**2))) if len(U) else 0.0
    return 4.0*mse(th, th_ref) + 1.5*mse(x, xr) + 0.01*u_rms

def simulate(pars: dict,
             controller: TSFuzzyController,
             x0: np.ndarray,
             x_ref: np.ndarray,
             T: float,
             dt: float,
             wind: Optional[Callable[[float], float]] = None,
             early_stop: Optional[Tuple[float, float]] = None
             ) -> tuple[np.ndarray, np.ndarray, bool, np.ndarray, float, float]:
    """
    Zwraca: X, U, ok, Fw_tr, ctrl_time_total, sim_time_wall
    """
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    forces: list[float] = []
    Fw_tr: list[float] = []
    t = 0.0
    ok = True

    ctrl_time_total = 0.0
    sim_t0 = perf_counter()

    for _ in range(steps):
        # czas kontrolera
        t0c = perf_counter()
        u = controller.step(t, x, x_ref)
        ctrl_time_total += perf_counter() - t0c

        forces.append(u)
        F_cur = float(wind(t)) if wind else 0.0
        Fw_tr.append(F_cur)

        x = rk4_step_wind(x, u, pars, dt, t, wind)
        traj.append(x.copy())
        t += dt

        if early_stop is not None:
            th_th, x_th = early_stop
            if abs(x[0]) > th_th or abs(x[2]) > x_th:
                ok = False
                break

    sim_time_wall = perf_counter() - sim_t0
    return np.vstack(traj), np.asarray(forces), ok, np.asarray(Fw_tr), ctrl_time_total, sim_time_wall

# =========================
# Autopick (znak + skala)
# =========================
def autopick_variant(pars: dict, base: TSParams16, dt: float, K_lqr: np.ndarray) -> TSParams16:
    best_p = None
    best_s = np.inf
    for sgn in (+1, -1):
        for sc in (0.35, 0.45, 0.65):
            cand = TSParams16(
                th_small=base.th_small,
                thd_small=base.thd_small,
                x_small=base.x_small,
                xd_small=base.xd_small,
                F_rules=base.F_rules.copy(),
                u_sat=base.u_sat,
                sign=sgn,
                flip_u=False,
                gain_scale=sc,
            )
            ctrl = TSFuzzyController(pars, cand, K_lqr, dt, du_max=1000.0, ramp_T=1.5)
            Xs, Us, ok, _, _, _ = simulate(
                pars,
                ctrl,
                SIM["x0"],
                SIM["x_ref"],
                T=3.0,
                dt=dt,
                wind=None,
                early_stop=(2.8, 4.0),
            )
            if not ok:
                continue
            score = evaluate_run(Xs, Us, SIM["x_ref"][2])
            if score < best_s:
                best_s = score
                best_p = cand
    return best_p if best_p is not None else base

# =========================
# Wykresy + animacja
# =========================
def plot_result(t: np.ndarray, X: np.ndarray, U: np.ndarray, title: str):
    fig = plt.figure(figsize=(9,7))
    fig.suptitle(title, fontsize=12, y=0.98)
    ax1 = fig.add_subplot(3,1,1); ax1.plot(t, X[:,0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3,1,2); ax2.plot(t, X[:,2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    tf = t[:-1] if len(U) == (len(t)-1) else np.linspace(0.0, t[-1], len(U))
    ax3 = fig.add_subplot(3,1,3); ax3.plot(tf, U); ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

def animate_cartpole(t: np.ndarray, X: np.ndarray, params: Optional[dict] = None, speed: float = 1.0):
    th = X[:, 0]; x = X[:, 2]
    p = params or {}; l = p.get("l", 0.36)
    cart_w, cart_h = 0.35, 0.18; wheel_r = 0.05
    pole_len = l * 1.5; pad = 0.8

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(True, alpha=0.3); ax.set_title("Cart-Pole  (TS-fuzzy + LQR)")
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
# Main (+ liczenie metryk)
# =========================
if __name__ == "__main__":
    plant = PLANT
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]

    # wiatr: wyłączony na start
    wind = None
    # wind = Wind(T, seed=23341, Ts=0.02, power=2e-3, smooth=7)

    K_lqr = lqr_from_plant(plant)
    print("LQR K (from CARE):", K_lqr)

    base_ts  = starter_ts_params16(u_sat=SIM["u_sat"])
    picked_ts = autopick_variant(plant, base_ts, dt, K_lqr)
    print("Użyty wariant TS:",
          f"sign={picked_ts.sign}, gain_scale={picked_ts.gain_scale:.2f}, flip_u={picked_ts.flip_u}")

    ctrl = TSFuzzyController(plant, picked_ts, K_lqr, dt, du_max=1000.0, ramp_T=2.0)

    X, U, ok, Fw_tr, ctrl_time_total, sim_time_wall = simulate(
        plant, ctrl, x0, x_ref, T, dt, wind=wind, early_stop=None
    )

    # siatka czasu
    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]  # siatka dla U i Fw_tr

    # wykres
    title = f"ts_fuzzy_art3ish  |  wind={'on' if wind else 'off'}  |  step=position_step  |  ok={ok}"
    plot_result(t, X, U, title)

    # ===== METRYKI =====
    th_ref_tr = np.zeros_like(t)
    x_ref_tr  = np.ones_like(t) * x_ref[2]

    # progi (spójne z innymi plikami)
    eps_theta = 0.01  # rad
    eps_x     = 0.01  # m
    hold_time = 0.5   # s
    ess_window_frac = 0.10
    ess_window_min  = 0.5
    is_step = True

    # błędy
    e_th = X[:,0] - th_ref_tr
    e_x  = X[:,2] - x_ref_tr

    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr),
        "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x":     mse(X[:,2], x_ref_tr),
        "mae_x":     mae(X[:,2], x_ref_tr),
        "iae_theta": iae(t, X[:,0], th_ref_tr),
        "ise_theta": ise(t, X[:,0], th_ref_tr),
        "iae_x":     iae(t, X[:,2], x_ref_tr),
        "ise_x":     ise(t, X[:,2], x_ref_tr),
        "e_u_l2":    control_energy_l2(tf, U),
        "e_u_l1":    control_energy_l1(tf, U),
        "sim_time_wall": sim_time_wall,
        "ctrl_time_total": ctrl_time_total,
    }

    metrics["t_s_theta"] = settling_time(t, X[:,0], th_ref_tr, eps=eps_theta, hold_time=hold_time) if is_step else float('nan')
    metrics["t_s_x"]     = settling_time(t, X[:,2], x_ref_tr,  eps=eps_x,     hold_time=hold_time) if is_step else float('nan')

    metrics["overshoot_theta"] = overshoot(X[:,0], th_ref_tr[-1]) if is_step else float('nan')  # NaN bo ref=0
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_tr[-1])  if is_step else float('nan')

    ess_th_mean, ess_th_rms = steady_state_error(t, X[:,0], th_ref_tr, window_frac=ess_window_frac, window_min=ess_window_min)
    ess_x_mean,  ess_x_rms  = steady_state_error(t, X[:,2], x_ref_tr,  window_frac=ess_window_frac, window_min=ess_window_min)
    metrics["ess_theta"] = ess_th_mean
    metrics["ess_x"]     = ess_x_mean
    metrics["ess_theta_rms"] = ess_th_rms
    metrics["ess_x_rms"]     = ess_x_rms

    if wind is not None and len(Fw_tr) == len(tf):
        # dopasuj błędy do siatki tf (bez pierwszego punktu stanu)
        e_th_tf = X[1:,0] - th_ref_tr[1:]
        e_x_tf  = X[1:,2] - x_ref_tr[1:]
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

    # animacja (opcjonalnie)
    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
