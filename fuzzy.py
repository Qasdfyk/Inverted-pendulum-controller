# controllers/ts_fuzzy_art3ish.py
# --------------------------------------------------------
# TS-fuzzy (16 reguł, 4 premise) + LQR (art3-ish) + AUTOPICK
# + animacja + metryki
# --------------------------------------------------------

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
# LQR jak w art3 (część klasyczna)
# =========================
def linearize_cartpole(pars: dict) -> tuple[np.ndarray, np.ndarray]:
    M, m, l, g = pars["M"], pars["m"], pars["l"], pars["g"]
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 0] = (M + m) * g / (M * l)   # dtheta_dot/dtheta
    A[2, 3] = 1.0
    A[3, 0] = -m * g / M
    B = np.zeros((4, 1))
    B[1, 0] = -1.0 / (M * l)
    B[3, 0] =  1.0 / M
    return A, B

def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rozwiązanie ARE metodą hamiltonianu (czysty numpy)."""
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([
        [ A,                -B @ Rinv @ B.T],
        [-Q,               -A.T          ],
    ])
    eigvals, eigvecs = np.linalg.eig(H)
    idx = [i for i, v in enumerate(eigvals) if np.real(v) < 0.0]
    V = eigvecs[:, idx]
    V1 = V[:n, :]
    V2 = V[n:, :]
    X = np.real(V2 @ np.linalg.inv(V1))
    return X

def lqr_from_pars(pars: dict,
                  Q: Optional[np.ndarray] = None,
                  R: Optional[np.ndarray] = None) -> np.ndarray:
    A, B = linearize_cartpole(pars)
    if Q is None:
        Q = np.diag([60.0, 6.0, 15.0, 2.0])   # mocno na kąt, trochę na x
    if R is None:
        R = np.array([[3.5]])                 # hamuje agresję
    X = solve_care(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ X)          # (1,4)
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
    F_rules: np.ndarray         # (16,2) -> [theta, theta_dot]
    u_sat: float
    sign: int
    flip_u: bool
    gain_scale: float

def starter_ts_params16(u_sat: float) -> TSParams16:
    # zbiory rozmyte (szerokie)
    th_small  = (-0.20, 0.0, 0.20)
    thd_small = (-1.5, 0.0, 1.5)
    x_small   = (-0.4, 0.0, 0.4)
    xd_small  = (-0.8, 0.0, 0.8)

    # 16 reguł: im więcej "L" w premise, tym większe wzmocnienie
    F_rules = np.zeros((16, 2), dtype=float)
    base_th = 65.0
    base_thd = 10.0
    for idx in range(16):
        bits = [(idx >> b) & 1 for b in (3, 2, 1, 0)]  # th, thd, x, xd
        largeness = sum(bits)
        F_rules[idx, 0] = base_th  + 35.0 * largeness
        F_rules[idx, 1] = base_thd +  6.0 * largeness

    return TSParams16(
        th_small=th_small,
        thd_small=thd_small,
        x_small=x_small,
        xd_small=xd_small,
        F_rules=F_rules,
        u_sat=u_sat,
        sign=-1,          # WAŻNE: tak, żeby miało ten sam kierunek co LQR
        flip_u=False,
        gain_scale=0.25,  # fuzzy tylko dopomaga
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

class TSFuzzyArt3Controller:
    def __init__(self,
                 pars: dict,
                 ts_params: TSParams16,
                 K_lqr: np.ndarray,
                 dt: float,
                 du_max: float = 800.0,
                 ramp_T: float = 2.0):
        self.pars = pars
        self.p = ts_params
        self.K = K_lqr.reshape(1, 4)
        self.dt = dt
        self.du_max = du_max
        self.ramp_T = ramp_T
        self.u_prev = 0.0

    def _u_ts(self, th: float, thd: float, x: float, xd: float) -> float:
        mu = ts_weights16(th, thd, x, xd, self.p)
        z = np.array([th, thd], dtype=float)
        u_rules = - self.p.sign * (self.p.F_rules @ z)
        u = float(mu @ u_rules) * self.p.gain_scale
        if self.p.flip_u:
            u = -u
        return u

    def step(self, t: float, state: np.ndarray, ref_state: np.ndarray) -> float:
        th, thd, x, xd = state
        # LQR na pełnym stanie, ale z referencją na x
        target = np.array([0.0, 0.0, ref_state[2], 0.0], dtype=float)
        err = state - target
        u_lqr = float(- self.K @ err)

        # fuzzy na [theta, theta_dot]
        u_ts = self._u_ts(th, thd, x, xd)

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
# Symulacja + metryki
# =========================
def mse(y, yref): return float(np.mean((np.asarray(y) - np.asarray(yref))**2))
def mae(y, yref): return float(np.mean(np.abs(np.asarray(y) - np.asarray(yref))))

def evaluate_run(X: np.ndarray, U: np.ndarray, x_ref: float) -> float:
    th = X[:, 0]; x = X[:, 2]
    th_ref = np.zeros_like(th); xr = np.ones_like(x) * x_ref
    u_rms = float(np.sqrt(np.mean(U**2))) if len(U) else 0.0
    # trochę jak w art3: duży nacisk na kąt
    return 4.0*mse(th, th_ref) + 1.5*mse(x, xr) + 0.01*u_rms

def simulate(pars: dict,
             controller: TSFuzzyArt3Controller,
             x0: np.ndarray,
             x_ref: np.ndarray,
             T: float,
             dt: float,
             wind: Optional[Callable[[float], float]] = None,
             early_stop: Optional[Tuple[float, float]] = None) -> tuple[np.ndarray, np.ndarray, bool]:
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    forces = []
    t = 0.0
    ok = True
    for _ in range(steps):
        u = controller.step(t, x, x_ref)
        forces.append(u)
        x = rk4_step_wind(x, u, pars, dt, t, wind)
        traj.append(x.copy())
        t += dt
        if early_stop is not None:
            th_th, x_th = early_stop
            if abs(x[0]) > th_th or abs(x[2]) > x_th:
                ok = False
                break
    return np.vstack(traj), np.asarray(forces), ok

# =========================
# Autopick (znak + skala)
# =========================
def autopick_variant(pars: dict, base: TSParams16, dt: float) -> TSParams16:
    best_p = None
    best_s = np.inf
    for sgn in (+1, -1):
        for sc in (0.15, 0.25, 0.35, 0.45):
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
            K_lqr = lqr_from_pars(pars)
            ctrl = TSFuzzyArt3Controller(pars, cand, K_lqr, dt, du_max=1000.0, ramp_T=1.5)
            Xs, Us, ok = simulate(
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
    ax.grid(True, alpha=0.3); ax.set_title("Cart–Pole  (TS-fuzzy + LQR)")
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

def print_metrics(X: np.ndarray, U: np.ndarray, x_ref: float):
    t_len = len(X)
    th_ref_tr = np.zeros(t_len)
    x_ref_tr  = np.ones(t_len) * x_ref
    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr),
        "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x":     mse(X[:,2], x_ref_tr),
        "mae_x":     mae(X[:,2], x_ref_tr),
        "u_rms":     float(np.sqrt(np.mean(U**2))) if len(U) else 0.0
    }
    print(f"MSE(theta)={metrics['mse_theta']:.6f}  MAE(theta)={metrics['mae_theta']:.6f}  "
          f"MSE(x)={metrics['mse_x']:.6f}  MAE(x)={metrics['mae_x']:.6f}  u_rms={metrics['u_rms']:.4f}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    plant = PLANT
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref = SIM["x0"], SIM["x_ref"]

    # jak w Twoim MPC – można włączyć/wyłączyć wiatr
    wind = None
    #wind = Wind(T, seed=23341, Ts=0.02, power=2e-3, smooth=7)

    base_ts  = starter_ts_params16(u_sat=SIM["u_sat"])
    picked_ts = autopick_variant(plant, base_ts, dt)
    K_lqr    = lqr_from_pars(plant)

    print("Użyty wariant TS:",
          f"sign={picked_ts.sign}, gain_scale={picked_ts.gain_scale:.2f}, flip_u={picked_ts.flip_u}")
    print("LQR K:", K_lqr)

    ctrl = TSFuzzyArt3Controller(plant, picked_ts, K_lqr, dt, du_max=1000.0, ramp_T=2.0)

    X, U, ok = simulate(plant, ctrl, x0, x_ref, T, dt,
                        wind=wind, early_stop=None)

    t = np.arange(0.0, T + dt, dt)
    if len(X) != len(t):
        t = np.linspace(0.0, T, len(X))

    title = f"ts_fuzzy_art3ish  |  wind={'on' if wind else 'off'}  |  step=position_step  |  ok={ok}"
    plot_result(t, X, U, title)
    print_metrics(X, U, x_ref[2])

    # animacja opcjonalna
    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
