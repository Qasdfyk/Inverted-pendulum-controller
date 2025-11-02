# controllers/ts_fuzzy_autopick.py
# --------------------------------------------------------
# TS-fuzzy (Takagi–Sugeno, art3 style) + AUTOPICK znaków/kierunków.
# - 4 reguły nad (theta, theta_dot); u_i = -sign * (F_i @ [th, thd, x, xd])
# - zewnętrzny PD na pozycję wózka
# - krótka autokalibracja (3 s) testuje 16 wariantów znaków i skalowania:
#       sign ∈ {+1, -1}, flip_u ∈ {0,1}, dex_sign ∈ {+1, -1}, gain_scale ∈ {0.6, 1.0, 1.4, 1.8}
#   i wybiera ten, który najszybciej zmniejsza |theta| bez wysadzenia trajektorii.
# - miękki rozruch (lambda ramp), ograniczenie du/dt, saturacja.
# --------------------------------------------------------

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import Optional, Callable, Sequence, Tuple

# ===== Plant & sim =====
PLANT = {"M": 2.4, "m": 0.23, "l": 0.36, "g": 9.81}
SIM = {
    "T": 10.0,
    "dt": 0.05,                          # na strojenie stabilniejsze
    "x0": np.array([0.03, 0.0, 0.0, 0.0]),   # [theta, theta_dot, x, x_dot]
    "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),
    "u_sat": 100.0
}

# ===== Wind (opcjonalnie na koniec) =====
class Wind:
    def __init__(self, t_end: float, seed=23341, Ts=0.02, power=2e-3, smooth=7):
        rng = np.random.default_rng(seed)
        self.tgrid = np.arange(0.0, t_end + Ts, Ts)
        sigma = np.sqrt(power / Ts)
        w = rng.normal(0.0, sigma, size=self.tgrid.shape)
        if smooth and smooth > 1:
            ker = np.ones(smooth) / smooth
            self.Fw = np.convolve(w, ker, mode='same')
        else:
            self.Fw = w
    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.tgrid, self.Fw))

# ===== Dynamics + RK4 (wiatr tylko w roślinie) =====
def f_nonlinear(x: np.ndarray, u: float, pars: dict, Fw: float = 0.0) -> np.ndarray:
    th, thd, pos, posd = x
    M, m, l, g = pars["M"], pars["m"], pars["l"], pars["g"]
    s, c = np.sin(th), np.cos(th)
    denom_x  = (M + m) - m * c * c
    denom_th = (m * l * c * c) - (M + m) * l
    thdd = (u * c - (M + m) * g * s + m * l * (c * s) * (thd ** 2) - (M / m) * Fw * c) / denom_th
    xdd  = (u + m * l * s * (thd ** 2) - m * g * c * s + Fw * (s * s)) / denom_x
    return np.array([thd, thdd, posd, xdd], dtype=float)

def rk4_step_wind(x, u, pars, dt, t, wind: Optional[Callable[[float], float]] = None):
    F1 = wind(t) if wind else 0.0
    k1 = f_nonlinear(x, u, pars, F1)
    F2 = wind(t + 0.5 * dt) if wind else 0.0
    k2 = f_nonlinear(x + 0.5 * dt * k1, u, pars, F2)
    F3 = wind(t + 0.5 * dt) if wind else 0.0
    k3 = f_nonlinear(x + 0.5 * dt * k2, u, pars, F3)
    F4 = wind(t + dt) if wind else 0.0
    k4 = f_nonlinear(x + dt * k3, u, pars, F4)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

# ===== TS fuzzy =====
EPS = 1e-12
def tri_mf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a + 1e-12)
    return (c - x) / (c - b + 1e-12)

@dataclass
class TSParams:
    theta_small: Sequence[float]      # [a,b,c]
    thdot_small: Sequence[float]      # [a,b,c]
    F_rules: np.ndarray               # (4,4): rows (S,S),(L,S),(S,L),(L,L); cols [kth,kthd,kx,kxd]
    Kp_cart: float
    Kd_cart: float
    u_sat: float
    sign: int         # +1 lub -1
    flip_u: bool      # True -> wyjście * (-1)
    dex_sign: int     # +1 lub -1, mnoży x_dot w pętli PD
    gain_scale: float # globalny mnożnik F_rules

class TSFuzzyController:
    def __init__(self, pars: dict, params: TSParams, dt: float,
                 du_max: float = 800.0, ramp_T: float = 2.0):
        self.pars = pars
        self.p = params
        self.dt = dt
        self.u_prev = 0.0
        self.du_max = float(du_max)            # rate limit [N/s]
        self.ramp_T = float(ramp_T)            # miękki rozruch (sekundy)

    def _weights(self, th: float, thd: float) -> np.ndarray:
        ms_t  = tri_mf(th,  *self.p.theta_small)
        ms_td = tri_mf(thd, *self.p.thdot_small)
        ml_t, ml_td = 1.0 - ms_t, 1.0 - ms_td
        w1 = ms_t * ms_td
        w2 = ml_t * ms_td
        w3 = ms_t * ml_td
        w4 = ml_t * ml_td
        W = w1 + w2 + w3 + w4 + EPS
        return np.array([w1, w2, w3, w4], dtype=float) / W

    def _u_ts(self, th: float, thd: float, x: float, xd: float) -> float:
        mu = self._weights(th, thd)
        z = np.array([th, thd, x, xd], float)
        F = self.p.F_rules * self.p.gain_scale
        u_rules = - self.p.sign * (F @ z)
        return float(mu @ u_rules)

    def _u_cart_pd(self, x: float, xd: float, x_ref: float) -> float:
        ex  = (x_ref - x)
        dex = self.p.dex_sign * xd * (-1.0)  # jeśli dex_sign = -1 -> jak wcześniej; jeśli +1 -> odwrócenie
        return self.p.Kp_cart * ex + self.p.Kd_cart * dex

    def step(self, t: float, state: np.ndarray, ref_state: np.ndarray) -> float:
        th, thd, x, xd = state
        u = self._u_cart_pd(x, xd, ref_state[2]) + self._u_ts(th, thd, x, xd)
        if self.p.flip_u:
            u = -u
        # miękki rozruch
        if self.ramp_T > 0.0:
            alpha = min(1.0, t / self.ramp_T)
            u *= (0.25 + 0.75 * alpha)
        # rate limit
        du = np.clip(u - self.u_prev, -self.du_max * self.dt, self.du_max * self.dt)
        u_limited = self.u_prev + du
        self.u_prev = float(np.clip(u_limited, -self.p.u_sat, self.p.u_sat))
        return self.u_prev

# ===== Metrics / simulate (z wczesnym odcięciem) =====
def mse(y, yref): return float(np.mean((np.asarray(y) - np.asarray(yref))**2))
def evaluate_run(X: np.ndarray, U: np.ndarray, x_ref: float) -> float:
    th = X[:, 0]; x = X[:, 2]
    th_ref = np.zeros_like(th); xr = np.ones_like(x) * x_ref
    u_rms = float(np.sqrt(np.mean(U**2))) if len(U) else 1e6
    return 3.0*float(np.mean((th - th_ref)**2)) + 1.0*float(np.mean((x - xr)**2)) + 0.02*u_rms

def simulate(pars, controller: TSFuzzyController, x0, x_ref, T, dt,
             wind: Optional[Callable[[float], float]] = None,
             early_thresh: Tuple[float, float] = (2.8, 4.0),
             early_T: float | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    steps = int(np.round(T / dt))
    early_steps = steps if early_T is None else min(steps, int(np.round(early_T / dt)))
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]; forces = []
    t = 0.0; ok = True
    th_th, x_th = early_thresh
    controller.u_prev = 0.0
    for k in range(steps):
        u = controller.step(t, x, x_ref)
        forces.append(u)
        x = rk4_step_wind(x, u, pars, dt, t, wind)
        traj.append(x.copy())
        t += dt
        if abs(x[0]) > th_th or abs(x[2]) > x_th:
            ok = False; break
        if (k+1) >= early_steps and early_T is not None:
            break
    return np.vstack(traj), np.asarray(forces), ok

# ===== Starter param (łagodny, jak w paperowych przykładach) =====
def starter_params(u_sat=SIM["u_sat"]) -> TSParams:
    theta_small = [-0.12, 0.0, 0.12]
    thdot_small = [-0.9,  0.0, 0.9]
    F = np.array([
        [ 65.0, 10.0,  0.0,  0.0],   # (S,S)
        [120.0, 16.0,  0.0,  0.0],   # (L,S)
        [ 85.0, 22.0,  0.0,  0.0],   # (S,L)
        [180.0, 32.0,  0.0,  0.0],   # (L,L)
    ], float)
    return TSParams(theta_small, thdot_small, F, Kp_cart=2.2, Kd_cart=6.5,
                    u_sat=u_sat, sign=+1, flip_u=False, dex_sign=-1, gain_scale=1.0)

# ===== Autopick wariantu znaków/skal =====
def autopick_variant() -> TSParams:
    base = starter_params()
    best_p = None; best_s = np.inf
    variants = []
    for sign in (+1, -1):
        for flip_u in (False, True):
            for dex_sign in (+1, -1):
                for scale in (0.6, 1.0, 1.4, 1.8):
                    p = starter_params()
                    p.sign = sign
                    p.flip_u = flip_u
                    p.dex_sign = dex_sign
                    p.gain_scale = scale
                    variants.append(p)

    for p in variants:
        ctrl = TSFuzzyController(PLANT, p, SIM["dt"], du_max=1000.0, ramp_T=1.5)
        Xs, Us, ok = simulate(PLANT, ctrl, SIM["x0"], SIM["x_ref"], SIM["T"], SIM["dt"],
                              wind=None, early_thresh=(2.8, 4.0), early_T=3.0)
        if not ok: 
            continue
        s = evaluate_run(Xs, Us, SIM["x_ref"][2])
        if s < best_s:
            best_s, best_p = s, p
    return best_p

# ===== Plot/anim =====
def plot_result(X, U, title):
    t = np.arange(0, SIM["T"] + SIM["dt"], SIM["dt"])
    if len(X) != len(t): t = np.linspace(0.0, SIM["T"], len(X))
    fig = plt.figure(figsize=(9,7))
    fig.suptitle(title, fontsize=12, y=0.98)
    ax1 = fig.add_subplot(3,1,1); ax1.plot(t, X[:,0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3,1,2); ax2.plot(t, X[:,2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    tf = t[:-1] if len(U) == (len(t)-1) else np.linspace(0.0, SIM["T"], len(U))
    ax3 = fig.add_subplot(3,1,3); ax3.plot(tf, U); ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

# ===== Main =====
if __name__ == "__main__":
    pick = autopick_variant()
    if pick is None:
        print("Autopick nie znalazł stabilnego wariantu w 3 s. Spróbuj ręcznie: ustaw flip_u=True albo sign=-1.")
        # awaryjnie odpal najbezpieczniejszy wariant:
        pick = starter_params(); pick.sign = -1; pick.flip_u = True; pick.dex_sign = -1; pick.gain_scale = 1.0

    print("Użyty wariant:",
          f"sign={pick.sign}, flip_u={pick.flip_u}, dex_sign={pick.dex_sign}, scale={pick.gain_scale}")
    ctrl = TSFuzzyController(PLANT, pick, SIM["dt"], du_max=1000.0, ramp_T=2.0)

    # końcowy bieg (bez wiatru; jak chcesz, włącz niżej wind)
    X, U, ok = simulate(PLANT, ctrl, SIM["x0"], SIM["x_ref"], SIM["T"], SIM["dt"],
                        wind=None, early_thresh=(3.14, 4.5), early_T=None)
    print("Stable:", ok)
    plot_result(X, U, "ts_fuzzy_autopick  |  wind=off  |  step=position_step")

    # test z wiatrem (opcjonalnie)
    # wind = Wind(SIM["T"], seed=23341, Ts=0.02, power=2e-3, smooth=7)
    # Xw, Uw, okw = simulate(PLANT, ctrl, SIM["x0"], SIM["x_ref"], SIM["T"], SIM["dt"], wind=wind)
    # plot_result(Xw, Uw, "ts_fuzzy_autopick  |  wind=on  |  step=position_step")
