# mpc_cartpole_physical.py
# --------------------------------------------------------
# Nonlinear MPC for the cart-pole (art2 model) + animation + metrics.
# Parameters fixed to your env config.
# --------------------------------------------------------

from __future__ import annotations
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# =========================
# Fixed physical & sim params
# =========================
PLANT = {"M": 2.4, "m": 0.23, "l": 0.36, "g": 9.81}
SIM = {
    "T": 10.0,
    "dt": 0.1,         # MPC + integration step
    "x0": np.array([0.03, 0.0, 0.0, 0.0]),  # [rad, rad/s, m, m/s]
    "x_ref": np.array([0.0, 0.0, 0.10, 0.0]),    # track 0.1 m cart step
    "u_sat": 100.0
}


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


# =========================
# Nonlinear MPC
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
    ax.grid(True, alpha=0.3); ax.set_title("Cart–Pole")
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
# Metrics
# =========================
def mse(y, yref): return float(np.mean((np.asarray(y) - np.asarray(yref))**2))
def mae(y, yref): return float(np.mean(np.abs(np.asarray(y) - np.asarray(yref))))
def print_summary(metrics):
    print(f"MSE(theta)={metrics['mse_theta']:.6f}  MAE(theta)={metrics['mae_theta']:.6f}  "
          f"MSE(x)={metrics['mse_x']:.6f}  MAE(x)={metrics['mae_x']:.6f}")


# =========================
# Simulation loop
# =========================
def simulate_mpc(pars, controller, x0, x_ref, T, dt, u0=0.0):
    steps = int(np.round(T / dt))
    x = np.asarray(x0, float).copy(); u_prev = float(u0)
    traj = [x.copy()]; forces = []
    for _ in range(steps):
        u_seq = controller.compute_control(x, x_ref, u_prev)
        u_apply = float(u_seq[0])
        forces.append(u_apply)
        x = rk4_step(f_nonlinear, x, u_apply, pars, dt)
        traj.append(x.copy())
        u_prev = u_apply
    return np.vstack(traj), np.asarray(forces)


# =========================
# Main
# =========================
if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT

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

    X, U = simulate_mpc(plant, ctrl, x0, x_ref, T, dt)
    t = np.arange(0, T + dt, dt)
    if len(X) != len(t): t = np.linspace(0, T, len(X))

    # ---- plotting same as env ----
    controller = "mpc_no"; disturbance = False; step_type = "position_step"
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(f"{controller}  |  wind={'on' if disturbance else 'off'}  |  step={step_type}",
                 fontsize=12, y=0.98)
    ax1 = fig.add_subplot(3, 1, 1); ax1.plot(t, X[:, 0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
    ax2 = fig.add_subplot(3, 1, 2); ax2.plot(t, X[:, 2]); ax2.grid(True); ax2.set_ylabel('x [m]')
    tf = t[:-1] if len(U) == (len(t)-1) else np.linspace(0.0, T, len(U))
    ax3 = fig.add_subplot(3, 1, 3); ax3.plot(tf, U); ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # ---- metrics ----
    th_ref_tr = np.zeros_like(t)
    x_ref_tr = np.ones_like(t) * x_ref[2]
    metrics = {
        "mse_theta": mse(X[:, 0], th_ref_tr),
        "mae_theta": mae(X[:, 0], th_ref_tr),
        "mse_x": mse(X[:, 2], x_ref_tr),
        "mae_x": mae(X[:, 2], x_ref_tr)
    }
    print_summary(metrics)

    # ---- optional animation ----
    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
