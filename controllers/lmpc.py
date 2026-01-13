from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import cont2discrete
import os

# Import shared infrastructure
from mpc_utils import (
    PLANT, SIM, Wind, simulate_mpc,
    print_summary, animate_cartpole, linearize_upright,
    mse, mae, iae, ise, control_energy_l2, control_energy_l1,
    settling_time, overshoot, steady_state_error, disturbance_robustness
)

# ==========================================
# LINEAR MPC CONTROLLER (ZOH + minimize L-BFGS-B)
# ==========================================
class LinearMPCController:
    def __init__(self, pars: dict, dt: float, N: int, Nu: int,
                 umin: float, umax: float, Q: np.ndarray, R: float):
        self.pars = pars
        self.dt = dt
        self.N = N
        self.Nu = Nu
        self.umin = umin
        self.umax = umax
        self.Q = Q
        self.R = R

        # --- LINEARIZATION ---
        Ac, Bc = linearize_upright(pars["M"], pars["m"], pars["l"], pars["g"])

        # --- DISCRETIZATION: ZOH ---
        # x[k+1] = Ad x[k] + Bd u[k] (assuming u is constant on [k*dt, (k+1)*dt])
        Ac = np.asarray(Ac, dtype=float)
        Bc = np.asarray(Bc, dtype=float).reshape(4, 1)
        Cc = np.eye(4)
        Dc = np.zeros((4, 1))
        Ad, Bd, *_ = cont2discrete((Ac, Bc, Cc, Dc), dt, method="zoh")

        self.Ad = np.asarray(Ad, dtype=float)
        self.Bd = np.asarray(Bd, dtype=float).reshape(4, 1)
        self.Bd_flat = self.Bd.flatten()

    def _rollout_linear(self, x0, u_seq):
        """Symulacja przyszłości przy użyciu modelu LINIOWEGO (dyskretny ZOH)."""
        traj = []
        x_curr = np.array(x0, dtype=float)

        for ui in u_seq:
            x_curr = self.Ad @ x_curr + self.Bd_flat * float(ui)
            traj.append(x_curr.copy())

        return np.asarray(traj)

    def _cost(self, du, x0, x_ref, u_prev):
        du = np.asarray(du, dtype=float)

        # Rekonstrukcja sekwencji sterowania z przyrostów (du)
        u_seq = np.zeros(self.N, dtype=float)
        if self.Nu > 0:
            u_cum = u_prev + np.cumsum(du)
            upto = min(self.Nu, self.N)
            u_seq[:upto] = u_cum[:upto]
            if self.N > self.Nu:
                u_seq[self.Nu:] = u_cum[upto - 1]
        else:
            u_seq[:] = u_prev

        # Predykcja (liniowa, dyskretna)
        preds = self._rollout_linear(x0, u_seq)

        # Błąd i koszt
        err = preds - np.asarray(x_ref, dtype=float).reshape(1, -1)

        cost_y = 0.0
        for e in err:
            cost_y += float(e.T @ self.Q @ e)

        cost_u = float(self.R * np.sum(du * du))
        return cost_y + cost_u

    def compute_control(self, x0, x_ref, u_prev):
        # Bounds na du (tak jak w Twojej wersji)
        du_min = self.umin - u_prev
        du_max = self.umax - u_prev
        bounds = [(du_min, du_max)] * self.Nu
        du0 = np.zeros(self.Nu, dtype=float)

        res = minimize(
            self._cost, du0,
            args=(x0, x_ref, u_prev),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-6, "disp": False},
        )
        du_opt = res.x if res.success else du0

        # Pierwszy ruch
        u_next = u_prev + float(du_opt[0]) if self.Nu > 0 else float(u_prev)
        u_next = float(np.clip(u_next, self.umin, self.umax))
        return [u_next]


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT

    #wind = Wind(T, seed=42, Ts=0.01, power=5e-3)
    wind = None

    ctrl = LinearMPCController(
        pars=plant, dt=dt, N=12, Nu=4, umin=-u_sat, umax=u_sat,
        Q=np.diag([15.0, 1.0, 15.0, 1.0]),
        R=0.1
    )

    print("Rozpoczynam symulację Liniowego MPC (ZOH + L-BFGS-B)...")
    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(
        plant, ctrl, x0, x_ref, T, dt, wind=wind
    )

    # --- Wykresy ---
    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    controller_name = "linear_mpc_zoh_lbfgsb_simple"
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

    # ---- metrics ----
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
    metrics["ess_theta_rms"] = ess_th_rms
    metrics["ess_x_rms"]     = ess_x_rms

    if disturbance:
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

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
