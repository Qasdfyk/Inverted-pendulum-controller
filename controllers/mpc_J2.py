from __future__ import annotations
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from mpc_utils import (PLANT, SIM, Wind, f_nonlinear, rk4_step, simulate_mpc, 
                       print_summary, animate_cartpole,
                       mse, mae, iae, ise, control_energy_l2, control_energy_l1,
                       settling_time, overshoot, steady_state_error, disturbance_robustness)

class MPCControllerJ2:
    def __init__(self, pars: dict, dt: float, N: int, Nu: int,
                 umin: float, umax: float, 
                 q_theta: float, q_x: float, q_thd: float = 1.0, q_xd: float = 1.0,
                 r: float = 0.001, r_abs: float = 0.0):
        self.pars = pars
        self.dt = dt
        self.N = N
        self.Nu = Nu
        self.umin = umin
        self.umax = umax
        self.q_theta = q_theta
        self.q_x = q_x
        self.q_thd = q_thd
        self.q_xd = q_xd
        self.r = r
        self.r_abs = r_abs

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
        
        e_th  = preds[:, 0] - x_ref[0]
        e_thd = preds[:, 1] - x_ref[1]
        e_x   = preds[:, 2] - x_ref[2]
        e_xd  = preds[:, 3] - x_ref[3]

        cost_state = np.sum(self.q_theta * e_th**2 + self.q_thd * e_thd**2 + 
                            self.q_x * e_x**2 + self.q_xd * e_xd**2)
        
        cost_du = self.r * np.sum(du**2)
        cost_u_abs = self.r_abs * np.sum(u_seq**2)

        return cost_state + cost_du + cost_u_abs

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
            options={'maxiter': 100, 'ftol': 1e-3, 'disp': False}
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

if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    #wind = None
    wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)

    ctrl = MPCControllerJ2(
        pars=plant, dt=dt, N=15, Nu=7, umin=-u_sat, umax=u_sat,
        q_theta=80.0, q_x=120.0, q_thd=5.0, q_xd=5.0,
        r=0.0001, r_abs=0.0
    )

    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)

    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    controller_name = "mpc_J2"
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

    th_ref_tr = np.zeros_like(t)
    x_ref_tr  = np.ones_like(t) * x_ref[2]

    eps_theta = 0.01; eps_x = 0.01; hold_time = 0.5
    ess_window_frac = 0.10; ess_window_min = 0.5
    is_step = True

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