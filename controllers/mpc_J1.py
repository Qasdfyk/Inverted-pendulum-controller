from __future__ import annotations
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import utilsów
from mpc_utils import (PLANT, SIM, Wind, f_nonlinear, rk4_step, simulate_mpc, 
                       print_summary, animate_cartpole,
                       mse, mae, iae, ise, control_energy_l2, control_energy_l1,
                       settling_time, overshoot, steady_state_error, disturbance_robustness)

class MPCControllerJ1:
    def __init__(self, pars, dt, N, Nu, umin, umax, q_theta, q_x, r):
        self.pars = pars
        self.dt = dt
        self.N = N
        self.Nu = Nu
        self.umin = umin
        self.umax = umax
        # Wagi
        self.q_theta_base = float(q_theta)
        self.q_x = float(q_x)
        self.r = float(r)

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
        
        # Pobranie zmiennych (theta:0, x:2)
        theta_k = preds[:, 0]
        x_k     = preds[:, 2]

        # WZÓR: J = q_th * th^2 + q_x * (x-ref)^2 + r * du^2
        # Usunięto "panic mode" -> teraz jest stabilnie.
        
        cost_theta = np.sum(self.q_theta_base * (theta_k**2))
        cost_x     = np.sum(self.q_x * ((x_k - x_ref[2])**2))
        cost_u     = self.r * np.sum(du**2)

        return cost_theta + cost_x + cost_u

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
        
        # Zwracamy pełną sekwencję (pierwszy element zostanie użyty)
        return np.array([u_prev + du_opt[0]])

if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT

    # Wind setup
    # wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)
    wind = None

    # Parametry zgodne z oryginałem (plus lekkie dostrojenie q_x żeby dojeżdżał)
    ctrl = MPCControllerJ1(
        pars=plant, dt=dt, N=10, Nu=8, 
        umin=-u_sat, umax=u_sat,
        q_theta=60.0, 
        q_x=25.0, # Lekko podbite żeby lepiej trzymał pozycję
        r=1e-3
    )

    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)

    # --- PEŁNY OUTPUT JAK W ORYGINALE ---
    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    controller_name = "mpc_J1_formula"
    disturbance = wind is not None
    step_type = "position_step"
    fig = plt.figure(figsize=(9, 7))
    fig.suptitle(f"{controller_name}  |  wind={'on' if disturbance else 'off'}", fontsize=12, y=0.98)
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
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, e_th_tf, Fw_tr)
        snr_x,  rms_e_x,  _     = disturbance_robustness(tf, e_x_tf,  Fw_tr)
        metrics["snr_theta"] = snr_th
        metrics["snr_x"]     = snr_x
        metrics["rms_e_theta"] = rms_e_th
        metrics["rms_e_x"]     = rms_e_x
        metrics["rms_Fw"]      = rms_F
    else:
        metrics["snr_theta"] = float('nan'); metrics["snr_x"] = float('nan')
        metrics["rms_e_theta"] = float('nan'); metrics["rms_e_x"] = float('nan'); metrics["rms_Fw"] = float('nan')

    print_summary(metrics)

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)