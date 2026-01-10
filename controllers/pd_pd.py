from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import os
from mpc_utils import (PLANT, SIM, Wind, f_nonlinear, rk4_step, simulate_mpc, 
                       print_summary, animate_cartpole,
                       mse, mae, iae, ise, control_energy_l2, control_energy_l1,
                       settling_time, overshoot, steady_state_error, disturbance_robustness)

class PDPDController:
    def __init__(self, pars: dict, dt: float, 
                 ang_pid: dict, cart_pid: dict,
                 u_limit: float = 80.0, integ_limit: float = 5.0):
        self.pars = pars
        self.dt = dt
        self.ang = ang_pid
        self.cart = cart_pid
        self.u_limit = u_limit
        self.integ_limit = integ_limit
        
        self.z_ang = 0.0
        self.z_cart = 0.0

    def _pid_step(self, e: float, de_dt: float, z: float, gains: dict, dt: float) -> tuple[float, float]:
        Ki = float(gains.get("Ki", 0.0))
        if Ki != 0.0:
            z = z + e * dt
            z = float(np.clip(z, -self.integ_limit, self.integ_limit))
        
        u = float(gains["Kp"]) * e + Ki * z + float(gains["Kd"]) * de_dt
        return u, z

    def compute_control(self, x0, x_ref, u_prev):
        # x0 = [th, thd, x, xd]
        th, thd, pos, posd = x0
        
        # References from x_ref
        # x_ref vector is [th_ref, thd_ref, x_ref, xd_ref] usually?
        # In mpc_utils.SIM, x_ref = [0.0, 0.0, 0.10, 0.0]
        th_ref = x_ref[0]
        target_pos = x_ref[2]

        # Angle PD
        e_th = th_ref - th
        de_th = -thd # assuming d(th_ref)/dt = 0
        u_ang, self.z_ang = self._pid_step(e_th, de_th, self.z_ang, self.ang, self.dt)

        # Cart PD
        e_x = target_pos - pos
        de_x = -posd
        u_cart, self.z_cart = self._pid_step(e_x, de_x, self.z_cart, self.cart, self.dt)

        u = u_ang + u_cart
        u = float(np.clip(u, -self.u_limit, self.u_limit))
        
        return np.array([u])

if __name__ == "__main__":
    dt, T = SIM["dt"], SIM["T"]
    x0, x_ref, u_sat = SIM["x0"], SIM["x_ref"], SIM["u_sat"]
    plant = PLANT
    wind = None
    #wind = Wind(T, seed=23341, Ts=0.01, power=1e-3, smooth=5)

    # Gains from config/controllers/pd_pd.yaml
    ang_pid = {"Kp": -40.0, "Ki": -1.0, "Kd": -8.0}
    cart_pid = {"Kp": -1.0, "Ki": -0.1, "Kd": -3.0}

    # ang_pid = {"Kp": -95.0, "Ki": 0.0, "Kd": -14.0}
    # cart_pid = {"Kp": -16.0, "Ki": 0.0, "Kd": -14.0}

    ctrl = PDPDController(
        pars=plant, dt=dt, 
        ang_pid=ang_pid, cart_pid=cart_pid,
        u_limit=80.0, integ_limit=5.0
    )

    X, U, Fw_tr, ctrl_time_total, sim_time_wall = simulate_mpc(plant, ctrl, x0, x_ref, T, dt, wind=wind)

    steps = len(U)
    t = np.linspace(0.0, T, steps + 1)
    tf = t[:-1]

    controller_name = "pd_pd"
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
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_tr[-1])  if is_step else float('nan') # Fixed index error potential if overshoot uses end

    ess_th_mean, ess_th_rms = steady_state_error(t, X[:,0], th_ref_tr, window_frac=ess_window_frac, window_min=ess_window_min)
    ess_x_mean,  ess_x_rms  = steady_state_error(t, X[:,2], x_ref_tr,  window_frac=ess_window_frac, window_min=ess_window_min)
    metrics["ess_theta"] = ess_th_mean
    metrics["ess_x"]     = ess_x_mean
    
    if disturbance:
        e_th_tf = X[1:, 0] - th_ref_tr[1:]
        e_x_tf  = X[1:, 2] - x_ref_tr[1:]
        snr_th, rms_e_th, rms_F = disturbance_robustness(tf, e_th_tf, Fw_tr, window_frac=0.5, window_min=1.0)
        snr_x,  rms_e_x,  _     = disturbance_robustness(tf, e_x_tf,  Fw_tr, window_frac=0.5, window_min=1.0)
        metrics["snr_theta"] = snr_th
        metrics["snr_x"]     = snr_x
    else:
        metrics["snr_theta"] = float('nan')
        metrics["snr_x"]     = float('nan')

    print_summary(metrics)

    if os.environ.get("ANIMATE", "0") == "1":
        animate_cartpole(t, X, params=plant)
