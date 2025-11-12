from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.integrate import solve_ivp
from time import perf_counter  # <-- NEW

from utils.cfg import load_configs
from utils import signals
from env.model import f_nonlinear
from env.disturbance import Wind
from env.metrics import (
    mse, mae, print_summary,
    iae, ise, control_energy_l2, control_energy_l1,
    settling_time, overshoot, steady_state_error, disturbance_robustness
)
from env.animation import animate_cartpole
from env.archives import archive
from controllers import REGISTRY
from dataclasses import dataclass
import numpy as np
from typing import Dict

@dataclass
class Result:
    t: np.ndarray
    X: np.ndarray
    U: np.ndarray
    Fw: np.ndarray
    refs: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    meta: Dict[str, object]
def _ref_fun(step_type: str, cfg):
    if step_type == "position_step":
        return lambda t: (cfg["sim"]["x_ref"], cfg["sim"]["theta_ref"])
    if step_type == "angle_step":
        return lambda t: (0.0, 0.05)
    if step_type == "track_sine":
        return lambda t: (signals.track_sine(t, 0.05, 1.0), 0.0)
    if step_type == "impulse":
        return lambda t: (signals.impulse(t, 0.5, 0.05), 0.0)
    raise ValueError(f"Unknown step_type: {step_type}")

def run(controller: str, disturbance: bool, animation: bool,
        step_type: str, archive_results: bool, config_variant: Optional[str]=None):
    cfg = load_configs(controller, config_variant)
    plant = cfg["plant"]; sim = cfg["sim"]; u_sat = sim["u_sat"]

    build_fn = REGISTRY[controller](cfg)
    ctrl = build_fn(plant)
    ctrl.reset()

    if disturbance and cfg["disturbance"]["type"] == "wind":
        wind = Wind(sim["t_end"], seed=cfg["disturbance"]["seed"],
                    Ts=cfg["disturbance"]["Ts"], power=cfg["disturbance"]["power"],
                    smooth=cfg["disturbance"]["smooth"])
        Fw_fun = wind
    else:
        Fw_fun = lambda t: 0.0

    ref_fun = _ref_fun(step_type, cfg)

    t0, tf = 0.0, sim["t_end"]
    dt_log = sim["dt_log"]
    t_grid = np.arange(t0, tf + 1e-12, dt_log)
    x = np.array(sim["x0"], dtype=float)

    # ====== Pomiar czasu kontrolera (akumulowany) ======
    ctrl_time_total = 0.0

    def rhs(t, x):
        nonlocal ctrl_time_total
        xref, thref = ref_fun(t)
        t_start = perf_counter()
        u = ctrl.step(t, x, {"x_ref": xref, "theta_ref": thref, "dt": dt_log})
        ctrl_time_total += perf_counter() - t_start
        u = np.clip(u, -u_sat, u_sat)
        Fw = Fw_fun(t)
        return f_nonlinear(x, u, plant, Fw)

    # ====== Start całościowego timera ======
    t_sim_start = perf_counter()
    sol = solve_ivp(rhs, (t0, tf), x, t_eval=t_grid, max_step=dt_log, rtol=1e-8, atol=1e-9)
    t = sol.t; X = sol.y.T

    # Rekonstrukcja torów sterowania/referencji do logów + dalsze liczenie ctrl_time
    U = np.zeros_like(t)
    F = np.zeros_like(t)
    x_ref_tr = np.zeros_like(t)
    th_ref_tr = np.zeros_like(t)
    for i, ti in enumerate(t):
        xref, thref = ref_fun(ti)
        t_start = perf_counter()
        u = ctrl.step(ti, X[i], {"x_ref": xref, "theta_ref": thref, "dt": dt_log})
        ctrl_time_total += perf_counter() - t_start
        U[i] = np.clip(u, -u_sat, u_sat)
        F[i] = Fw_fun(ti)
        x_ref_tr[i] = xref
        th_ref_tr[i] = thref

    sim_time_wall = perf_counter() - t_sim_start

    # ====== METRYKI ======
    # Błędy
    e_th = X[:,0] - th_ref_tr
    e_x  = X[:,2] - x_ref_tr

    # Progi i okna domyślne (możesz później przenieść do cfg["metrics"])
    eps_theta = 0.01   # rad
    eps_x     = 0.01   # m
    hold_time = 0.5    # s
    ess_window_frac = 0.10
    ess_window_min  = 0.5

    # Czy scenariusz krokowy (overshoot/settling sensowne)
    is_step = step_type in ("position_step", "angle_step")

    # ---- bazowe
    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr),
        "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x": mse(X[:,2], x_ref_tr),
        "mae_x": mae(X[:,2], x_ref_tr),
        "iae_theta": iae(t, X[:,0], th_ref_tr),
        "ise_theta": ise(t, X[:,0], th_ref_tr),
        "iae_x": iae(t, X[:,2], x_ref_tr),
        "ise_x": ise(t, X[:,2], x_ref_tr),
        "e_u_l2": control_energy_l2(t, U),
        "e_u_l1": control_energy_l1(t, U),
        "sim_time_wall": sim_time_wall,
        "ctrl_time_total": ctrl_time_total,
    }

    # ---- czas ustalania
    metrics["t_s_theta"] = settling_time(t, X[:,0], th_ref_tr, eps=eps_theta, hold_time=hold_time) if is_step else float('nan')
    metrics["t_s_x"]     = settling_time(t, X[:,2], x_ref_tr, eps=eps_x,     hold_time=hold_time) if is_step else float('nan')

    # ---- overshoot (tylko dla stepów i gdy ref_final != 0)
    th_ref_final = th_ref_tr[-1]
    x_ref_final  = x_ref_tr[-1]
    metrics["overshoot_theta"] = overshoot(X[:,0], th_ref_final) if is_step else float('nan')
    metrics["overshoot_x"]     = overshoot(X[:,2], x_ref_final)  if is_step else float('nan')

    # ---- błąd ustalony (mean i RMS na końcu)
    ess_th_mean, ess_th_rms = steady_state_error(t, X[:,0], th_ref_tr, window_frac=ess_window_frac, window_min=ess_window_min)
    ess_x_mean,  ess_x_rms  = steady_state_error(t, X[:,2], x_ref_tr,  window_frac=ess_window_frac, window_min=ess_window_min)
    metrics["ess_theta"] = ess_th_mean
    metrics["ess_x"]     = ess_x_mean
    metrics["ess_theta_rms"] = ess_th_rms
    metrics["ess_x_rms"]     = ess_x_rms

    # ---- odporność na zakłócenia (SNR = RMS(e)/RMS(Fw) na końcówce)
    if disturbance:
        snr_th, rms_e_th, rms_F = disturbance_robustness(t, e_th, F, window_frac=0.5, window_min=1.0)
        snr_x,  rms_e_x,  _     = disturbance_robustness(t, e_x,  F, window_frac=0.5, window_min=1.0)
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

    print(f"Controller: {controller} | Disturbance: {'on' if disturbance else 'off'} | Step: {step_type}")
    print_summary(metrics)

    import matplotlib.pyplot as plt
    headless = os.environ.get("CARTPOLE_HEADLESS") == "1"

    if not headless:
        fig = plt.figure(figsize=(9, 7))
        fig.suptitle(f"{controller}  |  wind={'on' if disturbance else 'off'}  |  step={step_type}", fontsize=12, y=0.98)

        ax1 = fig.add_subplot(3,1,1); ax1.plot(t, X[:,0]); ax1.grid(True); ax1.set_ylabel('theta [rad]')
        ax2 = fig.add_subplot(3,1,2); ax2.plot(t, X[:,2]); ax2.grid(True); ax2.set_ylabel('x [m]')
        ax3 = fig.add_subplot(3,1,3); ax3.plot(t, U);       ax3.grid(True); ax3.set_ylabel('u [N]'); ax3.set_xlabel('t [s]')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    if animation and not headless:
        animate_cartpole(t, X, params=plant)

    if archive_results:
        refs = {"x_ref": x_ref_tr, "theta_ref": th_ref_tr}
        meta = {"controller": controller, "disturbance": disturbance, "step_type": step_type}
        archive("runs", t, X, U, F, refs, metrics, meta)

    return Result(t=t, X=X, U=U, Fw=F, refs={"x_ref": x_ref_tr, "theta_ref": th_ref_tr},
                  metrics=metrics, meta={"controller": controller})
