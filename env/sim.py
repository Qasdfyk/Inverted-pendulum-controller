
from __future__ import annotations
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from scipy.integrate import solve_ivp

from utils.cfg import load_configs
from utils import signals
from env.model import f_nonlinear
from env.disturbance import Wind
from env.metrics import mse, mae, print_summary
from env.animation import animate_cartpole
from env.archives import archive
from controllers import REGISTRY

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

    def rhs(t, x):
        xref, thref = ref_fun(t)
        u = ctrl.step(t, x, {"x_ref": xref, "theta_ref": thref, "dt": dt_log})
        u = np.clip(u, -u_sat, u_sat)
        Fw = Fw_fun(t)
        return f_nonlinear(x, u, plant, Fw)

    sol = solve_ivp(rhs, (t0, tf), x, t_eval=t_grid, max_step=dt_log, rtol=1e-8, atol=1e-9)
    t = sol.t; X = sol.y.T

    U = np.zeros_like(t)
    F = np.zeros_like(t)
    x_ref_tr = np.zeros_like(t)
    th_ref_tr = np.zeros_like(t)
    for i, ti in enumerate(t):
        xref, thref = ref_fun(ti)
        u = ctrl.step(ti, X[i], {"x_ref": xref, "theta_ref": thref, "dt": dt_log})
        U[i] = np.clip(u, -u_sat, u_sat)
        F[i] = Fw_fun(ti)
        x_ref_tr[i] = xref
        th_ref_tr[i] = thref

    metrics = {
        "mse_theta": mse(X[:,0], th_ref_tr),
        "mae_theta": mae(X[:,0], th_ref_tr),
        "mse_x": mse(X[:,2], x_ref_tr),
        "mae_x": mae(X[:,2], x_ref_tr),
    }
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

    # Optional true animation (cart & pole)
    if animation and not headless:
        animate_cartpole(t, X, params=plant)


    if archive_results:
        refs = {"x_ref": x_ref_tr, "theta_ref": th_ref_tr}
        meta = {"controller": controller, "disturbance": disturbance, "step_type": step_type}
        archive("runs", t, X, U, F, refs, metrics, meta)

    return Result(t=t, X=X, U=U, Fw=F, refs={"x_ref": x_ref_tr, "theta_ref": th_ref_tr},
                  metrics=metrics, meta={"controller": controller})
