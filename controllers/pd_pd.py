# controllers/pd_pd.py
from __future__ import annotations
import numpy as np
from .base import Controller
from env.model import f_nonlinear  # used by env; controller is model-free

def sat(v: float, vmin: float, vmax: float) -> float:
    return float(np.clip(v, vmin, vmax))

class PDPD(Controller):
    """
    Two-loop PD-PD (sum of loops):
      u = u_angle + u_cart
      e_th = theta_ref - theta,  d(e_th)/dt = -theta_dot
      e_x  = x_ref     - x,      d(e_x)/dt  = -x_dot

    Anti-windup: integrators active only if Ki!=0 and clamped to ±integ_limit.   [AW]
    Saturation:  final control clipped to ±u_limit.                               [SAT]
    """

    def __init__(self, cfg, plant):
        super().__init__(cfg)

        # ---- CONFIG: controller.pid_pid.* ----
        c = cfg["controller"]["pid_pid"]
        self.ang  = dict(c["angle_pid"])   # {Kp, Ki, Kd}
        self.cart = dict(c["cart_pid"])    # {Kp, Ki, Kd}

        # References
        self.theta_ref = float(c.get("theta_ref", 0.0))
        self.x_ref_default = float(c.get("x_ref", 0.1))

        # Limits
        self.u_limit = float(c.get("u_limit", 80.0))     # [SAT]
        self.integ_limit = float(c.get("integ_limit", 5.0))  # [AW]

        # Integrator states (for optional Ki)
        self.z_ang = 0.0
        self.z_cart = 0.0

    def reset(self):
        self.z_ang = 0.0
        self.z_cart = 0.0

    def _pid_step(self, e: float, de_dt: float, z: float, gains: dict, dt: float) -> tuple[float, float]:
        # ---- ANTI-WINDUP [AW] ----
        Ki = float(gains.get("Ki", 0.0))
        if Ki != 0.0:
            z = z + e * dt
            z = sat(z, -self.integ_limit, self.integ_limit)
        u = float(gains["Kp"]) * e + Ki * z + float(gains["Kd"]) * de_dt
        return u, z

    def step(self, t, state, ref):
        th, thd, x, xd = state
        dt = float(max(1e-4, ref.get("dt", 0.01)))
        th_ref = float(ref.get("theta_ref", self.theta_ref))
        x_ref  = float(ref.get("x_ref", self.x_ref_default))

        # Angle PD/PID
        e_th  = th_ref - th
        de_th = -thd
        u_ang,  self.z_ang  = self._pid_step(e_th,  de_th,  self.z_ang,  self.ang,  dt)

        # Cart PD/PID
        e_x   = x_ref - x
        de_x  = -xd
        u_cart, self.z_cart = self._pid_step(e_x,   de_x,   self.z_cart, self.cart, dt)

        u = u_ang + u_cart

        # ---- SATURATION [SAT] ----
        u = sat(u, -self.u_limit, self.u_limit)
        return float(u)

def build(cfg: dict):
    return lambda plant: PDPD(cfg, plant)
