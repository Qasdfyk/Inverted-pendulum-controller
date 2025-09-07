from __future__ import annotations
import numpy as np
from .base import Controller

def sat(v, vmin, vmax):
    return max(vmin, min(v, vmax))

class PDPD(Controller):
    """
    Cascade PD–PD with inner-loop inversion of linearized angle dynamics:
        thdd ≈ a*th + b*u,  a=g/den,  b=-1/((M+m)*den)
        den = l*(4/3 - m/(M+m))

    Outer loop (cart PD) -> theta_ref (limited, rate-limited, filtered).
    Inner loop (angle PD) -> desired thdd, then u = (thdd_des - a*th)/b.
    """
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        c = cfg["controller"]

        # Plant constants for inner-loop inversion (match env.model)
        M = plant["M"]; m = plant["m"]; l = plant["l"]; g = plant["g"]
        den = l * (4.0/3.0 - m/(M + m))
        self.a = g / den
        self.b = -1.0 / ((M + m) * den)   # < 0

        # Outer loop (cart PD) -> theta_ref
        self.cart = c["cart_pid"]                         # {Kp, Ki (opt), Kd}
        self.theta_ref_limit = c.get("theta_ref_limit", 0.15)   # [rad]
        self.theta_ref_rate  = c.get("theta_ref_rate", 0.6)     # [rad/s] slew limit
        self.theta_ref_tau   = c.get("theta_ref_filter_tau", 0.12)  # [s] LPF

        # Inner loop (angle PD) gains used in thdd_des
        self.pend = c["pend_pid"]                         # {Kp, Ki (opt), Kd}

        # States
        self.z_cart = 0.0
        self.z_pend = 0.0
        self.theta_ref_filt = 0.0   # filtered theta_ref

    def reset(self):
        self.z_cart = 0.0
        self.z_pend = 0.0
        self.theta_ref_filt = 0.0

    def step(self, t, state, ref):
        th, thd, x, xd = state
        dt = max(1e-4, ref.get("dt", 0.01))

        # ---------- Outer: cart position -> raw theta_ref
        e_x  = ref["x_ref"] - x
        de_x = -xd
        self.z_cart += e_x * dt
        theta_ref_raw = ( self.cart["Kp"] * e_x
                        + self.cart.get("Ki", 0.0) * self.z_cart
                        + self.cart["Kd"] * de_x )
        theta_ref_raw = sat(theta_ref_raw, -self.theta_ref_limit, self.theta_ref_limit)

        # Rate-limit theta_ref
        max_step = self.theta_ref_rate * dt
        dcmd = theta_ref_raw - self.theta_ref_filt
        if dcmd >  max_step: theta_cmd = self.theta_ref_filt + max_step
        elif dcmd < -max_step: theta_cmd = self.theta_ref_filt - max_step
        else: theta_cmd = theta_ref_raw

        # Low-pass filter theta_ref (first-order)
        if self.theta_ref_tau > 1e-6:
            alpha = dt / (self.theta_ref_tau + dt)
            self.theta_ref_filt += alpha * (theta_cmd - self.theta_ref_filt)
        else:
            self.theta_ref_filt = theta_cmd

        theta_ref = self.theta_ref_filt

        # ---------- Inner: PD on (theta - theta_ref) -> thdd_des
        e_th  = th - theta_ref
        de_th = thd
        self.z_pend += e_th * dt

        thdd_des = - self.pend["Kp"] * e_th \
                   - self.pend.get("Ki", 0.0) * self.z_pend \
                   - self.pend["Kd"] * de_th

        # Linearized inversion: thdd ≈ a*th + b*u -> u = (thdd_des - a*th)/b
        u = (thdd_des - self.a * th) / self.b
        return float(u)

def build(cfg: dict):
    return lambda plant: PDPD(cfg, plant)
