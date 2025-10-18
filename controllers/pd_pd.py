from __future__ import annotations
import numpy as np
from .base import Controller

def sat(v, vmin, vmax):
    return max(vmin, min(v, vmax))

class PDPD(Controller):
    """
    Minimal cascade PD-PD (art2 linearization for inner-loop inversion):
        thdd ≈ a*th + b*u, with
            a = (M + m)*g/(M*l)
            b = -1/(M*l)
    Outer loop: cart PD -> theta_ref (optional clamp).
    Inner loop: angle PD -> thdd_des; then u = (thdd_des - a*th)/b.
    """
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        c = cfg["controller"]

        # art2 upright linearization constants
        M = plant["M"]; m = plant["m"]; l = plant["l"]; g = plant["g"]
        self.a = (M + m) * g / (M * l)
        self.b = -1.0 / (M * l)  # < 0

        # Outer cart PD and optional theta_ref clamp
        self.cart = c["cart_pid"]                 # {Kp, Ki (opt), Kd}
        self.theta_ref_limit = c.get("theta_ref_limit", None)  # None = no clamp

        # Inner angle PD
        self.pend = c["pend_pid"]                 # {Kp, Ki (opt), Kd}

        # Integrators (only if Ki > 0)
        self.z_cart = 0.0
        self.z_pend = 0.0

    def reset(self):
        self.z_cart = 0.0
        self.z_pend = 0.0

    def step(self, t, state, ref):
        th, thd, x, xd = state
        dt = max(1e-4, ref.get("dt", 0.01))

        # ----- Outer: cart PD -> theta_ref
        e_x  = ref["x_ref"] - x
        de_x = -xd
        if self.cart.get("Ki", 0.0) != 0.0:
            self.z_cart += e_x * dt

        theta_ref = ( self.cart["Kp"] * e_x
                    + self.cart.get("Ki", 0.0) * self.z_cart
                    + self.cart["Kd"] * de_x )

        if self.theta_ref_limit is not None:
            theta_ref = sat(theta_ref, -self.theta_ref_limit, self.theta_ref_limit)

        # ----- Inner: angle PD on (theta - theta_ref) -> thdd_des
        e_th  = th - theta_ref
        de_th = thd
        if self.pend.get("Ki", 0.0) != 0.0:
            self.z_pend += e_th * dt

        thdd_des = - self.pend["Kp"] * e_th \
                   - self.pend.get("Ki", 0.0) * self.z_pend \
                   - self.pend["Kd"] * de_th

        # Inversion: thdd ≈ a*th + b*u  =>  u = (thdd_des - a*th)/b
        u = (thdd_des - self.a * th) / self.b
        return float(u)

def build(cfg: dict):
    return lambda plant: PDPD(cfg, plant)
