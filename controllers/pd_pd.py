from __future__ import annotations
import numpy as np
from .base import Controller
from env.model import linearize_upright

def sat(v, vmin, vmax):
    return float(np.clip(v, vmin, vmax))

class PDPD(Controller):
    """
    Cascade PD–PD using env linearization.
    Inner loop inversion: thdd ≈ a*th + b*u.
    Outer loop: cart PD -> theta_ref.
    Inner loop: pendulum PD -> thdd_des.
    """

    def __init__(self, cfg, plant):
        super().__init__(cfg)
        c = cfg["controller"]

        # use env linearization to get a,b
        A, B = linearize_upright(plant["M"], plant["m"], plant["l"], plant["g"])
        self.a = float(A[1, 0])
        self.b = float(B[1, 0])

        self.cart = c["cart_pid"]
        self.theta_ref_limit = c.get("theta_ref_limit", None)
        self.pend = c["pend_pid"]

        self.z_cart = 0.0
        self.z_pend = 0.0

    def reset(self):
        self.z_cart = 0.0
        self.z_pend = 0.0

    def step(self, t, state, ref):
        th, thd, x, xd = state
        dt = float(max(1e-4, ref.get("dt", 0.01)))

        # Outer: cart PD
        e_x = ref["x_ref"] - x
        de_x = -xd
        if self.cart.get("Ki", 0.0) != 0.0:
            self.z_cart += e_x * dt

        theta_ref = (
            self.cart["Kp"] * e_x
            + self.cart.get("Ki", 0.0) * self.z_cart
            + self.cart["Kd"] * de_x
        )

        if self.theta_ref_limit is not None:
            theta_ref = sat(theta_ref, -self.theta_ref_limit, self.theta_ref_limit)

        # Inner: pendulum PD
        e_th = th - theta_ref
        de_th = thd
        if self.pend.get("Ki", 0.0) != 0.0:
            self.z_pend += e_th * dt

        thdd_des = -(
            self.pend["Kp"] * e_th
            + self.pend.get("Ki", 0.0) * self.z_pend
            + self.pend["Kd"] * de_th
        )

        # Inversion
        u = (thdd_des - self.a * th) / self.b
        return float(u)

def build(cfg: dict):
    return lambda plant: PDPD(cfg, plant)
