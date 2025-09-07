
from __future__ import annotations
import numpy as np
from .base import Controller

def tri_mf(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b: return (x - a) / (b - a)
    return (c - x) / (c - b)

class TSFuzzy(Controller):
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        c = cfg["controller"]
        self.cart = c["cart_loop"]
        ts = c["ts"]
        self.theta_breaks = ts["theta_breaks"]
        self.theta_dot_breaks = ts["theta_dot_breaks"]
        self.rules = ts["rules"]
        self.K_rules = ts["K_rules"]

    def reset(self): pass

    def _memberships(self, th, thd):
        tb = self.theta_breaks
        tdb = self.theta_dot_breaks
        m1 = tri_mf(th, tb[0], tb[1], tb[2]) * tri_mf(thd, tdb[0], tdb[1], tdb[2])
        m2 = 1.0 - m1
        return np.array([m1, m2])

    def step(self, t, state, ref):
        th, thd, x, xd = state
        e_x = ref["x_ref"] - x
        u_cart = self.cart["Kp"]*e_x + self.cart["Kd"]*(-xd)
        mu = self._memberships(th, thd)
        u_rules = []
        for i in range(self.rules):
            kr = self.K_rules[i]
            u_rules.append(kr["kth"]*th + kr["kthd"]*thd + kr.get("kx",0.0)*x + kr.get("kxd",0.0)*xd)
        u_pend = - float(np.dot(mu, u_rules))
        return u_cart + u_pend

def build(cfg: dict):
    return lambda plant: TSFuzzy(cfg, plant)
