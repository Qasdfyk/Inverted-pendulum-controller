
from __future__ import annotations
import numpy as np
from scipy.linalg import solve_continuous_are
from .base import Controller
from env.model import linearize_upright

class PIDLQR(Controller):
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        self.Kp = cfg["controller"]["pid"]["Kp"]
        self.Ki = cfg["controller"]["pid"]["Ki"]
        self.Kd = cfg["controller"]["pid"]["Kd"]
        Q = np.diag(cfg["controller"]["lqr"]["Q"])
        R = np.array([[cfg["controller"]["lqr"]["R"]]], dtype=float)
        A, B = linearize_upright(plant["M"], plant["m"], plant["l"], plant["g"])
        P = solve_continuous_are(A, B, Q, R)
        self.Klqr = (np.linalg.solve(R, B.T @ P)).ravel()
        self.z = 0.0

    def reset(self):
        self.z = 0.0

    def step(self, t, state, ref):
        th, thd, x, xd = state
        e_x = ref["x_ref"] - x
        de_x = -xd
        self.z += e_x * ref["dt"]
        u_pid = self.Kp*e_x + self.Ki*self.z + self.Kd*de_x
        x_ref = ref["x_ref"]
        state_dev = np.array([th - 0.0, thd - 0.0, x - x_ref, xd - 0.0])
        u_lqr = - float(self.Klqr @ state_dev)

        return u_pid + u_lqr

def build(cfg: dict):
    return lambda plant: PIDLQR(cfg, plant)
