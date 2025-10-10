from __future__ import annotations
import numpy as np
from .base import Controller
from env.model import f_nonlinear

class MPCNO(Controller):
    """
    Ultra-simple MPC-NO (fast).
    - Horizon H small (3-4)
    - Candidates = [-umax, 0, +umax]
    - Cost = theta^2 + 0.1*x^2 + 0.01*u^2
    """
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        self.plant = plant
        c = cfg["controller"]["mpc_no"]

        self.dt = float(c.get("dt", 0.02))
        self.H = int(c.get("H", 3))
        self.u_max = float(c.get("u_max", 10.0))
        self.u_min = float(c.get("u_min", -10.0))

        self.Q_th = float(c.get("Q_theta", 10.0))
        self.Q_x = float(c.get("Q_x", 1.0))
        self.R = float(c.get("R", 0.01))

        self.u_prev = 0.0

    def reset(self):
        self.u_prev = 0.0

    def _rollout_cost(self, state, u, ref):
        x = np.array(state, dtype=float)
        th_ref, x_ref = ref["theta_ref"], ref["x_ref"]
        J = 0.0
        for _ in range(self.H):
            dx = f_nonlinear(x, u, self.plant, 0.0)
            x = x + self.dt * dx
            # deviation cost
            th_err = x[0] - th_ref
            x_err = x[2] - x_ref
            J += self.Q_th * th_err**2 + self.Q_x * x_err**2 + self.R * u**2
        return J

    def step(self, t, state, ref):
        # candidates: just {-u_max, 0, +u_max}
        candidates = [self.u_min, 0.0, self.u_max]

        bestJ, bestu = np.inf, 0.0
        for u in candidates:
            J = self._rollout_cost(state, u, ref)
            if J < bestJ:
                bestJ, bestu = J, u

        self.u_prev = bestu
        return bestu

def build(cfg: dict):
    return lambda plant: MPCNO(cfg, plant)
