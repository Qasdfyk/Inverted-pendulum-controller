from __future__ import annotations
import numpy as np
from typing import Dict
from .base import Controller
from env.model import linearize_upright
from scipy.linalg import expm, solve_discrete_are

try:
    import cvxpy as cp
except Exception as e:
    cp = None
    _cvxpy_import_error = e
else:
    _cvxpy_import_error = None

def c2d(A, B, Ts):
    n = A.shape[0]
    M = np.block([[A, B], [np.zeros((B.shape[1], n + B.shape[1]))]])
    Md = expm(M * Ts)
    Ad = Md[:n, :n]
    Bd = Md[:n, n:n + B.shape[1]]
    return Ad, Bd

class LMPC(Controller):
    """
    Linear MPC with efficient OSQP setup (warm start, modest horizon).
    """
    def __init__(self, cfg: Dict, plant: Dict):
        if cp is None:
            raise ImportError(
                f"cvxpy is required for LMPC. {_cvxpy_import_error}\n"
                "Install with: pip install cvxpy osqp ecos"
            )

        super().__init__(cfg)
        c = cfg["controller"]["mpc"]

        self.Ts = float(c.get("Ts", 0.03))
        self.N = int(c.get("N", 1))
        self.Q = np.diag(c.get("Q", [20, 1, 300, 30]))
        self.R = np.array([[float(c.get("R", 0.5))]])
        self.dR = float(c.get("dR", 0.5))
        self.u_min = float(c.get("u_min", -10))
        self.u_max = float(c.get("u_max", 10))

        # linearization
        A, B = linearize_upright(plant["M"], plant["m"], plant["l"], plant["g"])
        self.Ad, self.Bd = c2d(A, B, self.Ts)

        # terminal weight
        if c.get("use_lqr_terminal", True):
            self.P = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        else:
            self.P = float(c.get("terminal_weight", 0.0)) * np.eye(4)

        # build QP once
        self._build_qp()

        self.u_prev = 0.0

    def _build_qp(self):
        n, N = 4, self.N
        Ad, Bd = self.Ad, self.Bd

        self.X = cp.Variable((n, N + 1))
        self.U = cp.Variable((1, N))
        self.x0_par = cp.Parameter(n)
        self.u_prev_par = cp.Parameter(1)

        cost = 0
        cons = [self.X[:, 0] == self.x0_par]
        for k in range(N):
            cost += cp.quad_form(self.X[:, k], self.Q) + cp.quad_form(self.U[:, k], self.R)
            if self.dR > 0:
                du = self.U[:, k] - (self.u_prev_par if k == 0 else self.U[:, k - 1])
                cost += self.dR * cp.sum_squares(du)
            cons += [self.X[:, k + 1] == Ad @ self.X[:, k] + Bd @ self.U[:, k]]
            cons += [self.U[:, k] <= self.u_max, self.U[:, k] >= self.u_min]
        cost += cp.quad_form(self.X[:, N], self.P)

        self.prob = cp.Problem(cp.Minimize(cost), cons)
        self._solver_opts = dict(
            solver=cp.OSQP,
            warm_start=True,
            verbose=False,
            eps_abs=1e-2,  # looser tolerances = faster
            eps_rel=1e-2,
            max_iter=100,
        )

    def reset(self):
        self.u_prev = 0.0

    def step(self, t, state, ref):
        th, thd, x, xd = state
        x_ref = float(ref["x_ref"]); theta_ref = float(ref["theta_ref"])
        x_dev = np.array([th - theta_ref, thd, x - x_ref, xd], dtype=float)

        self.x0_par.value = x_dev
        self.u_prev_par.value = [self.u_prev]

        try:
            self.prob.solve(**self._solver_opts)
        except Exception:
            return float(self.u_prev)

        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return float(self.u_prev)

        u = float(self.U[:, 0].value)
        self.u_prev = u
        return u

def build(cfg: dict):
    return lambda plant: LMPC(cfg, plant)
