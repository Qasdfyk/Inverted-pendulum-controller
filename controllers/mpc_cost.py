"""NMPC with an alternative (M-estimator) penalty in the control-increment term.

J(k) = sum_p e_p^T Q e_p + R * sum_r rho(du(k+r|k)),

with rho taken from Nebeluk, "Exploring the Benefits of Alternative Cost
Functions for Nonlinear Model Predictive Control", ICCMA 2024, eqs. (3), (5)-(19).
For rho(du) = du^2 the controller is identical to the NMPC-L2 (MPCController).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

from mpc import MPCController

HALF_PI = 0.5 * np.pi


def _piecewise(a, inner, outer, c):
    return np.where(a <= c, inner, outer)


# Each function takes du (array) and the shape parameter c (default 1, as in the paper)
RHO = {
    "L2": lambda du, c: du**2,                                               # (3)
    "L1": lambda du, c: np.abs(du),                                          # (5)
    "Ch": lambda du, c: 2.0 * (np.sqrt(1.0 + du**2 / 2.0) - 1.0),            # (6)
    "F":  lambda du, c: c**2 * (np.abs(du) / c - np.log1p(np.abs(du) / c)),  # (7)
    "Hu": lambda du, c: _piecewise(np.abs(du), du**2 / 2.0,                  # (8)
                                   c * np.abs(du) - c**2 / 2.0, c),
    "MH": lambda du, c: _piecewise(np.abs(du),                               # (9)
                                   c**2 * (1.0 - np.cos(np.clip(du / c, -HALF_PI, HALF_PI))),
                                   c * np.abs(du) + c**2 * (1.0 - HALF_PI), HALF_PI * c),
    "K":  lambda du, c: du**2 / (c + np.abs(du)),                            # (10)
    "L":  lambda du, c: c**2 * np.log(np.cosh(du / c)),                      # (11)
    "M":  lambda du, c: _piecewise(np.abs(du), du**2 / 2.0,                  # (12)
                                   c * np.abs(du) / 2.0, c),
    "Z":  lambda du, c: du**2 / (2.0 + c * np.abs(du)),                      # (13)
    "BH": lambda du, c: _piecewise(np.abs(du), du**2 / 2.0,                  # (14)
                                   c**2 * (np.log(np.maximum(np.abs(du), c) / c) + 0.5), c),
    "Be": lambda du, c: 5.0 * c**2 / 4.0 * (1.0 - (1.0 + du**2 / (5.0 * c**2))**-2),  # (15)
    "C":  lambda du, c: c**2 / 2.0 * np.log1p(du**2 / c**2),                 # (16)
    "MS": lambda du, c: _piecewise(np.abs(du), du**2 / 2.0,                  # (17)
                                   2.0 * c**1.5 * np.sqrt(np.abs(du)) - 1.5 * c**2, c),
    "R":  lambda du, c: (1.0 - (1.0 + c * np.abs(du)) * np.exp(-c * np.abs(du))) / c**2,  # (18)
    "Y":  lambda du, c: _piecewise(np.abs(du), du**2 / 2.0,                  # (19)
                                   np.log(du**2 + 1.0) + (c**2 / 2.0 - np.log(c**2 + 1.0)), c),
}

FULL_NAMES = {
    "L2": "L$_2$ norm", "L1": "L$_1$ norm", "Ch": "Charbonnier", "F": "Fair",
    "Hu": "Huber", "MH": "Modified Huber", "K": "Kong", "L": "Logistic",
    "M": "M\\\"uller", "Z": "Zhang", "BH": "Bab-Hadiashar", "Be": "Bell",
    "C": "Cauchy", "MS": "Merril-Schweppe", "R": "Ramsay", "Y": "Youssef",
}


class MPCCostController(MPCController):
    """NMPC-rho: same prediction model, horizons, Q and R as NMPC-L2; only rho changes."""
    def __init__(self, pars: dict, dt: float, N: int, Nu: int,
                 umin: float, umax: float, Q: np.ndarray, R: float,
                 rho: str = "L2", c: float = 1.0):
        super().__init__(pars, dt, N, Nu, umin, umax, Q, R)
        self.rho_name = rho
        self.rho = RHO[rho]
        self.c = c

    def _cost(self, du, x0, x_ref, u_prev):
        du = np.asarray(du, dtype=float)
        cost_y, _ = self._cost_state(du, x0, x_ref, u_prev)
        return cost_y + float(self.R * np.sum(self.rho(du, self.c)))

    def _solve(self, x0, x_ref, u_prev):
        if self.rho_name != "L1":
            return super()._solve(x0, x_ref, u_prev)
        return self._solve_l1(x0, x_ref, u_prev)

    def _solve_l1(self, x0, x_ref, u_prev):
        """Exact L1 via epigraph: min J_y + R*sum(s), s.t. -s <= du <= s (smooth for SLSQP)."""
        Nu = self.Nu
        z0 = np.zeros(2 * Nu)
        span = self.umax - self.umin

        def cost(z):
            cost_y, _ = self._cost_state(z[:Nu], x0, x_ref, u_prev)
            return cost_y + float(self.R * np.sum(z[Nu:]))

        I = np.eye(Nu)
        L = np.hstack([self._L, np.zeros((Nu, Nu))])
        A_abs = np.vstack([np.hstack([-I, I]), np.hstack([I, I])])  # s - du >= 0, s + du >= 0
        cons = [
            {'type': 'ineq', 'fun': lambda z: A_abs @ z, 'jac': lambda z: A_abs},
            {'type': 'ineq', 'fun': lambda z: self.umax - u_prev - L @ z, 'jac': lambda z: -L},
            {'type': 'ineq', 'fun': lambda z: u_prev + L @ z - self.umin, 'jac': lambda z: L},
        ]
        bounds = [(-span, span)] * Nu + [(0.0, span)] * Nu
        res = minimize(cost, z0, method='SLSQP', bounds=bounds,
                       constraints=cons, options=self.solver_options)
        self.n_calls += 1
        if not res.success:
            self.n_fail += 1
            return np.zeros(Nu)
        return res.x[:Nu]
