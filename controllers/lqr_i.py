from __future__ import annotations
import numpy as np

from lqr import discretize_upright, dlqr_gain


class LQRIController:
    """DLQR augmented with an integrator on the cart-position error.

    Augmented state x_aug = [theta, theta_dot, x, x_dot, z]^T with
    z(k+1) = z(k) + (x_ref - x_3(k)) * Tp; control law u(k) = -K_aug x_aug(k).
    """
    def __init__(self, pars: dict, dt: float,
                 lqr_gains: dict,
                 u_limit: float = 80.0):
        self.pars = pars
        self.dt = dt
        self.lqr = lqr_gains
        self.u_limit = u_limit

        Ad, Bd = discretize_upright(pars, dt)
        n = Ad.shape[0]
        # Integrator row: z(k+1) = z(k) - Tp * (x_3(k) - x_ref)
        A_aug = np.zeros((n + 1, n + 1))
        A_aug[:n, :n] = Ad
        A_aug[n, 2] = -dt
        A_aug[n, n] = 1.0
        B_aug = np.vstack([Bd, [[0.0]]])

        Q = np.diag(lqr_gains.get("Q", [1, 1, 1, 1, 1]))
        R = np.array([[lqr_gains.get("R", 1.0)]], dtype=float)
        self.Kaug = dlqr_gain(A_aug, B_aug, Q, R)
        self.reset()

    def reset(self):
        """Clears the integrator state before a new simulation run."""
        self.z = 0.0

    def compute_control(self, x0, x_ref, u_prev):
        th, thd, pos, posd = x0
        target_pos = x_ref[2]

        x_aug = np.array([th, thd, pos - target_pos, posd, self.z])
        u = -float(self.Kaug @ x_aug)
        u = float(np.clip(u, -self.u_limit, self.u_limit))

        # Integrator update after computing u(k)
        self.z += (target_pos - pos) * self.dt
        return np.array([u])
