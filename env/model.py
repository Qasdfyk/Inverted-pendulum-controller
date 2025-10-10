from __future__ import annotations
import numpy as np
from typing import Tuple

def linearize_upright(M: float, m: float, l: float, g: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearization around theta=0 for the SAME model used in f_nonlinear (art2 model).
    States: x=[th, thd, x, xd]
    """
    A = np.array([
        [0.0,                 1.0, 0.0, 0.0],
        [(M + m)*g/(M*l),     0.0, 0.0, 0.0],
        [0.0,                 0.0, 0.0, 1.0],
        [-m*g/M,              0.0, 0.0, 0.0]
    ], dtype=float)

    B = np.array([
        [0.0],
        [-1.0/(M*l)],
        [0.0],
        [1.0/M]
    ], dtype=float)

    return A, B

def f_nonlinear(x: np.ndarray, u: float, pars: dict, Fw: float=0.0) -> np.ndarray:
    th, thd, pos, posd = x
    M = pars["M"]; m = pars["m"]; l = pars["l"]; g = pars["g"]
    s, c = np.sin(th), np.cos(th)

    # art2 (massless rod, point mass) denominators
    denom_x  = (M + m) - m * c * c
    denom_th = (m * l * c * c) - (M + m) * l

    # art2 nonlinear equations
    thdd = (u * c
            - (M + m) * g * s
            + m * l * (c * s) * (thd ** 2)
            - (M / m) * Fw * c) / denom_th

    xdd  = (u
            + m * l * s * (thd ** 2)
            - m * g * c * s
            + Fw * (s * s)) / denom_x

    return np.array([thd, thdd, posd, xdd])