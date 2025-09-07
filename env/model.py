
from __future__ import annotations
import numpy as np
from typing import Tuple

def linearize_upright(M: float, m: float, l: float, g: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearization around theta=0 for the SAME model used in f_nonlinear:

        thdd = ( g*sin(th) - cos(th)*(u + m*l*thd^2*sin(th))/(M+m) ) / ( l*(4/3 - m*cos^2(th)/(M+m)) )
        xdd  = ( u + m*l*(thd^2*sin(th) - thdd*cos(th)) ) / (M+m)

    At (th=0, thd=0), let:
        den = l * ( 4/3 - m/(M+m) )
        a   = g / den
        b   = - 1 / ( (M+m) * den )     # thdd = a*th + b*u

    Then:
        xdd = c_u * u + c_th * th
        c_th = - (m*l*a) / (M+m)
        c_u  =  1/(M+m) - (m*l*b)/(M+m) = 1/(M+m) + m*l / ( (M+m)^2 * den )

    States: x=[th, thd, x, xd]
    """
    den = l * (4.0/3.0 - m/(M + m))
    a = g / den
    b = -1.0 / ((M + m) * den)

    c_th = - (m * l * a) / (M + m)
    c_u  =  1.0/(M + m) + (m * l) / (((M + m)**2) * den)

    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [a,   0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [c_th,0.0, 0.0, 0.0]
    ], dtype=float)

    B = np.array([
        [0.0],
        [b],
        [0.0],
        [c_u]
    ], dtype=float)

    return A, B

def f_nonlinear(x: np.ndarray, u: float, pars: dict, Fw: float=0.0) -> np.ndarray:
    th, thd, pos, posd = x
    M = pars["M"]; m = pars["m"]; l = pars["l"]; g = pars["g"]
    s, c = np.sin(th), np.cos(th)
    den = l * (4.0/3.0 - (m * c * c) / (M + m))
    thdd = (g*s - c*(u + Fw + m*l*thd*thd*s)/(M + m)) / den
    xdd  = (u + Fw + m*l*(thd*thd*s - thdd*c)) / (M + m)
    return np.array([thd, thdd, posd, xdd])
