from __future__ import annotations
import numpy as np
from .base import Controller

def _sinh(x): return np.sinh(x)
def _cosh(x): return np.cosh(x)

class MPCNO(Controller):
    """
    Explicit SIMO MPC (wg artykułu): u = N1*theta + N2*theta_dot + N3*(x - x_ref) + N4*x_dot
    Zgodny interfejs: step(t, state, ref) oraz build(cfg).
    Kolejność stanu jak u Ciebie: state = [theta, theta_dot, x, x_dot].
    """
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        self.plant = plant
        c = cfg["controller"]["mpc_no"]

        # limity zgodne z Twoim stylem kluczy
        umax = float(c.get("umax", c.get("u_max", 10.0)))
        self.u_max = +abs(umax)
        self.u_min = -abs(umax)

        # referencje
        self.x_ref_cfg = float(c.get("x_ref", 0.0))

        # parametry artykułu
        self.h = float(c.get("h", 0.06))
        self.rho1 = float(c.get("rho1", 0.08))
        self.r = float(c.get("r", 2.0))
        self.rho2 = float(c.get("rho2", 0.6))
        self.gm_min = float(c.get("gm_min", 1.6))

        # parametry rośliny (liniaryzacja wokół pionu)
        M = plant["M"]; m = plant["m"]; L = plant["l"]; g = plant["g"]

        # prosta liniaryzacja SIMO:
        #   theta_dd = a1*theta + b1*u
        #   x_dd     = b2*u
        a1 = g / L
        # b1 i b2 przyjęte praktycznie; jeśli masz inne w modelu liniowym, podmień tutaj
        b1 = 1.0 / ((M + m) * L)
        b2 = 1.0 / (M + m)

        h = self.h
        ah = np.sqrt(a1) * h

        # bloki predykcji dla podukładu kąta
        A1 = _cosh(ah)
        A2 = np.sqrt(a1) * _sinh(ah)
        B1 = (1.0/np.sqrt(a1)) * _sinh(ah)
        B2 = _cosh(ah)

        # współczynniki E1..E4 z integracji impulsu sterowania na h
        E1 = 2.0*(b1/a1)*(_sinh(ah/2.0)**2)
        E2 = (b1/np.sqrt(a1)) * _sinh(ah)
        E3 = 0.5*b2*h*h
        E4 = b2*h

        # znak w gałęzi wózka (k = -1 w artykule)
        k = -1.0

        rho1 = self.rho1; rho2 = self.rho2; r = self.r
        D = (E1**2 + rho1*E2**2 + r*(k**2)*(E3**2 + rho2*E4**2) + 1e-12)

        N1 = (A1*E1 + rho1*A2*E2) / D
        N2 = (B1*E1 + rho1*B2*E2) / D
        N3 = -(r*k*E3) / D
        N4 = -(r*k*(h*E3 + rho2*E4)) / D

        # szybka korekta marginesu GM przez skalowanie par (N1,N2) jeśli potrzeba
        def _gm(n1,n2,n3,n4):
            alpha = (n1/n2) - (n3/n4)
            return ((b1*n2)/(b2*n4)) / (1.0 + a1/(b2*n4*alpha))

        gm = _gm(N1,N2,N3,N4)
        if gm < self.gm_min:
            s = max(self.gm_min / max(gm, 1e-6), 1.0)
            N1 *= s; N2 *= s
            gm = _gm(N1,N2,N3,N4)
            if gm < self.gm_min:
                t = max(gm / self.gm_min, 1e-2)
                N3 *= t; N4 *= t

        self.N1, self.N2, self.N3, self.N4 = float(N1), float(N2), float(N3), float(N4)

    def reset(self):
        pass

    def step(self, t, state, ref):
        # state: [theta, theta_dot, x, x_dot]
        th = float(state[0])
        thd = float(state[1])
        x = float(state[2])
        xd = float(state[3])

        # referencje (domyślnie theta_ref=0)
        theta_ref = float(ref.get("theta_ref", 0.0)) if isinstance(ref, dict) else 0.0
        x_ref = float(ref.get("x_ref", self.x_ref_cfg)) if isinstance(ref, dict) else self.x_ref_cfg

        u = (
            self.N1*(th - theta_ref)
            + self.N2*thd
            + self.N3*(x - x_ref)
            + self.N4*xd
        )
        return float(np.clip(u, self.u_min, self.u_max))

def build(cfg: dict):
    return lambda plant: MPCNO(cfg, plant)
