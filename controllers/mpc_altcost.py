
from __future__ import annotations
from .base import Controller
import numpy as np

class MPCAltCost(Controller):
    def __init__(self, cfg, plant):
        super().__init__(cfg)
        self.fallback_Kp = 1.0
        self.fallback_Kd = 2.0

    def reset(self): pass

    def step(self, t, state, ref):
        th, thd, x, xd = state
        e = ref["x_ref"] - x
        return self.fallback_Kp*e + self.fallback_Kd*(-xd)

def build(cfg: dict):
    return lambda plant: MPCAltCost(cfg, plant)
