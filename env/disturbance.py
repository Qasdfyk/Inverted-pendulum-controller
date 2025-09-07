
from __future__ import annotations
import numpy as np

class Wind:
    def __init__(self, t_end: float, seed=23341, Ts=0.01, power=1e-3, smooth=5):
        rng = np.random.default_rng(seed)
        self.tgrid = np.arange(0.0, t_end + Ts, Ts)
        sigma = np.sqrt(power / Ts)
        w = rng.normal(0.0, sigma, size=self.tgrid.shape)
        if smooth and smooth > 1:
            kernel = np.ones(smooth) / smooth
            self.Fw = np.convolve(w, kernel, mode='same')
        else:
            self.Fw = w

    def __call__(self, t: float) -> float:
        return float(np.interp(t, self.tgrid, self.Fw))
