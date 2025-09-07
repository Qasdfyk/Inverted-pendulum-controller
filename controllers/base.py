
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np

class Controller(ABC):
    def __init__(self, cfg: Dict):
        self.cfg = cfg

    @abstractmethod
    def reset(self):
        ...

    @abstractmethod
    def step(self, t: float, state: np.ndarray, ref: Dict[str, float]) -> float:
        ...
