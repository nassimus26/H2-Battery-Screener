"""Numbers with provenance and an uncertainty range.

kind is one of
  database   - taken from a database (Materials Project, potential table, CoolProp)
  physics    - computed from a physical law with database inputs
  measured   - a published measurement (only used to validate the model)
  assumption - an engineering or design assumption; editable in targets.toml
  unknown    - no data or law fixes it; a wide range is swept instead
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Value:
    nominal: float
    low: float
    high: float
    kind: str
    source: str
    log: bool = False   # sample log-uniformly (for quantities spanning decades)

    @staticmethod
    def fixed(v: float, kind: str, source: str) -> "Value":
        return Value(v, v, v, kind, source)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.low == self.high:
            return np.full(n, self.nominal)
        if self.log:
            return np.exp(rng.uniform(math.log(self.low), math.log(self.high), n))
        return rng.uniform(self.low, self.high, n)
