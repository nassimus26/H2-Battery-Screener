"""Price of the cell hardware per m2 and of the power electronics per kW,
estimated from published cost studies, so the user does not have to guess.

The stack price depends on what kind of cell it is and on how many are built:

  gas cell      one electrode breathes hydrogen gas: a PEM fuel-cell-type
                stack (membrane, gas diffusion layers, bipolar plates, seals,
                assembly). Platinum is priced separately in the design.
                Source: US DOE Program Record 26002 (2026), heavy-duty PEM
                fuel-cell system, 2024 status (Strategic Analysis, Inc.):
                275 kW net, 52.5 m2 of cells, stack ~60% of system cost,
                catalyst more than half of the stack at high volume;
                155 USD/kW at 50,000 systems/year, 282 USD/kW below 1,000.
                  mass production: 155 x 275 x 0.6 = 25,600 USD of stack, less
                    the catalyst (> half) -> < 12,800 USD / 52.5 m2 = ~240 USD/m2
                  small series: 282 x 275 x 0.6 = 46,500 USD, less the same
                    catalyst -> ~640 USD/m2
  liquid cell   both sides are liquids pumped past carbon felts (flow cell):
                  mass production: material prices at volume, x1.5 for frames,
                    seals and assembly (that factor is our assumption):
                    membrane: 15 um PFSA, ionomer 658 USD/kg + ePTFE
                      9.68 USD/m2 (DOE 26002) = ~29 USD/m2
                    two carbon electrodes: 8.48 USD/m2 each (DOE 26002 gas
                      diffusion layer price, a carbon-paper proxy for felt)
                    bipolar plate: carbon-plastic 25-75 EUR/m2 (Noack et al.)
                    -> (29 + 17 + 55) x 1.5 = ~150 USD/m2
                  small series: the built prototype stack of Noack et al.,
                    Energies 9 (2016) 627 (Fraunhofer ICT, 10 kW vanadium flow
                    battery): 52,648 EUR for 306 cells x 580 cm2 = 17.7 m2,
                    i.e. ~2,970 EUR/m2 = ~3,260 USD/m2 (felts 150 EUR/m2,
                    graphite plates 418 EUR/m2, membrane 250 EUR/m2, gaskets
                    32% of the stack)

Each value carries a range (low-high) from the spread of the study numbers.

Power electronics: inverter ~97.5 USD/kW (NREL, 2023 commercial PV-plus-
storage cost benchmark, central inverter). A car would need a DC/DC converter
between the stack and the motor inverter, of similar order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    usd: float          # central value
    lo: float
    hi: float
    basis: str          # one sentence: where it comes from

    def range_text(self, unit: str) -> str:
        return f"{self.usd:,.0f} {unit} (range {self.lo:,.0f}-{self.hi:,.0f})"


SCALES = {"mass": "mass production (like Li-ion packs)", "small": "small series or first prototypes"}

_STACK = {
    ("gas", "mass"): Estimate(240.0, 170.0, 340.0,
                              "US DOE 2024 fuel-cell truck stack at 50,000 systems a year, without its platinum"),
    ("gas", "small"): Estimate(640.0, 450.0, 900.0,
                               "US DOE 2024 fuel-cell truck stack below 1,000 systems a year, without its platinum"),
    ("liquid", "mass"): Estimate(150.0, 90.0, 250.0,
                                 "membrane, two carbon electrodes and a carbon-plastic plate at volume prices "
                                 "(US DOE 2024, Noack et al. 2016), plus 50% for frames, seals and assembly"),
    ("liquid", "small"): Estimate(3260.0, 1600.0, 4500.0,
                                  "a built 10 kW flow-battery stack (Noack et al., Fraunhofer ICT, 2016)"),
}

POWER_ELECTRONICS = Estimate(97.5, 70.0, 150.0, "inverter price, NREL 2023 storage cost benchmark")


def stack(kind: str, scale: str = "mass") -> Estimate:
    """USD per m2 of cell for a 'gas' or 'liquid' cell."""
    return _STACK[(kind, scale)]


def geometric_mid(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)
