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

Purpose-built ("target"): a battery designed for its one chemistry and made at
volume, priced at the published cost targets instead of today's hardware:
  gas cell stack  US DOE ultimate heavy-duty fuel-cell system target 60 USD/kW
                  (vs 155 USD/kW in the 2024 status above, Program Record 26002;
                  NREL/TP-5400-91146, 2026): the same share of the 2024 stack,
                  240 x 60/155 = ~93 USD/m2 without platinum; the light-duty
                  ultimate target (30 vs ~57 USD/kW status) gives a similar
                  ratio (0.53) -> range 70-130 USD/m2
  power electronics  US DOE 2025 traction power-electronics target 2.7 USD/kW
                  (100 kW/L), a DC/DC converter matched to the cell instead of a
                  grid inverter; range 1.5-6 USD/kW. Note: lithium-ion pack prices
                  include no inverter either
  carbon-fibre tanks  US DOE ultimate onboard storage target 8 USD/kWh vs
                  ~16-17 USD/kWh status for 700 bar Type IV (Records 13010, 19008):
                  composite price x 0.5 (heavy-duty: 266 vs 378 USD/kg H2, x 0.7)
  liquid cell stack, tanks and pumps: no published target found; kept at the
                  mass-production values above
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


SCALES = {"mass": "today's parts, mass production (like Li-ion packs)",
          "target": "purpose-built, at published cost targets (US DOE)",
          "small": "small series or first prototypes"}

TANK_COST_FACTOR = {"mass": 1.0, "small": 1.0, "target": 0.5}   # carbon-fibre tanks: DOE 8 vs ~16 USD/kWh

_STACK = {
    ("gas", "mass"): Estimate(240.0, 170.0, 340.0,
                              "US DOE 2024 fuel-cell truck stack at 50,000 systems a year, without its platinum"),
    ("gas", "small"): Estimate(640.0, 450.0, 900.0,
                               "US DOE 2024 fuel-cell truck stack below 1,000 systems a year, without its platinum"),
    ("gas", "target"): Estimate(93.0, 70.0, 130.0,
                                "US DOE ultimate fuel-cell cost target (60 USD/kW heavy-duty vs 155 today) applied "
                                "to the 2024 stack, without its platinum"),
    ("liquid", "target"): Estimate(150.0, 90.0, 250.0,
                                   "no published target for flow-cell stacks: kept at the mass-production price"),
    ("liquid", "mass"): Estimate(150.0, 90.0, 250.0,
                                 "membrane, two carbon electrodes and a carbon-plastic plate at volume prices "
                                 "(US DOE 2024, Noack et al. 2016), plus 50% for frames, seals and assembly"),
    ("liquid", "small"): Estimate(3260.0, 1600.0, 4500.0,
                                  "a built 10 kW flow-battery stack (Noack et al., Fraunhofer ICT, 2016)"),
}

POWER_ELECTRONICS = Estimate(97.5, 70.0, 150.0, "inverter price, NREL 2023 storage cost benchmark")
POWER_ELECTRONICS_TARGET = Estimate(2.7, 1.5, 6.0, "US DOE 2025 traction power-electronics cost target "
                                                   "(a DC/DC converter matched to the cell)")


def power_electronics(scale: str = "mass") -> Estimate:
    """USD per kW of power electronics: today's inverter, or the purpose-built target."""
    return POWER_ELECTRONICS_TARGET if scale == "target" else POWER_ELECTRONICS


def stack(kind: str, scale: str = "mass") -> Estimate:
    """USD per m2 of cell for a 'gas' or 'liquid' cell."""
    return _STACK[(kind, scale)]


def geometric_mid(lo: float, hi: float) -> float:
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------ tanks, pumps, pipes

EUR_USD = 1.1           # 2016 exchange rate, for the Noack et al. prices
SIZE_EXPONENT = 0.6     # process equipment cost ~ size^0.6 (Darling et al., EES 2014)
REF_LITRES = 8257.0     # Noack et al. 2016: electrolyte volume of their 10 kW / 120 kWh system
REF_KW = 10.0
# Their itemised prices (EUR, Tables 1-2), for two liquid loops:
REF_TANKS_EUR = 9082.0                  # 1.1 EUR per litre of tank
REF_FLUID_EUR = 2 * 1000 + 50 * 20 + 14 * 30   # 2 pumps, 50 m of piping, 14 valves = 3,420
REF_HEAT_EUR = 2 * 1500                 # 2 heat exchangers
REF_SENSORS_EUR = 10 * 250 + 2 * 330    # 10 sensors, 2 actuators


def flow_hardware(litres: float, power_kw: float, loops: int = 2) -> dict[str, float]:
    """USD for the tanks and the liquid loops of a flow cell.

    Prices from the itemised 10 kW / 120 kWh flow battery of Noack et al.
    (Energies 9 (2016) 627), scaled to this system with the 0.6 power law:
    tanks by litres of liquid, pumps, pipes, valves, heat exchangers and
    sensors by power. 'loops' is the number of pumped liquids (2 for two
    molecules, 1 for a molecule against hydrogen gas). These are small-series
    prices; no mass-production discount is applied."""
    if litres <= 0:
        return {}
    share = loops / 2
    tanks = REF_TANKS_EUR * share * (litres / loops / (REF_LITRES / 2)) ** SIZE_EXPONENT
    p = (power_kw / REF_KW) ** SIZE_EXPONENT
    loop = (REF_FLUID_EUR + REF_HEAT_EUR + REF_SENSORS_EUR) * share * p
    return {"electrolyte tanks": tanks * EUR_USD, "pumps, pipes and controls": loop * EUR_USD}
