"""Physics-based estimates of the cell inputs the performance model needs.

Each input is derived from a law where one applies and gets a wide, labelled
range where none does. Ranges for engineering choices (gap, separator,
solubility) come from targets.toml so they can be changed without editing code.
"""

from __future__ import annotations

import math

from . import realgas
from .constants import F, P_STD
from .performance import stokes_einstein
from .provenance import Value
from .reactions import CellReaction

DURST_2014 = ("Durst et al., Energy Environ. Sci. 7 (2014) 2255, doi:10.1039/C4EE00440J: "
              "HOR/HER i0 on Pt ~1e2 mA/cm2_Pt in acid, ~2 orders lower in alkaline")


def h2_exchange_current(medium: str, roughness: tuple[float, float]) -> Value:
    """A/cm2 of geometric area: specific i0 x catalyst roughness factor."""
    if medium == "acid":
        lo, hi = 0.03 * roughness[0], 0.3 * roughness[1]
    elif medium in ("alkaline", "aqueous (pH-free)"):
        lo, hi = 3e-4 * roughness[0], 3e-3 * roughness[1]
    else:
        return Value(1e-3, 1e-5, 1e-1, "unknown", "no data for hydrogen electrodes in this medium", log=True)
    return Value(math.sqrt(lo * hi), lo, hi, "physics", DURST_2014 + " x roughness (targets)", log=True)


def asr(medium: str, t: dict) -> Value:
    """Ohm cm2: electrolyte gap / conductivity + separator and contacts."""
    k = t["conductivity_s_cm"].get(medium, t["conductivity_s_cm"]["other"])
    g = t["gap_cm"]
    s = t["separator_ohm_cm2"]
    lo = g[0] / k[1] + s[0]
    hi = g[1] / k[0] + s[1]
    return Value(math.sqrt(lo * hi), lo, hi, "physics",
                 "gap/conductivity + separator (ranges in targets.toml)", log=True)


def limiting_current(cell: CellReaction, t_k: float, p_min: float, p_max: float, t: dict) -> Value:
    """A/cm2: the smallest transport limit among the reactants (Fick's law)."""
    limits: list[tuple[float, float, str]] = []
    n = cell.n
    for sp, nu in list(cell.reactants.items()) + list(cell.products.items()):
        if sp.phase == "aq" and sp.formula not in ("H", "OH", "H2O"):
            # Dissolved species: i = (n/nu) F D c / delta, D by Stokes-Einstein.
            d_lo = stokes_einstein(t_k, t["hydrodynamic_radius_nm"][1] * 1e-9)
            d_hi = stokes_einstein(t_k, t["hydrodynamic_radius_nm"][0] * 1e-9)
            c_lo, c_hi = (x * 1000 for x in t["solubility_mol_l"])
            if cell.solubility_mol_l:
                c_hi = min(cell.solubility_mol_l, 4.0) * 1000
                c_lo = min(c_lo, 0.5 * c_hi)
            dl_lo, dl_hi = (x * 1e-6 for x in t["boundary_layer_um"])
            k = (n / nu) * F / 1e4
            limits.append((k * d_lo * c_lo / dl_hi, k * d_hi * c_hi / dl_lo,
                           f"diffusion of {sp.label} (Stokes-Einstein, Fick)"))
        elif sp.phase == "g" and realgas.fluid(sp.formula) and sp in cell.reactants:
            # Gas-fed porous electrode: limit scales with dissolved concentration,
            # i.e. with fugacity (Henry's law); base value at 1 atm from targets.
            p = p_min if sp.formula == "H2" else p_max
            scale = realgas.fugacity(sp.formula, t_k, p) / P_STD
            b = t["gas_electrode_ilim_1atm_a_cm2"]
            limits.append((b[0] * scale, b[1] * scale, f"{sp.formula} supply (Henry's law x fugacity)"))
        elif sp.phase == "s" and sp in cell.reactants:
            b = t["solid_ilim_a_cm2"]
            limits.append((b[0], b[1], f"solid-state transport in {sp.label} (unknown)"))
    if not limits:
        return Value(10.0, 10.0, 10.0, "assumption", "no transport-limited reactant")
    lo = min(x[0] for x in limits)
    hi = min(x[1] for x in limits)
    hi = max(hi, lo)
    worst = min(limits, key=lambda x: math.sqrt(x[0] * x[1]))
    kind = "unknown" if "unknown" in worst[2] else "physics"
    return Value(math.sqrt(lo * hi), lo, hi, kind, worst[2], log=True)


ORR_2024 = ("Nature Communications 15 (2024), doi:10.1038/s41467-024-51605-1: ORR on Pt "
            "i0 = 13 +/- 4 uA/cm2_Pt with a 120 mV/decade Tafel slope")


def positive_exchange_current(cell: CellReaction, e: dict) -> Value:
    if any(sp.formula == "O2" and sp.phase == "g" for sp in cell.reactants):
        r = e["catalyst_roughness"]
        lo, hi = 9e-6 * r[0], 17e-6 * r[1]
        return Value(math.sqrt(lo * hi), lo, hi, "physics", ORR_2024 + " x roughness; the OER on charge is assumed no faster", log=True)
    return Value(1e-3, e["unknown_i0_a_cm2"][0], e["unknown_i0_a_cm2"][1], "unknown",
                 "no law predicts exchange currents of an untested couple", log=True)


def cell_params(cell: CellReaction, t_k: float, p_min: float, p_max: float, t: dict) -> dict[str, Value]:
    e = t["estimates"]
    return {
        "i0_pos": positive_exchange_current(cell, e),
        "i0_neg": h2_exchange_current(cell.medium, tuple(e["catalyst_roughness"])),
        "asr": asr(cell.medium, e),
        "ilim": limiting_current(cell, t_k, p_min, p_max, e),
        "ce": Value(sum(e["coulombic_efficiency"]) / 2, *e["coulombic_efficiency"], "assumption",
                    "self-discharge by H2 crossover (targets.toml)"),
    }
