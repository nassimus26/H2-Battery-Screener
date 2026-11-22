"""Stage 3: real-gas behaviour of stored reactants (CoolProp equations of state).

Gives density, fugacity (for the Nernst term) and the stored-gas volume that
a given number of moles needs when the vessel swings between two pressures.
Also tells whether a reactant condenses at the storage temperature, which
changes how it is stored (Cl2, for example, is a liquid above ~7 bar at 20 C).
"""

from __future__ import annotations

from functools import lru_cache

import CoolProp.CoolProp as CP

# Species name in our reactions -> CoolProp fluid name.
FLUIDS = {
    "H2": "Hydrogen",
    "O2": "Oxygen",
    "Cl2": "Chlorine",
    "F2": "Fluorine",
    "NH3": "Ammonia",
    "HCl": "HydrogenChloride",
    "H2S": "HydrogenSulfide",
    # gases of the sealed two-gas screener (formulas with a single isomer only)
    "CH4": "Methane",
    "CO2": "CarbonDioxide",
    "CO": "CarbonMonoxide",
    "N2": "Nitrogen",
    "C2H6": "Ethane",
    "C2H4": "Ethylene",
    "C3H8": "n-Propane",
    "N2O": "NitrousOxide",
    "SO2": "SulfurDioxide",
    "COS": "CarbonylSulfide",
}


def fluid(species: str) -> str | None:
    return FLUIDS.get(species)


@lru_cache(maxsize=4096)
def molar_density(species: str, t_k: float, p_pa: float) -> float:
    """mol/m3 of the stored phase at (T, p). For a condensing species above its
    vapour pressure this is the liquid density (storage as liquid)."""
    f = FLUIDS[species]
    if t_k < CP.PropsSI("Tcrit", f):
        p_sat = CP.PropsSI("P", "T", t_k, "Q", 0, f)
        if p_pa >= p_sat:
            return CP.PropsSI("Dmolar", "T", t_k, "Q", 0, f)
    return CP.PropsSI("Dmolar", "T", t_k, "P", p_pa, f)


@lru_cache(maxsize=32)
def _state(species: str):
    return CP.AbstractState("HEOS", FLUIDS[species])


def _phi(species: str, t_k: float, p_pa: float | None) -> float:
    st = _state(species)
    if p_pa is None:
        st.update(CP.QT_INPUTS, 1.0, t_k)
    else:
        st.update(CP.PT_INPUTS, p_pa, t_k)
    return st.fugacity_coefficient(0)


@lru_cache(maxsize=4096)
def fugacity(species: str, t_k: float, p_pa: float) -> float:
    """Fugacity in Pa. Above the vapour pressure the gas fugacity is capped at
    the saturated value (the liquid holds the pressure)."""
    f = FLUIDS[species]
    if t_k < CP.PropsSI("Tcrit", f):
        p_sat = CP.PropsSI("P", "T", t_k, "Q", 0, f)
        if p_pa >= p_sat:
            # Gas side of the saturation line (the two-phase state has no
            # single-phase fugacity coefficient).
            return _phi(species, t_k, p_sat * (1 - 1e-6)) * p_sat
    return _phi(species, t_k, p_pa) * p_pa


def condenses(species: str, t_k: float, p_pa: float) -> bool:
    f = FLUIDS[species]
    if t_k >= CP.PropsSI("Tcrit", f):
        return False
    return p_pa >= CP.PropsSI("P", "T", t_k, "Q", 0, f)


def pressure_at_density(species: str, t_k: float, d_molar: float) -> float:
    return CP.PropsSI("P", "T", t_k, "Dmolar", d_molar, FLUIDS[species])


def swing_volume(species: str, moles: float, t_k: float, p_min: float, p_max: float) -> float:
    """m3 of vessel volume that delivers `moles` when the pressure falls from
    p_max to p_min."""
    d = molar_density(species, t_k, p_max) - molar_density(species, t_k, p_min)
    if d <= 0:
        raise ValueError("p_max must exceed p_min")
    return moles / d
