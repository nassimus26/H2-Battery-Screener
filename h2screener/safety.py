"""Safety as a function of storage pressure.

Mechanical: energy released if a vessel bursts = isentropic expansion work of
the stored gas from p to 1 atm (real-gas internal energy and entropy from
CoolProp), also given as kg TNT (4.184 MJ/kg).
Chemical: combustion energy of the hydrogen inventory (lower heating value,
H2 + 1/2 O2 -> H2O(g), 241.8 kJ/mol, NIST-JANAF).
Leak: choked compressible flow through a reference hole,
  m_dot = Cd A p sqrt(gamma M / (R T)) (2/(gamma+1))^((gamma+1)/(2(gamma-1)))
and the time it takes to fill a closed room to 25 % of the lower flammability
limit (4 vol % H2 in air), and the steady concentration with ventilation.
Toxic: inventory of stored species with severe GHS hazard statements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import CoolProp.CoolProp as CP

from . import realgas
from .constants import M_H2, P_STD, R

TNT_J_PER_KG = 4.184e6
H2_LHV_J_MOL = 241.8e3
H2_LFL = 0.04


@dataclass
class SafetyPoint:
    burst_mj: float
    burst_tnt_kg: float
    combustion_mj: float
    leak_g_s: float
    minutes_to_quarter_lfl: float
    ventilated_h2_vol_pct: float
    pv_bar_l: float


def expansion_work(gas: str, moles: float, t_k: float, p_pa: float) -> float:
    """J released by isentropic expansion of `moles` of gas from (T, p) to 1 atm."""
    f = realgas.FLUIDS[gas]
    if realgas.condenses(gas, t_k, p_pa):
        p_pa = CP.PropsSI("P", "T", t_k, "Q", 1, f) * (1 - 1e-6)  # vapour space only
    u1 = CP.PropsSI("Umolar", "T", t_k, "P", p_pa, f)
    s1 = CP.PropsSI("Smolar", "T", t_k, "P", p_pa, f)
    if p_pa <= P_STD:
        return 0.0
    u2 = CP.PropsSI("Umolar", "P", P_STD, "Smolar", s1, f)
    return moles * max(0.0, u1 - u2)


def choked_flow(gas: str, t_k: float, p_pa: float, hole_m2: float, cd: float) -> float:
    """kg/s"""
    f = realgas.FLUIDS[gas]
    g = CP.PropsSI("Cpmass", "T", t_k, "P", p_pa, f) / CP.PropsSI("Cvmass", "T", t_k, "P", p_pa, f)
    m = CP.PropsSI("molar_mass", f)
    crit = (2 / (g + 1)) ** ((g + 1) / (2 * (g - 1)))
    if p_pa * crit < P_STD:           # not choked: subsonic, use the incompressible bound
        rho = CP.PropsSI("Dmass", "T", t_k, "P", p_pa, f)
        return cd * hole_m2 * math.sqrt(2 * rho * max(0.0, p_pa - P_STD))
    return cd * hole_m2 * p_pa * math.sqrt(g * m / (R * t_k)) * crit


def safety_point(h2_moles: float, v_gas_m3: float, t_k: float, p_max: float, s: dict) -> SafetyPoint:
    w = expansion_work("H2", h2_moles, t_k, p_max)
    leak = choked_flow("H2", t_k, p_max, s["leak_hole_mm2"] * 1e-6, s["discharge_coefficient"])
    rho_air_h2 = realgas.molar_density("H2", t_k, P_STD) * M_H2            # kg/m3 of H2 at 1 atm
    to_quarter = 0.25 * H2_LFL * s["room_m3"] * rho_air_h2 / leak if leak > 0 else math.inf
    vent = s["room_m3"] * s["air_changes_per_hour"] / 3600                 # m3/s of air
    conc = (leak / rho_air_h2) / (vent + leak / rho_air_h2) * 100 if leak > 0 else 0.0
    return SafetyPoint(w / 1e6, w / TNT_J_PER_KG, h2_moles * H2_LHV_J_MOL / 1e6,
                       leak * 1000, to_quarter / 60, conc, p_max / 1e5 * v_gas_m3 * 1000)
