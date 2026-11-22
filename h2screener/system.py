"""Weight, volume and material cost per kWh as a function of hydrogen pressure.

Per kWh delivered at average discharge voltage V:
  charge         Q = 3.6e6 / V  coulomb
  reactions      N = Q / (n F)
  active mass    N x mass of reactants (heavier of charged/discharged state)
  electrolyte    dissolved moles / solubility x water density
  gas storage    every stored gas: swing volume between p_min and p_max from the
                 real-gas density (CoolProp), vessel wall mass from the thin-wall
                 membrane stress: m = phi rho_wall p V / sigma_allow
                 (phi = 3/2 for a sphere, 2 for a long cylinder)
  stack          electrode area = Q / areal capacity, x areal mass of the stack
  catalyst       area x Pt loading

Raising the pressure shrinks the gas volume but thickens the wall in
proportion; for an ideal gas the wall mass is nearly constant, and real-gas
compressibility (Z > 1 for H2) makes it rise. The sweep shows where the
optimum lies for each couple.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import materials, realgas
from .constants import F, WH
from .reactions import CellReaction


@dataclass
class SystemPoint:
    p_max_bar: float
    kg_per_kwh: float
    l_per_kwh: float
    usd_per_kwh: float | None
    parts: dict[str, float]


def system_point(cell: CellReaction, v_avg: float, p_min: float, p_max: float,
                 t_k: float, t: dict) -> SystemPoint:
    s = t["system"]
    q = 1000 * WH / v_avg
    n_rxn = q / (cell.n * F)
    parts: dict[str, float] = {}
    vol = 0.0
    usd = 0.0
    usd_known = True

    active = n_rxn * cell.active_mass()
    parts["active"] = active
    for sp, c in cell.reactants.items():
        if sp.is_solvent or sp.formula in ("H", "OH"):
            continue
        price = materials.cost_per_kg(sp.formula)
        if price is None:
            usd_known = False
        else:
            usd += price * n_rxn * c * sp.molar_mass

    dissolved = n_rxn * cell.dissolved_moles()
    water = dissolved / s["solubility_mol_l"] * 1.0   # kg (1 kg/L)
    parts["electrolyte"] = water
    vol += water / 1.0 / 1000

    vessel = 0.0
    for gas, nu in cell.gas_terms().items():
        if nu <= 0 or not realgas.fluid(gas):
            continue
        moles = n_rxn * nu
        lo = p_min if gas == "H2" else max(p_min, 1e5)
        v_gas = realgas.swing_volume(gas, moles, t_k, lo, p_max)
        vol += v_gas
        vessel += s["shape_factor"] * s["wall_density_kg_m3"] * p_max * v_gas / (s["wall_allowable_stress_mpa"] * 1e6)
    parts["vessel"] = vessel
    usd += vessel * s["wall_usd_per_kg"]

    area_cm2 = (q / 3.6) / s["areal_capacity_mah_cm2"]
    stack = area_cm2 * s["stack_areal_mass_g_cm2"] / 1000
    parts["stack"] = stack
    vol += area_cm2 * s["stack_thickness_cm"] / 1e6
    pt = area_cm2 * s["pt_loading_mg_cm2"] / 1e6
    usd += pt * materials.prices().get("Pt", 0)
    usd += stack * s["stack_usd_per_kg"]

    total = sum(parts.values())
    return SystemPoint(p_max / 1e5, total, vol * 1000, usd if usd_known else None, parts)


def pressure_sweep(cell: CellReaction, v_of_p, p_min: float, p_grid_bar: list[float],
                   t_k: float, t: dict) -> list[SystemPoint]:
    """v_of_p(p_max_pa) -> average discharge voltage at that pressure."""
    out = []
    for pb in p_grid_bar:
        p = pb * 1e5
        if p <= p_min:
            continue
        out.append(system_point(cell, v_of_p(p), p_min, p, t_k, t))
    return out
