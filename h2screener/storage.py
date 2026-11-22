"""Where the hydrogen sits, and how the cell is recharged.

Storage options for the hydrogen a design needs (`moles` used per discharge):

  tank       sealed gas vessel; pressure swings p_min (empty) -> p_max (full)
  piston     gas at one constant pressure in a variable-volume vessel
             (piston or bellows); wall sized for that pressure over the full volume
  hydride    metal hydride powder in a vessel; the two-phase plateau keeps the
             pressure constant at p_eq(T) from the van 't Hoff law
                 ln(p_eq / 1 atm) = -dH / RT + dS / R      (desorption, per mol H2)
             with dH, dS and capacity from ML-HydPARK (Witman et al., Sandia,
             doi:10.5281/zenodo.10680097). Heat dH must be removed on charge
             and supplied on discharge.
  electrode  the hydride is the negative electrode itself (no gas, no platinum):
             its potential equals a hydrogen electrode at p_eq, so the cell
             voltage shifts by (RT/2F) ln(p_eq / 1 atm)

Recharge modes:
  electric   sealed; electricity runs the reaction backwards; nothing is added
  h2-refuel  hydrogen is bought or made off board (electrolysis + compression)
             and filled into the tank; the positive side is oxygen from air
             (fuel cell). Recharge time = filling time.
  swap       hydrogen refuel plus exchange of the charged liquid electrolyte
             (couples with a dissolved positive side); regenerated off board.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import CoolProp.CoolProp as CP

from . import maintenance, materials, realgas, safety
from .constants import M_H2, P_STD, R

ROOT = Path(__file__).resolve().parent.parent
HYDPARK = ROOT / "data" / "raw" / "hydpark" / "ML-HYDPARK_v0.0.5.csv"
HYDPARK_SOURCE = "ML-HydPARK v0.0.5 (Witman et al., Sandia), doi:10.5281/zenodo.10680097"


@dataclass
class Storage:
    kind: str
    label: str
    mass_kg: float
    volume_l: float
    capex_usd: float
    pressure: str                      # "2 -> 60 bar" or "constant 8.3 bar"
    p_design_pa: float                 # pressure the vessel is built for
    voltage_shift_v: float             # change of cell voltage vs a 1 atm H2 electrode
    burst_tnt_kg: float
    leak_g_s: float
    h2_loss_per_year: float
    heat_kw: float = 0.0               # heat to move at design power (hydride)
    h2_inventory_mol: float = 0.0      # hydrogen inside when full (bought once for a sealed cell)
    needs_pt: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Alloy:
    formula: str
    wt_pct: float
    dh_kj: float
    ds_j: float
    cls: str
    reference: str

    def p_eq(self, t_k: float) -> float:
        """Pa"""
        return P_STD * math.exp(-self.dh_kj * 1000 / (R * t_k) + self.ds_j / R)


@lru_cache(maxsize=1)
def alloys() -> list[Alloy]:
    out = []
    with open(HYDPARK, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                a = Alloy(row["Composition_Formula"], float(row["Hydrogen_Weight_Percent"]),
                          float(row["Heat_of_Formation_kJperMolH2"]),
                          float(row["Entropy_of_Formation_JperMolH2perK"]),
                          row["Material_Class"], row.get("Reference") or "")
            except (ValueError, KeyError):
                continue
            if a.wt_pct > 0:
                out.append(a)
    return out


def alloy_cost_usd_kg(a: Alloy) -> float | None:
    try:
        return materials.cost_per_kg(a.formula)
    except Exception:
        return None


def pick_alloys(t_k: float, p_lo: float, p_hi: float, n: int = 3) -> list[Alloy]:
    """Alloys whose plateau lies in [p_lo, p_hi] at T, best by cost per kg of
    hydrogen stored (elements without a price are skipped), radioactive ones excluded."""
    from pymatgen.core import Composition
    scored = []
    for a in alloys():
        p = a.p_eq(t_k)
        if not (p_lo <= p <= p_hi):
            continue
        try:
            els = {str(e) for e in Composition(a.formula).elements}
        except Exception:
            continue
        if els & materials.RADIOACTIVE:
            continue
        c = alloy_cost_usd_kg(a)
        if c is None:
            continue
        scored.append((c / (a.wt_pct / 100), a))
    scored.sort(key=lambda x: x[0])
    seen, out = set(), []
    for _, a in scored:
        if a.formula not in seen:
            seen.add(a.formula)
            out.append(a)
        if len(out) >= n:
            break
    return out


def _vessel(p: float, v: float, sm: dict) -> float:
    return sm["shape_factor"] * sm["wall_density_kg_m3"] * p * v / (sm["wall_allowable_stress_mpa"] * 1e6)


def _common(moles_inventory, v_gas, t_k, p, t):
    sm, sf, mt = t["system"], t["safety"], t["maintenance"]
    sigma = sm["wall_allowable_stress_mpa"] * 1e6
    sp = safety.safety_point(moles_inventory, v_gas, t_k, p, sf) if v_gas > 0 else None
    if v_gas > 0 and moles_inventory > 0:
        perm, seal, _, _ = maintenance.hydrogen_loss(moles_inventory, v_gas, t_k, p, mt, sigma)
        loss = perm[1] + seal[1]
    else:
        loss = 0.0
    return sp, loss


def tank(moles: float, t_k: float, p_min: float, p_max: float, t: dict) -> Storage:
    sm, d = t["system"], t["design"]
    v = realgas.swing_volume("H2", moles, t_k, p_min, p_max)
    inv = v * realgas.molar_density("H2", t_k, p_max)
    mass = _vessel(p_max, v, sm)
    sp, loss = _common(inv, v, t_k, p_max, t)
    return Storage("tank", f"gas tank {p_min/1e5:g}->{p_max/1e5:g} bar", mass, v * 1000,
                   mass * sm["wall_usd_per_kg"] * d["vessel_fabrication_factor"],
                   f"{p_min/1e5:g} -> {p_max/1e5:g} bar", p_max, 0.0,
                   sp.burst_tnt_kg, sp.leak_g_s, loss, h2_inventory_mol=inv)


TROUTON = 88.0      # J/mol/K: evaporation entropy at the boiling point (Trouton's rule)
LIQUID_KG_L = 1.0   # density assumed for a condensed gas without equation-of-state data


def vapour_pressure(tb_k: float, t_k: float) -> float:
    """Pa, Clausius-Clapeyron with Trouton's rule (+/- a factor ~2)."""
    return P_STD * math.exp(TROUTON / R * (1 - tb_k / t_k))


def gas_tank(gas: str, moles: float, t_k: float, p_min: float, p_max: float, t: dict,
             tb_k: float | None = None) -> tuple[float, float, float]:
    """(kg, litres, USD) of a vessel for a gas a sealed cell must keep
    (oxygen of a hydrogen-oxygen cell, NO of a nitrate cell...). Same wall law
    and material as the hydrogen tank; the gas swings between max(p_min, 1 atm)
    and p_max. A gas with no equation of state in CoolProp is ideal, and is
    kept as a liquid at its vapour pressure when that is below p_max
    (boiling point tb_k, from the measured table when not given)."""
    sm, d = t["system"], t["design"]
    lo = max(p_min, P_STD)
    if realgas.fluid(gas):
        v = realgas.swing_volume(gas, moles, t_k, lo, p_max)
        p_wall = p_max
    else:
        if tb_k is None:
            from . import gases
            tb_k = gases.measured_tb(gas) or 0.0
        p_vap = vapour_pressure(tb_k, t_k) if tb_k > 0 else math.inf
        if p_vap < p_max:                       # condensed: liquid at its own vapour pressure
            from pymatgen.core import Composition
            v = moles * Composition(gas).weight / 1000 / (LIQUID_KG_L * 1000)
            p_wall = max(p_vap, P_STD)
        else:
            v = moles * R * t_k / (p_max - lo)
            p_wall = p_max
    mass = _vessel(p_wall, v, sm)
    return mass, v * 1000, mass * sm["wall_usd_per_kg"] * d["vessel_fabrication_factor"]


def piston(moles: float, t_k: float, p: float, t: dict) -> Storage:
    sm, d, s = t["system"], t["design"], t["storage"]
    v = moles / realgas.molar_density("H2", t_k, p)
    mass = _vessel(p, v, sm) * s["piston_mass_factor"]
    sp, loss = _common(moles, v, t_k, p, t)
    shift = R * t_k / (2 * 96485.33212) * math.log(realgas.fugacity("H2", t_k, p) / P_STD)
    return Storage("piston", f"variable volume at {p/1e5:g} bar", mass, v * 1000,
                   mass * sm["wall_usd_per_kg"] * d["vessel_fabrication_factor"] * s["piston_cost_factor"],
                   f"constant {p/1e5:g} bar", p, shift, sp.burst_tnt_kg, sp.leak_g_s, loss,
                   h2_inventory_mol=moles,
                   notes=["moving seal: leak rate and wear not modelled beyond the seal class"])


def hydride(moles: float, t_k: float, a: Alloy, power_w: float, e_kwh: float, t: dict,
            as_electrode: bool = False) -> Storage:
    sm, d, s = t["system"], t["design"], t["storage"]
    h2_kg = moles * M_H2 / s["hydride_usable_fraction"]
    alloy_kg = h2_kg / (a.wt_pct / 100)
    p = a.p_eq(t_k)
    rho = s["hydride_bulk_density_kg_l"]
    v_l = alloy_kg / rho
    price = alloy_cost_usd_kg(a) or d["unknown_material_usd_kg"]
    capex = alloy_kg * price * s["hydride_processing_factor"]
    # heat flow at design power: H2 moles per second x dH
    mol_s = moles / (e_kwh / (power_w / 1000) * 3600)
    heat = mol_s * a.dh_kj
    shift = R * t_k / (2 * 96485.33212) * math.log(p / P_STD)
    gas_v = v_l / 1000 * s["hydride_gas_void_fraction"]
    p_vessel = max(p, P_STD) * s["hydride_pressure_margin"]
    mass = alloy_kg
    if not as_electrode:
        vessel = _vessel(p_vessel, v_l / 1000, sm)
        mass += vessel
        capex += vessel * sm["wall_usd_per_kg"] * d["vessel_fabrication_factor"]
    gas_moles = gas_v * realgas.molar_density("H2", t_k, max(p, 1.0))
    sp, loss = _common(gas_moles if gas_moles > 0 else 0, gas_v, t_k, max(p, P_STD * 1.0001), t)
    kind = "electrode" if as_electrode else "hydride"
    label = f"{a.formula} {'electrode' if as_electrode else 'hydride'} ({a.wt_pct:g} wt%, {p/1e5:.3g} bar)"
    st = Storage(kind, label, mass, v_l, capex, f"constant {p/1e5:.3g} bar", p_vessel, shift,
                 sp.burst_tnt_kg if sp else 0.0, sp.leak_g_s if sp else 0.0, loss, heat,
                 needs_pt=not as_electrode, h2_inventory_mol=h2_kg / M_H2,
                 notes=[f"{HYDPARK_SOURCE}: dH {a.dh_kj:g} kJ/mol H2, dS {a.ds_j:.0f} J/mol K",
                        "cycle life of the alloy (decrepitation, capacity fade) not modelled"])
    if as_electrode:
        st.notes.append("electrode kinetics of the hydride unknown; no platinum needed")
    return st


# ------------------------------------------------------------------ off-board chain

def compression_work_j_per_mol(t_k: float, p_in: float, p_out: float, stages: int, eff: float) -> float:
    """Multi-stage isentropic compression with intercooling to t_k (real gas)."""
    if p_out <= p_in:
        return 0.0
    r = (p_out / p_in) ** (1 / stages)
    w, p = 0.0, p_in
    for _ in range(stages):
        h1 = CP.PropsSI("Hmolar", "T", t_k, "P", p, "Hydrogen")
        s1 = CP.PropsSI("Smolar", "T", t_k, "P", p, "Hydrogen")
        h2 = CP.PropsSI("Hmolar", "P", p * r, "Smolar", s1, "Hydrogen")
        w += (h2 - h1) / eff
        p *= r
    return w
