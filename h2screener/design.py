"""System design: a couple built for a given power and duration at a given pressure.

For each maximum pressure the stack current density is chosen to minimise the
levelised cost of storage (LCOS): a larger stack runs at lower current, so it
is more efficient but costs more. Everything else follows from the laws in
the other modules:

  area       A = P / (i V_dis(i))
  capacity   Q = P t / V_dis
  hydrogen   swing volume between p_min and p_max (real gas), vessel wall
             from the membrane stress, permeation and seal loss
  safety     burst energy, leak, flammable and toxic inventory at p_max
  capex      materials, vessel, stack, catalyst, power electronics
  opex       O&M share of capex + hydrogen make-up
  LCOS       (capex + PV(opex + charging electricity)) / PV(energy delivered)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import estimate, maintenance, materials, performance, realgas, safety, synthesis
from .constants import F, M_H2, WH
from .performance import CellInputs, CellModel
from .reactions import CellReaction

E_O2 = 1.229


@dataclass
class DesignPoint:
    p_max_bar: float
    i_a_cm2: float
    area_m2: float
    rte: float
    rte_band: tuple[float, float, float]
    v_dis: float
    mass_kg: float
    volume_l: float
    parts_kg: dict
    capex_usd: float
    capex_usd_kwh: float
    opex_usd_year: float
    lcos_usd_kwh: float
    min_charge_h: float
    safety: safety.SafetyPoint
    h2_loss_per_year: float
    years_to_refill: float
    tank_volume_l: float = 0.0          # hydrogen tank inner volume
    wall_mm: float = 0.0                # tank wall thickness (cylinder, hoop stress)
    tank_diameter_m: float = 0.0
    h2_kg: float = 0.0                  # hydrogen inside when full
    cost_parts: dict = field(default_factory=dict)   # USD per part (build cost)
    synth: list = field(default_factory=list)        # (SynthCost, kg) per organic molecule


@dataclass
class Design:
    cell: CellReaction
    points: list[DesignPoint] = field(default_factory=list)
    best: DesignPoint | None = None
    ragone: tuple[list[float], list[float]] = ((), ())
    curve: dict = field(default_factory=dict)
    drift: list[str] = field(default_factory=list)
    recombination: list[str] = field(default_factory=list)
    permeation_sources: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)


def _nominal(params) -> CellInputs:
    a = lambda k: np.array([params[k].nominal])
    return CellInputs(a("i0_pos"), a("i0_neg"), a("asr"), a("ilim"), a("ce"))


def _pv(rate: float, years: int) -> float:
    return sum(1 / (1 + rate) ** y for y in range(1, years + 1))


def design(cell: CellReaction, t: dict, rng: np.random.Generator) -> Design:
    u, d, sm, sf, mt = t["use"], t["design"], t["system"], t["safety"], t["maintenance"]
    t_k = u["temperature_c"] + 273.15
    p_min = u["p_min_bar"] * 1e5
    aqueous = cell.medium in ("acid", "alkaline", "aqueous (pH-free)")
    v_ch_max = E_O2 + u["oxygen_margin_v"] if aqueous else cell.e0 + 0.5
    out = Design(cell)
    e_neg = 0.0 if cell.medium == "acid" else (-0.8277 if cell.medium == "alkaline" else 0.0)
    out.drift, out.recombination = maintenance.chemical_closure(cell, e_neg, E_O2 if aqueous else None)
    power_w = d["power_kw"] * 1000
    e_kwh = d["power_kw"] * d["hours"]
    sigma = sm["wall_allowable_stress_mpa"] * 1e6
    pv = _pv(d["discount_rate"], d["lifetime_years"])
    delivered_pv = e_kwh * d["cycles_per_year"] * pv

    for pb in u["pressure_grid_bar"]:
        p_max = pb * 1e5
        if p_max <= p_min:
            continue
        params = estimate.cell_params(cell, t_k, p_min, p_max, t)
        model = CellModel(cell, t_k, p_min, p_max, params)
        x = _nominal(params)
        ocv = model.ocv(performance.SOC)
        grid = np.geomspace(1e-4, 0.95 * x.ilim[0], 60)
        best = None
        curve_rte, curve_v = [], []
        for i in grid:
            rte, vd, vc = performance.efficiency(model, x, np.array([i]), ocv)
            rte, vd, vc = float(rte[0]), float(vd[0]), float(vc[0])
            curve_rte.append(rte)
            curve_v.append(vd)
            if rte <= 0 or vc > v_ch_max:
                continue
            pt = _point(cell, t, t_k, p_min, p_max, i, rte, vd, power_w, e_kwh, sigma, pv, delivered_pv)
            if best is None or pt.lcos_usd_kwh < best.lcos_usd_kwh:
                best = pt
        if best is None:
            continue
        # Fastest charge at this stack size (charge voltage limit), and the
        # efficiency band over the uncertain inputs at the chosen current.
        ok = [i for i, r, in zip(grid, curve_rte) if r > 0]
        i_fast = max([i for i in ok if performance.efficiency(model, x, np.array([i]), ocv)[2][0] <= v_ch_max],
                     default=best.i_a_cm2)
        best.min_charge_h = d["hours"] * best.i_a_cm2 / i_fast
        xs = model.sample(rng, 300)
        r_mc, _, _ = performance.efficiency(model, xs, np.full(300, best.i_a_cm2), ocv)
        best.rte_band = performance.pct(r_mc)
        out.points.append(best)
        out.params = params
        if pb == max(u["pressure_grid_bar"]) or not out.curve:
            out.curve = {"i": grid.tolist(), "rte": curve_rte, "v_dis": curve_v, "p_bar": pb}

    if not out.points:
        out.flags.append("no current density meets the charge-voltage limit")
        return out
    out.best = min(out.points, key=lambda p: p.lcos_usd_kwh)
    b = out.best
    # Power vs duration for the built system (fixed area and capacity).
    c = out.curve
    area_cm2 = b.area_m2 * 1e4
    cap_ah = e_kwh * 1000 / b.v_dis
    pw, hrs = [], []
    for i, v in zip(c["i"], c["v_dis"]):
        if v <= 0:
            continue
        h = cap_ah / (i * area_cm2)
        if not 0.001 <= h <= 1000:      # a minute to six weeks
            continue
        pw.append(i * area_cm2 * v / 1000)
        hrs.append(h)
    out.ragone = (pw, hrs)
    return out


def _point(cell, t, t_k, p_min, p_max, i, rte, vd, power_w, e_kwh, sigma, pv, delivered_pv) -> DesignPoint:
    d, sm, sf, mt = t["design"], t["system"], t["safety"], t["maintenance"]
    area_cm2 = power_w / (i * vd)
    q_c = e_kwh * 1000 * WH / vd
    n_rxn = q_c / (cell.n * F)
    parts, vol, capex = {}, 0.0, 0.0
    cp: dict = {}
    synth: list = []

    act = n_rxn * cell.active_mass() * (1 + d["active_excess"])
    # hydrogen is shown on its own (the tank inventory below), not inside "chemicals"
    act -= n_rxn * max(cell.h2_consumed(), 0) * M_H2 * (1 + d["active_excess"])
    parts["active"] = max(act, 0.0)
    for sp, c in cell.reactants.items():
        if sp.is_solvent or sp.formula in ("H", "OH", "H2"):
            continue
        price, sc = synthesis.species_price(sp, d["unknown_material_usd_kg"], t.get("synthesis"))
        kg = n_rxn * c * sp.molar_mass * (1 + d["active_excess"])
        cc = price * kg
        capex += cc
        cp["chemicals"] = cp.get("chemicals", 0.0) + cc
        if sc:
            synth.append((sc, kg))
    sol = min(cell.solubility_mol_l, 4.0) if cell.solubility_mol_l else sm["solubility_mol_l"]
    water = n_rxn * cell.dissolved_moles() / max(sol, 1e-3)
    parts["electrolyte"] = water
    vol += water / 1000

    h2_moles, v_h2, vessel = 0.0, 0.0, 0.0
    for gas, nu in cell.gas_terms().items():
        if nu <= 0 or not realgas.fluid(gas):
            continue
        lo = p_min if gas == "H2" else max(p_min, 1e5)
        v = realgas.swing_volume(gas, n_rxn * nu, t_k, lo, p_max)
        vol += v
        vessel += sm["shape_factor"] * sm["wall_density_kg_m3"] * p_max * v / sigma
        if gas == "H2":
            v_h2 = v
            h2_moles = v * realgas.molar_density("H2", t_k, p_max)
    parts["vessel"] = vessel
    parts["hydrogen"] = h2_moles * M_H2
    cp["hydrogen tank"] = vessel * sm["wall_usd_per_kg"] * d["vessel_fabrication_factor"]
    capex += cp["hydrogen tank"]

    parts["stack"] = area_cm2 * sm["stack_areal_mass_g_cm2"] / 1000
    vol += area_cm2 * sm["stack_thickness_cm"] / 1e6
    cp["cell stack"] = area_cm2 / 1e4 * d["stack_usd_m2"]
    cp["platinum catalyst"] = area_cm2 * sm["pt_loading_mg_cm2"] / 1e6 * materials.prices().get("Pt", 0)
    cp["power electronics"] = d["power_kw"] * d["power_electronics_usd_kw"]
    capex += cp["cell stack"] + cp["platinum catalyst"] + cp["power electronics"]

    if h2_moles > 0:
        perm, seal, years, _ = maintenance.hydrogen_loss(h2_moles, v_h2, t_k, p_max, mt, sigma)
        loss = perm[1] + seal[1]
        sp_ = safety.safety_point(h2_moles, v_h2, t_k, p_max, sf)
    else:
        loss, years = 0.0, math.inf
        sp_ = safety.SafetyPoint(0, 0, 0, 0, math.inf, 0, 0)
    cp["first hydrogen fill"] = h2_moles * M_H2 * d["h2_usd_kg"]   # sealed cell
    capex += cp["first hydrogen fill"]
    makeup = loss * h2_moles * M_H2 * d["h2_usd_kg"]
    opex = d["om_fraction_capex"] * capex + makeup
    charging = e_kwh * d["cycles_per_year"] / rte * d["electricity_usd_kwh"]
    lcos = (capex + pv * (opex + charging)) / delivered_pv
    pt = DesignPoint(p_max / 1e5, float(i), area_cm2 / 1e4, rte, (rte, rte, rte), vd,
                     sum(parts.values()), vol * 1000, parts, capex, capex / e_kwh, opex,
                     lcos, d["hours"], sp_, loss, years)
    pt.cost_parts = cp
    pt.synth = synth
    if v_h2 > 0:
        area, wall = maintenance.vessel_geometry(v_h2, p_max, sigma, mt["vessel_aspect"])
        pt.tank_volume_l, pt.wall_mm, pt.h2_kg = v_h2 * 1000, wall * 1000, h2_moles * M_H2
        pt.tank_diameter_m = (4 * v_h2 / (math.pi * mt["vessel_aspect"])) ** (1 / 3)
    return pt


def pareto(points: list[tuple[float, ...]]) -> list[int]:
    """Indices of points not dominated (all objectives minimised)."""
    keep = []
    for a, pa in enumerate(points):
        if not any(all(q <= p for q, p in zip(pb, pa)) and any(q < p for q, p in zip(pb, pa))
                   for b, pb in enumerate(points) if b != a):
            keep.append(a)
    return keep
