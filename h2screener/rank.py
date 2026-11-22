"""Evaluate every candidate through all stages and rank them.

Score: weighted geometric mean of desirabilities in (0, 1]
  cost          1 / (1 + usd_per_kwh / ref)
  efficiency    median round-trip efficiency at the fastest charge that meets the target
  charge_speed  min(1, target hours / median charge hours)
  safety        product of hazard factors (water oxidation, toxic species, ...)
  supply        1 - HHI / 20000
  mass, volume  1 / (1 + value / ref)
A zero in any term (radioactive, never meets the efficiency target, ...) gives
a zero score but the candidate stays in the table with its reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import estimate, materials, performance, system
from .constants import F, WH
from .performance import CellModel
from .provenance import Value
from .reactions import CellReaction

E_O2 = 1.229   # V, O2/H2O vs the hydrogen electrode in the same electrolyte
# Small molecules with very strong bonds (N#N, C=O in CO2, ...) whose
# electrochemical oxidation/reduction back to the charged state is an open
# research problem; a cell that makes them on discharge is hard to recharge.
INERT_PRODUCTS = {"N2", "N2O", "CO2", "CH4", "Kr", "Xe"}


@dataclass
class Result:
    cell: CellReaction
    score: float = 0.0
    wh_kg_active: float = 0.0
    perf: dict = field(default_factory=dict)
    sweep: list = field(default_factory=list)
    best: system.SystemPoint | None = None
    flags: list[str] = field(default_factory=list)
    factors: dict[str, float] = field(default_factory=dict)
    params: dict[str, Value] = field(default_factory=dict)
    hazards: dict[str, list[str]] = field(default_factory=dict)


def evaluate(cell: CellReaction, t: dict, rng: np.random.Generator, ghs: bool = False) -> Result:
    u, rk = t["use"], t["ranking"]
    t_k = u["temperature_c"] + 273.15
    p_min, p_max = u["p_min_bar"] * 1e5, u["p_max_bar"] * 1e5
    r = Result(cell)
    els = cell.elements()
    safety = 1.0

    if els & materials.RADIOACTIVE:
        r.flags.append("radioactive: " + ",".join(sorted(els & materials.RADIOACTIVE)))
        safety = 0.0
    if els & materials.NOBLE:
        r.flags.append("noble-gas compound (unstable or explosive)")
        safety *= 0.1
    inert = {sp.formula for sp in cell.products if sp.formula in INERT_PRODUCTS}
    if inert:
        r.flags.append("discharge makes " + ",".join(sorted(inert)) +
                       ": no known electrode turns it back at useful rates (recharge doubtful)")
        r.factors["reversibility"] = 0.3
    else:
        r.factors["reversibility"] = 1.0
    aqueous = cell.medium in ("acid", "alkaline", "aqueous (pH-free)")
    from . import maintenance
    e_neg = -0.8277 if cell.medium == "alkaline" else 0.0
    drift, _ = maintenance.chemical_closure(cell, e_neg, E_O2 if aqueous else None)
    r.factors["closure"] = 0.5 ** min(len(drift), 3)
    r.flags += drift
    if aqueous:
        v_ch_max = E_O2 + u["oxygen_margin_v"]
        if cell.e0 > v_ch_max:
            r.flags.append(f"positive side oxidises water (E0 {cell.e0:.2f} V > {v_ch_max:.2f} V)")
            safety *= 0.3
    else:
        v_ch_max = cell.e0 + 0.5
    if ghs:
        for sp in cell.species():
            if sp.formula in ("H", "OH", "H2O", "H2", "e"):
                continue
            hz = materials.ghs_hazards(sp.formula) or []
            r.hazards[sp.formula] = hz
            f = materials.safety_factor(hz)
            if f < 1:
                r.flags.append(f"{sp.formula}: {','.join(h for h in hz if h in materials.SEVERE | materials.TOXIC)}")
            safety *= f

    r.params = estimate.cell_params(cell, t_k, p_min, p_max, t)
    model = CellModel(cell, t_k, p_min, p_max, r.params)
    q = Value.fixed(t["system"]["areal_capacity_mah_cm2"] / 1000, "assumption", "targets.toml")
    r.perf = performance.evaluate(model, u["rte_target"], v_ch_max, q, rng, rk["monte_carlo_draws"])
    ocv_mean = r.perf["ocv_v"][1]
    v_avg = r.perf["v_dis_avg"][1] if r.perf["v_dis_avg"][1] > 0 else ocv_mean
    loss = ocv_mean - v_avg

    def v_of_p(p):
        m = CellModel(cell, t_k, p_min, p, r.params)
        return max(1e-3, float(m.ocv(performance.SOC).mean()) - loss)

    r.sweep = system.pressure_sweep(cell, v_of_p, p_min, u["pressure_grid_bar"], t_k, t)
    if r.sweep:
        key = (lambda s: s.usd_per_kwh) if all(s.usd_per_kwh is not None for s in r.sweep) else (lambda s: s.kg_per_kwh)
        r.best = min(r.sweep, key=key)
    r.wh_kg_active = cell.e0 * cell.n * F / cell.active_mass() / WH

    f = r.factors
    f["safety"] = safety
    usd = r.best.usd_per_kwh if r.best else None
    f["cost"] = 1 / (1 + usd / rk["cost_ref_usd_kwh"]) if usd is not None else 0.3
    if usd is None:
        r.flags.append("no price data for some element")
    f["efficiency"] = r.perf["rte_at_i_charge"][1]
    th = r.perf["t_charge_h"][1]
    f["charge_speed"] = min(1.0, u["charge_hours_target"] / th) if math.isfinite(th) and th > 0 else 0.0
    h = materials.supply_hhi(els - {"H", "O"})
    f["supply"] = 1 - h / 20000 if math.isfinite(h) else 0.5
    f["mass"] = 1 / (1 + r.best.kg_per_kwh / rk["mass_ref_kg_kwh"]) if r.best else 0
    f["volume"] = 1 / (1 + r.best.l_per_kwh / rk["volume_ref_l_kwh"]) if r.best else 0
    wsum = sum(rk[k] for k in f)
    if any(v <= 0 for v in f.values()):
        r.score = 0.0
    else:
        r.score = math.exp(sum(rk[k] * math.log(v) for k, v in f.items()) / wsum)
    return r


def rank(cells: list[CellReaction], t: dict, seed: int = 1, ghs: bool = False) -> list[Result]:
    rng = np.random.default_rng(seed)
    out = []
    for c in cells:
        try:
            out.append(evaluate(c, t, rng, ghs))
        except Exception as e:   # a candidate the models cannot handle is reported, not hidden
            r = Result(c)
            r.flags.append(f"evaluation failed: {type(e).__name__}: {e}")
            out.append(r)
    out.sort(key=lambda r: -r.score)
    return out
