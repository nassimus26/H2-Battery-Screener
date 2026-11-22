"""Every way to build and recharge a couple, sized for the same job and compared.

For one couple and the design in targets.toml (power x hours) this builds:

  electric + tank       sealed, pressure swings p_min -> p_max
  electric + piston     sealed, constant pressure, variable volume
  electric + hydride    sealed, constant plateau pressure (ML-HydPARK alloys)
  electric + electrode  sealed, hydride negative electrode, no gas, no platinum
                        (alkaline or pH-free media: hydride alloys corrode in acid)
  h2-refuel + tank      fuel cell on air; recharge = filling hydrogen made off
                        board (electrolysis + compression) at 350/700 bar
  h2-refuel + hydride   the same with a hydride tank
  swap + tank           hydrogen refuel plus exchange of the charged electrolyte

Each option gets its own current density, chosen for the lowest levelised
cost of storage, and the same outputs: mass, volume, build cost, LCOS,
efficiency (whole chain for refuel modes), recharge time, safety, H2 loss.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from . import estimate, materials, performance, realgas, stackcost, storage, synthesis
from .constants import F, M_H2, P_STD, WH
from .performance import CellModel
from .provenance import Value
from .reactions import CellReaction

E_O2 = 1.229


@dataclass
class Config:
    mode: str
    store: storage.Storage
    i_a_cm2: float
    area_m2: float
    v_dis: float
    rte: float
    rte_band: tuple
    mass_kg: float
    volume_l: float
    capex_usd: float
    capex_usd_kwh: float
    lcos_usd_kwh: float
    recharge_h: float
    wh_kg: float
    wh_l: float
    w_kg: float
    notes: list[str] = field(default_factory=list)
    cost_parts: dict = field(default_factory=dict)   # USD per part (build cost)
    synth: list = field(default_factory=list)        # (SynthCost, kg) per organic molecule

    @property
    def name(self) -> str:
        return f"{self.mode} + {self.store.label}"


def _pv(rate, years):
    return sum(1 / (1 + rate) ** y for y in range(1, years + 1))


def _nominal(params):
    a = lambda k: np.array([params[k].nominal])
    return performance.CellInputs(a("i0_pos"), a("i0_neg"), a("asr"), a("ilim"), a("ce"))


def air_cell(cells: list[CellReaction], medium: str = "acid") -> CellReaction | None:
    for c in cells:
        if c.medium == medium and {sp.formula for sp in c.reactants} == {"O2", "H2"} and \
                {sp.formula for sp in c.products} == {"H2O"}:
            return c
    return None


def _optimise(cell, t, mode, p_lo, p_hi, build_store, needs_pt=True, i0_neg=None,
              fixed=None, w_comp_v=0.0, rng=None, recharge=None, liquid=False) -> Config | None:
    """Scan current density; return the lowest-LCOS feasible Config."""
    u, d, sm = t["use"], t["design"], t["system"]
    t_k = u["temperature_c"] + 273.15
    aqueous = cell.medium in ("acid", "alkaline", "aqueous (pH-free)")
    v_ch_max = E_O2 + u["oxygen_margin_v"] if aqueous else cell.e0 + 0.5
    if mode == "h2-refuel":
        v_ch_max = 3.0          # the electrolyser runs off board, no water-window limit
    if getattr(cell, "v_ch_max", None):
        v_ch_max = cell.v_ch_max
    params = estimate.cell_params(cell, t_k, p_lo, p_hi, t)
    if i0_neg is not None:
        params["i0_neg"] = i0_neg
    if fixed and "O2" in fixed:
        b = t["estimates"]["gas_electrode_ilim_1atm_a_cm2"]
        frac = fixed["O2"] / P_STD
        params["ilim"] = Value(math.sqrt(b[0] * b[1]) * frac, b[0] * frac, b[1] * frac, "physics",
                               "O2 from air (Henry's law x partial pressure)", log=True)
    model = CellModel(cell, t_k, p_lo, p_hi, params, fixed_fugacity=fixed or {})
    x = _nominal(params)
    ocv = model.ocv(performance.SOC)
    power_w = d["power_kw"] * 1000
    e_kwh = d["power_kw"] * d["hours"]
    pv = _pv(d["discount_rate"], d["lifetime_years"])
    delivered = e_kwh * d["cycles_per_year"] * pv
    grid = np.geomspace(1e-4, 0.95 * x.ilim[0], 50)
    feas = []
    for i in grid:
        r, vd, vc = performance.efficiency(model, x, np.array([i]), ocv)
        r, vd, vc = float(r[0]), float(vd[0]), float(vc[0])
        if r > 0 and vc <= v_ch_max:
            feas.append((i, r, vd, vc))
    if not feas:
        return None
    i_fast = max(f[0] for f in feas)
    best = None
    for i, r, vd, vc in feas:
        rte = r * vc / (vc + w_comp_v) if w_comp_v else r
        area_cm2 = power_w / (i * vd)
        area = area_cm2 / 1e4                       # m2
        q_c = e_kwh * 1000 * WH / vd
        n_rxn = q_c / (cell.n * F)
        h2_mol = n_rxn * cell.h2_consumed()
        st = build_store(h2_mol)
        if st is None:
            return None
        mass, vol, capex = st.mass_kg, st.volume_l / 1000, st.capex_usd
        parts = {"hydrogen storage": st.capex_usd}
        synth: list = []
        if mode == "electric":                 # first fill of a sealed cell
            fill = st.h2_inventory_mol * M_H2 * d["h2_usd_kg"]
            capex += fill
            parts["first hydrogen fill"] = fill
        if mode != "h2-refuel":
            act = n_rxn * cell.active_mass() * (1 + d["active_excess"])
            mass += act
            for sp, c in cell.reactants.items():
                if sp.is_solvent or sp.formula in ("H", "OH", "H2"):
                    continue
                price, sc = synthesis.species_price(sp, d["unknown_material_usd_kg"], t.get("synthesis"))
                kg = n_rxn * c * sp.molar_mass
                cc = price * kg
                capex += cc
                parts["chemicals"] = parts.get("chemicals", 0.0) + cc
                if sc:
                    synth.append((sc, kg))
            sol = min(cell.solubility_mol_l, 4.0) if cell.solubility_mol_l else sm["solubility_mol_l"]
            water = n_rxn * cell.dissolved_moles() / max(sol, 1e-3)
            mass += water
            vol += water / 1000
            for k, v in stackcost.flow_hardware(water, d["power_kw"], 2 if liquid else 1).items():
                capex += v
                parts[k] = v
        stack = area_cm2 * sm["stack_areal_mass_g_cm2"] / 1000
        mass += stack
        vol += area_cm2 * sm["stack_thickness_cm"] / 1e6
        parts["cell stack"] = area * d["stack_usd_m2_liquid" if liquid else "stack_usd_m2"]
        parts["power electronics"] = d["power_kw"] * d["power_electronics_usd_kw"]
        capex += parts["cell stack"] + parts["power electronics"]
        if needs_pt:
            parts["platinum catalyst"] = area_cm2 * sm["pt_loading_mg_cm2"] / 1e6 * materials.prices().get("Pt", 0)
            capex += parts["platinum catalyst"]
        makeup = st.h2_loss_per_year * abs(h2_mol) * M_H2 * d["h2_usd_kg"]
        opex = d["om_fraction_capex"] * capex + makeup
        charging = e_kwh * d["cycles_per_year"] / rte * d["electricity_usd_kwh"]
        lcos = (capex + pv * (opex + charging)) / delivered
        rh = recharge(h2_mol, n_rxn) if recharge else d["hours"] * i / i_fast
        cfg = Config(mode, st, float(i), area, vd, rte, (rte, rte, rte), mass, vol * 1000, capex,
                     capex / e_kwh, lcos, rh, e_kwh * 1000 / mass, e_kwh * 1000 / (vol * 1000),
                     power_w / mass, cost_parts=parts, synth=synth)
        if best is None or cfg.lcos_usd_kwh < best.lcos_usd_kwh:
            best = cfg
    if rng is not None:
        xs = model.sample(rng, 200)
        r_mc, _, vc_mc = performance.efficiency(model, xs, np.full(200, best.i_a_cm2), ocv)
        if w_comp_v:
            r_mc = r_mc * vc_mc / (vc_mc + w_comp_v)
        best.rte_band = performance.pct(r_mc)
    return best


def configurations(cell: CellReaction, t: dict, rng: np.random.Generator,
                   all_cells: list[CellReaction] | None = None) -> list[Config]:
    u, d, rf = t["use"], t["design"], t["refuel"]
    t_k = u["temperature_c"] + 273.15
    p_min, p_max = u["p_min_bar"] * 1e5, u["p_max_bar"] * 1e5
    power_w, e_kwh = d["power_kw"] * 1000, d["power_kw"] * d["hours"]
    out: list[Config] = []
    has_h2 = cell.h2_consumed() > 0

    def add(c, note=None):
        if c is not None:
            if note:
                c.notes.append(note)
            out.append(c)

    if has_h2:
        add(_optimise(cell, t, "electric", p_min, p_max,
                      lambda m: storage.tank(m, t_k, p_min, p_max, t), rng=rng))
        add(_optimise(cell, t, "electric", p_max * 0.999, p_max,
                      lambda m: storage.piston(m, t_k, p_max, t), rng=rng))
        win = [x * 1e5 for x in t["storage"]["plateau_window_bar"]]
        for a in storage.pick_alloys(t_k, win[0], min(win[1], p_max), 2):
            p = a.p_eq(t_k)
            add(_optimise(cell, t, "electric", p * 0.999, p,
                          lambda m, a=a: storage.hydride(m, t_k, a, power_w, e_kwh, t), rng=rng))
        if cell.medium in ("alkaline", "aqueous (pH-free)"):
            for a in storage.pick_alloys(t_k, 1e3, P_STD * 2, 2):
                p = a.p_eq(t_k)
                add(_optimise(cell, t, "electric", p * 0.999, p,
                              lambda m, a=a: storage.hydride(m, t_k, a, power_w, e_kwh, t, as_electrode=True),
                              needs_pt=False,
                              i0_neg=Value(1e-2, 1e-4, 1e-1, "unknown", "hydride electrode kinetics", log=True),
                              rng=rng))
        if cell.dissolved_moles() > 0:
            w = storage.compression_work_j_per_mol(t_k, rf["electrolyser_outlet_bar"] * 1e5, p_max,
                                                   rf["compressor_stages"], rf["compressor_efficiency"])

            def swap_time(h2_mol, n_rxn, cell=cell):
                sol = min(cell.solubility_mol_l, 4.0) if cell.solubility_mol_l else t["system"]["solubility_mol_l"]
                litres = n_rxn * cell.dissolved_moles() / max(sol, 1e-3)
                return max(h2_mol * M_H2 / rf["fill_kg_per_min"], litres / rf["swap_l_per_min"]) / 60
            add(_optimise(cell, t, "swap", p_min, p_max, lambda m: storage.tank(m, t_k, p_min, p_max, t),
                          w_comp_v=w * cell.h2_consumed() / (cell.n * F), rng=rng, recharge=swap_time),
                "station regenerates the electrolyte with the same reaction run backwards (cost not counted)")
    return out


TOYOTA_TANK = ("Toyota, Mirai launch (Nov 2014): 700 bar tanks, 5.7 wt% storage density, "
               "https://global.toyota/en/detail/4198334")


def composite_stress_mpa(t: dict) -> float:
    """Effective wall stress that makes the thin-wall model hold the published
    hydrogen weight fraction of a real 700 bar car tank (liner, bosses and the
    2.25 burst safety factor are folded into it)."""
    rf, sm = t["refuel"], t["system"]
    t_k = t["use"]["temperature_c"] + 273.15
    p = rf["composite_reference_bar"] * 1e5
    wt = rf["composite_reference_wt_pct"] / 100
    rho_h2 = realgas.molar_density("H2", t_k, p) * M_H2
    ratio = (1 - wt) / wt                         # wall kg per kg H2
    return sm["shape_factor"] * rf["composite_wall_density_kg_m3"] * p / (ratio * rho_h2) / 1e6


def refuel_configurations(all_cells: list[CellReaction], t: dict, rng) -> list[Config]:
    """Hydrogen car style: fuel cell on air, tank refilled at a station."""
    u, d, rf = t["use"], t["design"], t["refuel"]
    t_k = u["temperature_c"] + 273.15
    cell = air_cell(all_cells, "acid")
    if cell is None:
        return []
    tc = copy.deepcopy(t)
    tc["system"].update(wall_density_kg_m3=rf["composite_wall_density_kg_m3"],
                        wall_allowable_stress_mpa=composite_stress_mpa(t),
                        wall_usd_per_kg=rf["composite_usd_per_kg"])
    fixed = {"O2": rf["air_o2_fraction"] * P_STD}
    p_min = u["p_min_bar"] * 1e5
    out = []
    fill = lambda h2_mol, n_rxn: h2_mol * M_H2 / rf["fill_kg_per_min"] / 60
    for pb in rf["tank_pressures_bar"]:
        p = pb * 1e5
        w = storage.compression_work_j_per_mol(t_k, rf["electrolyser_outlet_bar"] * 1e5, p,
                                               rf["compressor_stages"], rf["compressor_efficiency"])
        c = _optimise(cell, tc, "h2-refuel", p_min, p, lambda m, p=p: storage.tank(m, t_k, p_min, p, tc),
                      fixed=fixed, w_comp_v=w * cell.h2_consumed() / (cell.n * F), rng=rng, recharge=fill)
        if c:
            c.store.label = f"carbon-fibre tank {pb:g} bar"
            c.notes += ["efficiency = electrolysis x compression x fuel cell",
                        "off-board electrolyser, compressor and station are not in the build cost",
                        "tank wall fitted to " + TOYOTA_TANK]
            out.append(c)
    win = [x * 1e5 for x in t["storage"]["plateau_window_bar"]]
    for a in storage.pick_alloys(t_k, win[0], win[1], 1):
        pe = a.p_eq(t_k)
        w = storage.compression_work_j_per_mol(t_k, rf["electrolyser_outlet_bar"] * 1e5, pe * 1.2,
                                               rf["compressor_stages"], rf["compressor_efficiency"])
        c = _optimise(cell, t, "h2-refuel", pe * 0.999, pe,
                      lambda m, a=a: storage.hydride(m, t_k, a, d["power_kw"] * 1000, d["power_kw"] * d["hours"], t),
                      fixed=fixed, w_comp_v=w * cell.h2_consumed() / (cell.n * F), rng=rng, recharge=fill)
        if c:
            c.notes += ["efficiency = electrolysis x compression x fuel cell",
                        "off-board electrolyser, compressor and station are not in the build cost",
                        "fill rate limited by hydride heat removal (not modelled)"]
            out.append(c)
    return out


def pair_configuration(pr, t: dict, rng) -> Config | None:
    """Two-molecule cell: both sides liquid, no gas, no platinum, no vessel."""
    from . import pairs
    cell = pairs.as_cell(pr)
    # charging may push each electrode only up to its water limit; split evenly
    head = min(1.229 + pairs.OER_MARGIN - pr.pos_range[1], pr.neg_range[0] + pairs.HER_MARGIN)
    cell.v_ch_max = pr.voltage + 2 * max(head, 0.0)
    none = lambda m: storage.Storage("none", "no gas: two liquid electrolytes", 0.0, 0.0, 0.0,
                                     "atmospheric, no gas", P_STD, 0.0, 0.0, 0.0, 0.0, needs_pt=False,
                                     notes=["no free hydrogen: nothing to burn or burst"])
    unknown = Value(1e-3, 1e-5, 1e-1, "unknown", "organic redox on carbon felt (no data for new molecules)", log=True)
    c = _optimise(cell, t, "electric", P_STD * 0.999, P_STD, none, needs_pt=False, i0_neg=unknown, rng=rng,
                  liquid=True)
    if c is not None:
        c.notes += pr.notes
    return c
