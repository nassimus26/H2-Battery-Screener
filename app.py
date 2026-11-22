"""Visual simulator: run the hydrogen battery search, see the trade-offs, and
compare every candidate with the lithium-ion packs in current cars.

    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import copy
import json
import math
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from h2screener import ai_search, gas_search, gases, design, discover, explain, options, pairs, rank, reference, runs, stackcost, synthesis, animation
from h2screener import molecules as M

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Hydrogen battery simulator", layout="wide")
# compact layout: the top-right toolbar floats over the page instead of taking a full-width band
st.markdown("""<style>
.js-plotly-plot path.point[style*="rgb(225, 29, 73)"] {stroke-dasharray: 5 4 !important;}
.js-plotly-plot .legend path.scatterpts[style*="rgb(225, 29, 73)"] {stroke-dasharray: 3.5 2.5 !important;}
header[data-testid="stHeader"] {background: transparent; height: 0; min-height: 0; pointer-events: none;}
header[data-testid="stHeader"] [data-testid="stToolbar"], header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a {pointer-events: auto;}
[data-testid="stToolbar"] {top: 0.4rem; right: 0.6rem; background: rgba(17,24,39,0.85); border-radius: 10px;
                           padding: 0 0.3rem; backdrop-filter: blur(4px);}
[data-testid="stDecoration"] {display: none;}
[data-testid="stMainBlockContainer"], .block-container {padding-top: 1.2rem !important;}
[data-testid="stSidebarHeader"] {height: 2.2rem; min-height: 2.2rem; padding-top: 0.4rem; padding-bottom: 0;}
[data-testid="stSidebarUserContent"] {padding-top: 0 !important;}
h1 {padding-top: 0 !important; margin-top: 0 !important;}
[data-testid="stMetricValue"] {font-size: 1.45rem !important; line-height: 1.25;}
[data-testid="stMetricValue"] > div {white-space: normal !important; overflow: visible !important;
                                     text-overflow: clip !important; overflow-wrap: anywhere;}
[data-testid="stMetricDelta"] > div {white-space: normal !important; overflow: visible !important;
                                     text-overflow: clip !important;}
</style>""", unsafe_allow_html=True)

BASE = tomllib.load(open(ROOT / "targets.toml", "rb"))
REF = reference.load()
LI = reference.pack_targets(REF)

PRESETS = {
    "Stationary storage (100 kW for 6 h)": dict(power_kw=100.0, hours=6.0, charge_hours_target=6.0),
    "Car pack (75 kWh, 250 kW peak)": dict(power_kw=250.0, hours=0.3, charge_hours_target=0.5),
}

# ------------------------------------------------------------------ controls

SS = st.session_state
SS["_full_run"] = True                      # set to False at the end of every full page run
SETTING_KEYS = ["cfg_tab", "mol_live", "pair_live", "cfg_preset", "cfg_power", "cfg_hours", "cfg_pmax", "cfg_pmin", "cfg_rte", "cfg_life", "cfg_elec",
                "cfg_topn", "cfg_scale", "watchlist", "mol_gens", "mol_batch", "mol_heavy", "mol_kappa",
                "pair_gens", "pair_batch", "pair_heavy", "pair_kappa", "cfg_kind", "gas_live",
                "gas_gens", "gas_batch", "gas_heavy", "gas_kappa"] + [f"w_{k}" for k in BASE["ranking"]
                                                                         if isinstance(BASE["ranking"][k], (int, float))]
if "_settings_loaded" not in SS:            # restore the last settings after a page reload
    for k, v in runs.load_settings().items():
        if k in SETTING_KEYS:
            SS[k] = v
    SS["_settings_loaded"] = True


@st.cache_resource
def search_controls():
    runs.mark_interrupted()          # runs once per app start
    return {"positive": ai_search.Control(), "pair": ai_search.Control()}

@st.cache_resource
def gas_control():
    return ai_search.Control()


def reset_everything():
    """Sidebar 'Reset' (a button callback: runs before the page is rebuilt)."""
    msgs = []
    if SS.get("rs_stop", True):
        ctls = list(search_controls().values()) + [gas_control()]
        n_ = 0
        for c in ctls:
            if c.thread is not None and c.thread.is_alive():
                c.stopped, c.paused = True, False
                n_ += 1
        msgs.append(f"{n_} running search(es) asked to stop (after the molecule in progress)" if n_ else
                    "no search was running")
    if SS.get("rs_data", False):
        ai_search.bump_epoch()                    # searches still finishing may not write the old results back
        b1, b2 = ai_search.reset_store(), gas_search.reset()
        msgs.append("AI results moved to backups" + (f": {', '.join(str(x).split('/')[-1] for x in (b1, b2) if x)}"
                                                     if (b1 or b2) else " (nothing to move)"))
    if SS.get("rs_settings", False):
        for k in SETTING_KEYS:
            SS.pop(k, None)
        SS["watchlist"] = []
        runs.save_settings({})
        msgs.append("candidates and settings cleared")
    if SS.get("rs_cache", True):
        st.cache_data.clear()
        if not SS.get("rs_stop", True):
            msgs.append("computed tables cleared (kept the search controls)")
        else:
            st.cache_resource.clear()
            msgs.append("all computed results cleared")
    SS["_reset_msg"] = "; ".join(msgs)


def default(key: str, value):
    """Give a widget its default once; afterwards its own (saved) value is used."""
    if key not in SS:
        SS[key] = value


def apply_preset():
    pre_ = PRESETS[SS["cfg_preset"]]
    SS["cfg_power"], SS["cfg_hours"] = pre_["power_kw"], pre_["hours"]


def restore_settings(rec: dict):
    for k, v in rec.get("settings", {}).items():
        if k in SETTING_KEYS:
            SS[k] = v


with st.sidebar:
    st.header("What to build")
    default("cfg_preset", list(PRESETS)[0])
    preset = st.selectbox("Use", list(PRESETS), key="cfg_preset", on_change=apply_preset)
    pre = PRESETS[preset]
    default("cfg_power", pre["power_kw"])
    default("cfg_hours", pre["hours"])
    power = st.number_input("Power, kW", 1.0, 100000.0, key="cfg_power")
    hours = st.number_input("Delivers full power for, hours", 0.05, 200.0, key="cfg_hours")
    st.caption(f"Stored energy: {power * hours:,.0f} kWh")
    default("cfg_pmax", int(BASE["use"]["p_max_bar"]))
    p_max = st.slider("Highest hydrogen pressure, bar", 5, 200, key="cfg_pmax")
    default("cfg_pmin", int(BASE["use"]["p_min_bar"]))
    p_min = st.slider("Pressure left when empty, bar (gas tanks only)", 1, 20, key="cfg_pmin",
                      help="Sealed gas tanks swing between this and the highest pressure as they discharge and "
                           "charge. Kept above 1 bar so a leak never pulls air in, and so the hydrogen electrode "
                           "still delivers power at the end of discharge. Hydride and piston storage hold one "
                           "constant pressure and ignore it.")
    default("cfg_rte", BASE["use"]["rte_target"])
    rte = st.slider("Round-trip efficiency it must keep", 0.5, 0.95, step=0.01, key="cfg_rte")
    default("cfg_life", BASE["design"]["lifetime_years"])
    life = st.slider("Lifetime, years", 5, 40, key="cfg_life")
    default("cfg_elec", BASE["design"]["electricity_usd_kwh"])
    elec = st.number_input("Charging electricity, USD/kWh", 0.0, 1.0, key="cfg_elec")
    with st.expander("Ranking weights"):
        st.caption("How much each criterion counts in the ranking score, relative to the others: "
                   "0 = ignored, 5 = most important. Only ratios matter: cost 3 and mass 0.5 means cost "
                   "counts 6x as much as weight. They change the order of the list, not the physics.")
        WEIGHT_HELP = {
            "cost": ("Cost", "Build cost per kWh (materials, tank, stack, catalyst, electronics)."),
            "efficiency": ("Efficiency", "Share of the charging energy you get back."),
            "safety": ("Safety", "Penalties for toxic or radioactive species, water splitting, unstable chemistry."),
            "charge_speed": ("Charge speed", "How fast it can be recharged compared with your target."),
            "reversibility": ("Rechargeability", "Penalty when discharge makes very stable products (N2, CO2...) that are hard to turn back."),
            "closure": ("No maintenance", "Penalty when side reactions slowly move material out of the couple (needs topping up)."),
            "supply": ("Supply risk", "Penalty when an element is mined in only a few countries."),
            "volume": ("Volume", "Litres per kWh of the whole system."),
            "mass": ("Weight", "kg per kWh of the whole system (tank, hydrogen, chemicals, stack). Raise it for vehicles."),
        }
        for k in WEIGHT_HELP:
            default(f"w_{k}", float(BASE["ranking"][k]))
        weights = {k: st.slider(WEIGHT_HELP[k][0], 0.0, 5.0, step=0.5, help=WEIGHT_HELP[k][1], key=f"w_{k}")
                   for k in WEIGHT_HELP}
    default("cfg_scale", BASE["design"]["production_scale"])
    st.radio("Built as", list(stackcost.SCALES), key="cfg_scale", format_func=stackcost.SCALES.get,
             help="Today's parts: priced like current fuel-cell and flow-battery hardware. Purpose-built: a "
                  "battery designed for its one chemistry, at the published cost targets (US DOE) for its stack, "
                  "power electronics and tanks. Li-ion prices are for mass production.")
    with st.expander("Prices the model uses (estimated, with sources)"):
        for kind_, label_ in (("gas", "Cell with a hydrogen-gas electrode"), ("liquid", "Two-liquid cell (two tanks)")):
            e_ = stackcost.stack(kind_, SS["cfg_scale"])
            st.markdown(f"**{label_}:** {e_.range_text('USD per m2 of cell')}  \n<small>{e_.basis}</small>",
                        unsafe_allow_html=True)
        pe_ = stackcost.power_electronics(SS["cfg_scale"])
        st.markdown(f"**Power electronics:** {pe_.range_text('USD per kW')}  \n<small>{pe_.basis}</small>",
                    unsafe_allow_html=True)
        st.markdown("**Tanks, pumps, pipes (liquid cells):** itemised prices of a built 10 kW flow battery "
                    "(Noack et al. 2016: tanks 1.1 EUR per litre, pumps, piping, valves, heat exchangers, sensors), "
                    "scaled with size^0.6.")
        if stackcost.TANK_COST_FACTOR.get(SS["cfg_scale"], 1.0) != 1.0:
            st.markdown(f"**Carbon-fibre tanks:** x{stackcost.TANK_COST_FACTOR[SS['cfg_scale']]:g} "
                        "(US DOE ultimate target 8 USD/kWh vs ~16-17 today for 700 bar tanks).")
        st.markdown("**Molecules:** estimated for each molecule from how hard it is to make (shown in its details).")
    default("cfg_topn", 20)
    top_n = st.slider("Candidates to size in detail", 5, 60, key="cfg_topn")
    st.caption("Settings are saved automatically and come back after a page reload.")
    with st.popover("Reset...", width="stretch",
                    help="Stop searches, clear computed results, or start over. Nothing is deleted for good."):
        st.checkbox("Stop running searches", value=True, key="rs_stop")
        st.checkbox("Clear computed results (the next page load recomputes them, about 2 min)", value=True,
                    key="rs_cache")
        st.checkbox("Clear my ⭐ candidates and all settings", value=False, key="rs_settings")
        st.checkbox("Forget the AI search results (molecules, pairs, gas molecules); kept as dated backup files "
                    "in cache/", value=False, key="rs_data")
        st.button("Reset now", type="primary", on_click=reset_everything, width="stretch", key="rs_go")

def make_targets(s: dict) -> dict:
    """Model inputs from the sidebar values (also used by the one-setting sweep)."""
    t_ = copy.deepcopy(BASE)
    pmx, pmn = float(s["p_max"]), float(s["p_min"])
    t_["use"].update(p_max_bar=pmx, p_min_bar=pmn, rte_target=float(s["rte"]),
                     charge_hours_target=s["charge_hours_target"],
                     pressure_grid_bar=sorted({x for x in [3, 5, 10, 20, 30, 45, 60, 100, 150, 200]
                                               if pmn < x <= pmx} | {pmx}))
    t_["design"].update(power_kw=float(s["power"]), hours=float(s["hours"]), lifetime_years=int(s["life"]),
                        electricity_usd_kwh=float(s["elec"]), production_scale=s["scale"],
                        stack_usd_m2=stackcost.stack("gas", s["scale"]).usd,
                        stack_usd_m2_liquid=stackcost.stack("liquid", s["scale"]).usd,
                        power_electronics_usd_kw=stackcost.power_electronics(s["scale"]).usd)
    t_["refuel"]["composite_usd_per_kg"] *= stackcost.TANK_COST_FACTOR.get(s["scale"], 1.0)
    t_["ranking"].update(s["weights"])
    return t_


def targets_at(scale: str) -> dict:
    """The current settings, priced at another level ('mass' = today's parts, 'target' = purpose-built)."""
    return make_targets({**SETTINGS_NOW, "scale": scale})


SETTINGS_NOW = dict(power=power, hours=hours, p_max=p_max, p_min=p_min, rte=rte, life=life, elec=elec,
                    scale=SS["cfg_scale"], weights=weights, charge_hours_target=pre["charge_hours_target"])
t = make_targets(SETTINGS_NOW)
E_KWH = power * hours
runs.save_settings({k: SS[k] for k in SETTING_KEYS if k in SS})   # now, not only at the end of a (long) run


MODEL_VERSION = 8   # bump when model outputs change shape, so cached results are recomputed


def tkey(t_: dict) -> str:
    return json.dumps({**t_, "_model_version": MODEL_VERSION}, sort_keys=True)


@st.cache_data(show_spinner=False)
def screen(targets_repr: str):
    tt = json.loads(targets_repr)
    cells = discover.table_pairs()
    return rank.rank(cells, tt)


@st.cache_data(show_spinner=False)
def size(targets_repr: str, names: tuple[str, ...], extra_smiles: tuple[str, ...]):
    tt = json.loads(targets_repr)
    rng = np.random.default_rng(3)
    by_name = {c.name: c for c in discover.table_pairs()}
    cells = [by_name[n] for n in names if n in by_name]
    for s in extra_smiles:
        r = M.evaluate(s)
        if math.isfinite(r.e0):
            cells.append(M.as_cell(r))
    return [design.design(c, tt, rng) for c in cells]


def design_rows(designs):
    rows = []
    for d in designs:
        b = d.best
        if b is None:
            continue
        cmp = reference.compare(b, E_KWH, REF)
        rows.append({
            "couple": d.cell.name, "origin": d.cell.origin, "medium": d.cell.medium,
            "E0 V": round(d.cell.e0, 3),
            "best p bar": b.p_max_bar,
            "Wh/kg (system)": round(E_KWH * 1000 / b.mass_kg, 1),
            "Wh/L (system)": round(E_KWH * 1000 / b.volume_l, 1),
            "build USD/kWh": round(b.capex_usd_kwh),
            "LCOS USD/kWh": round(b.lcos_usd_kwh, 3),
            "RTE p50": round(b.rte_band[1], 2), "RTE p10": round(b.rte_band[0], 2), "RTE p90": round(b.rte_band[2], 2),
            "fastest charge h": round(b.min_charge_h, 2),
            "kg per kWh": round(b.mass_kg / E_KWH, 2),
            "kg per kW": round(b.mass_kg / power, 2),
            "stack kg per kW": round(b.parts_kg.get("stack", 0) / power, 3),
            "burst kg TNT": round(b.safety.burst_tnt_kg, 2),
            "leak g/s (1 mm2)": round(b.safety.leak_g_s, 2),
            "H2 refill years": round(b.years_to_refill) if math.isfinite(b.years_to_refill) else "never",
            "chemistry drifts": "; ".join(d.drift) or "no",
            **{f"vs Li-ion: {k}": v for k, v in cmp.items()},
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def option_table(targets_json: str, name: str | None, smiles: tuple[str, ...]):
    tt = json.loads(targets_json)
    cells = discover.table_pairs()
    rng = np.random.default_rng(5)
    cfgs = []
    if name:
        by = {c.name: c for c in cells}
        if name in by:
            cfgs += [(name, c) for c in options.configurations(by[name], tt, rng)]
        else:
            for s in smiles:
                r = M.evaluate(s)
                if r.smiles == name or s == name:
                    cfgs += [(name, c) for c in options.configurations(M.as_cell(r), tt, rng)]
    cfgs += [("H2-air fuel cell", c) for c in options.refuel_configurations(cells, tt, rng)]
    return cfgs


def recharge_options(name: str, kp: str):
    """Every way to store the hydrogen and to recharge this couple, for the same job, vs lithium-ion."""
    st.caption("How the hydrogen gets back into the battery: **electric** = sealed, recharged with electricity "
               "only; **h2-refuel** = fuel cell on air, tank refilled with hydrogen made elsewhere (efficiency "
               "counts electrolysis, compression and the fuel cell); **swap** = hydrogen refill plus exchange of "
               "the charged liquid. A plain hydrogen-air fuel cell is added for reference.")
    with st.spinner("Sizing every storage and recharge option..."):
        cfgs = option_table(tkey(t), name, ())
    if not cfgs:
        st.info("No option can be sized for these settings.")
        return
    rows = [{"option": c.name if n_ == name else f"{n_}: {c.name}", "pressure": c.store.pressure,
             "Wh/kg": round(c.wh_kg, 1), "Wh/L": round(c.wh_l, 1), "build USD/kWh": round(c.capex_usd_kwh),
             "full battery USD": round(c.capex_usd_kwh * E_KWH), "lifetime USD/kWh": round(c.lcos_usd_kwh, 3),
             "efficiency": f"{c.rte:.2f} ({c.rte_band[0]:.2f}-{c.rte_band[2]:.2f})",
             "recharge min": round(c.recharge_h * 60), "burst kg TNT": round(c.store.burst_tnt_kg, 3),
             "leak g/s": round(c.store.leak_g_s, 3), "H2 loss %/yr": round(c.store.h2_loss_per_year * 100, 4),
             "notes": "; ".join(c.store.notes + c.notes)} for n_, c in cfgs]
    li_usd, li_name = li_price_now()
    rows.append({"option": li_name, "Wh/kg": LI["wh_kg"], "Wh/L": LI["wh_l"], "build USD/kWh": li_usd,
                 "full battery USD": round(li_usd * E_KWH), "recharge min": round(LI["charge_min_10_80"] / 0.7)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Li-ion recharge time is the 10-80 % fast-charge time scaled to a full charge. Refuel options do "
               "not include the off-board electrolyser and station in the build cost.")


def render_couple(d, kp: str):
    """Everything about one sized couple (used by the detail tab and the pop-up)."""
    novelty_banner("couple", [d.cell.name])
    b = d.best
    st.write(f"**{d.cell.name}** — {d.cell.origin}, {d.cell.medium}; E0 {d.cell.e0:.3f} V; data: {d.cell.source}")
    cmp = reference.compare(b, E_KWH, REF)
    cols = st.columns(len(cmp))
    for c, (k, v) in zip(cols, cmp.items()):
        c.metric(f"vs Li-ion: {k}", v)
    P = pd.DataFrame([{"p_max bar": p.p_max_bar, "mass kg": p.mass_kg, "volume L": p.volume_l,
                       "build USD/kWh": p.capex_usd_kwh, "LCOS USD/kWh": p.lcos_usd_kwh,
                       "burst kg TNT": p.safety.burst_tnt_kg, "leak g/s": p.safety.leak_g_s,
                       "H2 loss %/yr": p.h2_loss_per_year * 100, "RTE": p.rte} for p in d.points])
    fig = make_subplots(rows=2, cols=4, subplot_titles=list(P.columns[1:]))
    for k, col in enumerate(P.columns[1:]):
        fig.add_trace(go.Scatter(x=P["p_max bar"], y=P[col], mode="lines+markers", name=col),
                      row=k // 4 + 1, col=k % 4 + 1)
    fig.update_layout(height=520, showlegend=False, title="Everything vs maximum hydrogen pressure")
    st.plotly_chart(fig, width="stretch", key=f"{kp}_press")
    a, c = st.columns(2)
    pw, hr = d.ragone
    rg = go.Figure(go.Scatter(x=pw, y=hr, mode="lines", name="this design"))
    rg.add_vline(x=math.log10(power), line_dash="dot", annotation_text="design power")  # log axis: exponent
    rg.update_layout(xaxis_title="power delivered, kW", yaxis_title="how long, hours",
                     xaxis_type="log", yaxis_type="log", title="Power delivery vs duration (built system)")
    a.plotly_chart(rg, width="stretch", key=f"{kp}_rag")
    cv = d.curve
    ef = go.Figure()
    ef.add_trace(go.Scatter(x=cv["i"], y=cv["rte"], name="round-trip efficiency"))
    ef.add_trace(go.Scatter(x=cv["i"], y=cv["v_dis"], name="discharge voltage, V", yaxis="y2"))
    ef.update_layout(xaxis_type="log", xaxis_title="current density, A/cm2",
                     yaxis2=dict(overlaying="y", side="right"), title=f"Efficiency and voltage ({cv['p_bar']} bar, nominal inputs)")
    c.plotly_chart(ef, width="stretch", key=f"{kp}_eff")
    show_cost(b.cost_parts, {"i_a_cm2": b.i_a_cm2, "area_m2": b.area_m2, "p_max_bar": b.p_max_bar,
                             "v_dis": b.v_dis, "synth": b.synth}, kp)
    names = {"active": "chemicals", "electrolyte": "water / electrolyte", "vessel": "hydrogen tank (steel)",
             "hydrogen": "hydrogen gas", "stack": "cell stack"}
    rows = []
    for p in d.points:
        for k, v in p.parts_kg.items():
            rows.append({"max pressure, bar": p.p_max_bar, "part": names.get(k, k), "kg per kWh": v / E_KWH})
    wf = px.bar(pd.DataFrame(rows), x="max pressure, bar", y="kg per kWh", color="part",
                title="Weight per kWh stored vs maximum hydrogen pressure: the tank gets smaller but thicker-walled, "
                      "so its weight levels off")
    wf.add_hline(y=1000 / LI["wh_kg"], line_dash="dash", line_color="red",
                 annotation_text=f"Li-ion pack {1000 / LI['wh_kg']:.1f} kg/kWh")
    wf.update_xaxes(type="category")
    st.plotly_chart(wf, width="stretch", key=f"{kp}_wparts")
    if any(p.tank_volume_l for p in d.points):
        st.markdown("**Gas and tank vs pressure** (one cylindrical steel tank, length = 3 x diameter)")
        st.caption("Wall thickness = pressure x radius / allowed steel stress. Higher pressure: smaller tank, "
                   "thicker wall; the two almost cancel in the steel weight. At low pressure much of the tank "
                   "holds gas that never leaves (the 'empty' pressure), so the tank must be larger.")
        st.dataframe(pd.DataFrame([{
            "max pressure, bar": p.p_max_bar,
            "hydrogen gas, kg/kWh": round(p.h2_kg / E_KWH, 4),
            "tank volume, L/kWh": round(p.tank_volume_l / E_KWH, 1),
            "tank diameter, m": round(p.tank_diameter_m, 2),
            "wall thickness, mm": round(p.wall_mm, 1),
            "tank steel, kg/kWh": round(p.parts_kg.get("vessel", 0) / E_KWH, 2),
            "tank + gas, kg/kWh": round((p.parts_kg.get("vessel", 0) + p.h2_kg) / E_KWH, 2)}
            for p in d.points]), hide_index=True, width="stretch")
    k1, k2, k3 = st.columns(3)
    k1.metric("Weight per kWh stored", f"{b.mass_kg / E_KWH:.1f} kg/kWh", f"{b.mass_kg / E_KWH - 1000 / LI['wh_kg']:+.1f} vs Li-ion",
              delta_color="inverse")
    k2.metric("Weight per kW of power", f"{b.mass_kg / power:.1f} kg/kW",
              help="Whole system divided by its power. For long storage this is dominated by the stored energy, not the power.")
    k3.metric("Cell stack per kW", f"{b.parts_kg.get('stack', 0) / power:.2f} kg/kW",
              help="Only the part that sets the power: the stack of cells. The tank does not change it.")
    st.plotly_chart(px.pie(names=list(b.parts_kg), values=list(b.parts_kg.values()),
                           title=f"Mass at {b.p_max_bar:g} bar: {b.mass_kg:,.0f} kg"), width="content", key=f"{kp}_pie")
    st.markdown("**Maintenance and chemical closure**")
    st.write("Inventory drift: " + ("; ".join(d.drift) if d.drift else "no drift path found in the data (measured and derived half-reactions)"))
    st.write("Recombination: " + "; ".join(d.recombination))
    st.write(f"H2 loss {b.h2_loss_per_year * 100:.4f} %/yr (permeation upper bound + seals) → refill after "
             f"{b.years_to_refill:,.0f} years")
    st.markdown("**Inputs and where they come from**")
    st.dataframe(pd.DataFrame([{"input": k, "kind": v.kind, "low": v.low, "high": v.high, "source": v.source}
                               for k, v in d.params.items()]), hide_index=True, width="stretch")
    with st.expander("Other ways to recharge it (sealed, refuel with hydrogen, swap the liquid)"):
        recharge_options(d.cell.name, kp)




def ck(key: str) -> str:
    """Widget key of a clickable chart or table (changes after each click)."""
    return f"{key}_{st.session_state.get(f'nonce_{key}', 0)}"


def _points(event):
    try:
        sel = event.selection
        return sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", [])
    except Exception:
        return []


NEW_RING = "#ffd400"
NEW_RED = "#e11d49"   # this exact colour is dashed by the CSS at the top of the page


def add_new_rings(fig, xs, ys, cds, size=20, width=3):
    """Yellow ring around every NEW molecule (not in PubChem); clicks on it open the same candidate."""
    if not len(xs):
        return
    xs, ys, cds = list(xs), list(ys), [list(c) for c in cds]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", customdata=cds, legendgroup="newring",
                             marker=dict(symbol="circle-open", size=size, color=NEW_RING, line=dict(width=width, color=NEW_RING)),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", customdata=cds, legendgroup="newring",
                             marker=dict(symbol="circle-open", size=size, color=NEW_RED, line=dict(width=width, color=NEW_RED)),
                             showlegend=False, hoverinfo="skip"))   # dashed by the CSS at the top of the page
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", legendgroup="newring", hoverinfo="skip",
                             marker=dict(symbol="circle", size=14, color=NEW_RING, line=dict(width=2.5, color=NEW_RED)),
                             name="yellow ring, red dashes = NEW molecule (never reported in PubChem)"))   # legend only


def take(event, key: str, kind: str):
    """A chart was clicked: remember which candidate; the pop-up opens at the
    end of the page run. The chart is redrawn unselected next time."""
    pts = _points(event)
    if pts:
        SS[f"nonce_{key}"] = SS.get(f"nonce_{key}", 0) + 1     # always consume the click
        cd = pts[0].get("customdata")
        if cd is not None:
            SS["pending_dialog"] = (kind, list(cd) if isinstance(cd, (list, tuple)) else [cd])
            open_now()


def take_any(event, key: str):
    """Like take(), for charts that mix kinds: each point's customdata is [kind, *ids]."""
    pts = _points(event)
    if pts:
        SS[f"nonce_{key}"] = SS.get(f"nonce_{key}", 0) + 1
        cd = pts[0].get("customdata")
        if isinstance(cd, (list, tuple)) and len(cd) >= 2 and cd[0]:
            SS["pending_dialog"] = (cd[0], [x for x in cd[1:] if x not in (None, "")])
            open_now()


def take_row_any(event, key: str):
    """A table row was clicked in a table that mixes kinds: SS['rows_<key>'] holds [kind, *ids] per row."""
    try:
        sel = event.selection
        rows = sel.get("rows", []) if isinstance(sel, dict) else getattr(sel, "rows", [])
        if not rows:                                   # a click on a cell counts as a click on its row
            cells = sel.get("cells", []) if isinstance(sel, dict) else getattr(sel, "cells", [])
            rows = [c[0] for c in cells]
    except Exception:
        rows = []
    values = SS.get(f"rows_{key}", [])
    if rows:
        SS[f"nonce_{key}"] = SS.get(f"nonce_{key}", 0) + 1
        if rows[0] < len(values):
            v = values[rows[0]]
            SS["pending_dialog"] = (v[0], list(v[1:]))
            open_now()


def take_row(event, key: str, kind: str):
    try:
        sel = event.selection
        rows = sel.get("rows", []) if isinstance(sel, dict) else getattr(sel, "rows", [])
        if not rows:                                   # a click on a cell counts as a click on its row
            cells = sel.get("cells", []) if isinstance(sel, dict) else getattr(sel, "cells", [])
            rows = [c[0] for c in cells]
    except Exception:
        rows = []
    values = SS.get(f"rows_{key}", [])
    if rows:
        SS[f"nonce_{key}"] = SS.get(f"nonce_{key}", 0) + 1     # always consume the click
        if rows[0] < len(values):
            SS["pending_dialog"] = (kind, list(values[rows[0]]))
            open_now()


DIALOG_WHAT = {"couple": "couple", "molecule": "molecule", "pair": "pair", "gasmol": "gas molecule",
               "gascell": "sealed gas cell"}


def open_now():
    """A click was taken: say so at once, and open the pop-up now instead of after the rest of the
    page (during a panel's own 2-second refresh the full page run opens it instead)."""
    pend = SS.get("pending_dialog")
    if not pend:
        return
    kind, v = pend
    st.toast(f"Opening the {DIALOG_WHAT.get(kind, kind)} {' | '.join(str(x) for x in v[:2])}... "
             "(computing its details, a few seconds)", icon="⏳")
    if SS.get("_full_run"):
        open_pending_dialog()


CLICKABLE = ["tradeoff", "best_all", "best_tbl", "f1", "f2", "f3", "molsc", "mol_tbl", "pairsc", "pair_tbl", "mol_prog", "pair_prog",
             "gas_prog", "gas_sc", "gas_tbl"]


def click_waiting() -> bool:
    """True when a chart or table holds a click the page has not acted on yet."""
    for k in CLICKABLE:
        stt = SS.get(ck(k))
        if not stt:
            continue
        try:
            sel = stt.get("selection", {}) if isinstance(stt, dict) else getattr(stt, "selection", {})
            if (sel.get("points") or sel.get("rows")):
                return True
        except Exception:
            continue
    return False


def running_any() -> bool:
    return any(c.thread is not None and c.thread.is_alive() for c in search_controls().values())


def open_pending_dialog():
    pend = SS.pop("pending_dialog", None)
    if not pend:
        return
    kind, v = pend
    if kind == "couple":
        couple_dialog(v[0])
    elif kind == "molecule":
        molecule_dialog(v[0])      # extra ids (if any) are ignored
    elif kind == "pair":
        pair_dialog(v[0], v[1])
    elif kind == "gasmol":
        gas_molecule_dialog(v[0])
    elif kind == "gascell":
        gas_cell_dialog(v[0])


@st.cache_data(show_spinner=False)
def molecule_options(targets_json: str, smiles: str):
    tt = json.loads(targets_json)
    r = ai_search.load_store().get(smiles) or M.evaluate(smiles)
    return options.configurations(M.as_cell(r), tt, np.random.default_rng(11)) if math.isfinite(r.e0) else []


def show_cost(parts: dict, ctx: dict, kp: str):
    """Build cost split by part, and why it is high (with the setting to change)."""
    if not parts:
        return
    ctx = {"e_kwh": E_KWH, "power_kw": power, "li_usd_kwh": LI["usd_kwh_stationary"],
           "stack_est": stackcost.stack(ctx.pop("stack_kind", "gas"), t["design"]["production_scale"]), **ctx}
    df_ = pd.DataFrame([{"part": k, "USD per kWh": v / E_KWH} for k, v in parts.items() if v > 0])
    fc = px.bar(df_.sort_values("USD per kWh"), x="USD per kWh", y="part", orientation="h",
                title=f"Where the build cost goes ({sum(parts.values()) / E_KWH:,.0f} USD/kWh)")
    fc.add_vline(x=LI["usd_kwh_stationary"], line_dash="dash", line_color="red",
                 annotation_text=f"Li-ion pack {LI['usd_kwh_stationary']:.0f}")
    fc.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fc, width="stretch", key=f"{kp}_cost")
    st.markdown("**Why it costs this much**\n" + "\n".join(f"- {x}" for x in explain.explain_cost(parts, ctx)))


def band_text(band) -> str:
    """'lo-hi' build cost in USD/kWh over the reaction-speed range."""
    if not band:
        return ""
    f = lambda x: f"{x:,.0f}" if math.isfinite(x) else "not buildable"
    return f"{f(band[0])}-{f(band[2])}"


PRICE_LEVELS = {"mass": "today's parts", "target": "purpose-built", "both": "both (line = saving)"}


def li_price_now() -> tuple[float, str]:
    """Li-ion price to compare with: the car pack for a car, the stationary pack otherwise."""
    if "car" in str(SS.get("cfg_preset", "")).lower():
        return LI["usd_kwh_bev"], "Li-ion car pack"
    return LI["usd_kwh_stationary"], "Li-ion stationary pack"



# ------------------------------------------------------------------ known or new? and is it interesting?

from urllib.parse import quote as _q


@st.cache_data(show_spinner=False)
def molecule_known(smiles: str) -> bool | None:
    """True: a known compound (measured data, or found in PubChem); False: never reported; None: not checked."""
    if smiles in gases.MEASURED or smiles in ("O", "[H][H]"):
        return True
    st_ = ai_search.load_store().get(smiles)
    if st_ is not None and st_.in_pubchem is not None:
        return st_.in_pubchem
    try:
        return M.in_pubchem(smiles)
    except Exception:
        return None


def _mol_label(smiles: str) -> str:
    try:
        return gases.formula_of(smiles) + f" ({smiles})"
    except Exception:
        return smiles


@st.cache_data(show_spinner=False)
def novelty(kind: str, ids: tuple) -> dict:
    """Is each molecule known or new, and has this battery chemistry been shown to work?"""
    mols, battery, refs = [], ("unknown", "no experimental record in the model"), []
    if kind == "molecule":
        mols = [ids[0]]
        battery = ("not demonstrated", "this molecule against a hydrogen electrode: no record in the model")
    elif kind == "pair":
        mols = list(ids[:2])
        battery = ("not demonstrated", "this two-molecule cell: no record in the model")
    elif kind == "gascell":
        c, _ = gas_cell_by_key(ids[0])
        if c is not None:
            mols = [x for x in c.stoich if x not in ("O", "[H][H]")]
            battery = (c.recharge, "; ".join(f"{side}: {why}" for (_, why), side in
                                            zip(c.evidence, ("oxidant side", "fuel side"))))
    elif kind == "couple":
        battery = ("known reactions", "both half-reactions come from published standard-potential tables or "
                                      "Materials Project; whether they run as a rechargeable cell is not recorded")
    status = {m: molecule_known(m) for m in mols}
    return {"molecules": status, "new": [m for m, k in status.items() if k is False],
            "unchecked": [m for m, k in status.items() if k is None], "battery": battery}


def novelty_tag(kind: str, ids: list) -> str:
    """Short label for tables and hovers."""
    n = novelty(kind, tuple(ids))
    if n["new"]:
        return "🆕 NEW molecule"
    if kind == "couple":
        return "✔ known reactions"
    return "✔ known molecules" if not n["unchecked"] else "? not checked"


def novelty_banner(kind: str, ids: list, skip: tuple = ()):
    """skip: molecules the page has already announced (their known/new line is not repeated)."""
    n = novelty(kind, tuple(ids))
    new_ = [m for m in n["new"] if m not in skip]
    if new_:
        st.success("🆕 **NEW: contains a molecule never reported** (not in PubChem): "
                   + ", ".join(_mol_label(m) for m in new_)
                   + ". Its energy comes from quantum chemistry only.")
    elif n["new"] or skip:
        pass                                            # already said at the top of the page
    elif kind == "couple":
        st.info("✔ **KNOWN chemistry**: " + n["battery"][1] + ".")
    elif n["molecules"]:
        st.info("✔ **KNOWN molecules**: every molecule is a known compound (measured data or PubChem)"
                + (f"; not checked: {', '.join(n['unchecked'])}" if n["unchecked"] else "") + ".")
    kind_b, why = n["battery"]
    icon = {"demonstrated": "✔", "both directions separately": "◐", "one direction": "⚠"}.get(kind_b, "❓")
    if kind != "couple":
        st.caption(f"{icon} **As a rechargeable battery: {kind_b}** — {why}")


def literature_links(kind: str, ids: list, name: str) -> str:
    mols = list(novelty(kind, tuple(ids))["molecules"])
    links = [f"[PubChem: {gases.formula_of(m) if kind == 'gascell' else m[:24]}]"
             f"(https://pubchem.ncbi.nlm.nih.gov/#query={_q(m)})" for m in mols[:4]]
    q = name if kind in ("gascell", "couple") else " ".join(mols[:2])
    links.append(f"[Google Scholar: '{q[:40]}' battery](https://scholar.google.com/scholar?q={_q(q + ' battery')})")
    return " · ".join(links)


def assessment(kind: str, ids: list, name: str, now_, tgt_):
    """'Is it interesting?' from the model's own numbers, with what the model cannot tell."""
    li_usd, li_name = li_price_now()
    b_ = tgt_ or now_
    wins, losses = [], []
    for lab, ok, txt in (("weight", b_.wh_kg > LI["wh_kg"], f"{b_.wh_kg:.0f} vs {LI['wh_kg']:.0f} Wh/kg"),
                         ("volume", b_.wh_l > LI["wh_l"], f"{b_.wh_l:.0f} vs {LI['wh_l']:.0f} Wh/L"),
                         ("build price", b_.capex_usd_kwh < li_usd,
                          f"${b_.capex_usd_kwh:,.0f} vs ${li_usd:.0f}/kWh, x{b_.capex_usd_kwh / li_usd:.1f}"),
                         ("recharge time", b_.recharge_h * 60 * 0.7 < LI["charge_min_10_80"],
                          f"{b_.recharge_h * 60:.0f} min full vs {LI['charge_min_10_80']:.0f} min 10-80 %")):
        (wins if ok else losses).append(f"{lab} ({txt})")
    n = novelty(kind, tuple(ids))
    st.markdown("#### Is it interesting?")
    lines = [f"**Against the {li_name}** ({'purpose-built' if tgt_ else 'today'}'s prices): "
             + (f"better on {', '.join(wins)}" if wins else "better on nothing")
             + (f"; worse on {', '.join(losses)}." if losses else ".")]
    if n["new"]:
        lines.append("**New molecule:** never reported, so it may not exist, may decompose, or may be hard to make. "
                     "Its energy is a quantum-chemistry estimate (±80 kJ/mol for gases, ±0.3 V for liquids).")
    else:
        lines.append("**Known molecules:** the chemistry is not new; check the literature below before treating it "
                     "as a discovery.")
    lines.append(f"**Efficiency {b_.rte:.0%}** against about {LI_RTE:.0%} for lithium-ion: it needs "
                 f"x{LI_RTE / max(b_.rte, 1e-3):.1f} the charging electricity for the same energy delivered.")
    kb, why = n["battery"]
    if kind != "couple":
        lines.append(f"**Recharging: {kb}.** {why}.")
    burst = getattr(getattr(b_, "store", None), "burst_tnt_kg", 0) or 0
    if burst > 0.05:
        lines.append(f"**Safety:** {burst:.2f} kg TNT of stored gas burst energy"
                     + ("; it keeps pure oxygen under pressure next to a fuel" if kind == "gascell"
                        and "O=O" in name.replace("O2(g)", "O=O") else "")
                     + ". The model does not price the protection this needs.")
    lines.append("**Lifetime:** not predicted by the model (catalyst and membrane wear, cycles); a real cell may "
                 "fade long before the years assumed in the lifetime cost.")
    if tgt_ and now_:
        lines.append(f"**Price:** ${tgt_.capex_usd_kwh:,.0f}/kWh is a target (US DOE cost goals for the parts); "
                     f"with today's parts it is ${now_.capex_usd_kwh:,.0f}/kWh.")
    if hours < 2:
        lines.append("**Use:** with short hours the power parts dominate the price; try the stationary preset "
                     "(long hours), where weight matters less and the price per kWh falls.")
    st.markdown("\n".join(f"- {x}" for x in lines))
    st.caption("Check the literature: " + literature_links(kind, ids, name))


def li_metrics(c, kp: str, at=None, kind: str | None = None, ids: list | None = None, name: str = ""):
    """Whole-battery figures against lithium-ion. at(scale) -> the same design priced at 'mass' (today's parts)
    or 'target' (purpose-built); with it the user can see either price level or both."""
    li_usd, li_name = li_price_now()
    view, a_, b_ = "now", c, None
    if at is not None:
        default(f"{kp}_price", "both")
        view = st.radio("Prices", list(PRICE_LEVELS), key=f"{kp}_price", horizontal=True, format_func=PRICE_LEVELS.get,
                        help="Today's parts: current fuel-cell and flow-battery hardware at mass production. "
                             "Purpose-built: designed for this chemistry, at the published US DOE cost targets "
                             "(sources in the sidebar).")
        try:
            got = {sc: at(sc) for sc in ("mass", "target")}
        except Exception:
            got = {}
        if view == "both" and got.get("mass") and got.get("target"):
            a_, b_ = got["mass"], got["target"]
        elif view in got and got[view]:
            a_ = got[view]
    show = b_ or a_
    arrow = lambda f_: (f"{f_(a_)} → {f_(b_)}" if b_ else f_(a_))
    cols = st.columns(5)
    cols[0].metric("Energy per kg", arrow(lambda x: f"{x.wh_kg:.0f}") + " Wh/kg", f"{show.wh_kg - LI['wh_kg']:+.0f} vs Li-ion")
    cols[1].metric("Energy per litre", arrow(lambda x: f"{x.wh_l:.0f}") + " Wh/L", f"{show.wh_l - LI['wh_l']:+.0f} vs Li-ion")
    if b_:
        cols[2].metric("Build cost", arrow(lambda x: f"${x.capex_usd_kwh:,.0f}") + "/kWh",
                       f"{b_.capex_usd_kwh - li_usd:+,.0f} vs {li_name}", delta_color="inverse",
                       help="Today's parts → purpose-built. The difference to Li-ion is for purpose-built.")
    else:
        v_, d_, h_ = explain.cost_range(show.capex_usd_kwh, show.capex_band, li_usd)
        cols[2].metric("Build cost", v_, d_.replace("vs Li-ion", f"vs {li_name}"), delta_color="inverse", help=h_)
    tot = lambda x: x.capex_usd_kwh * E_KWH
    money = lambda v: f"${v / 1000:,.1f}k" if v >= 10_000 else f"${v:,.0f}"
    cols[3].metric("Full battery price", arrow(lambda x: money(tot(x))),
                   f"{money(abs(tot(show) - li_usd * E_KWH))} {'more' if tot(show) > li_usd * E_KWH else 'less'} "
                   f"than {li_name} ({money(li_usd * E_KWH)})", delta_color="off",
                   help=f"The whole battery for {E_KWH:,.1f} kWh ({power:g} kW for {hours:g} h): cells, catalyst, "
                        f"tanks, chemicals and power electronics. {li_name}: {E_KWH:,.1f} kWh at ${li_usd:.0f}/kWh.")
    cols[4].metric("Recharge", f"{show.recharge_h * 60:.0f} min",
                   f"{show.recharge_h * 60 * 0.7 - LI['charge_min_10_80']:+.0f} vs Li-ion 10-80 %", delta_color="inverse")
    if kind:
        g_ = got if at is not None else {}
        assessment(kind, ids or [], name, g_.get("mass") or c, g_.get("target"))


def draw(smiles_list, legends):
    from rdkit import Chem
    from rdkit.Chem import Draw
    mols = [Chem.MolFromSmiles(x) for x in smiles_list]
    return Draw.MolsToGridImage(mols, molsPerRow=min(4, len(mols)), subImgSize=(260, 200), legends=legends)


def launch_verify(args: list) -> None:
    """Start a first-principles (DFT) recomputation in a separate background process."""
    import subprocess
    import sys as _sys
    log = ROOT / "scripts" / "dev" / "verify.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    subprocess.Popen([_sys.executable, str(ROOT / "scripts" / "verify_molecules.py"), *map(str, args)], cwd=str(ROOT),
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def launch_gas_fp() -> None:
    """DFT / CCSD(T) for every gas molecule not computed yet (skips itself if a batch is already running)."""
    import subprocess
    import sys as _sys
    log = ROOT / "scripts" / "dev" / "gas_fp.log"
    subprocess.Popen([_sys.executable, str(ROOT / "scripts" / "compute_gas_fp.py"), "3"], cwd=str(ROOT),
                     stdout=open(log, "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def dft_check(r, kp: str):
    """First-principles voltage of the first hydrogen uptake, or a button to compute it."""
    from h2screener import abinitio as _A
    red = r.forms[0] if r.forms else r.reduced
    if not red:
        return
    v = _A.hydrogenation_voltage(r.smiles, red, 1, "dft", compute=False)
    x = (r.steps or [r.e0])[0]
    if v is not None:
        st.caption(f"🔬 **First-principles check** ({_A.LEVELS['dft']}, no reference couple): first H2 at "
                   f"**{v:.2f} V** vs {x:.2f} V from the fast xTB search (difference {v - x:+.2f} V).")
    elif st.button("🔬 Verify with DFT (10-30 min, in the background)", key=f"{kp}_dft"):
        launch_verify([r.smiles])
        st.toast("DFT started in the background: reopen this molecule later to see the result.")


def stability_table(r):
    """Computed reactions with water of the charged and discharged forms."""
    from h2screener import stability as _S
    red = r.forms[0] if r.forms else r.reduced
    rows = []
    for state, x in (("charged", r.smiles), ("discharged", red)):
        if not x:
            continue
        verdict, rx = _S.summary(x)
        if not rx:
            rows.append({"form": state, "reaction with water": "none of the common reactions applies",
                         "dG kJ/mol": None, "verdict": "stable", "method": "-"})
        for q in rx:
            rows.append({"form": state, "reaction with water": q["reaction"], "products": " + ".join(q["products"]),
                         "dG kJ/mol": None if q["dG_kj"] is None else round(q["dG_kj"]),
                         "verdict": q["verdict"] or "not computed yet", "method": q["level"]})
    st.markdown("**Does it survive in water?** Computed reactions of water with the molecule "
                "(dG < -10 kJ/mol: it reacts; thermodynamics only, not how fast).")
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if any(x["verdict"] == "not computed yet" for x in rows):
        st.caption("Computed by '🔬 Verify with DFT' (with the voltage check).")


def render_molecule(smiles: str, kp: str, pin: bool = False):
    r = ai_search.load_store().get(smiles) or next((x for x in view_pool("mol") + view_pool("pair")
                                                    if x.smiles == smiles), None) or M.evaluate(smiles)
    steps = r.steps or [r.e0]
    sc = synthesis.estimate(r.smiles, t["synthesis"])
    e_mean = sum(steps[:r.best_k]) / max(r.best_k, 1)
    # header: picture | name, known/new, key numbers | ⭐ button
    h0, h1, h2 = st.columns([1.1, 3, 1.1], vertical_alignment="top")
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        h0.image(Draw.MolToImage(Chem.MolFromSmiles(r.smiles), size=(240, 180)))
    except Exception:
        pass
    with h1:
        st.markdown(f"**{r.smiles}**  ·  {r.mw:.0f} g/mol  ·  takes up to {len(steps)} H2, uses {r.best_k}")
        novelty_banner("molecule", [r.smiles])
        dft_check(r, kp)
        c = st.columns(5)
        c[0].metric("Voltage (used)", f"{e_mean:.2f} V", help="Average of the hydrogen uptakes used, V vs SHE.")
        c[1].metric("Solubility", f"{r.solubility_mol_l:.2f} M", help="ESOL estimate.")
        c[2].metric("Synthesis", f"{r.sa:.1f}/10", help="SA score: 1 = easy, 10 = very hard to make.")
        c[3].metric("Must cost <", f"${pairs.CHEMICAL_SHARE * LI['usd_kwh_stationary'] / pairs.kg_per_kwh(r.mw, max(e_mean, 0.05), 2 * r.best_k):.1f}/kg",
                    help="Price per kg above which the chemical alone costs more than a lithium-ion pack's share.")
        c[4].metric("Est. price", f"${sc.usd_kg:,.0f}/kg", help=sc.describe())
    if pin:
        with h2:
            pin_button("molecule", [r.smiles], kp)
    if r.notes:
        st.warning("; ".join(r.notes))
    cfgs = molecule_options(tkey(t), r.smiles)
    tb = st.tabs(["⚡ vs lithium-ion", "💰 Cost breakdown", "🔧 Build options", "🧪 Chemistry"])
    with tb[3]:
        a_, b_ = st.columns([3, 2])
        a_.image(draw([r.smiles] + r.forms, ["charged (oxidised)"] + [f"+{i + 1} H2 at {e:.2f} V" for i, e in enumerate(steps)]))
        sf = go.Figure(go.Bar(x=[f"H2 #{i + 1}" for i in range(len(steps))], y=steps))
        sf.add_hline(y=1.229 + 0.4, line_dash="dash", line_color="red", annotation_text="water splits above (O2)")
        sf.add_hline(y=-0.3, line_dash="dash", line_color="red", annotation_text="water splits below (H2)")
        sf.update_layout(title="Voltage of each hydrogen uptake (V vs SHE)", height=260, margin=dict(t=40, b=10))
        b_.plotly_chart(sf, width="stretch", key=f"{kp}_steps")
    with tb[3]:
        stability_table(r)
    if not cfgs:
        tb[0].info("Cannot be built: no voltage inside the water window or no usable hydrogen uptake.")
        return
    best = min(cfgs, key=lambda x: x.lcos_usd_kwh)
    with tb[0]:
        st.caption("Built as a battery against hydrogen. Every way to store the hydrogen and recharge is in "
                   "Build options.")

        def at_(sc_, sm=r.smiles):
            cf_ = molecule_options(tkey(targets_at(sc_)), sm)
            return min(cf_, key=lambda x: x.lcos_usd_kwh) if cf_ else None
        li_metrics(best, kp, at=at_, kind="molecule", ids=[r.smiles], name=r.smiles)
    with tb[1]:
        k_ = max(r.best_k, 1)
        show_cost(best.cost_parts, {"i_a_cm2": best.i_a_cm2, "area_m2": best.area_m2, "v_dis": best.v_dis,
                                    "p_max_bar": best.store.p_design_pa / 1e5 if best.store.kind != "none" else None,
                                    "chem_kg_per_kwh": pairs.kg_per_kwh(r.mw, max(sum(steps[:k_]) / k_, 0.05), 2 * k_),
                                    "chem_usd_kg": sc.usd_kg, "synth": best.synth}, kp)
    with tb[2]:
        st.dataframe(pd.DataFrame([{"option": x.name, "Wh/kg": round(x.wh_kg, 1), "Wh/L": round(x.wh_l, 1),
                                    "build USD/kWh": round(x.capex_usd_kwh),
                                    "build p10-p90 (reaction speed)": band_text(x.capex_band),
                                    "LCOS": round(x.lcos_usd_kwh, 3),
                                    "efficiency": f"{x.rte:.2f} ({x.rte_band[0]:.2f}-{x.rte_band[2]:.2f})",
                                    "recharge min": round(x.recharge_h * 60), "burst kg TNT": round(x.store.burst_tnt_kg, 3)}
                                   for x in cfgs]), hide_index=True, width="stretch")


def render_pair(pr, kp: str):
    novelty_banner("pair", [pr.neg.smiles, pr.pos.smiles])
    st.markdown(f"**Negative** {pr.neg.smiles} (releases {pr.kn} H2, steps {', '.join(f'{x:.2f}' for x in (pr.neg.steps or [pr.neg.e0]))} V)  \n"
                f"**Positive** {pr.pos.smiles} (takes {pr.kp} H2, steps {', '.join(f'{x:.2f}' for x in (pr.pos.steps or [pr.pos.e0]))} V)")
    st.image(draw([pr.neg.smiles, pr.pos.smiles], ["negative", "positive"]))
    c = st.columns(6)
    c[0].metric("Cell voltage", f"{pr.voltage:.2f} V")
    c[1].metric("Liquids", f"{pr.wh_l:.0f} Wh/L", f"{pr.wh_l - LI['wh_l']:+.0f} vs Li-ion pack")
    c[2].metric("Must cost below", f"${pairs.price_ceiling_usd_kg(pr, LI['usd_kwh_stationary']):.1f}/kg")
    c[3].metric("Free hydrogen", "none")
    sn, sp_ = synthesis.estimate(pr.neg.smiles, t["synthesis"]), synthesis.estimate(pr.pos.smiles, t["synthesis"])
    c[4].metric("Negative costs", f"${sn.usd_kg:,.0f}/kg",
                help=f"Estimated price to make the negative molecule: {sn.describe()}.")
    c[5].metric("Positive costs", f"${sp_.usd_kg:,.0f}/kg",
                help=f"Estimated price to make the positive molecule: {sp_.describe()}.")
    st.caption("; ".join(pr.notes + pr.neg.notes + pr.pos.notes))
    cfg = pair_cfg(f"{pr.neg.smiles}|{pr.pos.smiles}|{pr.kn}|{pr.kp}", tkey(t), pr)
    st.markdown("**Built as a full battery, vs lithium-ion**")
    if cfg is None:
        st.info("Cannot be charged inside the water window.")
    else:
        li_metrics(cfg, kp, kind="pair", ids=[pr.neg.smiles, pr.pos.smiles], name=f"{pr.neg.smiles} {pr.pos.smiles}",
                   at=lambda sc: pair_cfg(f"{pr.neg.smiles}|{pr.pos.smiles}|{pr.kn}|{pr.kp}", tkey(targets_at(sc)), pr))
        show_cost(cfg.cost_parts, {"i_a_cm2": cfg.i_a_cm2, "area_m2": cfg.area_m2, "v_dis": cfg.v_dis,
                                   "chem_kg_per_kwh": pairs.kg_per_kwh(pr.neg.mw, pr.voltage, 2 * pr.kn)
                                   + pairs.kg_per_kwh(pr.pos.mw, pr.voltage, 2 * pr.kp), "synth": cfg.synth,
                                   "stack_kind": "liquid"}, kp)
        st.write(f"Efficiency {cfg.rte:.2f} (p10-p90 {cfg.rte_band[0]:.2f}-{cfg.rte_band[2]:.2f}); "
                 f"lifetime cost {cfg.lcos_usd_kwh:.3f} USD/kWh; mass {cfg.mass_kg:,.0f} kg, volume {cfg.volume_l:,.0f} L.")




def search_buttons(mode: str, kw: dict, kp: str):
    """Start / Pause-Resume / Stop / Reset on one line (inside the settings fragment: every press reruns the page)."""
    ctl = search_controls()[mode]
    running = ctl.thread is not None and ctl.thread.is_alive()
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Start", type="primary", disabled=running, key=f"{kp}_start", width="stretch"):
        ctl.history.clear()
        rid = runs.new_run(mode, {k: SS[k] for k in SETTING_KEYS if k in SS}, kw)
        ctl.on_update = lambda c, store, rid=rid: runs.update_run(rid, c, store)
        ai_search.start_background(ctl, mode=mode, **kw)
        SS[f"{kp}_view_run"] = None
        st.rerun(scope="app")
    if b2.button("Resume" if ctl.paused else "Pause", disabled=not running, key=f"{kp}_pause", width="stretch"):
        ctl.paused = not ctl.paused
        st.rerun(scope="app")
    if b3.button("Stop", disabled=not running, key=f"{kp}_stop", width="stretch"):
        ctl.stopped, ctl.paused = True, False
        st.rerun(scope="app")
    with b4.popover("Reset", disabled=running, width="stretch"):
        st.write("Start over with an empty molecule list. The current results are kept as a dated backup "
                 "file in cache/; computed energies stay cached.")
        if st.button("Yes, reset", key=f"{kp}_reset"):
            for c in search_controls().values():
                c.history.clear()
                c.status = "idle"
            ai_search.bump_epoch()
            ai_search.reset_store()
            st.rerun(scope="app")


def search_panel(mode: str, kp: str):
    """The live progress chart and the saved searches (the buttons are in the settings, search_buttons)."""
    ctl = search_controls()[mode]
    running = ctl.thread is not None and ctl.thread.is_alive()
    default(f"{kp}_live", True)
    live = st.toggle("Live progress (refresh every 2 s)", key=f"{kp}_live",
                     help="While on, the progress panel refreshes itself every 2 seconds, and clicks on charts "
                          "can be lost. Turn it off to explore results while the search keeps running in the "
                          "background; turn it on again to watch progress.") if running else False
    if running and live:
        st.fragment(live_progress, run_every=2)(mode, kp)
    else:
        live_progress(mode, kp)
    saved = runs.list_runs(mode)
    with st.expander(f"Saved searches ({len(saved)})"):
        if not saved:
            st.caption("Every search you start is saved here with its settings and results.")
        else:
            ids = [r["id"] for r in saved]
            pick = st.selectbox("Search", ids, format_func=lambda i: runs.label(next(r for r in saved if r["id"] == i)),
                                key=f"{kp}_runpick")
            rec = next(r for r in saved if r["id"] == pick)
            c1, c2, c3 = st.columns(3)
            c1.button("Restore these settings", key=f"{kp}_restore", on_click=restore_settings, args=(rec,),
                      width="stretch", help="Puts back every setting this search was started with (sidebar and search).")
            if c2.button("Show only this search's results", key=f"{kp}_view", width="stretch"):
                SS[f"{kp}_view_run"] = pick
                st.rerun()
            any_running = any(c.thread is not None and c.thread.is_alive() for c in search_controls().values())
            with c3.popover("Delete", disabled=any_running, width="stretch",
                            help="Removes this saved search and its results" + (" (stop the running search first)" if any_running else "")):
                st.write(f"Delete the saved search started {rec['started']} and its "
                         f"{len(rec.get('molecules', []))} molecules?")
                also = st.checkbox("Also remove its molecules from the library", value=True, key=f"{kp}_delmols_{pick}",
                                   help="Molecules that another saved search also found are kept.")
                st.caption("The record and its results move to cache/runs/trash/, so they can be recovered.")
                if st.button("Yes, delete", key=f"{kp}_delete_{pick}"):
                    if also:
                        n_ = runs.remove_results(rec)
                        st.toast(f"{n_} molecules removed from the library")
                    runs.delete_run(pick)
                    for k_ in ("mol_view_run", "pair_view_run"):
                        if SS.get(k_) == pick:
                            SS[k_] = None
                    st.rerun()
            st.json({"settings": rec.get("settings", {}), "search": rec.get("search", {})}, expanded=False)
    vr = SS.get(f"{kp}_view_run")
    if vr:
        rec = next((r for r in saved if r["id"] == vr), None)
        if rec:
            a, b = st.columns([4, 1])
            a.info(f"Showing only the results of the search started {rec['started']} "
                   f"({len(rec.get('molecules', []))} molecules).")
            if b.button("Show all", key=f"{kp}_viewall", width="stretch"):
                SS[f"{kp}_view_run"] = None
                st.rerun()


def view_pool(kp: str) -> list:
    """Molecules to show: one saved search's results, or the whole library."""
    vr = SS.get(f"{kp}_view_run")
    if vr:
        rec = next((r for r in runs.list_runs() if r["id"] == vr), None)
        if rec:
            out = []
            for d_ in rec.get("molecules", []):
                try:
                    out.append(M.MolResult(**d_))
                except TypeError:
                    continue
            return out
    return library_snapshot()


def library_snapshot() -> list:
    """The molecule library as shown on the page. While a search runs it is
    re-read only when a generation finishes, so charts do not change under
    the mouse (a chart whose data changes loses its click)."""
    ctls = search_controls().values()
    if not any(c.thread is not None and c.thread.is_alive() for c in ctls):
        SS.pop("snap_sig", None)
        return list(ai_search.load_store().values())
    sig = tuple(len(c.history) for c in ctls)
    if SS.get("snap_sig") != sig or "snap_pool" not in SS:
        SS["snap_pool"] = list(ai_search.load_store().values())
        SS["snap_sig"] = sig
    return SS["snap_pool"]


def funnel_text(ctl, library: int, unit: str = "molecules") -> str:
    """How much chemical space a search has looked at: written -> passed the filters (scored by the model)
    -> computed with xTB."""
    w, p_ = getattr(ctl, "written", 0), getattr(ctl, "passed", 0)
    done = len(getattr(ctl, "evaluated", []))
    if not w and not done:
        return ""
    wg, pg = getattr(ctl, "written_gen", 0), getattr(ctl, "passed_gen", 0)
    return (f"**This run:** the AI wrote **{w:,}** {unit} (mutations and crossovers) → **{p_:,}** were new, valid "
            f"and stable, and the model scored them all → the **{done:,}** most promising were computed with "
            f"xTB. This generation: {wg:,} written, {pg:,} scored. Library: **{library:,}** {unit} computed "
            f"in all runs.")


def live_progress(mode: str, kp: str):
    # During the panel's own 2-second refresh (not a full page run) a click on a
    # chart elsewhere would wait unseen: hand it to a full page run. Never do
    # this inside a full run, or the page would restart itself endlessly.
    if not SS.get("_full_run") and (click_waiting() or SS.get("pending_dialog")):
        st.rerun(scope="app")
    ctl = search_controls()[mode]
    last = st.session_state.get(f"{kp}_status")
    st.session_state[f"{kp}_status"] = ctl.status
    if last == "running" and ctl.status in ("finished", "stopped", "failed"):
        st.rerun()                       # refresh tables with the new molecules
    label = {"idle": "not running", "running": "running", "paused": "paused",
             "finished": "finished", "stopped": "stopped (results so far are kept)", "failed": "failed"}
    st.caption(f"Status: **{label.get(ctl.status, ctl.status)}**"
               + (f" — {ctl.history[-1]['molecules']} molecules in the library" if ctl.history else "")
               + (f" — {ctl.error}" if ctl.error else ""))
    gens_total = getattr(ctl, "generations", 0)
    if ctl.status in ("running", "paused") and gens_total:
        import time as _t
        phase = getattr(ctl, "phase", "")
        done, size = getattr(ctl, "done_in_phase", 0), getattr(ctl, "phase_size", 0)
        gen = getattr(ctl, "generation", 0)
        batch = max(getattr(ctl, "batch", 1), 1)
        if phase == "starting molecules":
            frac = 0.0
            text = f"Computing the starting molecules: {done} of {size}"
        else:
            frac = ((gen - 1) * batch + min(done, batch)) / (gens_total * batch)
            text = f"Generation {gen} of {gens_total} — molecule {min(done + 1, size)} of {size} in this generation"
        st.progress(min(max(frac, 0.0), 1.0), text=text)
        elapsed = _t.time() - getattr(ctl, "started", _t.time())
        eta = ai_search.eta_seconds(ctl) if hasattr(ai_search, "eta_seconds") else None
        fmt = lambda x: f"{int(x // 3600)} h {int(x % 3600 // 60)} min" if x >= 3600 else f"{int(x // 60)} min {int(x % 60)} s"
        n_t = len(getattr(ctl, "mol_times", []))
        st.caption(f"Elapsed {fmt(elapsed)}"
                   + (f" — about {fmt(eta)} left (average {sum(ctl.mol_times) / n_t:.0f} s per molecule over {n_t})"
                      if eta is not None else " — time left: estimated after the first molecule")
                   + (" — paused: the clock keeps running" if ctl.status == "paused" else ""))
    ft = funnel_text(ctl, len(ai_search.load_store()))
    if ft:
        st.caption(ft)
    elif ctl.status in ("running", "paused"):
        st.caption(f"{len(ctl.history)} generation(s) completed. This run started before progress tracking was "
                   "added; the next run shows the progress bar and time left.")
    if ctl.history:
        unit = "Wh per litre of both liquids" if mode == "pair" else "Wh per litre of charged electrolyte"
        kind = "pair" if mode == "pair" else "molecule"
        fallback = None
        if any(not h.get("best_id") for h in ctl.history):     # runs from before best_id was recorded
            if mode == "pair":
                bp = pairs.best_pairs(view_pool("pair"), 1)
                fallback = [bp[0].neg.smiles, bp[0].pos.smiles] if bp else None
            else:
                top = max(view_pool("mol"), key=lambda r: r.objective, default=None)
                fallback = [top.smiles] if top else None
        ids = [h.get("best_id") or fallback for h in ctl.history]
        keep = [i for i, x in enumerate(ids) if x]
        gx = [ctl.history[i]["generation"] for i in keep]
        gy = [ctl.history[i]["best"] for i in keep]
        f = go.Figure()
        # every molecule computed in each generation (grey), then the best so far (blue)
        pool_ = {r.smiles: r for r in ai_search.load_store().values()}
        if mode == "pair":
            val = {}
            for p_ in pairs.best_pairs(list(pool_.values()), 2000):
                for m_ in (p_.neg.smiles, p_.pos.smiles):
                    val[m_] = max(val.get(m_, 0.0), p_.wh_l)
        else:
            val = {k: (r.objective if math.isfinite(r.objective) else 0.0) for k, r in pool_.items()}
        import random as _r
        _rg = _r.Random(0)
        ax_, ay_, ac_ = [], [], []
        for h in ctl.history:
            for m_ in h.get("ids") or []:
                if m_ in pool_:
                    ax_.append(h["generation"] + _rg.uniform(-0.3, 0.3))
                    ay_.append(val.get(m_, 0.0))
                    ac_.append(m_)
        if ac_:
            f.add_trace(go.Scatter(x=ax_, y=ay_, mode="markers", marker=dict(size=9, color="rgba(150,150,150,0.55)"),
                                   name=f"every molecule computed ({len(ac_)}, click to open)"
                                        + (" — value = its best pair" if mode == "pair" else ""),
                                   customdata=[[m_, m_] for m_ in ac_] if mode != "pair" else None,
                                   hovertext=ac_, hovertemplate="generation %{x:.0f}<br>%{y:.0f}<br>%{hovertext}<extra></extra>"))
            if mode != "pair":
                _nx = [i for i, m_ in enumerate(ac_) if pool_[m_].in_pubchem is False]
                add_new_rings(f, [ax_[i] for i in _nx], [ay_[i] for i in _nx], [[ac_[i], ac_[i]] for i in _nx],
                              size=12, width=2)
        f.add_trace(go.Scatter(x=gx, y=gy, mode="lines+markers", marker=dict(size=12, color="#3b82f6"),
                               name="best so far (click to open)",
                               customdata=[[ids[i][0], ids[i][-1]] for i in keep],
                               hovertext=[" | ".join(ids[i]) for i in keep],
                               hovertemplate="generation %{x}<br>%{y:.0f}<br>%{hovertext}<extra></extra>"))
        f.add_hline(y=LI["wh_l"], line_dash="dash", line_color="red", annotation_text=f"Li-ion car pack {LI['wh_l']:.0f} Wh/L")
        f.update_layout(xaxis_title="generation", yaxis_title=unit, height=380, legend=dict(orientation="h", y=-0.3),
                        title="Search progress vs lithium-ion: every molecule computed, by generation (click to open)")
        f.update_xaxes(dtick=1)
        evg = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=ck(f"{kp}_prog"))
        take(evg, f"{kp}_prog", kind)
    if not SS.get("_full_run") and SS.get("pending_dialog"):
        st.rerun(scope="app")            # progress chart clicked during a panel refresh


def best_so_far(mode: str, kp: str):
    """The search's best candidate, what it gives as a whole battery, and a link to the trade-off chart."""
    pool_ = view_pool("pair" if mode == "pair" else "mol")
    if mode == "pair":
        bp = pairs.best_pairs(pool_, 1)
        if not bp:
            return
        label_ = f"{bp[0].neg.smiles} | {bp[0].pos.smiles}"
        cf_ = pair_cfg(f"{bp[0].neg.smiles}|{bp[0].pos.smiles}|{bp[0].kn}|{bp[0].kp}", tkey(t), bp[0])
        head = f"Best pair so far: **{label_}**, {bp[0].wh_l:.0f} Wh per litre of the two liquids (liquids only)"
    else:
        top = max((r for r in pool_ if not pairs._unstable(r.smiles)), key=lambda r: r.objective, default=None)
        if top is None:
            return
        label_ = top.smiles
        cfs = molecule_options(tkey(t), top.smiles)
        cf_ = min(cfs, key=lambda x: x.lcos_usd_kwh) if cfs else None
        head = f"Best molecule so far: **{label_}**, {top.objective:.0f} Wh per litre of charged liquid (liquid only)"
    tail = (f". As a whole battery (stack, tanks, pumps, electronics): **{cf_.wh_kg:.0f} Wh/kg, {cf_.wh_l:.0f} Wh/L, "
            f"${cf_.capex_usd_kwh:,.0f}/kWh** (Li-ion pack {LI['wh_kg']:.0f} Wh/kg, {LI['wh_l']:.0f} Wh/L)."
            if cf_ else ". It cannot be built as a whole battery with these settings.")
    c1, c2 = st.columns([4, 1])
    c1.markdown(head + tail)
    c2.button("Show it in the trade-off chart", key=f"{kp}_show_trade", on_click=show_in_tradeoff, args=(label_,),
              help="Opens the Trade-offs tab with this candidate ringed and labelled.", width="stretch")


def show_in_tradeoff(name: str):
    """Button callback (runs before the page is rebuilt, so it may set widget states)."""
    SS["trade_hl"] = [name]
    SS["cfg_tab"] = "Trade-offs"


def save_settings_now():
    """Save the settings at once (normally saved at the end of each page run, which a reload can cut short)."""
    runs.save_settings({k: SS[k] for k in SETTING_KEYS if k in SS})


def pin_button(kind: str, ids: list, kp: str = ""):
    """Add or remove this candidate from the interesting-candidates list."""
    item = [kind, *ids]
    wl = list(SS.get("watchlist") or [])
    if item in wl:
        if st.button("Remove from interesting candidates", key=f"unwatch_{kind}{kp}"):
            SS["watchlist"] = [x for x in wl if x != item]
            save_settings_now()
            st.rerun()
        st.caption(f"This is **⭐{wl.index(item) + 1}** in your list: the same gold star and number mark it on the "
                   "trade-off chart and in the tables, recalculated whenever you change a setting.")
    elif st.button("⭐ Add to interesting candidates", key=f"watch_{kind}{kp}", type="primary",
                   help="Puts it in the ⭐ Candidates tab, where all of them are recalculated and "
                        "compared every time you change a setting on the left."):
        SS["watchlist"] = wl + [item]
        save_settings_now()
        st.rerun()


@st.dialog("Candidate detail", width="large")
def couple_dialog(name: str):
    with st.spinner("Computing the details: a few seconds the first time, instant afterwards..."):
        st.session_state["selected_couple"] = name
        d = next((x for x in designs if x.cell.name == name), None)
        if d is None or d.best is None:
            st.info("This couple was not sized.")
        else:
            pin_button("couple", [name])
            render_couple(d, "dlg")


@st.dialog("Molecule detail", width="large")
def molecule_dialog(smiles: str):
    with st.spinner("Computing the details: a few seconds the first time, instant afterwards..."):
        render_molecule(smiles, "mdlg", pin=True)


@st.dialog("Pair detail", width="large")
def pair_dialog(neg: str, pos: str):
    with st.spinner("Computing the details: a few seconds the first time, instant afterwards..."):
        pool_ = view_pool("pair")
        pr = next((p for p in pairs.best_pairs(pool_, 200) if p.neg.smiles == neg and p.pos.smiles == pos), None)
        if pr is None:
            st.info("Pair no longer available.")
        else:
            pin_button("pair", [neg, pos, pr.kn, pr.kp])
            render_pair(pr, "pdlg")


st.title("Hydrogen battery simulator")
st.caption("Candidates come from thermodynamic data and physical laws; every result is compared with the "
           "lithium-ion packs in current cars (Tesla NMC/NCA, BYD Blade LFP).")

TAB_NAMES = ["🧪 AI discovery: 3 battery types", "⭐ Candidates (best + interesting + details)", "Trade-offs",
             "Li-ion reference and validation"]
if SS.get("cfg_tab") not in TAB_NAMES:
    SS["cfg_tab"] = TAB_NAMES[0]
# the open tab is a saved setting, so a page reload comes back to it
st.segmented_control("Section", TAB_NAMES, key="cfg_tab", label_visibility="collapsed", width="stretch")
TAB_NOW = SS.get("cfg_tab") or TAB_NAMES[0]      # only this section is computed on each page run
if SS.get("_reset_msg"):
    st.toast("Reset: " + SS.pop("_reset_msg"), icon="🧹")

with st.spinner("Generating and screening every hydrogen couple..."):
    results = screen(tkey(t))
names = tuple(r.cell.name for r in results if r.score > 0)[:top_n]
extra = tuple(st.session_state.get("sized_molecules", []))
with st.spinner(f"Sizing {len(names) + len(extra)} systems at every pressure..."):
    designs = size(tkey(t), names, extra)
df = design_rows(designs).sort_values("LCOS USD/kWh") if designs else pd.DataFrame()


# ------------------------------------------------------------------ pinned candidate

SWEEPS = {   # setting -> (label, values around the current one)
    "p_max": ("Highest hydrogen pressure, bar", lambda v: [10, 20, 30, 45, 60, 100, 150, 200]),
    "power": ("Power, kW", lambda v: [v * f for f in (0.25, 0.5, 1, 2, 4)]),
    "hours": ("Delivers full power for, hours", lambda v: [v * f for f in (0.25, 0.5, 1, 2, 4, 8)]),
    "rte": ("Round-trip efficiency it must keep", lambda v: [0.55, 0.65, 0.75, 0.85, 0.9]),
    "life": ("Lifetime, years", lambda v: [5, 10, 15, 20, 30]),
    "elec": ("Charging electricity, USD/kWh", lambda v: [0.02, 0.05, 0.1, 0.2]),
    "scale": ("Built as", lambda v: list(stackcost.SCALES)),
}


def evaluate_pinned(pin: list, tt: dict) -> dict | None:
    """Key results of the pinned candidate for these model inputs, with the total build price."""
    r_ = _evaluate_pinned(pin, tt)
    if r_:
        r_ = {"build USD (total)": r_["build USD/kWh"] * tt["design"]["power_kw"] * tt["design"]["hours"], **r_}
    return r_


def _evaluate_pinned(pin: list, tt: dict) -> dict | None:
    kind, rng = pin[0], np.random.default_rng(5)
    e = tt["design"]["power_kw"] * tt["design"]["hours"]
    if kind == "couple":
        cell = next((c for c in discover.table_pairs() if c.name == pin[1]), None)
        b = design.design(cell, tt, rng).best if cell else None
        if b is None:
            return None
        return {"build USD/kWh": b.capex_usd_kwh, "lifetime cost USD/kWh": b.lcos_usd_kwh,
                "Wh/kg": e * 1000 / b.mass_kg, "Wh/L": e * 1000 / b.volume_l, "efficiency": b.rte,
                "pressure bar": b.p_max_bar}
    if kind == "gascell":
        c, sp = gas_cell_by_key(pin[1])
        cfgs = options.gas_configurations(c, sp, tt, rng) if c else []
        cf = min(cfgs, key=lambda x: x.lcos_usd_kwh) if cfgs else None
        if cf is None:
            return None
        return {"build USD/kWh": cf.capex_usd_kwh, "lifetime cost USD/kWh": cf.lcos_usd_kwh,
                "Wh/kg": cf.wh_kg, "Wh/L": cf.wh_l, "efficiency": cf.rte}
    store = ai_search.load_store()
    if kind == "molecule":
        r = store.get(pin[1]) or M.evaluate(pin[1])
        cfgs = options.configurations(M.as_cell(r), tt, rng) if math.isfinite(r.e0) else []
        c = min(cfgs, key=lambda x: x.lcos_usd_kwh) if cfgs else None
    else:
        n, p_ = store.get(pin[1]), store.get(pin[2])
        c = options.pair_configuration(pairs.make_pair(n, p_, int(pin[3]), int(pin[4])), tt, rng) if n and p_ else None
    if c is None:
        return None
    return {"build USD/kWh": c.capex_usd_kwh, "lifetime cost USD/kWh": c.lcos_usd_kwh,
            "Wh/kg": c.wh_kg, "Wh/L": c.wh_l, "efficiency": c.rte}


@st.cache_data(show_spinner=False)
def pinned_now(pin_json: str, targets_repr: str, version: int = MODEL_VERSION):
    return evaluate_pinned(json.loads(pin_json), json.loads(targets_repr))


@st.cache_data(show_spinner=False)
def pinned_sweep(pin_json: str, settings_json: str, key: str, version: int = MODEL_VERSION):
    pin, s = json.loads(pin_json), json.loads(settings_json)
    rows = []
    for v in sorted(set(SWEEPS[key][1](s[key])) | {s[key]}, key=str if key == "scale" else float):
        if key == "p_max" and v <= s["p_min"]:
            continue
        res = evaluate_pinned(pin, make_targets({**s, key: v}))
        if res:
            rows.append({"value": stackcost.SCALES[v] if key == "scale" else v, "now": v == s[key], **res})
    return pd.DataFrame(rows)


def short_label(pin: list) -> str:
    if pin[0] == "couple":
        return pin[1]
    if pin[0] == "gascell":
        c, _ = gas_cell_by_key(pin[1])
        name = c.name if c else pin[1]
        return name if len(name) <= 48 else name[:46] + "…"
    if pin[0] == "molecule":
        return pin[1] if len(pin[1]) <= 28 else pin[1][:26] + "…"
    cut = lambda x: x if len(x) <= 16 else x[:14] + "…"
    return f"{cut(pin[1])} | {cut(pin[2])}"


def show_details(pin: list, kp: str):
    if pin[0] == "gascell":
        c, sp = gas_cell_by_key(pin[1])
        if c is None:
            st.info("This cell is no longer in the screen (a molecule was removed or reset).")
            return
        gas_cost_study(c, sp, kp)
        return
    if pin[0] == "couple":
        d_ = size(tkey(t), (pin[1],), ())
        if d_ and d_[0].best:
            render_couple(d_[0], kp)
    elif pin[0] == "molecule":
        render_molecule(pin[1], kp)
    else:
        store_ = ai_search.load_store()
        n_, p_ = store_.get(pin[1]), store_.get(pin[2])
        if n_ and p_:
            render_pair(pairs.make_pair(n_, p_, int(pin[3]), int(pin[4])), kp)
        else:
            st.info("Its molecules are no longer in the library.")


def candidate_choices(src: str) -> list:
    """Candidates (as pins) to pick from in the details picker."""
    if src == "interesting":
        return [x for x in (SS.get("watchlist") or []) if x]
    if src == "couple":
        return [["couple", d.cell.name] for d in designs if d.best]
    if src == "molecule":
        st_ = [r for r in ai_search.load_store().values() if math.isfinite(r.e0)]
        return [["molecule", r.smiles] for r in sorted(st_, key=lambda r: -r.objective)[:60]]
    if src == "pair":
        return [["pair", p.neg.smiles, p.pos.smiles, p.kn, p.kp]
                for p in pairs.best_pairs(view_pool("pair"), 30, max_per_molecule=2)]
    if src == "gascell":
        n_e, n_s = gas_energy_count(), len(gases.searched_species())
        if n_e == 0:
            return []
        with st.spinner("Loading the sealed gas cells..."):
            cells, _, _ = gas_screen(n_e, n_s)
            keys = [c.key for c in cells[:30]]
            keys += [v["key"] for v in sorted(gas_ai_best_cells(n_e, n_s).values(), key=lambda v: -v["wh_kg"])[:30]]
        return [["gascell", k_] for k_ in dict.fromkeys(keys)]
    return []


CAND_SOURCES = {"interesting": "⭐ my interesting candidates", "couple": "best table couples (Best candidates tab)",
                "molecule": "AI molecules (vs hydrogen)", "pair": "AI pairs (two molecules)",
                "gascell": "sealed gas cells (best + found by the AI)"}


def candidate_details():
    st.markdown("#### Details of any candidate")
    st.caption("Everything about one candidate, recalculated with your settings. Pin it to compare it above.")
    default("cand_src", "interesting" if SS.get("watchlist") else "couple")
    src = st.radio("From", list(CAND_SOURCES), key="cand_src", horizontal=True, format_func=CAND_SOURCES.get)
    ch = candidate_choices(src)
    if not ch:
        st.info({"interesting": "No interesting candidates yet: pick one from another list and press "
                                "'⭐ Add to interesting candidates'.",
                 "gascell": "No sealed gas cell yet: compute the gas molecules in the AI discovery tab."}.get(
                    src, "Nothing here yet: run the corresponding search in the AI discovery tab."))
        return
    labels = [json.dumps(x) for x in ch]
    last = SS.get("selected_couple")
    idx = next((i for i, x in enumerate(ch) if src == "couple" and x[1] == last), 0)
    pk = st.selectbox("Candidate", labels, index=idx, key=f"cand_pick_{src}",
                      format_func=lambda j: short_label(json.loads(j)))
    pin = json.loads(pk)
    pin_button(pin[0], pin[1:], "_cand")
    show_details(pin, "cand")


def best_all_types():
    st.markdown("#### Best candidates of every battery type, against lithium-ion")
    li_usd, li_name = li_price_now()
    packs = li_packs(t)
    li_lc = min(p_["lifetime cost USD/kWh"] for p_ in packs)
    nk = tradeoff_args()
    with st.spinner("Sizing every battery type for your settings (first time only)..."):
        now_ = tradeoff_points(tkey(t), *nk)
        tgt_ = tradeoff_points(tkey(targets_at("target")), *nk) if SS.get("cfg_scale") != "target" else now_
    if not now_:
        st.info("Nothing sized yet.")
        return
    tg = {(r["kind"], tuple(r["ids"])): r for r in tgt_}
    rows = []
    for r in now_:
        g = tg.get((r["kind"], tuple(r["ids"])), r)
        wins = [k_ for k_, ok in (("weight", r["Wh/kg"] > LI["wh_kg"]), ("volume", r["Wh/L"] > LI["wh_l"]),
                                  ("build price", r["build USD/kWh"] < li_usd),
                                  ("lifetime cost", r["lifetime cost USD/kWh"] < li_lc)) if ok]
        wins_t = [k_ for k_, ok in (("weight", g["Wh/kg"] > LI["wh_kg"]), ("volume", g["Wh/L"] > LI["wh_l"]),
                                    ("build price", g["build USD/kWh"] < li_usd),
                                    ("lifetime cost", g["lifetime cost USD/kWh"] < li_lc)) if ok]
        pn_ = pin_number(r["kind"], r["ids"])
        rows.append({"candidate": (f"⭐{pn_} " if pn_ else "") + ("🆕 " if r.get("new") else "") + r["name"],
                     "battery type": r["type"], "known or new": novelty_tag(r["kind"], r["ids"]),
                     "beats Li-ion on (today's parts)": ", ".join(wins) or "-",
                     "beats Li-ion on (purpose-built)": ", ".join(wins_t) or "-",
                     "n": len(wins_t) + 0.1 * len(wins),
                     "Wh/kg": round(r["Wh/kg"]), "Wh/L": round(r["Wh/L"]),
                     "build USD/kWh today": round(r["build USD/kWh"]),
                     "build USD/kWh purpose-built": round(g["build USD/kWh"]),
                     "full battery USD today": round(r["build USD/kWh"] * E_KWH),
                     "full battery USD purpose-built": round(g["build USD/kWh"] * E_KWH),
                     "lifetime USD/kWh": round(r["lifetime cost USD/kWh"], 3),
                     "efficiency": round(r["efficiency"], 2), "notes": r["extra"],
                     "_pin": [r["kind"], *r["ids"]]})
    df_ = pd.DataFrame(rows)
    types = [k_ for k_ in TYPE_COLORS if k_ in set(df_["battery type"])]
    f1, f2 = st.columns([3, 1])
    shown = f1.multiselect("Battery types", types, default=types, key="best_types")
    default("best_sort", "beats Li-ion")
    sort_ = f2.selectbox("Sort by", ["beats Li-ion", "Wh/kg", "build price", "lifetime cost"], key="best_sort")
    df_ = df_[df_["battery type"].isin(shown)]
    if df_.empty:
        return
    by_, asc_ = {"beats Li-ion": (["n", "build USD/kWh purpose-built"], [False, True]),
                 "Wh/kg": ("Wh/kg", False), "build price": ("build USD/kWh today", True),
                 "lifetime cost": ("lifetime USD/kWh", True)}[sort_]
    df_ = df_.sort_values(by=by_, ascending=asc_)
    c1, c2, c3, c4 = st.columns(4)
    bk, bl = df_["Wh/kg"].max(), df_["Wh/L"].max()
    bc, bt = df_["build USD/kWh today"].min(), df_["build USD/kWh purpose-built"].min()
    c1.metric("Lightest", f"{bk:.0f} Wh/kg", f"{bk - LI['wh_kg']:+.0f} vs Li-ion pack {LI['wh_kg']:.0f}")
    c2.metric("Most compact", f"{bl:.0f} Wh/L", f"{bl - LI['wh_l']:+.0f} vs Li-ion pack {LI['wh_l']:.0f}")
    c3.metric("Cheapest, today's parts", f"${bc:,.0f}/kWh", f"{bc - li_usd:+,.0f} vs {li_name}", delta_color="inverse")
    c4.metric("Cheapest, purpose-built", f"${bt:,.0f}/kWh", f"{bt - li_usd:+,.0f} vs {li_name}", delta_color="inverse")
    st.caption(f"Each figure is the best single candidate for that figure, so no one candidate need reach all four. "
               f"Whole batteries sized for {power:g} kW x {hours:g} h ({E_KWH:,.1f} kWh). Lithium-ion: "
               f"{li_name} ${li_usd:.0f}/kWh, lifetime cost {li_lc:.3f} USD/kWh (same formula). Per type: every table "
               "couple, the 15 best AI molecules and pairs, and about 35 sealed gas cells. Click a row to open it.")
    SS["rows_best_all"] = df_["_pin"].tolist()
    ev = st.dataframe(df_.drop(columns=["_pin", "n"]), hide_index=True, width="stretch", on_select="rerun",
                      selection_mode=["single-row", "single-cell"], key=ck("best_all"))
    take_row_any(ev, "best_all")
    st.download_button("Download table (CSV)", df_.drop(columns=["_pin", "n"]).to_csv(index=False),
                       "best_candidates_all_types.csv")
    with st.expander("Table couples: every sized couple with pressure, safety, leaks and refill"):
        table_couples_detail()


def watch_tab():
    best_all_types()
    st.divider()
    wl = [x for x in (SS.get("watchlist") or []) if x]
    st.markdown("#### ⭐ Your interesting candidates")
    if not wl:
        st.info("No interesting candidates yet. Pick one below (or click any candidate in a table or chart) and "
                "press '⭐ Add to interesting candidates'. They are then compared here and recalculated every "
                "time you change a setting on the left.")
        st.divider()
        candidate_details()
        return
    st.caption(f"Recalculated with your settings: {power:g} kW for {hours:g} h ({E_KWH:,.0f} kWh), up to "
               f"{p_max} bar, efficiency target {rte:.2f}, {stackcost.SCALES[SS['cfg_scale']]}.")
    fmt = {"build USD (total)": "{:,.0f}", "build USD/kWh": "{:,.0f}", "lifetime cost USD/kWh": "{:,.3f}", "Wh/kg": "{:,.1f}",
           "Wh/L": "{:,.1f}", "efficiency": "{:.2f}", "pressure bar": "{:g}"}
    rows = []
    with st.spinner(f"Recalculating {len(wl)} candidates..."):
        for pin in wl:
            now = pinned_now(json.dumps(pin), tkey(t))
            row = {"#": f"⭐{len(rows) + 1}", "candidate": short_label(pin), "kind": {"couple": "table couple", "molecule": "molecule vs H2",
                                                         "pair": "two molecules", "gascell": "two gases (sealed)"}[pin[0]]}
            row.update({k: fmt[k].format(v) for k, v in now.items()} if now else {"build USD/kWh": "cannot be built"})
            if SS.get("cfg_scale") != "target":
                tg = pinned_now(json.dumps(pin), tkey(targets_at("target")))
                row["build USD/kWh, purpose-built"] = f"{tg['build USD/kWh']:,.0f}" if tg else "-"
            rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Li-ion pack for comparison: about {LI['usd_kwh_stationary']:.0f} USD/kWh, {LI['wh_kg']:.0f} Wh/kg, "
               f"{LI['wh_l']:.0f} Wh/L.")

    keys = list(SWEEPS)
    default("pin_sweep", "p_max")
    key = st.selectbox("How one setting changes them", keys, format_func=lambda k: SWEEPS[k][0], key="pin_sweep")
    if key == "p_max" and any(p[0] == "pair" for p in wl):
        st.caption("Two-molecule batteries hold no gas, so pressure does not change them (flat lines).")
    if key == "p_max" and any(p[0] == "gascell" for p in wl):
        st.caption("Sealed gas batteries use carbon-fibre tanks at 350 or 700 bar (the cheaper of the two is kept), "
                   "so this setting does not change them; see their details below for 350 vs 700 bar.")
    metrics_ = ["build USD (total)", "build USD/kWh", "lifetime cost USD/kWh", "Wh/kg", "efficiency"]
    fig = make_subplots(rows=1, cols=5, subplot_titles=metrics_)
    sj = json.dumps(SETTINGS_NOW, sort_keys=True)
    palette = px.colors.qualitative.Plotly
    with st.spinner(f"Recalculating at several values of '{SWEEPS[key][0]}' (first time only)..."):
        for j, pin in enumerate(wl):
            sw = pinned_sweep(json.dumps(pin), sj, key)
            if sw.empty:
                continue
            for i_, m_ in enumerate(metrics_):
                if m_ not in sw.columns:          # results cached by an older version of the model
                    continue
                fig.add_trace(go.Scatter(x=sw["value"], y=sw[m_], mode="lines+markers", name=f"⭐{j + 1} {short_label(pin)}",
                                         legendgroup=str(j), showlegend=i_ == 0,
                                         line=dict(color=palette[j % len(palette)]),
                                         marker=dict(size=[13 if x else 6 for x in sw["now"]],
                                                     symbol=["star" if x else "circle" for x in sw["now"]])),
                              row=1, col=i_ + 1)
    if key in ("power", "hours", "p_max"):
        fig.update_xaxes(type="log")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=60, b=10),
                      title=f"x axis: {SWEEPS[key][0]} (star = your current setting)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch", key="watch_sweep_chart")
    st.caption("Total price vs price per kWh: the cell stack, platinum and power electronics are sized by the power "
               "(kW), the tanks and chemicals by the energy (kWh). Fewer hours at the same power cut the total only a "
               "little, so the price per kWh rises; less power at the same hours shrinks both together.")

    c1, c2, c3 = st.columns([4, 1, 1])
    pick = c1.selectbox("Remove from the list", range(len(wl)), format_func=lambda i: f"⭐{i + 1} {short_label(wl[i])}",
                        key="watch_pick", label_visibility="collapsed")
    if c2.button("Remove", key="watch_remove"):
        SS["watchlist"] = [x for k, x in enumerate(wl) if k != pick]
        save_settings_now()
        st.rerun()
    if c3.button("Clear all", key="watch_clear"):
        SS["watchlist"] = []
        save_settings_now()
        st.rerun()
    st.divider()
    candidate_details()



# ------------------------------------------------------------------ gas batteries

@st.cache_resource
def gas_job() -> dict:
    """Background xTB computation of the generated gas molecules."""
    return {"thread": None, "done": 0, "total": 0}


def gas_cell_by_key(key: str):
    """(cell, species) of a screened sealed gas cell, or (None, species)."""
    cells, sp, _ = gas_screen(gas_energy_count(), len(gases.searched_species()))
    idx = gas_index(gas_energy_count(), len(gases.searched_species()))
    return idx.get(key), sp


@st.cache_resource(show_spinner=False, max_entries=2)
def gas_index(n_energies: int, n_searched: int) -> dict:
    cells, _, _ = gas_screen(n_energies, n_searched)
    return {c.key: c for c in cells}


@st.cache_resource(show_spinner=False, max_entries=256)
def gas_cfgs(key: str, targets_json: str) -> list:
    """Every sizing of one sealed gas cell for these settings (kept in memory)."""
    c, sp = gas_cell_by_key(key)
    return options.gas_configurations(c, sp, json.loads(targets_json), np.random.default_rng(3)) if c else []


@st.cache_resource(show_spinner=False, max_entries=256)
def pair_cfg(key: str, targets_json: str, _pr):
    """The full-battery sizing of one pair for these settings (kept in memory)."""
    return options.pair_configuration(_pr, json.loads(targets_json), np.random.default_rng(13))


def gas_energy_count() -> int:
    """Cache key of the gas screen: xTB energies plus first-principles gas results (in steps of 100, so a
    background DFT batch refreshes the screen every 100 molecules, not on every page run)."""
    from h2screener import abinitio as _A
    n_fp = sum(1 for k, v in _A.load().items() if k.endswith("|g") and "g_kj" in v and "|xtb|" not in k)
    return xtb_energy_count() + 1_000_000 * (n_fp // 100)


def xtb_energy_count() -> int:
    return sum(1 for v in gases.load_energies().values() if v is not None)


@st.cache_resource(show_spinner=False, max_entries=2)
def gas_screen(n_energies: int, n_searched: int = 0):
    cells, sp = gases.screen()
    fit = gases.fit_atom_equivalents(gases.load_energies())
    return cells, sp, fit


@st.cache_resource(show_spinner=False, max_entries=8)      # kept in memory: gas cells are not copied
def gas_sized(targets_json: str, n_energies: int, n_searched: int, top: int, measured_only: bool, no_toxic: bool,
              proven_only: bool):
    tt = json.loads(targets_json)
    cells, sp, _ = gas_screen(n_energies, len(gases.searched_species()))
    pick = [c for c in cells if (not measured_only or not c.computed_only) and (not no_toxic or not c.toxic)
            and (not proven_only or c.recharge == "demonstrated")]
    ref = gases.reference_h2_o2(cells)
    if ref is not None and ref not in pick[:top]:
        pick = pick[:top - 1] + [ref]
    out = []
    for c in pick[:top]:
        cfgs = options.gas_configurations(c, sp, tt, np.random.default_rng(3))
        if cfgs:
            out.append((c, max(cfgs, key=lambda x: x.wh_kg)))
    return out, sp




def gas_progress():
    if not SS.get("_full_run") and (click_waiting() or SS.get("pending_dialog")):
        st.rerun(scope="app")            # a chart was clicked during a panel refresh
    ctl = gas_control()
    last = SS.get("gas_status")
    SS["gas_status"] = ctl.status
    if last == "running" and ctl.status in ("finished", "stopped", "failed"):
        launch_gas_fp()                  # recompute the new molecules from first principles, in the background
        st.rerun(scope="app")            # refresh the cell table with the new molecules
    label = {"idle": "not running", "running": "running", "paused": "paused", "finished": "finished",
             "stopped": "stopped (results so far are kept)", "failed": "failed"}
    store = gas_search.load()
    recs = store["molecules"]
    st.caption(f"Status: **{label.get(ctl.status, ctl.status)}** — {len(recs)} larger molecules computed so far"
               + (f" — {ctl.error}" if ctl.error else ""))
    if ctl.status in ("running", "paused") and ctl.generations:
        gen, done, size = max(ctl.generation, 1), ctl.done_in_phase, max(ctl.phase_size, 1)
        if ctl.phase == "preparing":
            frac, text = 0.0, "Screening the current pool and training the model..."
        elif ctl.phase == "choosing":
            frac = (gen - 1) / ctl.generations
            text = f"Generation {gen} of {ctl.generations} — the model scores {size} new molecules ({done} done)"
        else:
            frac = ((gen - 1) + min(done, size) / size) / ctl.generations
            text = f"Generation {gen} of {ctl.generations} — xTB on molecule {min(done + 1, size)} of {size}"
        st.progress(min(max(frac, 0.0), 1.0), text=text)
        import time as _t
        eta = gas_search.eta_seconds(ctl)
        fmt = lambda x: f"{int(x // 3600)} h {int(x % 3600 // 60)} min" if x >= 3600 else f"{int(x // 60)} min {int(x % 60)} s"
        st.caption(f"Elapsed {fmt(_t.time() - ctl.started)}"
                   + (f" — about {fmt(eta)} left" if eta is not None else " — time left: estimated after the first molecule"))
    ft = funnel_text(ctl, len(recs), "larger gas molecules")
    if ft:
        st.caption(ft)
    hist = store.get("history", [])
    ok_ = {s_: r for s_, r in recs.items() if r.get("g_gas_kj") is not None}
    if hist and ok_:
        hd = pd.DataFrame(hist)
        per = {}                               # best molecule computed in each generation
        for s_, r in ok_.items():
            g_ = r.get("generation")
            if g_ is not None and r.get("value_wh_kg", 0) > per.get(g_, ("", -1))[1]:
                per[g_] = (s_, r.get("value_wh_kg", 0))
        gx = sorted(per)
        f = go.Figure()
        import random as _r
        _rg = _r.Random(0)
        allm = [(s_, r) for s_, r in ok_.items() if r.get("generation") is not None]
        f.add_trace(go.Scatter(x=[r["generation"] + _rg.uniform(-0.3, 0.3) for _, r in allm],
                               y=[r.get("value_wh_kg", 0) for _, r in allm], mode="markers",
                               marker=dict(size=9, color="rgba(150,150,150,0.55)"),
                               name=f"every molecule computed ({len(allm)}, click to open)",
                               customdata=[[s_] for s_, _ in allm],
                               hovertext=[f"{gases.formula_of(s_)}  {s_}" for s_, _ in allm],
                               hovertemplate="generation %{x:.0f}<br>%{y:.0f} Wh/kg<br>%{hovertext}<extra></extra>"))
        f.add_trace(go.Scatter(x=gx, y=[per[g_][1] for g_ in gx], mode="markers", marker=dict(size=12),
                               name="best molecule of this generation (click to open)",
                               customdata=[[per[g_][0]] for g_ in gx],
                               text=[f"{gases.formula_of(per[g_][0])}  {per[g_][0]}" for g_ in gx],
                               hovertemplate="generation %{x}<br>%{y:.0f} Wh/kg<br>%{text}<extra></extra>"))
        _nx = [i for i, (s_, _) in enumerate(allm) if molecule_known(s_) is False]
        add_new_rings(f, [f.data[0].x[i] for i in _nx], [f.data[0].y[i] for i in _nx], [[allm[i][0]] for i in _nx], size=12, width=2)
        f.add_trace(go.Scatter(x=hd["generation"], y=hd["best_new"], mode="lines", hoverinfo="skip",
                               name="best larger molecule so far"))
        f.add_trace(go.Scatter(x=hd["generation"], y=hd["best_overall"], mode="lines", line=dict(dash="dot"),
                               hoverinfo="skip", name="best cell of the whole pool (small molecules included)"))
        f.add_hline(y=LI["wh_kg"], line_dash="dash", line_color="red",
                    annotation_text=f"Li-ion car pack {LI['wh_kg']:.0f} Wh/kg (whole pack)")
        f.update_yaxes(rangemode="tozero")
        f.update_layout(height=360, xaxis_title="generation", yaxis_title="Wh/kg, chemicals + tank walls",
                        title="Search progress vs lithium-ion: every molecule computed, by generation (click a point to open)",
                        legend=dict(orientation="h", y=-0.25))
        evg = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=ck("gas_prog"))
        take(evg, "gas_prog", "gasmol")
        st.caption("Grey dots: every molecule computed in that generation; blue: the best of the generation. Wh/kg "
                   "here counts only the chemicals and the carbon-fibre tank walls (an upper bound). The chart below "
                   "shows the same molecules as whole batteries (cells, platinum, electronics), so it sits much lower.")
    if not SS.get("_full_run") and SS.get("pending_dialog"):
        st.rerun(scope="app")


def gas_settings():
    """The gas search settings and buttons: a fragment, so changing a value reruns only this part."""
    g1, g2, g3, g4 = st.columns(4)
    default("gas_gens", 5)
    gens = g1.number_input("Generations", 1, 200, key="gas_gens",
                           help="Rounds of evolution: write new molecules, score them with the model, compute the best few.")
    default("gas_batch", 6)
    batch = g2.number_input("Molecules computed per generation", 2, 64, key="gas_batch",
                            help="How many get a full xTB calculation each round (a few seconds each for small gases).")
    default("gas_heavy", 8)
    heavy = g3.number_input("Largest molecule (heavy atoms)", 5, 14, key="gas_heavy",
                            help="Non-hydrogen atoms. Molecules much above 8 rarely stay gases at room temperature.")
    default("gas_kappa", 1.0)
    kappa = g4.slider("Exploration (0 = exploit, 3 = explore)", 0.0, 3.0, step=0.1, key="gas_kappa",
                      help="0 = compute only molecules the model is confident are good; higher = also molecules it is unsure about.")
    save_settings_now()
    ctl = gas_control()
    running = ctl.thread is not None and ctl.thread.is_alive()
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Start", type="primary", disabled=running or gas_energy_count() == 0, key="gas_start",
                 width="stretch"):
        ctl.history.clear()
        gas_search.start_background(ctl, generations=int(gens), batch=int(batch), max_heavy=int(heavy), kappa=kappa)
        st.rerun(scope="app")
    if b2.button("Resume" if ctl.paused else "Pause", disabled=not running, key="gas_pause", width="stretch"):
        ctl.paused = not ctl.paused
        st.rerun(scope="app")
    if b3.button("Stop", disabled=not running, key="gas_stop", width="stretch"):
        ctl.stopped, ctl.paused = True, False
        st.rerun(scope="app")
    with b4.popover("Reset", disabled=running, width="stretch"):
        st.write("Forget the molecules the AI found (kept as a dated backup in cache/; energies stay cached). "
                 "The small molecules are not affected.")
        if st.button("Yes, reset", key="gas_reset"):
            ai_search.bump_epoch()
            gas_search.reset()
            ctl.history.clear()
            ctl.status = "idle"
            gas_screen.clear()
            st.rerun(scope="app")


def gas_search_panel():
    st.markdown("#### AI search for larger gas molecules")
    st.write("Every molecule with up to 4 heavy atoms is already computed (above). The AI writes larger ones "
             "(SELFIES mutations and crossovers of the molecules in the lightest cells), keeps those that boil "
             "below 127 C (a gas at 25 C, or a liquid product), predicts their energy with a model trained on "
             "every xTB result, builds each into cells with the best oxidant and fuel couples, and computes the "
             "most promising ones with xTB. They then join the table below like any other molecule.")
    if gas_energy_count() == 0:
        st.info("Compute the small molecules first (button above): the AI learns from them.")
        return
    st.fragment(gas_settings)()
    ctl = gas_control()
    running = ctl.thread is not None and ctl.thread.is_alive()
    plan = gas_search.load().get("plan")
    if plan and not running and plan.get("done", 0) < plan.get("generations", 0):
        left = plan["generations"] - plan.get("done", 0)
        w1, w2 = st.columns([3, 1], vertical_alignment="center")
        w1.warning(f"The search started {plan.get('started', '')} was interrupted after {plan.get('done', 0)} of "
                   f"{plan['generations']} generations (the app was restarted; the search runs inside it). "
                   "Everything computed so far is kept.")
        if w2.button(f"Resume: {left} generations left", key="gas_resume", type="primary", width="stretch"):
            ctl.history.clear()
            gas_search.start_background(ctl, generations=left, batch=int(plan.get("batch", SS.get("gas_batch", 6))),
                                        max_heavy=int(plan.get("max_heavy", SS.get("gas_heavy", 8))),
                                        kappa=float(plan.get("kappa", SS.get("gas_kappa", 1.0))))
            st.rerun()
    default("gas_live", True)
    live = st.toggle("Live progress (refresh every 2 s)", key="gas_live") if running else False
    if running and live:
        st.fragment(gas_progress, run_every=2)()
    else:
        gas_progress()
    recs = gas_search.load()["molecules"]
    if recs:
        st.markdown("**What the AI found.** A gas battery is always a couple: an oxidant gas against a fuel gas. "
                    "The AI writes one new molecule at a time and pairs it with every oxidant and fuel couple of "
                    "the pool; each row is that molecule in its best cell, with the partner that gives it. "
                    "Open a molecule to see its other cells: a different partner gives a different battery.")
        bc = gas_ai_best_cells(gas_energy_count(), len(gases.searched_species()))
        with st.spinner("Sizing each molecule's best cell as a whole battery..."):
            fz = gas_ai_sized(tkey(t), gas_energy_count(), len(gases.searched_species()))
        if fz:
            bk_ = max(fz.values(), key=lambda x: x["wh_kg"])
            bl_ = max(fz.values(), key=lambda x: x["wh_l"])
            bc_ = min(fz.values(), key=lambda x: x["usd"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Best AI battery, energy per kg", f"{bk_['wh_kg']:.0f} Wh/kg",
                      f"{bk_['wh_kg'] - LI['wh_kg']:+.0f} vs Li-ion pack {LI['wh_kg']:.0f}")
            m2.metric("Energy per litre", f"{bl_['wh_l']:.0f} Wh/L", f"{bl_['wh_l'] - LI['wh_l']:+.0f} vs Li-ion pack {LI['wh_l']:.0f}")
            m3.metric("Build cost", f"${bc_['usd']:.0f}/kWh", f"{bc_['usd'] - LI['usd_kwh_stationary']:+.0f} vs Li-ion "
                      f"${LI['usd_kwh_stationary']:.0f}", delta_color="inverse")
            st.caption("Whole batteries sized for your settings (cells, platinum, tanks, electronics), best design per "
                       "figure, so no single design need reach all three. Green = better than lithium-ion.")
        found = pd.DataFrame([{"molecule": s_, "formula": r.get("formula"),
                               "its side": bc.get(s_, {}).get("side", "-"),
                               "stored or product": bc.get(s_, {}).get("role", "-"),
                               "partner (stored gases)": bc.get(s_, {}).get("partners", "-"),
                               "best cell (discharge)": bc.get(s_, {}).get("cell") or (
                                   "xTB failed" if r.get("g_gas_kj") is None else "no cell passes the screen"),
                               "full battery Wh/kg": fz.get(s_, {}).get("wh_kg"),
                               "full battery Wh/L": fz.get(s_, {}).get("wh_l"),
                               "vs Li-ion (Wh/kg)": ("better" if fz[s_]["wh_kg"] > LI["wh_kg"] else "worse") if s_ in fz else "-",
                               "build USD/kWh": fz.get(s_, {}).get("usd"),
                               "efficiency": fz.get(s_, {}).get("rte"),
                               "lightest cell Wh/kg (chem. + tanks)": bc.get(s_, {}).get("wh_kg", 0.0),
                               "cell V": bc.get(s_, {}).get("voltage"),
                               "recharge proven?": bc.get(s_, {}).get("recharge", "-"),
                               "heavy atoms": r.get("heavy"),
                               "boils at C (Joback)": round(r.get("tb_k", 0) - 273.15),
                               "gas at 25 C": "yes" if r.get("tb_k", 999) < gases.T25 else "no (liquid)",
                               "dfG gas kJ/mol (xTB)": r.get("g_gas_kj"),
                               "model predicted": r.get("predicted_g_kj"),
                               "generation": r.get("generation")} for s_, r in recs.items()])
        found = found.sort_values(["full battery Wh/kg", "lightest cell Wh/kg (chem. + tanks)"], ascending=False,
                                  na_position="last")
        okf = found[found["full battery Wh/kg"].notna()].copy()
        if len(okf):
            okf["size"] = okf["heavy atoms"].clip(lower=1)
            sc = px.scatter(okf, x="lightest cell Wh/kg (chem. + tanks)", y="full battery Wh/kg", color="gas at 25 C",
                            size="size", custom_data=["molecule"],
                            hover_data=["formula", "molecule", "partner (stored gases)", "best cell (discharge)",
                                        "boils at C (Joback)", "efficiency"],
                            title="Every molecule the AI computed, as a whole battery vs lithium-ion (click to open; "
                                  "yellow ring = new molecule)")
            _nw = okf[[molecule_known(m_) is False for m_ in okf["molecule"]]]
            add_new_rings(sc, _nw["lightest cell Wh/kg (chem. + tanks)"], _nw["full battery Wh/kg"], _nw[["molecule"]].values, size=26)
            sc.add_hline(y=LI["wh_kg"], line_dash="dash", line_color="red",
                         annotation_text=f"Li-ion car pack {LI['wh_kg']:.0f} Wh/kg")
            sc.update_yaxes(rangemode="tozero")
            sc.update_layout(xaxis_title="→ Across: Wh per kg of chemicals + tanks only (best possible, same as the chart above)",
                             yaxis_title="↑ Up: Wh per kg of the whole battery",
                             legend_title_text="state at 25 C (dot size = molecule size)")
            sc.for_each_trace(lambda t: t.update(name={"yes": "gas", "no (liquid)": "liquid"}.get(t.name, t.name)))
            evs = st.plotly_chart(sc, width="stretch", on_select="rerun", selection_mode="points", key=ck("gas_sc"))
            st.caption("**How to read it.** Each dot is one molecule the AI computed, in its best cell. "
                       "**Across** = energy per kg counting only the stored chemicals and their carbon-fibre tanks: "
                       "the most the chemistry could ever give (the figure the AI optimises, as in the chart above). "
                       "**Up** = energy per kg of the complete battery sized for your settings (power, hours, pressures): "
                       "chemicals and tanks plus the fuel-cell stack, platinum, pumps and electronics. "
                       "The gap between the two is the weight of the hardware. **Above the red line** = lighter than a "
                       "lithium-ion pack. Top-right is best; a dot far right but low has great chemistry that the "
                       "hardware drags down (often a slow or low-voltage cell that needs a big stack).")
            take(evs, "gas_sc", "gasmol")
        SS["rows_gas_tbl"] = found[["molecule"]].values.tolist()
        found.insert(0, "details", "🔍 open")
        evt = st.dataframe(found, hide_index=True, width="stretch", on_select="rerun",
                           selection_mode=["single-row", "single-cell"], key=ck("gas_tbl"))
        take_row(evt, "gas_tbl", "gasmol")
        st.caption("Click 🔍 open (or any cell of a row) to see the molecule and its cells. The search ranks molecules with xTB and atom "
                   "constants fitted on half of the known gases; every molecule is then recomputed with DFT (and "
                   "CCSD(T) when small) in the background, and the table uses the best result available (its method "
                   "is shown in the molecule's details). A search that keeps the best of many uncertain numbers "
                   "tends to keep overestimated ones: the accurate recalculation is the check.")
    st.divider()


def render_gas_cell(c, cf, sp: dict, kp: str, banner: bool = True):
    if banner:
        novelty_banner("gascell", [c.key])
    st.markdown(f"**{c.name}** — {c.n_e} electrons per reaction, {c.voltage:.2f} V (± {c.band_v:.2f} V)")
    for (kind, why), side in zip(c.evidence, ("positive (oxidant) side", "negative (fuel) side")):
        st.markdown(f"- Recharge, {side}: **{kind}** — {why}")
    for s in sorted({s for s in c.stoich if s in sp and s != "O"}):
        g = sp[s]
        st.markdown(f"- {g.name} `{s}`: boils at {g.tb_k - 273.15:.0f} C ({g.tb_source}), "
                    f"{'gas' if g.gas else 'liquid'} at 25 C; energy: {g.source}"
                    + ("; " + "; ".join(g.notes) if g.notes else ""))
    def at_(sc):
        cf_ = gas_cfgs(c.key, tkey(targets_at(sc)))
        same = [x for x in cf_ if x.store.label == cf.store.label]          # the same tank option as shown
        return (same or cf_ or [None])[0]
    li_metrics(cf, kp, at=at_, kind="gascell", ids=[c.key], name=c.name)
    show_cost(cf.cost_parts, {"i_a_cm2": cf.i_a_cm2, "area_m2": cf.area_m2, "v_dis": cf.v_dis}, kp)
    st.write(f"Efficiency {cf.rte:.2f} (p10-p90 {cf.rte_band[0]:.2f}-{cf.rte_band[2]:.2f}); mass {cf.mass_kg:,.0f} kg, "
             f"volume {cf.volume_l:,.0f} L; hydrogen burst energy {cf.store.burst_tnt_kg:.2f} kg TNT.")
    st.caption("; ".join(cf.notes))


@st.cache_data(show_spinner=False)
def gas_ai_best_cells(n_energies: int, n_searched: int) -> dict:
    """For each molecule the AI found: its lightest cell in the full screen, its side and its partners."""
    cells, sp, _ = gas_screen(n_energies, n_searched)
    ai = {s_ for s_, g in sp.items() if "proposed by the AI search" in g.notes}
    out = {}
    for c in cells:                                   # lightest first
        hit = [s_ for s_ in c.stoich if s_ in ai and s_ not in out]
        for s_ in hit:
            side = ("oxidant side" if s_ in (c.pos.ox, c.pos.red) else "fuel side")
            role = "stored (charged)" if c.stoich[s_] > 0 else "product (discharged)"
            nm = lambda x: "H2O" if x == "O" else ("H2" if x == "[H][H]" else sp[x].formula if x in sp else x)
            partners = [nm(x) for x, n in c.stoich.items() if x != s_ and n > 0 and x != "O"]
            out[s_] = {"cell": c.name, "key": c.key, "side": side, "role": role,
                       "partners": ", ".join(partners) or "-", "wh_kg": round(c.est_wh_kg, 1),
                       "voltage": round(c.voltage, 2), "recharge": c.recharge}
        if len(out) == len(ai):
            break
    return out


def gas_cost_study(c, sp: dict, kp: str, skip: tuple = ()):
    """Everything about one sealed gas cell for the current settings: the two tank options, the itemised
    build cost, and what changes it."""
    novelty_banner("gascell", [c.key], skip)                 # known or new: first, before the numbers
    with st.spinner("Sizing this cell for your settings..."):
        cfgs = gas_cfgs(c.key, tkey(t))
        both = {sc: gas_cfgs(c.key, tkey(targets_at(sc)))
                for sc in ("mass", "target")}
    if not cfgs:
        st.info("This cell cannot be sized for the current settings.")
        return
    if all(both.values()):
        st.markdown("**Two estimates of the build price**: today's parts vs a battery purpose-built for this "
                    "chemistry at the published cost targets (cheapest lifetime option of each).")
        cmp_ = []
        for sc, lab in (("mass", "today's parts (mass production)"), ("target", "purpose-built (US DOE targets)")):
            x = min(both[sc], key=lambda y: y.lcos_usd_kwh)
            cmp_.append({"estimate": lab, "build USD (total)": round(x.capex_usd_kwh * E_KWH),
                         "build USD/kWh": round(x.capex_usd_kwh), "lifetime cost USD/kWh": round(x.lcos_usd_kwh, 3),
                         "stack": round(x.cost_parts.get("cell stack", 0)),
                         "power electronics": round(x.cost_parts.get("power electronics", 0)),
                         "platinum": round(x.cost_parts.get("platinum catalyst", 0)),
                         "tanks": round(sum(v_ for k_, v_ in x.cost_parts.items() if "storage" in k_)),
                         "vs Li-ion car pack": f"x{x.capex_usd_kwh / LI['usd_kwh_bev']:.1f}"})
        st.dataframe(pd.DataFrame(cmp_), hide_index=True, width="stretch")
        st.caption("Purpose-built uses the US DOE ultimate fuel-cell stack target, the 2025 power-electronics "
                   "target (a DC/DC converter, as lithium-ion pack prices include no inverter either) and the "
                   "ultimate carbon-fibre tank target. Platinum and the electricity are unchanged. Sources in the "
                   "sidebar under 'Prices the model uses'.")
    st.markdown("**Tank options** (the sealed gases are kept in carbon-fibre tanks)")
    st.dataframe(pd.DataFrame([{"tanks": x.store.label, "Wh/kg": round(x.wh_kg), "Wh/L": round(x.wh_l),
                                "build USD (total)": round(x.capex_usd_kwh * E_KWH),
                                "build USD/kWh": round(x.capex_usd_kwh),
                                "build p10-p90 (reaction speed)": band_text(x.capex_band),
                                "lifetime cost USD/kWh": round(x.lcos_usd_kwh, 3), "efficiency": round(x.rte, 2)}
                               for x in cfgs]), hide_index=True, width="stretch")
    cf = min(cfgs, key=lambda x: x.lcos_usd_kwh)
    total = sum(cf.cost_parts.values()) or 1.0
    big = sorted(cf.cost_parts.items(), key=lambda kv: -kv[1])[:3]
    kw_parts = {"cell stack", "platinum catalyst", "power electronics"}
    by_kw = sum(v_ for k_, v_ in cf.cost_parts.items() if k_ in kw_parts)
    st.markdown(f"**Build price: ${cf.capex_usd_kwh * E_KWH:,.0f} in total** for {E_KWH:,.1f} kWh "
                f"(${cf.capex_usd_kwh:,.0f}/kWh). {by_kw / total:.0%} of it is sized by the power "
                f"({power:g} kW: stack, platinum, electronics) and {1 - by_kw / total:.0%} by the energy "
                "(tanks, chemicals): with fewer hours at the same power the total drops only a little, so the "
                "price per kWh rises.")
    st.markdown("**Where the money goes** (cheapest lifetime option): " + ", ".join(
        f"{k_} {v_ / total:.0%}" for k_, v_ in big)
        + ". To lower the cost, act on the largest parts first: the cell stack shrinks with faster reactions "
          "(better catalysts) and with a lower power per kWh (longer hours); the tanks with lower pressure or "
          "fewer moles of gas per kWh; the efficiency sets the electricity cost over the lifetime. "
          "Pin this cell to see how each setting moves its cost in the ⭐ tab.")
    render_gas_cell(c, cf, sp, kp, banner=False)


@st.cache_data(show_spinner=False)
def gas_ai_sized(targets_json: str, n_energies: int, n_searched: int) -> dict:
    """Each AI molecule's lightest cell sized as a whole battery for these settings (comparable with Li-ion)."""
    tt = json.loads(targets_json)
    cells, sp, _ = gas_screen(n_energies, n_searched)
    bc = gas_ai_best_cells(n_energies, n_searched)
    keys = {v["key"] for v in bc.values()}
    cellmap = {c.key: c for c in cells if c.key in keys}
    out = {}
    for s_, v in bc.items():
        c = cellmap.get(v["key"])
        cfgs = options.gas_configurations(c, sp, tt, np.random.default_rng(3)) if c else []
        if cfgs:
            cf = max(cfgs, key=lambda x: x.wh_kg)
            out[s_] = {"wh_kg": round(cf.wh_kg, 1), "wh_l": round(cf.wh_l, 1), "usd": round(cf.capex_usd_kwh),
                       "rte": round(cf.rte, 2)}
    return out


@st.cache_data(show_spinner=False)
def gas_cells_of(smiles: str, n_energies: int, n_searched: int, k: int = 8) -> list[str]:
    """Keys of the lightest cells this molecule is part of."""
    cells, _, _ = gas_screen(n_energies, n_searched)
    out = []
    for c in cells:
        if smiles in c.stoich:
            out.append(c.key)
            if len(out) >= k:
                break
    return out


@st.dialog("Sealed gas cell detail", width="large")
def gas_cell_dialog(key: str):
    with st.spinner("Computing the details: a few seconds the first time, instant afterwards..."):
        c, sp = gas_cell_by_key(key)
        if c is None:
            st.info("This cell is no longer in the screen.")
            return
        pin_button("gascell", [key], "_gcdlg")
        gas_cost_study(c, sp, "gcdlg")


@st.dialog("Gas molecule detail", width="large")
def gas_molecule_dialog(smiles: str):
    with st.spinner("Computing the details: a few seconds the first time, instant afterwards..."):
        rec = gas_search.load()["molecules"].get(smiles, {})
        n_e, n_s = gas_energy_count(), len(gases.searched_species())
        with st.spinner("Loading the screened cells..."):
            cells, sp, _ = gas_screen(n_e, n_s)
        g = sp.get(smiles)
        a, b = st.columns([1, 2])
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw
            a.image(Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(260, 220)))
        except Exception:
            pass
        h1, h2 = b.columns([3, 2], vertical_alignment="center")
        h1.markdown(f"### {gases.formula_of(smiles)}\n`{smiles}`")
        pin_slot = h2.empty()                          # the ⭐ button, for the cell chosen below
        kn_ = molecule_known(smiles)
        if kn_ is False:
            b.success("🆕 **NEW molecule**: never reported (not in PubChem), found by the AI.")
        elif kn_:
            b.info("✔ **KNOWN compound**" + (" (its measured data are used only to test the method)"
                                              if smiles in gases.MEASURED else " (in PubChem)"))
        else:
            b.caption("? Known or new: not checked (PubChem could not be reached)")
        if g is not None:
            b.markdown(f"- boils at **{g.tb_k - 273.15:.0f} C** ({g.tb_source}): {'gas' if g.gas else 'liquid'} at 25 C\n"
                       f"- Gibbs energy of formation: **{g.g_kj:.0f} kJ/mol** in that state ({g.source})\n"
                       f"- molar mass {g.mw:.1f} g/mol" + ("; toxic" if g.toxic else ""))
        if rec:
            b.markdown(f"- found by the AI in generation {rec.get('generation')}; the model predicted "
                       f"{rec.get('predicted_g_kj')} kJ/mol, xTB gave {rec.get('g_gas_kj')} kJ/mol")
        if g is None:
            st.warning("This molecule has no usable energy (xTB failed or a storage alert): it makes no cell.")
            return
        with st.spinner("Finding the cells this molecule is part of..."):
            keys = gas_cells_of(smiles, n_e, n_s)
        if not keys:
            st.info("No sealed cell with this molecule passes the screen (voltage, gases, water window).")
            return
        by_key = {c.key: c for c in cells if c.key in set(keys)}
        rows, sized_ = [], []
        rng = np.random.default_rng(3)
        bar = st.progress(0.0, text="Sizing its cells as whole batteries for your settings...")
        for i_, k_ in enumerate(keys):
            bar.progress((i_ + 1) / len(keys), text=f"Sizing cell {i_ + 1} of {len(keys)} as a whole battery...")
            c = by_key[k_]
            cfgs = gas_cfgs(c.key, tkey(t))
            cf = max(cfgs, key=lambda x: x.wh_kg) if cfgs else None
            sized_.append((c, cf))
            rows.append({"partner (stored gases)": ", ".join(sp[x].formula if x in sp else x for x, n_ in c.stoich.items()
                                                             if x not in (smiles, "O") and n_ > 0) or "-",
                         "its side": "oxidant" if smiles in (c.pos.ox, c.pos.red) else "fuel",
                         "cell (discharge)": c.name, "voltage V": round(c.voltage, 2),
                         "Wh/kg chem. + tanks": round(c.est_wh_kg),
                         "Wh/kg full system": round(cf.wh_kg) if cf else None,
                         "Wh/L full system": round(cf.wh_l) if cf else None,
                         "build USD/kWh": round(cf.capex_usd_kwh) if cf else None,
                         "efficiency": round(cf.rte, 2) if cf else None,
                         "recharge proven?": c.recharge})
        bar.empty()
        st.markdown("**The same molecule with different partners** (its 8 lightest cells, sized for your settings): "
                    "the partner changes the voltage, the electrons per reaction and the tanks, so the battery changes.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        ok_ = [(c, cf) for c, cf in sized_ if cf is not None]
        if not ok_:
            st.info("None of these cells can be sized for the current settings.")
            return
        best_i = max(range(len(sized_)), key=lambda i: sized_[i][1].wh_kg if sized_[i][1] else -1)
        pick_i = st.selectbox("Study one of these cells (cost, tanks, against lithium-ion)", range(len(sized_)),
                              index=best_i, format_func=lambda i: f"{rows[i]['partner (stored gases)']}: {sized_[i][0].name}",
                              key="gdlg_pick")
        c, cf = sized_[pick_i]
        with pin_slot.container():
            pin_button("gascell", [c.key], "_gdlg")
            st.caption(f"adds the cell chosen below: {rows[pick_i]['partner (stored gases)']}")
        if cf is None:
            st.info("This cell cannot be sized for the current settings.")
            return
        gas_cost_study(c, sp, "gdlg", skip=(smiles,))


def gas_tab():
    st.subheader("Sealed batteries that store their energy in two gases")
    st.caption("Still hydrogen batteries: each gas couple takes up or gives back hydrogen (A + H2 -> B + H2O), "
               "and hydrogen crosses the membrane as protons. Only some cells keep H2 itself as a gas.")
    st.markdown(
        "Every molecule with up to 4 atoms of C, N, O, F, S or Cl is generated (nothing is picked by hand), "
        "and an AI search (below) proposes larger ones. "
        "Its energy comes from measured tables when the molecule is known, otherwise from a quantum-chemistry "
        "calculation (GFN2-xTB). Every pair of an **oxidant gas** and a **fuel gas** that exchange hydrogen "
        "through an acid membrane is built as a **sealed battery**: each gas in its own carbon-fibre tank, "
        "liquids (water...) in a small tank, then sized for your power and hours like every other design. "
        "A tank wall weighs the same per mole of gas at any pressure, so what makes a gas battery light is "
        "**energy per mole of stored gas** (electrons x voltage).")
    job = gas_job()
    running = job["thread"] is not None and job["thread"].is_alive()
    n_e = xtb_energy_count()
    todo = len([x for x in gases.to_compute() if x not in gases.load_energies()])
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{len(gases.all_species()):,} molecules generated; {n_e:,} computed with xTB; "
               f"{todo:,} still to compute." + (f" Computing: {job['done']}/{job['total']}..." if running else ""))
    if c2.button("Compute missing molecules", disabled=running or todo == 0, width="stretch"):
        import threading

        def work():
            gases.compute_energies(progress=lambda k, n: job.update(done=k, total=n))
        job["thread"] = threading.Thread(target=work, daemon=True)
        job["thread"].start()
        st.rerun()
    if n_e == 0:
        st.info("No molecule computed yet: press 'Compute missing molecules' (a few minutes).")
        return
    gas_search_panel()
    st.markdown("#### Best sealed gas batteries, sized for your settings")
    f1, f2, f3, f4 = st.columns(4)
    top = f1.slider("Cells to size in detail", 5, 60, 25, key="gas_top")
    measured_only = f2.toggle("Only molecules that exist (measured)", False, key="gas_measured")
    no_toxic = f3.toggle("Hide toxic gases", False, key="gas_notoxic")
    proven_only = f4.toggle("Only proven rechargeable", False, key="gas_proven")
    with st.spinner("Sizing the best gas batteries..."):
        sized, sp = gas_sized(tkey(t), n_e, len(gases.searched_species()), top, measured_only, no_toxic, proven_only)
    _, _, fit = gas_screen(n_e, len(gases.searched_species()))
    if fit:
        st.caption(f"Computed molecules: Gibbs energy from xTB with atom equivalents fitted to {fit.n} measured "
                   f"molecules, leave-one-out error {fit.rmse_loo:.0f} kJ/mol (about "
                   f"{fit.rmse_loo * 1000 / (2 * 96485):.2f} V on a 2-electron step).")
    if not sized:
        st.warning("No sealed two-gas cell passes these filters.")
        return
    best = max(sized, key=lambda x: x[1].wh_kg)[1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Lightest gas battery", f"{best.wh_kg:.0f} Wh/kg", f"{best.wh_kg - LI['wh_kg']:+.0f} vs Li-ion pack {LI['wh_kg']:.0f}")
    k2.metric("Its volume", f"{best.wh_l:.0f} Wh/L", f"{best.wh_l - LI['wh_l']:+.0f} vs Li-ion pack {LI['wh_l']:.0f}")
    v_, d_, h_ = explain.cost_range(best.capex_usd_kwh, best.capex_band, LI["usd_kwh_stationary"])
    k3.metric("Its build cost", v_, d_, delta_color="inverse", help=h_)
    rows = []
    for c, cf in sized:
        rows.append({
            "reaction (discharge)": c.name,
            "voltage V": f"{c.voltage:.2f} ± {c.band_v:.2f}",
            "Wh per mol of gas": round(c.wh_per_mol_gas, 1),
            "Wh/kg (system)": round(cf.wh_kg),
            "Wh/L (system)": round(cf.wh_l),
            "build USD/kWh": round(cf.capex_usd_kwh),
            "build p10-p90 (reaction speed)": band_text(cf.capex_band),
            "efficiency": round(cf.rte, 2),
            "recharge proven?": c.recharge,
            "beats Li-ion on": ", ".join(k for k, ok in (("weight", cf.wh_kg > LI["wh_kg"]), ("volume", cf.wh_l > LI["wh_l"]),
                                                          ("cost", cf.capex_usd_kwh < LI["usd_kwh_stationary"])) if ok) or "-",
            "toxic": ", ".join(c.toxic) or "-",
            "computed only (may not exist)": ", ".join(sp[s].formula for s in c.computed_only) or "-",
            "found by the AI": ", ".join(sp[s].formula for s in c.computed_only
                                         if "proposed by the AI search" in sp[s].notes) or "-",
            "tanks": cf.store.label,
        })
    df_ = pd.DataFrame(rows).sort_values("Wh/kg (system)", ascending=False)
    st.dataframe(df_, hide_index=True, width="stretch")
    st.download_button("Download table (CSV)", df_.to_csv(index=False), "gas_batteries.csv")
    names = [c.name for c, _ in sized]
    pick = st.selectbox("Details of", range(len(sized)), format_func=lambda i: names[i], key="gas_pick")
    c, cf = sized[pick]
    pin_button("gascell", [c.key], "_gastab")
    render_gas_cell(c, cf, sp, "gas")


# ------------------------------------------------------------------ tab 1

def table_couples_detail():
    df = design_rows(designs).sort_values("LCOS USD/kWh") if designs else pd.DataFrame()
    st.caption(f"{len(results)} couples screened from the potential table and Materials Project; {len(df)} sized "
               f"for {power:g} kW x {hours:g} h, each at its best pressure.")
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        bk, bl = df["Wh/kg (system)"].max(), df["Wh/L (system)"].max()
        bc, bt = df["build USD/kWh"].min(), df["fastest charge h"].min() * 60 * 0.7
        c1.metric("Energy per kg", f"{bk:.0f} Wh/kg", f"{bk - LI['wh_kg']:+.0f} vs Li-ion pack {LI['wh_kg']:.0f}")
        c2.metric("Energy per litre", f"{bl:.0f} Wh/L", f"{bl - LI['wh_l']:+.0f} vs Li-ion pack {LI['wh_l']:.0f}")
        c3.metric("Build cost", f"${bc:.0f}/kWh", f"{bc - LI['usd_kwh_stationary']:+.0f} vs Li-ion ${LI['usd_kwh_stationary']:.0f}",
                  delta_color="inverse")
        c4.metric("Charge 10-80 %", f"{bt:.0f} min", f"{bt - LI['charge_min_10_80']:+.0f} vs Li-ion {LI['charge_min_10_80']:.0f} min",
                  delta_color="inverse")
        st.caption("Green = better than lithium-ion, red = worse. Each figure is the best single design for that "
                   "figure, so no one design need reach all four. Lifetime cannot be predicted and is not compared.")
        wins = df[[c for c in df.columns if c.startswith("vs Li-ion")]].eq("better").sum(axis=1)
        df.insert(0, "beats Li-ion on", wins)
        st.caption("Click a row (or any point in the charts) to open everything about that candidate.")
        SS["rows_best_tbl"] = df[["couple"]].values.tolist()
        ev = st.dataframe(df, width="stretch", hide_index=True, on_select="rerun",
                          selection_mode=["single-row", "single-cell"], key=ck("best_tbl"))
        take_row(ev, "best_tbl", "couple")
        st.download_button("Download table (CSV)", df.to_csv(index=False), "h2_candidates.csv")

# ------------------------------------------------------------------ the main comparison (all battery types)

TYPE_COLORS = {"table couple (H2 + inorganic)": "#60a5fa", "molecule + hydrogen gas": "#f59e0b",
               "two molecules (no gas)": "#a78bfa", "two gases (sealed)": "#2dd4bf"}


def _row(type_, name, kind, ids, wh_kg, wh_l, usd_kwh, lcos, rte, extra=""):
    return {"type": type_, "name": name, "kind": kind, "ids": list(ids), "Wh/kg": float(wh_kg),
            "Wh/L": float(wh_l), "build USD/kWh": float(usd_kwh), "lifetime cost USD/kWh": float(lcos),
            "efficiency": float(rte), "extra": extra}


def latest_finds(k: int = 30) -> tuple[tuple, tuple]:
    """Candidates for 'just found': the k best (by the search's own score; zero-score molecules never) computed by
    the newest molecule search and by the newest pair search, and the k best of the 30 newest larger gas
    molecules. tradeoff_points sizes them as whole batteries and keeps the most promising few."""
    store = ai_search.load_store()
    mols = []
    for mode in ("positive", "pair"):
        rr = [r for r in runs.list_runs(mode) if r.get("molecules")]
        if rr:
            mols += [m["smiles"] for m in rr[0]["molecules"] if m.get("smiles") in store]
    for c in search_controls().values():                       # a search running now: its molecules so far
        mols += [x for x in getattr(c, "evaluated", []) if x in store]
    mols = [x for x in dict.fromkeys(mols) if store[x].objective > 0 and not pairs._unstable(x)]
    mols = sorted(mols, key=lambda x: -store[x].objective)[:k]
    gm = gas_search.load()["molecules"]
    newest = sorted((x for x, r in gm.items() if r.get("g_gas_kj") is not None and r.get("value_wh_kg", 0) > 0),
                    key=lambda x: gm[x].get("found", ""), reverse=True)[:30]
    gas = sorted(newest, key=lambda x: -gm[x].get("value_wh_kg", 0))[:k]
    return tuple(mols), tuple(gas)


def tradeoff_args() -> tuple:
    """Everything the comparison depends on besides the settings (cache key)."""
    nm, ng = latest_finds()
    pins = tuple(tuple(x) for x in (SS.get("watchlist") or []) if x)
    return (len(ai_search.load_store()), gas_energy_count(), len(gases.searched_species()), nm, ng, pins)


def pin_number(kind: str, ids: list) -> int | None:
    """1-based position of a candidate in the ⭐ list (the same number everywhere), or None."""
    for i, p_ in enumerate([x for x in (SS.get("watchlist") or []) if x], 1):
        if p_[0] == kind and list(p_[1:1 + len(ids)]) == list(ids):
            return i
    return None


def _gas_key_has(key: str, smiles: set) -> bool:
    return any(part in smiles for part in key.replace("|", ">").split(">"))


@st.cache_data(show_spinner=False)
def tradeoff_points(targets_json: str, n_mol: int, n_gas_e: int, n_gas_s: int, new_mols: tuple = (),
                    new_gas: tuple = (), pins: tuple = (), version: int = MODEL_VERSION) -> list:
    """Every battery type, each candidate sized as a whole battery for these settings (plain dicts)."""
    tt = json.loads(targets_json)
    e = tt["design"]["power_kw"] * tt["design"]["hours"]
    rows = []
    names_ = tuple(d.cell.name for d in designs)
    for d in (designs if targets_json == tkey(t) else size(targets_json, names_, ())):   # table couples
        b = d.best
        if b is not None:
            rows.append(_row("table couple (H2 + inorganic)", d.cell.name, "couple", [d.cell.name],
                             e * 1000 / b.mass_kg, e * 1000 / b.volume_l, b.capex_usd_kwh, b.lcos_usd_kwh, b.rte,
                             f"{b.p_max_bar:g} bar, burst {b.safety.burst_tnt_kg:.2f} kg TNT"))
    store = [r for r in ai_search.load_store().values() if math.isfinite(r.e0)]
    rng = np.random.default_rng(11)
    by_s = {r.smiles: r for r in store}
    top_m = sorted((r for r in store if not pairs._unstable(r.smiles)), key=lambda r: -r.objective)[:15]
    base_m = {r.smiles for r in top_m}
    top_m += [by_s[x] for x in new_mols if x in by_s and by_s[x] not in top_m]
    top_m += [by_s[p_[1]] for p_ in pins if p_[0] == "molecule" and p_[1] in by_s and by_s[p_[1]] not in top_m]
    for r in top_m:
        cfgs = options.configurations(M.as_cell(r), tt, rng)
        if cfgs:
            c = min(cfgs, key=lambda x: x.lcos_usd_kwh)
            rows.append(_row("molecule + hydrogen gas", r.smiles, "molecule", [r.smiles], c.wh_kg, c.wh_l,
                             c.capex_usd_kwh, c.lcos_usd_kwh, c.rte, c.store.label))
            rows[-1]["new"] = r.smiles in new_mols
            rows[-1]["added"] = r.smiles not in base_m
            rows[-1]["ai"] = True
    prs = pairs.best_pairs(store, 15, max_per_molecule=2)
    base_p = {(p_.neg.smiles, p_.pos.smiles) for p_ in prs}
    if new_mols:                                   # the best pairs that use a molecule just found
        nset = set(new_mols)
        seen = {(p_.neg.smiles, p_.pos.smiles) for p_ in prs}
        prs += [p_ for p_ in pairs.best_pairs(store, 200, max_per_molecule=2)
                if (p_.neg.smiles in nset or p_.pos.smiles in nset) and (p_.neg.smiles, p_.pos.smiles) not in seen][:8]
    seen = {(p_.neg.smiles, p_.pos.smiles) for p_ in prs}
    for p_ in pins:                                  # ⭐ pairs, even when not among the best
        if p_[0] == "pair" and p_[1] in by_s and p_[2] in by_s and (p_[1], p_[2]) not in seen:
            try:
                prs.append(pairs.make_pair(by_s[p_[1]], by_s[p_[2]], int(p_[3]), int(p_[4])))
            except Exception:
                pass
    for pr in prs:
        c = options.pair_configuration(pr, tt, rng)
        if c is not None:
            rows.append(_row("two molecules (no gas)", f"{pr.neg.smiles} | {pr.pos.smiles}", "pair",
                             [pr.neg.smiles, pr.pos.smiles], c.wh_kg, c.wh_l, c.capex_usd_kwh, c.lcos_usd_kwh, c.rte))
            rows[-1]["new"] = pr.neg.smiles in new_mols or pr.pos.smiles in new_mols
            rows[-1]["added"] = (pr.neg.smiles, pr.pos.smiles) not in base_p
            rows[-1]["ai"] = True
    if n_gas_e:
        cells, sp, _ = gas_screen(n_gas_e, n_gas_s)
        abc = gas_ai_best_cells(n_gas_e, n_gas_s)
        ai_gas = set(gases.searched_species())
        keys = [c.key for c in cells[:20]] + [v["key"] for v in sorted(abc.values(), key=lambda v: -v["wh_kg"])[:15]]
        base_g = set(keys)
        keys += [abc[x]["key"] for x in new_gas if x in abc]
        keys += [p_[1] for p_ in pins if p_[0] == "gascell"]
        ref = gases.reference_h2_o2(cells)
        if ref is not None:
            keys.append(ref.key)
        idx = gas_index(n_gas_e, n_gas_s)
        for k_ in dict.fromkeys(keys):
            c = idx.get(k_)
            cfgs = options.gas_configurations(c, sp, tt, np.random.default_rng(3)) if c else []
            if cfgs:
                cf = min(cfgs, key=lambda x: x.lcos_usd_kwh)
                rows.append(_row("two gases (sealed)", c.name, "gascell", [k_], cf.wh_kg, cf.wh_l, cf.capex_usd_kwh,
                                 cf.lcos_usd_kwh, cf.rte, f"{cf.store.label}; recharge {c.recharge}"))
                rows[-1]["new"] = _gas_key_has(k_, set(new_gas))
                rows[-1]["added"] = k_ not in base_g
                rows[-1]["ai"] = _gas_key_has(k_, ai_gas)
    return keep_promising_new(rows)


LI_RTE = 0.92            # assumption: round-trip efficiency of a lithium-ion pack
LI_CYCLES_NMC = 1500     # assumption: full cycles of an NMC pack (LFP: LI["cycles_lfp"])


def li_lcos(price_usd_kwh: float, cycles: float, tt: dict) -> float:
    """Lifetime cost of a lithium-ion pack, USD per kWh delivered, with the same formula and settings as the
    hydrogen designs (options.py): build, replacements when the cycles run out, O&M, charging electricity."""
    d = tt["design"]
    rate, years, per_year = d["discount_rate"], int(d["lifetime_years"]), d["cycles_per_year"]
    life = max(cycles / per_year, 0.5)
    capex = price_usd_kwh
    repl = sum(price_usd_kwh / (1 + rate) ** y for y in np.arange(life, years, life))
    pv = sum(1 / (1 + rate) ** y for y in range(1, years + 1))
    opex = d["om_fraction_capex"] * capex
    charging = per_year / LI_RTE * d["electricity_usd_kwh"]
    return (capex + repl + pv * (opex + charging)) / (per_year * pv)


def li_packs(tt: dict) -> list[dict]:
    """Today's lithium-ion packs with every figure the comparison chart can use."""
    out = []
    for b in REF["battery"]:
        if b["level"] != "pack" or "wh_kg" not in b:
            continue
        lfp = "LFP" in b["name"]
        price = REF["prices"]["pack_lfp_usd_kwh"] if lfp else REF["prices"]["pack_bev_usd_kwh"]
        cyc = LI["cycles_lfp"] if lfp else LI_CYCLES_NMC
        out.append({"name": b["name"].split(" (")[0].split(",")[0], "Wh/kg": max(b["wh_kg"]),
                    "Wh/L": max(b["wh_l"]) if "wh_l" in b else None, "build USD/kWh": price,
                    "lifetime cost USD/kWh": li_lcos(price, cyc, tt), "cycles": cyc})
    return out


def keep_promising_new(rows: list, k: int = 5) -> list:
    """Of the rows flagged new, keep the flag on the k per battery type that come closest to lithium-ion as whole
    batteries (most Wh/kg per USD/kWh); rows that were only added because they were new and did not make it are
    dropped, so they do not clutter the chart. ⭐ candidates are never dropped."""
    for r in rows:
        r.setdefault("added", False)
    for ty in {r["type"] for r in rows}:
        new = sorted((r for r in rows if r["type"] == ty and r.get("new")),
                     key=lambda r: -r["Wh/kg"] / max(r["build USD/kWh"], 1e-9))
        for r in new[k:]:
            r["new"] = False
            if r.get("added") and pin_number(r["kind"], r["ids"]) is None:
                r["drop"] = True
    return [r for r in rows if not r.get("drop")]


def pareto_front(xs, ys) -> list[int]:
    """Indices of the points no other point beats on both: higher x and lower y."""
    order = sorted(range(len(xs)), key=lambda i: (-xs[i], ys[i]))
    out, best_y = [], math.inf
    for i in order:
        if ys[i] < best_y:
            out.append(i)
            best_y = ys[i]
    return sorted(out, key=lambda i: xs[i])


def comparison_chart():
    st.markdown("#### Did anything beat lithium-ion? Every battery type on one chart")
    st.caption("Each dot is one candidate sized as a whole battery for your settings (cells, catalyst, tanks, "
               "electronics). Red stars are today's lithium-ion packs. Anything in the green zone is lighter "
               "and cheaper than lithium-ion. The pale yellow line joins the best trade-offs: for each of them, no "
               "other candidate is both lighter and cheaper (the frontier to push towards the green zone). Click a "
               "dot to open it.")
    a1, a2, a0, a3 = st.columns([1, 1, 1.3, 2.2])
    default("trade_x", "Wh/kg")
    xk = a1.radio("Across", ["Wh/kg", "Wh/L"], key="trade_x", horizontal=True,
                  format_func=lambda k: {"Wh/kg": "energy per kg", "Wh/L": "energy per litre"}[k])
    default("trade_y", "build USD/kWh")
    yk = a2.radio("Up", ["build USD/kWh", "lifetime cost USD/kWh"], key="trade_y", horizontal=True,
                  format_func=lambda k: {"build USD/kWh": "build price", "lifetime cost USD/kWh": "lifetime cost"}[k])
    PRICE = {"mass": "today's parts", "target": "purpose-built", "both": "both (line = saving)"}
    default("trade_price", "both")
    pr_ = a0.radio("Prices", list(PRICE), key="trade_price", format_func=PRICE.get,
                   help="Today's parts: current fuel-cell and flow-battery hardware at mass production. "
                        "Purpose-built: a battery designed for its one chemistry, at the published US DOE cost "
                        "targets for the stack, power electronics and tanks (sources in the sidebar).")
    nk = tradeoff_args()
    with st.spinner("Sizing every battery type for your settings (first time only)..."):
        sets = {sc: tradeoff_points(tkey(targets_at(sc)), *nk) for sc in (("mass", "target") if pr_ == "both" else (pr_,))}
    rows = sets["target" if pr_ != "mass" else "mass"]
    ghost = sets.get("mass") if pr_ == "both" else None
    if not rows:
        st.info("Nothing sized yet.")
        return
    types = [k_ for k_ in TYPE_COLORS if any(r["type"] == k_ for r in rows)]
    shown = a3.multiselect("Battery types", types, default=types, key="trade_types")
    default("trade_show", "all")
    only_ai = st.radio("Show", ["all", "ai"], key="trade_show", horizontal=True,
                       format_func={"all": "everything (starting data + AI finds)",
                                    "ai": "only what the AI found + my ⭐"}.get,
                       help="Starting data: every couple of the published potential table and Materials Project "
                            "(blue) and the cells of every small gas molecule up to 4 heavy atoms (green). They are "
                            "computed from tables and physics, not found by the AI, so a reset does not remove them.")
    if only_ai == "ai":
        rows = [r for r in rows if r.get("ai") or pin_number(r["kind"], r["ids"])]
        ghost = [g for g in (ghost or []) if g.get("ai") or pin_number(g["kind"], g["ids"])]
    rows = [r for r in rows if r["type"] in shown and r[yk] > 0 and r[xk] > 0]
    if not rows:
        st.info("Nothing to show: the AI has not found anything yet with these filters. Run a search in "
                "'AI discovery', or show everything.")
        return
    ghost = [g for g in (ghost or []) if g["type"] in shown and g[yk] > 0 and g[xk] > 0]
    names_all = sorted({r["name"] for r in rows})
    n_new = sum(1 for r in rows if r.get("new"))
    SS["trade_hl"] = [x for x in SS.get("trade_hl", []) if x in names_all]
    hl = st.multiselect("Highlight candidates (type part of a name, formula or SMILES)", names_all, key="trade_hl",
                        help="Rings and labels them on the chart. The AI tabs' 'Show it in the trade-off chart' "
                             "button fills this in for you.")
    wl = [x for x in (SS.get("watchlist") or []) if x]
    pinned = lambda r: pin_number(r["kind"], r["ids"]) is not None
    packs = li_packs(t)
    li_lim = LI["wh_kg"] if xk == "Wh/kg" else LI["wh_l"]
    car = next((p_ for p_ in packs if "LFP" not in p_["name"] and "Blade" not in p_["name"]), packs[0])
    li_y = (LI["usd_kwh_bev"] if yk == "build USD/kWh" else car["lifetime cost USD/kWh"])
    li_floor = LI["usd_kwh_stationary"] if yk == "build USD/kWh" else min(p_[yk] for p_ in packs)
    x_max = max(max(r[xk] for r in rows), li_lim) * 1.12
    y_min = min(min(r[yk] for r in rows), li_floor) / 3
    y_max = max([r[yk] for r in rows] + [g[yk] for g in ghost]) * 1.6
    x_max = max([x_max] + [g[xk] * 1.12 for g in ghost])
    f = go.Figure()
    if li_y:
        f.add_shape(type="rect", x0=li_lim, x1=x_max, y0=y_min, y1=li_y, line_width=0,
                    fillcolor="rgba(34,197,94,0.13)", layer="below")
        f.add_annotation(x=(li_lim + x_max) / 2, y=(math.log10(y_min) + math.log10(li_y)) / 2,
                         showarrow=False,
                         text="<b>beats lithium-ion</b><br>lighter and cheaper", font=dict(color="#4ade80", size=12))
    if ghost:                                            # today's prices, faded, with a line to purpose-built
        now_ = {(r["kind"], tuple(r["ids"])): r for r in rows}
        lx, ly = [], []
        for g in ghost:
            r = now_.get((g["kind"], tuple(g["ids"])))
            if r:
                lx += [g[xk], r[xk], None]
                ly += [g[yk], r[yk], None]
        f.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="rgba(229,231,235,0.25)", width=1),
                               hoverinfo="skip", showlegend=False))
        f.add_trace(go.Scatter(x=[g[xk] for g in ghost], y=[g[yk] for g in ghost], mode="markers",
                               name="same candidates with today's parts",
                               marker=dict(size=7, color=[TYPE_COLORS[g["type"]] for g in ghost], opacity=0.3),
                               text=[g["name"] for g in ghost],
                               hovertemplate="<b>%{text}</b><br>today's parts: %{x:.0f} " + xk + ", %{y:,.2f} " + yk
                                             + "<extra></extra>"))
    for ty in shown:
        rs = [r for r in rows if r["type"] == ty]
        if not rs:
            continue
        f.add_trace(go.Scatter(
            x=[r[xk] for r in rs], y=[r[yk] for r in rs], mode="markers", name=ty,
            marker=dict(size=[15 if pinned(r) else 10 for r in rs], color=TYPE_COLORS[ty], opacity=0.85,
                        line=dict(color=["#ffffff" if pinned(r) else "rgba(0,0,0,0.4)" for r in rs],
                                  width=[3 if pinned(r) else 1 for r in rs])),
            customdata=[[r["kind"], *r["ids"]] + [""] * (3 - len(r["ids"])) for r in rs],
            meta=[r["name"] for r in rs],
            hovertext=[novelty_tag(r["kind"], r["ids"]) + f"<br>efficiency {r['efficiency']:.2f}"
                       + (f"; {r['extra']}" if r["extra"] else "")
                       + ("; ⭐ pinned" if pinned(r) else "") for r in rs],
            hovertemplate=("<b>%{meta}</b><br>" + ty + "<br>%{x:.0f} " + xk + "<br>%{y:,.2f} " + yk +
                           "<br>%{hovertext}<extra></extra>")))
    _nr = [r for ty in shown for r in rows if r["type"] == ty and novelty_tag(r["kind"], r["ids"]).startswith("🆕")]
    add_new_rings(f, [r[xk] for r in _nr], [r[yk] for r in _nr],
                  [[r["kind"], *r["ids"]] + [""] * (3 - len(r["ids"])) for r in _nr], size=19)
    # name the best of each type (furthest right) and every highlighted candidate
    lab = {}
    for ty in shown:
        rs = [r for r in rows if r["type"] == ty]
        if rs:
            b_ = max(rs, key=lambda r: r[xk])
            lab[b_["name"]] = (b_, f"best {ty.split(' (')[0]}: ")
    for r in rows:
        if r["name"] in hl:
            lab[r["name"]] = (r, "")
    newr = [r for r in rows if r.get("new")]
    pr_rows = [(pin_number(r["kind"], r["ids"]), r) for r in rows]
    pr_rows = sorted([(n_, r) for n_, r in pr_rows if n_], key=lambda x: x[0])
    if pr_rows:
        f.add_trace(go.Scatter(x=[r[xk] for _, r in pr_rows], y=[r[yk] for _, r in pr_rows], mode="markers+text",
                               name="⭐ your candidates (number = their place in your ⭐ list)",
                               text=[f"<b>{n_}</b>" for n_, _ in pr_rows], textposition="middle right",
                               textfont=dict(color="#facc15", size=20),
                               marker=dict(symbol="star", size=24, color="rgba(250,204,21,0.25)",
                                           line=dict(color="#facc15", width=2)),
                               customdata=[[r["kind"], *r["ids"]] + [""] * (3 - len(r["ids"])) for _, r in pr_rows],
                               hovertext=[f"⭐{n_} {r['name']}" for n_, r in pr_rows],
                               hovertemplate="<b>%{hovertext}</b><br>%{x:.0f} " + xk + ", %{y:,.2f} " + yk
                                             + "<extra></extra>"))
    if newr:                             # always shown; click its legend entry to hide it
        f.add_trace(go.Scatter(x=[r[xk] for r in newr], y=[r[yk] for r in newr], mode="markers+text",
                               name=f"🆕 latest AI finds ({len(newr)}: the most promising of your latest searches)", text=["NEW"] * len(newr),
                               textposition="top center", textfont=dict(color="#a3e635", size=10),
                               marker=dict(symbol="diamond-open", size=22, color="#a3e635", line=dict(width=3)),
                               customdata=[[r["kind"], *r["ids"]] + [""] * (3 - len(r["ids"])) for r in newr],
                               hovertext=[r["name"] for r in newr],
                               hovertemplate="🆕 <b>%{hovertext}</b><br>%{x:.0f} " + xk + ", %{y:,.2f} " + yk
                                             + "<extra></extra>"))
    if hl:
        hr = [r for r in rows if r["name"] in hl]
        f.add_trace(go.Scatter(x=[r[xk] for r in hr], y=[r[yk] for r in hr], mode="markers", name="highlighted",
                               marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color="#f97316", width=3)),
                               hoverinfo="skip"))
    for i_, (nm, (r, pre)) in enumerate(lab.items()):
        f.add_annotation(x=r[xk], y=math.log10(r[yk]), text=pre + (nm if len(nm) <= 38 else nm[:36] + "…"),
                         showarrow=True, arrowhead=0, ax=30 + 10 * (i_ % 3), ay=-30 - 16 * (i_ % 4),
                         font=dict(size=10, color="#f97316" if nm in hl else "#e5e7eb"),
                         bgcolor="rgba(17,24,39,0.85)")
    xs, ys = [r[xk] for r in rows], [r[yk] for r in rows]
    fr = pareto_front(xs, ys)
    f.add_trace(go.Scatter(x=[xs[i] for i in fr], y=[ys[i] for i in fr], mode="lines+markers",
                           name="best trade-off line: nothing else is both lighter and cheaper",
                           line=dict(color="#fde68a", width=2), marker=dict(size=5, color="#fde68a"),
                           hoverinfo="skip"))
    drawn = [p_ for p_ in packs if p_[xk] is not None]
    yfmt = "$%{y:.0f}/kWh" if yk == "build USD/kWh" else "$%{y:.3f}/kWh delivered"
    f.add_trace(go.Scatter(
        x=[p_[xk] for p_ in drawn], y=[p_[yk] for p_ in drawn], mode="markers+text",
        name="lithium-ion packs today", text=[p_["name"] for p_ in drawn],
        hovertext=[f"{p_['cycles']:,.0f} cycles" for p_ in drawn],
        textposition=[("middle left", "middle right")[i % 2] for i in range(len(drawn))],
        textfont=dict(color="#fca5a5", size=11),
        marker=dict(symbol="star", size=20, color="#ef4444", line=dict(color="white", width=1)),
        hovertemplate="%{text}<br>%{x:.0f} " + xk + ", " + yfmt + "<br>%{hovertext}<extra></extra>"))
    if yk == "build USD/kWh":
        f.add_hline(y=LI["usd_kwh_stationary"], line_dash="dot", line_color="#9ca3af",
                    annotation_text=f"Li-ion stationary ${LI['usd_kwh_stationary']:.0f}/kWh",
                    annotation_position="top right", annotation_font_color="#9ca3af")
    missing = [p_["name"] for p_ in packs if p_[xk] is None]
    notes_ = []
    if missing:
        notes_.append(f"{', '.join(missing)}: no {xk} figure in the reference data, so no star on this axis "
                      f"(the line is the best pack that has one, {li_lim:.0f} {xk}).")
    if yk == "lifetime cost USD/kWh":
        notes_.append(f"Lithium-ion lifetime cost uses the same formula and settings as the hydrogen designs "
                      f"(price, O&M, charging electricity, {t['design']['lifetime_years']} years), with a new pack "
                      f"each time the cycles run out (assumptions: LFP {LI['cycles_lfp']:,} cycles, NMC "
                      f"{LI_CYCLES_NMC:,}, efficiency {LI_RTE:.2f}).")
    if notes_:
        st.caption(" ".join(notes_))
    f.add_vline(x=li_lim, line_dash="dot", line_color="#ef4444", opacity=0.6,
                annotation_text=f"best Li-ion pack {li_lim:.0f} {xk}", annotation_position="top",
                annotation_font_color="#fca5a5")
    f.update_xaxes(title=("energy per kg of the whole battery, Wh/kg" if xk == "Wh/kg"
                          else "energy per litre of the whole battery, Wh/L"), range=[0, x_max])
    f.update_yaxes(title=("build price, USD per kWh (log scale)" if yk == "build USD/kWh"
                          else "lifetime cost, USD per kWh delivered (log scale)"),
                   type="log", range=[math.log10(y_min), math.log10(y_max)])
    f.update_layout(height=620, legend=dict(orientation="h", y=-0.14, x=0), margin=dict(t=30, r=20),
                    hoverlabel=dict(namelength=-1))
    ev = st.plotly_chart(f, width="stretch", on_select="rerun", selection_mode="points", key=ck("tradeoff"))
    take_any(ev, "tradeoff")
    best_x = max(rows, key=lambda r: r[xk])
    st.caption(f"Furthest right: **{best_x['name'][:60]}** ({best_x['type']}), {best_x[xk]:.0f} {xk} at "
               f"${best_x['build USD/kWh']:,.0f}/kWh, efficiency {best_x['efficiency']:.2f}. Gold stars with a number "
               "are your candidates, numbered as in your ⭐ list (pin any candidate from its details). A **yellow "
               "ring with red dashes** around a dot = it contains a new molecule, never reported (not in PubChem). Prices: " + {"mass": "today's parts at mass production.",
                                            "target": "purpose-built at the US DOE cost targets.",
                                            "both": "solid dots purpose-built, faded dots today's parts; the line "
                                                    "is the saving."}[pr_]
               + " The efficiency, and so the electricity cost, is the same in both: only the parts get cheaper.")


# ------------------------------------------------------------------ tab 2

if TAB_NOW == TAB_NAMES[2]:
    comparison_chart()
    st.divider()
    pts = []
    for d in designs:
        for p in d.points:
            pts.append({"couple": d.cell.name, "p_max bar": p.p_max_bar,
                        "Wh/kg": E_KWH * 1000 / p.mass_kg, "Wh/L": E_KWH * 1000 / p.volume_l,
                        "build USD/kWh": p.capex_usd_kwh, "LCOS": p.lcos_usd_kwh,
                        "burst kg TNT": p.safety.burst_tnt_kg, "RTE": p.rte})
    pdf = pd.DataFrame(pts)
    if not pdf.empty:
        front = design.pareto([(-r["Wh/kg"], r["build USD/kWh"], r["burst kg TNT"]) for _, r in pdf.iterrows()])
        pdf["Pareto best"] = [i in front for i in range(len(pdf))]
        old_ = st.expander("Table couples only, at every pressure (older charts; your ⭐ candidates of every type "
                           "are on the main chart above)")
        old_.__enter__()
        f1 = px.scatter(pdf, x="Wh/kg", y="build USD/kWh", color="burst kg TNT", symbol="Pareto best", custom_data=["couple"],
                        hover_data=["couple", "p_max bar", "RTE", "LCOS"],
                        title="Hydrogen designs (dots) vs today's lithium-ion packs (red stars): lighter = further right, "
                              "cheaper = lower",
                        labels={"burst kg TNT": "burst<br>kg TNT", "Pareto best": "best trade-off"})
        # lithium-ion packs as points on the same axes, so they can be compared directly
        price = {"LFP": REF["prices"]["pack_lfp_usd_kwh"], "other": REF["prices"]["pack_bev_usd_kwh"]}
        li_pts = [(b["name"].split(" (")[0].split(",")[0], max(b["wh_kg"]),
                   price["LFP"] if "LFP" in b["name"] else price["other"])
                  for b in REF["battery"] if b["level"] == "pack" and "wh_kg" in b]
        x_max = max(pdf["Wh/kg"].max(), LI["wh_kg"]) * 1.15
        f1.add_shape(type="rect", x0=LI["wh_kg"], x1=x_max, y0=0, y1=LI["usd_kwh_bev"], line_width=0,
                     fillcolor="rgba(34,197,94,0.15)", layer="below")
        f1.add_annotation(x=(LI["wh_kg"] + x_max) / 2, y=LI["usd_kwh_bev"], yanchor="bottom", showarrow=False,
                          text="beats Li-ion<br>on weight and cost", font=dict(color="#22c55e", size=11))
        f1.add_trace(go.Scatter(x=[p_[1] for p_ in li_pts], y=[p_[2] for p_ in li_pts], mode="markers+text",
                                marker=dict(symbol="star", size=18, color="#ef4444", line=dict(color="white", width=1)),
                                text=[p_[0] for p_ in li_pts], textposition=[("top left", "bottom left", "top right")[i % 3] for i in range(len(li_pts))],
                                textfont=dict(color="#fca5a5", size=11), name="Li-ion car packs today",
                                hovertemplate="%{text}<br>%{x:.0f} Wh/kg, $%{y:.0f}/kWh<extra></extra>"))
        f1.add_hline(y=LI["usd_kwh_stationary"], line_dash="dot", line_color="#9ca3af",
                     annotation_text=f"Li-ion stationary ${LI['usd_kwh_stationary']:.0f}/kWh",
                     annotation_position="bottom left", annotation_font_color="#9ca3af")
        f1.update_xaxes(range=[0, x_max])
        pc_ = [(pin_number("couple", [c_]), c_) for c_ in pdf["couple"].unique()]
        pc_ = [(n_, c_) for n_, c_ in pc_ if n_]
        if pc_:
            best_ = pdf.sort_values("build USD/kWh").drop_duplicates("couple").set_index("couple")
            f1.add_trace(go.Scatter(x=[best_.loc[c_, "Wh/kg"] for _, c_ in pc_],
                                    y=[best_.loc[c_, "build USD/kWh"] for _, c_ in pc_], mode="markers+text",
                                    text=[f"<b>{n_}</b>" for n_, _ in pc_], textposition="middle right",
                                    textfont=dict(color="#facc15", size=20), name="⭐ your candidates",
                                    marker=dict(symbol="star", size=22, color="rgba(250,204,21,0.25)",
                                                line=dict(color="#facc15", width=2))))
        f1.update_layout(height=520, legend=dict(orientation="h", y=-0.18, x=0),
                         coloraxis_colorbar=dict(thickness=12, len=0.6, y=0.65))
        ev1 = st.plotly_chart(f1, width="stretch", on_select="rerun", selection_mode="points", key=ck("f1"))
        take(ev1, "f1", "couple")
        pk3 = st.selectbox("Show details of…", [""] + sorted(pdf["couple"].unique().tolist()), key="couple_pick",
                           format_func=lambda x: "choose a couple" if not x else x)
        if pk3 and st.button("Open", key="couple_pick_open"):
            SS["pending_dialog"] = ("couple", [pk3])
        f2 = px.scatter(pdf, x="burst kg TNT", y="LCOS", color="p_max bar", hover_data=["couple"], custom_data=["couple"],
                        title="Safety (stored burst energy) vs levelised cost of storage")
        ev2 = st.plotly_chart(f2, width="stretch", on_select="rerun", selection_mode="points", key=ck("f2"))
        take(ev2, "f2", "couple")
        f3 = px.line(pdf, x="p_max bar", y="Wh/kg", color="couple", markers=True, custom_data=["couple"],
                     title="Energy per kg vs maximum hydrogen pressure")
        f3.add_hline(y=LI["wh_kg"], line_dash="dot", annotation_text="Li-ion pack")
        ev3 = st.plotly_chart(f3, width="stretch", on_select="rerun", selection_mode="points", key=ck("f3"))
        take(ev3, "f3", "couple")
        old_.__exit__(None, None, None)

# ------------------------------------------------------------------ tab 3


# ------------------------------------------------------------------ AI tabs

def mol_settings():
    """The search settings and buttons: a fragment, so changing a value reruns only this part."""
    g1, g2, g3, g4 = st.columns(4)

    default("mol_gens", 5)
    gens = g1.number_input("Generations", 1, 200, key="mol_gens", help='How many rounds of evolution to run. In each round the AI writes several hundred new molecules by mutating and crossing the best ones found so far, predicts which are promising, and computes the best few with quantum chemistry. More rounds = more molecules explored and usually better results; time grows in proportion.')
    default("mol_batch", 8)
    batch = g2.number_input("Molecules computed per generation", 2, 64, key="mol_batch", help='How many molecules get a full quantum-chemistry calculation in each round (about 5-30 s each). Larger = more thorough per round but slower.')
    default("mol_heavy", 24)
    heavy = g3.number_input("Largest molecule (heavy atoms)", 8, 60, key="mol_heavy", help='Largest molecule allowed, counted in non-hydrogen atoms (C, N, O, S, ...). Small molecules dissolve better and are lighter per kWh; larger ones can carry more hydrogen but compute slower and are harder to make.')
    default("mol_kappa", 1.0)
    kappa = g4.slider("Exploration (0 = exploit, 3 = explore)", 0.0, 3.0, step=0.1, key="mol_kappa", help='Balance between exploiting and exploring. 0 = only compute molecules the AI is confident are good (fast improvement, may get stuck). Higher = also compute molecules it is unsure about, to discover new kinds of chemistry (slower, broader). 1-1.5 is a good middle.')
    save_settings_now()
    search_buttons("positive", dict(generations=int(gens), batch=int(batch), max_heavy=int(heavy), kappa=kappa), "mol")


def molecule_h2_tab():
    st.write("A genetic algorithm writes new molecules (SELFIES mutations and crossovers); a machine-learning "
             "model predicts which are worth computing; GFN2-xTB quantum chemistry computes their voltage with "
             "hydrogen; the model learns from every result. Molecules not in PubChem have never been reported.")
    st.fragment(mol_settings)()
    search_panel("positive", "mol")
    best_so_far("positive", "mol")
    v1, v2 = st.columns([3, 1], vertical_alignment="center")
    v1.caption("🔬 The search ranks molecules with the fast xTB method. Recompute the best ones from first principles "
               "(DFT, no measured value used): each takes 10-30 min in the background; results appear in each "
               "molecule's details.")
    if v2.button("Verify the 10 best with DFT", key="mol_verify10", width="stretch"):
        launch_verify(["--top", 10])
        st.toast("DFT started in the background for the 10 best molecules.")
    store = {r.smiles: r for r in view_pool("mol")}
    if store:
        mol = pd.DataFrame([{"molecule": r.smiles, "reduced form": r.reduced, "E V vs SHE": round(r.e0, 3),
                             "solubility M (ESOL)": round(r.solubility_mol_l, 2), "Wh/L electrolyte": round(r.wh_l, 1),
                             "Wh/kg molecule": round(r.wh_kg), "synthesis difficulty (1-10)": round(r.sa, 1),
                             "score": 0.0 if pairs._unstable(r.smiles) else round(r.objective, 1),
                             "H2 per molecule": r.best_k,
                             "step voltages V": ", ".join(f"{x:.2f}" for x in (r.steps or [r.e0])),
                             "must cost below USD/kg": round(pairs.CHEMICAL_SHARE * LI["usd_kwh_stationary"]
                                                             / pairs.kg_per_kwh(r.mw, sum((r.steps or [r.e0])[:r.best_k]) / r.best_k, 2 * r.best_k), 1)
                                                       if r.e0 > 0.05 else None,
                             "estimated price USD/kg": round(synthesis.estimate(r.smiles, t["synthesis"]).usd_kg, 1),
                             "new (not in PubChem)": {True: "no", False: "yes", None: "?"}[r.in_pubchem],
                             "notes": "; ".join(r.notes + (["unstable: " + ", ".join(M.instability(r.smiles))]
                                                           if pairs._unstable(r.smiles) else []))}
                            for r in store.values() if math.isfinite(r.e0)])
        mol = mol.sort_values("score", ascending=False)
        mol["status"] = np.where(mol["score"] > 0, "usable",
                                 np.where(mol["E V vs SHE"] > 1.629, "rejected: voltage would split the water",
                                          "rejected: unstable, too low or insoluble"))
        mol["never reported"] = mol["new (not in PubChem)"]
        sc = px.scatter(mol, x="E V vs SHE", y="Wh/L electrolyte", color="status", symbol="never reported", custom_data=["molecule"],
                        size=mol["solubility M (ESOL)"].clip(0.05, 4), hover_data=["molecule", "notes"],
                        title="Every molecule computed: voltage vs energy per litre of electrolyte (yellow ring = new molecule)")
        _nw = mol[mol["new (not in PubChem)"] == "yes"]
        add_new_rings(sc, _nw["E V vs SHE"], _nw["Wh/L electrolyte"], _nw[["molecule"]].values, size=24)
        sc.add_hline(y=LI["wh_l"], line_dash="dash", line_color="red",
                     annotation_text=f"Li-ion car pack {LI['wh_l']:.0f} Wh/L")
        evm = st.plotly_chart(sc, width="stretch", on_select="rerun", selection_mode="points", key=ck("molsc"))
        take(evm, "molsc", "molecule")
        st.caption("Click 🔍 open (or any cell of a row) to see the molecule and its details. Wh/L of electrolyte is an upper bound: the full "
                   "battery (stack, hydrogen storage, case) holds less. Each molecule's details show it as a full battery "
                   "against lithium-ion; ⭐ adds it to your candidates, compared in every tab.")
        SS["rows_mol_tbl"] = mol[["molecule"]].values.tolist()
        mol.insert(0, "details", "🔍 open")
        evt = st.dataframe(mol, hide_index=True, width="stretch", on_select="rerun",
                           selection_mode=["single-row", "single-cell"], key=ck("mol_tbl"))
        take_row(evt, "mol_tbl", "molecule")

# ------------------------------------------------------------------ tab 6: two molecules

def pair_settings():
    """The search settings and buttons: a fragment, so changing a value reruns only this part."""
    q1, q2, q3, q4 = st.columns(4)

    default("pair_gens", 5)
    pg = q1.number_input("Generations", 1, 200, key="pair_gens", help='How many rounds of evolution to run. In each round the AI writes several hundred new molecules by mutating and crossing the best ones found so far, predicts which are promising, and computes the best few with quantum chemistry. More rounds = more molecules explored and usually better results; time grows in proportion.')
    default("pair_batch", 8)
    pb = q2.number_input("Molecules computed per generation", 2, 64, key="pair_batch", help='How many molecules get a full quantum-chemistry calculation in each round (about 5-30 s each). Larger = more thorough per round but slower.')
    default("pair_heavy", 28)
    ph = q3.number_input("Largest molecule (heavy atoms)", 8, 60, key="pair_heavy", help='Largest molecule allowed, counted in non-hydrogen atoms (C, N, O, S, ...). Small molecules dissolve better and are lighter per kWh; larger ones can carry more hydrogen but compute slower and are harder to make.')
    default("pair_kappa", 1.5)
    pk = q4.slider("Exploration (0 = exploit, 3 = explore)", 0.0, 3.0, step=0.1, key="pair_kappa", help='Balance between exploiting and exploring. 0 = only compute molecules the AI is confident are good (fast improvement, may get stuck). Higher = also compute molecules it is unsure about, to discover new kinds of chemistry (slower, broader). 1-1.5 is a good middle.')
    save_settings_now()
    search_buttons("pair", dict(generations=int(pg), batch=int(pb), max_heavy=int(ph), kappa=pk), "pair")


def two_molecule_tab():
    st.write("Hydrogen moves between two molecules as protons and electrons and is never a gas: "
             "no pressure, no tank, no platinum, a water-based liquid that cannot burn. "
             "Negative: N-H2 -> N + 2H+ + 2e-; positive: P + 2H+ + 2e- -> P-H2; voltage = E(P) - E(N), "
             f"kept inside the water window ({-pairs.HER_MARGIN + pairs.HEADROOM:.2f} to "
             f"{1.229 + pairs.OER_MARGIN - pairs.HEADROOM:.2f} V vs SHE).")
    st.fragment(pair_settings)()
    search_panel("pair", "pair")
    best_so_far("pair", "pair")
    pool = view_pool("pair")
    ps = pairs.best_pairs(pool, 40, max_per_molecule=2)
    if ps:
        pt = pd.DataFrame([{"negative molecule": p.neg.smiles, "E neg V": round(p.neg.e0, 2),
                            "positive molecule": p.pos.smiles, "E pos V": round(p.pos.e0, 2),
                            "cell V": round(p.voltage, 2), "e- per neg / pos": f"{2 * p.kn} / {2 * p.kp}",
                            "Wh/L (both liquids)": round(p.wh_l, 1),
                            "Wh/kg (both liquids)": round(p.wh_kg, 1),
                            "kg chemicals per kWh": round(pairs.kg_per_kwh(p.neg.mw, p.voltage, 2 * p.kn)
                                                          + pairs.kg_per_kwh(p.pos.mw, p.voltage, 2 * p.kp), 2),
                            "must cost below USD/kg": round(pairs.price_ceiling_usd_kg(p, LI["usd_kwh_stationary"]), 1),
                            "estimated USD/kg (neg / pos)": f"{synthesis.estimate(p.neg.smiles, t['synthesis']).usd_kg:,.0f} / "
                                                            f"{synthesis.estimate(p.pos.smiles, t['synthesis']).usd_kg:,.0f}",
                            "never reported": ", ".join(x for x, r in (("neg", p.neg), ("pos", p.pos)) if r.in_pubchem is False) or "-",
                            "notes": "; ".join(p.notes + p.neg.notes + p.pos.notes)} for p in ps])
        fp = px.scatter(pt, x="cell V", y="Wh/L (both liquids)", size="Wh/kg (both liquids)",
                        custom_data=["negative molecule", "positive molecule"],
                        hover_data=["negative molecule", "positive molecule", "must cost below USD/kg"],
                        title="Every pair: voltage vs energy per litre of the two liquids (yellow ring = contains a new molecule)")
        _nw = pt[pt["never reported"] != "-"]
        add_new_rings(fp, _nw["cell V"], _nw["Wh/L (both liquids)"], _nw[["negative molecule", "positive molecule"]].values, size=28)
        fp.add_hline(y=LI["wh_l"], line_dash="dash", line_color="red", annotation_text=f"Li-ion car pack {LI['wh_l']:.0f} Wh/L")
        evp = st.plotly_chart(fp, width="stretch", on_select="rerun", selection_mode="points", key=ck("pairsc"))
        take(evp, "pairsc", "pair")
        labels = [f"{a} | {b}" for a, b in zip(pt["negative molecule"], pt["positive molecule"])]
        st.caption(f"'must cost below' = the price per kg at which both molecules together stay under "
                   f"{pairs.CHEMICAL_SHARE:.0%} of a stationary Li-ion pack (${LI['usd_kwh_stationary']:.0f}/kWh). "
                   "Liquid-level figures are upper bounds. Click 🔍 open (or any cell) for the pair as a full battery against "
                   "lithium-ion.")
        SS["rows_pair_tbl"] = pt[["negative molecule", "positive molecule"]].values.tolist()
        pt.insert(0, "details", "🔍 open")
        evq = st.dataframe(pt, hide_index=True, width="stretch", on_select="rerun",
                           selection_mode=["single-row", "single-cell"], key=ck("pair_tbl"))
        take_row(evq, "pair_tbl", "pair")
    else:
        st.info("No usable pair yet: run pair discovery.")

if TAB_NOW == TAB_NAMES[1]:
    watch_tab()

# ------------------------------------------------------------------ AI discovery: the three battery types

BATTERY_TYPES = {
    "molecule": ("Molecule + hydrogen gas",
                 "A liquid molecule takes up hydrogen on the positive side; the negative side is hydrogen gas in a tank."),
    "pair": ("Two molecules (no gas)",
             "Hydrogen moves as protons between two dissolved molecules; no gas, no pressure, no platinum."),
    "gas": ("Two gases (sealed)",
            "An oxidant gas and a fuel gas exchange hydrogen as protons through a membrane; every gas stays in "
            "its own tank, so the battery can be recharged."),
}

if TAB_NOW == TAB_NAMES[0]:
    st.caption("Three ways to build a battery that moves hydrogen. In each one an AI proposes new molecules, "
               "quantum chemistry (GFN2-xTB) checks them, and the best ones are sized against lithium-ion.")
    default("cfg_kind", "molecule")
    if SS.get("cfg_kind") not in BATTERY_TYPES:
        SS["cfg_kind"] = "molecule"
    kind_ = st.radio("Battery type", list(BATTERY_TYPES), key="cfg_kind", horizontal=True,
                     format_func=lambda k: BATTERY_TYPES[k][0], captions=[v[1] for v in BATTERY_TYPES.values()])
    # save at once: the gas view takes a while to finish, and a reload before the end of the run
    # would otherwise lose the choice (settings are normally saved at the end of each run)
    runs.save_settings({k: SS[k] for k in SETTING_KEYS if k in SS})
    with st.expander(f"How it works: {BATTERY_TYPES[kind_][0].lower()} (animation)", expanded=False):
        anim_mode = st.radio("Show", ["discharge", "charge"], horizontal=True, key=f"anim_mode_{kind_}",
                             format_func=lambda m: {"discharge": "Discharging (gives power)",
                                                    "charge": "Charging (stores power)"}[m])
        draw_ = {"molecule": animation.molecule_h2_html, "pair": animation.two_tank_html,
                 "gas": animation.two_gas_html}[kind_]
        st.iframe(draw_(anim_mode, height=540), height=540)
    st.divider()
    {"molecule": molecule_h2_tab, "pair": two_molecule_tab, "gas": gas_tab}[kind_]()

# ------------------------------------------------------------------ Li-ion reference

def validation_section():
    """How far the computations are from measurement, on known molecules (their data are used only here)."""
    st.subheader("How accurate are the computations? Tested on known molecules")
    st.write("Every molecule, known or new, is computed the same way. The measured values of known molecules are "
             "used only to measure the error below; at most two element constants (carbon and sulfur, solids that "
             "cannot be computed as molecules) are fitted on a training half, and the errors are on the other half.")
    try:
        val = json.loads((Path(__file__).parent / "cache" / "validation.json").read_text())
    except Exception:
        st.info("Not computed yet (scripts/validate_abinitio.py).")
        return
    rows = []
    for lv, res in val.items():
        sm = res.get("summary", {})
        g = sm.get("gas reactions, V") or {}
        f_ = sm.get("formation test, kJ/mol") or {}
        aq = sm.get("aqueous voltage, V") or {}
        rows.append({"method": res.get("method", lv),
                     "gas cell voltage error, V (mean abs / max)": f"{g['mae']:.3f} / {g['max']:.2f}" if g else "-",
                     "systematic gas shift, V": f"{g['mean']:+.3f}" if g else "-",
                     "energy of formation, test half, kJ/mol (mean abs)": f"{f_['mae']:.0f}" if f_ else "-",
                     "voltage in water, V (mean abs)": f"{aq['mae']:.2f}" if aq else "not computed",
                     "known molecules": g.get("n", "-"), "computed on": res.get("date", "")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Gas cell voltage error: the complete oxidation of every known gas molecule (to CO2, water, N2, SO2, "
               "HF, HCl), computed with no fitted constant at all, against the measured reaction energy, per "
               "electron moved. GFN2-xTB alone is not usable for absolute energies, which is why the fast AI search "
               "uses it only with constants fitted on the training half, and the results are recomputed with DFT "
               "and, for small molecules, CCSD(T) (the 'gold standard' of quantum chemistry).")
    for lv, res in val.items():
        with st.expander(f"Details: {res.get('method', lv)}"):
            st.dataframe(pd.DataFrame([{"oxidation of": r["name"], "electrons": r.get("electrons"),
                                        "dG measured kJ/mol": round(r["dG_meas"], 1), "dG computed": round(r["dG_calc"], 1),
                                        "error V": round(r["err_v"], 3)} for r in res["reactions"] if "err_v" in r]),
                         hide_index=True, width="stretch")
            if res.get("aqueous"):
                st.dataframe(pd.DataFrame([{"quinone": r["name"], "measured E V": r["E_meas"],
                                            "computed E V": None if r["E_calc"] is None else round(r["E_calc"], 3),
                                            "error V": None if r["err_v"] is None else round(r["err_v"], 3),
                                            "source": r["source"]} for r in res["aqueous"]]),
                             hide_index=True, width="stretch")


if TAB_NOW == TAB_NAMES[3]:
    st.subheader("Lithium-ion batteries in current cars (liquid electrolyte, not solid-state)")
    st.dataframe(pd.DataFrame([{"battery": b["name"], "level": b["level"],
                                "Wh/kg": "-".join(f"{x:g}" for x in b.get("wh_kg", [])),
                                "Wh/L": "-".join(f"{x:g}" for x in b.get("wh_l", [])),
                                "cycles": "-".join(f"{x:g}" for x in b.get("cycles", [])),
                                "fast charge 10-80 % min": "-".join(f"{x:g}" for x in b.get("charge_minutes_10_80", [])),
                                "quality": b["quality"], "sources": " | ".join(b["sources"])} for b in REF["battery"]]),
                 hide_index=True, width="stretch")
    pr = REF["prices"]
    st.write(f"Pack prices, BloombergNEF 2025 survey: average ${pr['pack_avg_usd_kwh']:.0f}/kWh, LFP ${pr['pack_lfp_usd_kwh']:.0f}, "
             f"NMC ${pr['pack_nmc_usd_kwh']:.0f}, car packs ${pr['pack_bev_usd_kwh']:.0f}, stationary ${pr['pack_stationary_usd_kwh']:.0f} "
             f"({pr['source']})")
    st.subheader("Model checks against published hydrogen cells")
    st.write("- Mn–H2 (Chen et al., Nature Energy 2018): ~1.3 V measured; model OCV 1.22 V.\n"
             "- H2–Br2 (Cho et al., J. Electrochem. Soc. 2012): 1.4 W/cm² peak measured; model p90 0.72 W/cm² — the "
             "generic reaction-speed range is too pessimistic for fast halogen couples.\n"
             "- Molecules: the fast xTB voltages carry one offset fitted on half of the known quinones; the other half "
             "tests it (see below).")
    validation_section()


open_pending_dialog()
SS["_full_run"] = False
runs.save_settings({k: SS[k] for k in SETTING_KEYS if k in SS})
