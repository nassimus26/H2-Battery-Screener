"""Why a design costs what it costs, in plain words, with the setting to change.

explain_cost(parts, ctx) takes the build cost split by part (USD) and the
design context, and returns sentences that name the biggest parts, the
physical reason, and what to change in the app (sidebar or search settings).
"""

from __future__ import annotations

import math



def explain_cost(parts: dict, ctx: dict) -> list[str]:
    total = sum(parts.values())
    if total <= 0:
        return []
    e_kwh = ctx["e_kwh"]
    power = ctx["power_kw"]
    hours = e_kwh / power if power else 0.0
    ranked = sorted(parts.items(), key=lambda kv: -kv[1])
    out = ["Biggest costs: " + ", ".join(f"{k} {v / e_kwh:,.0f} USD/kWh ({v / total:.0%})"
                                          for k, v in ranked[:3] if v > 0) + "."]
    share = {k: v / total for k, v in parts.items()}
    stackish = share.get("cell stack", 0) + share.get("platinum catalyst", 0) + share.get("power electronics", 0)

    if stackish > 0.4:
        i = ctx.get("i_a_cm2", 0.0)
        area = ctx.get("area_m2", 0.0)
        v = ctx.get("v_dis", 0.0)
        est = ctx.get("stack_est")
        usd_m2 = est.usd if est else ctx.get("stack_usd_m2")
        w_m2 = i * 1e4 * v
        out.append(f"How the stack cost adds up: cell area = power / (current density x voltage) = "
                   f"{power * 1000:,.0f} W / ({i:.3g} A/cm2 x {v:.2f} V) = {area:,.0f} m2, so each m2 of cells "
                   f"delivers only {w_m2:,.0f} W."
                   + (f" At {usd_m2:,.0f} USD per m2 (plates, membrane, electrodes, seals, frame) that is "
                      f"{area * usd_m2:,.0f} USD." if usd_m2 else ""))
        if est:
            out.append(f"Where {est.usd:,.0f} USD per m2 comes from: {est.basis}. It pays for the cell hardware and "
                       "is the same for every molecule (the molecules are under 'chemicals'). Across its range "
                       f"({est.lo:,.0f}-{est.hi:,.0f} USD/m2) this stack costs "
                       f"{area * est.lo / e_kwh:,.0f}-{area * est.hi / e_kwh:,.0f} USD/kWh."
                       + (" The price per m2 is not the problem here; the number of m2 is (a fuel-cell stack "
                          "delivers ~6,000 W per m2)." if w_m2 < 2000 else ""))
        if hours < 2:
            out.append(f"You ask for full power for only {hours:.2f} h ({power / e_kwh:.1f} kW per kWh stored). "
                       "The stack does not shrink with the stored energy, so its cost per kWh is high. "
                       "-> Raise 'Delivers full power for, hours' or lower 'Power, kW' (sidebar).")
        if i < 0.05:
            out.append("The current density is low because this couple's electrode reaction speed is unknown and "
                       "the model samples it over a wide, mostly slow range; to keep the efficiency target the "
                       "cells must be large. -> Lowering 'Round-trip efficiency it must keep' allows a smaller, "
                       "cheaper stack (less efficient); a lab measurement of the reaction speed would settle it.")
        if share.get("platinum catalyst", 0) > 0.15:
            out.append("Platinum catalyst on the hydrogen electrode is a large part. Two-molecule cells need none; "
                       "for hydrogen-gas cells a platinum-free catalyst would be the lever (not yet a setting).")

    if share.get("hydrogen tank", 0) + share.get("hydrogen storage", 0) > 0.3:
        p = ctx.get("p_max_bar")
        out.append("The hydrogen tank is a large part" + (f" at {p:g} bar" if p else "") + ". Its steel weight and "
                   "cost barely change above ~20 bar (smaller tank, thicker wall); below ~10 bar the tank grows "
                   "fast because gas stays inside at the 'empty' pressure. -> Check the cost-vs-pressure chart; "
                   "hydride storage (constant low pressure) or the two-molecule mode (no gas) removes the tank.")

    loops = share.get("electrolyte tanks", 0) + share.get("pumps, pipes and controls", 0)
    if loops > 0.15:
        out.append("The tanks and liquid loops are a large part. Tank cost grows with the litres of liquid, which "
                   "shrink when the molecules dissolve better or carry more hydrogen per molecule; pumps, pipes, "
                   "heat exchangers and sensors grow with power. Prices are those of a built 10 kW flow battery "
                   "(Noack et al. 2016), scaled with size^0.6 (bigger systems cost less per unit); mass production "
                   "would be cheaper.")

    for sc, kg in ctx.get("synth") or []:
        out.append(f"Making the molecule {sc.smiles}: {sc.describe()}. {kg / e_kwh:.1f} kg per kWh, so "
                   f"{sc.usd_kg * kg / e_kwh:,.0f} USD/kWh ({sc.lo * kg / e_kwh:,.0f}-{sc.hi * kg / e_kwh:,.0f}). "
                   "The price is estimated from how hard the molecule is to make (synthetic accessibility score, "
                   "fitted to chemicals with known bulk prices); the score tracks price only loosely, hence the "
                   "wide range.")

    if share.get("chemicals", 0) > 0.3:
        kg = ctx.get("chem_kg_per_kwh")
        usd = ctx.get("chem_usd_kg")
        detail = (f" ({kg:.1f} kg per kWh" + (f" at about {usd:.1f} USD/kg" if usd else "") + ")") if kg else ""
        out.append("The chemicals are a large part" + detail + ". Each kWh needs more material when the molecule "
                   "is heavy, carries few electrons or has a low voltage. -> In the search, lower 'Largest "
                   "molecule (heavy atoms)'; molecules that take up 2-3 H2 or pairs with a higher cell voltage "
                   "need less material per kWh; molecules with a lower difficulty score are cheaper to make.")

    if ctx.get("li_usd_kwh") and total / e_kwh > ctx["li_usd_kwh"]:
        out.append(f"For comparison, a lithium-ion pack costs about {ctx['li_usd_kwh']:.0f} USD/kWh.")
    return out


def cost_range(capex_usd_kwh: float, band: tuple, li_usd_kwh: float) -> tuple[str, str, str]:
    """(value, delta, help) for the build-cost metric: the cost at the middle
    reaction speed, its gap to Li-ion, and the p10-p90 range over the
    reaction-speed (exchange-current) uncertainty. band = (p10, p50, p90)."""
    value = f"${capex_usd_kwh:,.0f}/kWh"
    gap = f"{capex_usd_kwh - li_usd_kwh:+,.0f}"
    if not band or not math.isfinite(band[0]):
        return value, f"{gap} vs Li-ion", "Reaction-speed range not computed."
    lo, mid, hi = band
    fmt = lambda x: f"${x:,.0f}" if math.isfinite(x) else "cannot be built"
    span = (f"{lo - li_usd_kwh:+,.0f} to {hi - li_usd_kwh:+,.0f}" if math.isfinite(hi)
            else f"{lo - li_usd_kwh:+,.0f} to (cannot be built)")
    delta = f"{gap} vs Li-ion (p10-p90 {span})"
    help_ = (f"Shown: cost when both electrodes react at the middle of their speed range (the cost split below "
             f"uses it). Nobody has measured how fast these molecules react, so the stack is re-sized for "
             f"random reaction speeds: 10 % of them cost less than {fmt(lo)}/kWh (fast), half less "
             f"than {fmt(mid)}/kWh, 90 % less than {fmt(hi)}/kWh (slow). Li-ion pack: ${li_usd_kwh:,.0f}/kWh.")
    return value, delta, help_
