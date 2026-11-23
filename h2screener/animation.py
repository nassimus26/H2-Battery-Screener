"""Animated diagrams of the three battery types, as self-contained HTML/SVG
for the app: two liquids (two_tank_html), a liquid against hydrogen gas
(molecule_h2_html) and a sealed two-gas cell (two_gas_html).

Discharge: pumps send each liquid from its tank through its half of the cell.
On the negative side the loaded molecule N-H2 gives up its hydrogen as two
protons (through the membrane) and two electrons (through the outside wire,
where they do work); on the positive side P picks them up and becomes P-H2.
The charged layer in each tank shrinks as the liquid comes back spent.
Charge runs everything backwards, driven by a charger.
"""

from __future__ import annotations

# path ids used by the moving dots
_PATHS = {
    # negative loop: tank bottom -> pump -> cell bottom-left, cell top-left -> tank top
    "neg_out": "M 100 300 L 100 330 L 250 330 L 250 285 L 318 285",
    "neg_back": "M 318 150 L 250 150 L 250 95 L 155 95 L 155 125",
    "pos_out": "M 660 300 L 660 330 L 510 330 L 510 285 L 442 285",
    "pos_back": "M 442 150 L 510 150 L 510 95 L 605 95 L 605 125",
    "wire": "M 330 140 L 330 50 L 430 50 L 430 140",
    "water": "M 430 293 L 430 352",
}


def _dots(path_id: str, n: int, dur: float, color: str, r: float, reverse: bool, label: str = "") -> str:
    kp = 'keyPoints="1;0" keyTimes="0;1" calcMode="linear"' if reverse else ""
    out = []
    for k in range(n):
        begin = -dur * k / n
        txt = (f'<text font-size="9" fill="#111" text-anchor="middle" dy="3">{label}</text>' if label else "")
        out.append(f'<g><circle r="{r}" fill="{color}"/>{txt}'
                   f'<animateMotion dur="{dur}s" begin="{begin:.2f}s" repeatCount="indefinite" {kp}>'
                   f'<mpath href="#{path_id}"/></animateMotion></g>')
    return "".join(out)


def _protons(reverse: bool) -> str:
    out = []
    for k, y in enumerate((170, 200, 230, 260)):
        x0, x1 = (395, 365) if reverse else (365, 395)
        out.append(f'<g><circle r="7" fill="#e5484d"/><text font-size="8" fill="#fff" text-anchor="middle" dy="3">H+</text>'
                   f'<animateTransform attributeName="transform" type="translate" values="{x0} {y};{x1} {y}" '
                   f'dur="1.6s" begin="{-0.4 * k:.1f}s" repeatCount="indefinite"/></g>')
    return "".join(out)



def _liquid_tank(x, fill_c, fill_s, label, h_from, h_to):
    return f"""
      <rect x="{x}" y="125" width="140" height="175" rx="14" fill="#1f2937" stroke="#9ca3af" stroke-width="2"/>
      <rect x="{x + 4}" y="150" width="132" height="146" rx="10" fill="{fill_s}"/>
      <rect x="{x + 4}" width="132" rx="10" fill="{fill_c}" y="{296 - h_from}" height="{h_from}">
        <animate attributeName="height" values="{h_from};{h_to};{h_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
        <animate attributeName="y" values="{296 - h_from};{296 - h_to};{296 - h_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
      </rect>
      <text x="{x + 70}" y="118" fill="#e5e7eb" font-size="13" text-anchor="middle" font-weight="600">{label}</text>
"""


def _gas_tank(x, color, label, gas, full_from, full_to):
    """Pressure tank: a capsule whose shade (pressure) falls or rises, with molecules moving inside."""
    o_from, o_to = (0.9, 0.25) if full_from else (0.25, 0.9)
    mol = []
    for k in range(7):
        cx, cy = x + 25 + (k * 37) % 95, 160 + (k * 53) % 115
        mol.append(f'<g><circle r="8" fill="{color}" opacity="0.95"/><text font-size="7" fill="#111" text-anchor="middle" dy="2.5">{gas}</text>'
                   f'<animateTransform attributeName="transform" type="translate" values="{cx} {cy};{cx + 18} {cy - 14};{cx - 10} {cy + 12};{cx} {cy}" '
                   f'dur="{2.2 + k * 0.3:.1f}s" repeatCount="indefinite"/></g>')
    return f"""
      <rect x="{x}" y="125" width="140" height="175" rx="60" fill="#1f2937" stroke="#cbd5e1" stroke-width="3"/>
      <rect x="{x + 5}" y="130" width="130" height="165" rx="56" fill="{color}" opacity="{o_from}">
        <animate attributeName="opacity" values="{o_from};{o_to};{o_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
      </rect>
      {''.join(mol)}
      <text x="{x + 70}" y="118" fill="#e5e7eb" font-size="13" text-anchor="middle" font-weight="600">{label}</text>
"""


def _pump(x):
    return (f'<circle cx="{x}" cy="330" r="11" fill="#374151" stroke="#9ca3af"/>'
            f'<path d="M {x - 5} 334 L {x} 324 L {x + 5} 334 Z" fill="#9ca3af">'
            f'<animateTransform attributeName="transform" type="rotate" from="0 {x} 330" to="360 {x} 330" dur="1.2s" repeatCount="indefinite"/></path>'
            f'<text x="{x}" y="355" fill="#9ca3af" font-size="10" text-anchor="middle">pump</text>')


def _valve(x):
    return (f'<path d="M {x - 10} 322 L {x + 10} 338 L {x + 10} 322 L {x - 10} 338 Z" fill="#9ca3af"/>'
            f'<text x="{x}" y="355" fill="#9ca3af" font-size="10" text-anchor="middle">valve</text>')


def _page(svg: str, legend: str, height: int, view_h: int = 365, aria: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:transparent;font-family:system-ui,-apple-system,sans-serif}}
.card{{background:#111827;border-radius:12px;padding:8px 8px 4px}}
.leg{{color:#cbd5e1;font-size:12px;padding:2px 8px 6px;line-height:1.5}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px;vertical-align:-1px}}
</style></head><body><div class="card">
<svg viewBox="0 0 760 {view_h}" width="100%" style="max-height:{height - 70}px" role="img" aria-label="{aria}">{svg}</svg>
<div class="leg">{legend}</div></div></body></html>"""


def _cell_and_load(rev: bool, left_fill: str, right_fill: str, label: bool = True) -> str:
    load = ("charger", "#3b82f6") if rev else ("motor / grid", "#22c55e")
    return f"""
  <path id="wire" d="{_PATHS['wire']}" fill="none" stroke="#d1d5db" stroke-width="3"/>
  <rect x="318" y="140" width="124" height="155" rx="6" fill="#0b1220" stroke="#9ca3af" stroke-width="2"/>
  <rect x="322" y="144" width="52" height="147" fill="{left_fill}" opacity="0.8"/>
  <rect x="386" y="144" width="52" height="147" fill="{right_fill}" opacity="0.8"/>
  <line x1="380" y1="142" x2="380" y2="293" stroke="#e5e7eb" stroke-width="3" stroke-dasharray="6 4"/>
  {'<text x="380" y="310" fill="#e5e7eb" font-size="11" text-anchor="middle">cell (membrane in the middle)</text>' if label else
   '<text x="372" y="310" fill="#e5e7eb" font-size="11" text-anchor="end">cell, membrane in the middle</text>'}
  <rect x="352" y="36" width="56" height="28" rx="8" fill="{load[1]}"/>
  <text x="380" y="55" fill="#0b1220" font-size="12" text-anchor="middle" font-weight="700">{'+ / -' if rev else 'M'}</text>
  <text x="380" y="26" fill="#e5e7eb" font-size="12" text-anchor="middle">{load[0]}</text>"""


def _pipes(keys) -> str:
    return "".join(f'<path id="{k}" d="{_PATHS[k]}" fill="none" stroke="#4b5563" stroke-width="8" stroke-linejoin="round"/>'
                   for k in keys)


def _legend_head(rev: bool) -> str:
    return f"<b>{'Charging' if rev else 'Discharging'}.</b>"


def two_tank_html(mode: str = "discharge", height: int = 400) -> str:
    """Two liquids (no gas). mode: 'discharge' or 'charge'."""
    rev = mode == "charge"
    h_from, h_to = (40, 150) if rev else (150, 40)
    neg_txt = "N + 2H+ + 2e- -> N-H2" if rev else "N-H2 -> N + 2H+ + 2e-"
    pos_txt = "P-H2 -> P + 2H+ + 2e-" if rev else "P + 2H+ + 2e- -> P-H2"
    svg = (_pipes(("neg_out", "neg_back", "pos_out", "pos_back"))
           + _liquid_tank(30, "#2563eb", "#1e3a5f", "Negative tank", h_from, h_to)
           + _liquid_tank(590, "#ea580c", "#5c2f12", "Positive tank", h_from, h_to)
           + _pump(175) + _pump(585) + _cell_and_load(rev, "#1e3a5f", "#5c2f12")
           + _dots("neg_out", 6, 4, "#60a5fa", 5, False) + _dots("neg_back", 6, 4, "#93c5fd", 5, False)
           + _dots("pos_out", 6, 4, "#fb923c", 5, False) + _dots("pos_back", 6, 4, "#fdba74", 5, False)
           + _protons(rev) + _dots("wire", 7, 3, "#facc15", 7, rev, "e-"))
    legend = f"""{_legend_head(rev)}
 <span class="sw" style="background:#60a5fa"></span>negative liquid
 <span class="sw" style="background:#fb923c"></span>positive liquid
 <span class="sw" style="background:#e5484d"></span>H+ through the membrane
 <span class="sw" style="background:#facc15"></span>e- through the wire<br>
 <span style="color:#bfdbfe">Negative side: {neg_txt}</span> &nbsp;·&nbsp; <span style="color:#fed7aa">Positive side: {pos_txt}</span><br>
 In the tanks, the bright layer is charged liquid (N-H2 on the left, P on the right) and the dark part is spent liquid;
 {'charging refills the bright layers' if rev else 'discharging empties the bright layers'}.<br>
 The hydrogen never becomes a gas: it travels as protons and electrons from one molecule to the other.
 Energy is set by the tanks (litres of liquid); power by the cell area (m2). Tanks, pumps and pipes are priced from a built 10 kW flow battery (Noack et al. 2016), scaled to the system size."""
    return _page(svg, legend, height, aria="Two-tank battery: liquids pumped through a cell; protons cross the membrane, electrons go through the outside wire")


def molecule_h2_html(mode: str = "discharge", height: int = 400) -> str:
    """Hydrogen gas (negative) against a liquid molecule (positive)."""
    rev = mode == "charge"
    h_from, h_to = (40, 150) if rev else (150, 40)
    neg_txt = "2H+ + 2e- -> H2 (back into the tank)" if rev else "H2 -> 2H+ + 2e-"
    pos_txt = "P-H2 -> P + 2H+ + 2e-" if rev else "P + 2H+ + 2e- -> P-H2"
    svg = (_pipes(("neg_out", "pos_out", "pos_back"))
           + _gas_tank(30, "#a5b4fc", "Hydrogen tank", "H2", not rev, rev)
           + _liquid_tank(590, "#ea580c", "#5c2f12", "Positive tank", h_from, h_to)
           + _valve(175) + _pump(585) + _cell_and_load(rev, "#312e81", "#5c2f12")
           + _dots("neg_out", 5, 3, "#c7d2fe", 7, rev, "H2")
           + _dots("pos_out", 6, 4, "#fb923c", 5, False) + _dots("pos_back", 6, 4, "#fdba74", 5, False)
           + _protons(rev) + _dots("wire", 7, 3, "#facc15", 7, rev, "e-"))
    legend = f"""{_legend_head(rev)}
 <span class="sw" style="background:#c7d2fe"></span>hydrogen gas
 <span class="sw" style="background:#fb923c"></span>positive liquid (the molecule)
 <span class="sw" style="background:#e5484d"></span>H+ through the membrane
 <span class="sw" style="background:#facc15"></span>e- through the wire<br>
 <span style="color:#c7d2fe">Negative side (platinum on carbon, like a fuel cell): {neg_txt}</span> &nbsp;·&nbsp;
 <span style="color:#fed7aa">Positive side: {pos_txt}</span><br>
 Discharging, the hydrogen tank loses pressure and the molecule takes up the hydrogen; charging pushes it back into
 the tank at pressure. Energy is set by the hydrogen tank and the litres of liquid; power by the cell area (m2).
 The tank's pressure range is the sidebar's "highest hydrogen pressure" and "pressure left when empty"."""
    return _page(svg, legend, height, aria="Hydrogen gas tank and a liquid tank feeding a cell; protons cross the membrane")


def two_gas_html(mode: str = "discharge", height: int = 400) -> str:
    """Sealed two-gas cell, drawn for the simplest one: hydrogen + oxygen -> water."""
    rev = mode == "charge"
    w_from, w_to = (40, 4) if rev else (4, 40)
    neg_txt = "4H+ + 4e- -> 2 H2" if rev else "2 H2 -> 4H+ + 4e-"
    pos_txt = "2 H2O -> O2 + 4H+ + 4e-" if rev else "O2 + 4H+ + 4e- -> 2 H2O"
    water_tank = f"""
      <rect x="400" y="352" width="140" height="50" rx="10" fill="#1f2937" stroke="#9ca3af" stroke-width="2"/>
      <rect x="404" width="132" rx="7" fill="#38bdf8" y="{398 - w_from}" height="{w_from}">
        <animate attributeName="height" values="{w_from};{w_to};{w_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
        <animate attributeName="y" values="{398 - w_from};{398 - w_to};{398 - w_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
      </rect>
      <text x="548" y="382" fill="#e5e7eb" font-size="12" font-weight="600">Water tank</text>"""
    svg = (_pipes(("neg_out", "pos_out", "water"))
           + _gas_tank(30, "#a5b4fc", "Fuel tank (H2)", "H2", not rev, rev)
           + _gas_tank(590, "#f87171", "Oxidant tank (O2)", "O2", not rev, rev)
           + water_tank + _valve(175) + _valve(585) + _cell_and_load(rev, "#312e81", "#7f1d1d", label=False)
           + _dots("neg_out", 5, 3, "#c7d2fe", 7, rev, "H2") + _dots("pos_out", 4, 3, "#fca5a5", 7, rev, "O2")
           + _dots("water", 3, 2, "#38bdf8", 6, rev, "")
           + _protons(rev) + _dots("wire", 7, 3, "#facc15", 7, rev, "e-"))
    legend = f"""{_legend_head(rev)}
 <span class="sw" style="background:#c7d2fe"></span>fuel gas
 <span class="sw" style="background:#fca5a5"></span>oxidant gas
 <span class="sw" style="background:#38bdf8"></span>water
 <span class="sw" style="background:#e5484d"></span>H+ through the membrane
 <span class="sw" style="background:#facc15"></span>e- through the wire<br>
 <span style="color:#c7d2fe">Negative side: {neg_txt}</span> &nbsp;·&nbsp; <span style="color:#fca5a5">Positive side: {pos_txt}</span><br>
 Drawn for hydrogen + oxygen. Other cells work the same way: the fuel (methane, acetylene...) gives up its hydrogen
 as protons and electrons and its product (CO2...) goes to its own tank; the oxidant (O2, Cl2...) takes them up.
 Nothing leaves the battery: every gas and the water are kept, so the same cell can be charged again
 ({'it now works as an electrolyser' if rev else 'it works as a fuel cell'}). Energy is set by the tanks; power by the cell area."""
    return _page(svg, legend, height, view_h=410, aria="Sealed gas battery: fuel and oxidant gas tanks, a cell and a water tank")


def switchable_html(draw, height: int = 480) -> str:
    """Both modes of one animation in a single page, with the discharge / charge switch in the page itself:
    switching happens in the browser, instantly, with no round trip to the server. Only one drawing is in the
    page at a time (the other waits in a <template>), so the SVG ids (pipe paths the dots follow) never clash."""
    def body(mode):
        page = draw(mode, height=height - 36)
        return page[page.index("<body>") + 6:page.index("</body>")]
    head = draw("discharge", height=height - 36)
    style = head[head.index("<style>") + 7:head.index("</style>")]
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{style}
.sw2{{display:flex;gap:22px;padding:4px 4px 10px;font-size:14px;color:#e5e7eb}}
.sw2 label{{cursor:pointer;display:flex;align-items:center;gap:6px}}
.sw2 input{{accent-color:#ff4b4b;margin:0;cursor:pointer}}
</style></head><body>
<div class="sw2" role="radiogroup" aria-label="Show">
 <label><input type="radio" name="m" value="discharge" checked>Discharging (gives power)</label>
 <label><input type="radio" name="m" value="charge">Charging (stores power)</label>
</div>
<div id="view"></div>
<template id="t-discharge">{body("discharge")}</template>
<template id="t-charge">{body("charge")}</template>
<script>
const show = m => {{
  document.getElementById("view").innerHTML = document.getElementById("t-" + m).innerHTML;
  try {{ sessionStorage.setItem("h2anim:{draw.__name__}", m); }} catch (e) {{}}
}};
let m0 = "discharge";
try {{ m0 = sessionStorage.getItem("h2anim:{draw.__name__}") || m0; }} catch (e) {{}}
document.querySelectorAll('input[name=m]').forEach(i => {{
  i.checked = i.value === m0;
  i.addEventListener("change", () => show(i.value));
}});
show(m0);
</script></body></html>"""
