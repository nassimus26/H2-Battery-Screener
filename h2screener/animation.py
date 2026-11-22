"""Animated diagram of a two-molecule (two-tank) battery, as self-contained
HTML/SVG for the app.

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


def two_tank_html(mode: str = "discharge", height: int = 400) -> str:
    """mode: 'discharge' or 'charge'."""
    rev = mode == "charge"
    load = ("charger", "#3b82f6") if rev else ("motor / grid", "#22c55e")
    # bright layer = charged liquid (N-H2 on the negative side, P on the positive):
    # it shrinks while discharging and grows while charging
    h_from, h_to = (40, 150) if rev else (150, 40)
    neg_txt = "N + 2H+ + 2e- -> N-H2" if rev else "N-H2 -> N + 2H+ + 2e-"
    pos_txt = "P-H2 -> P + 2H+ + 2e-" if rev else "P + 2H+ + 2e- -> P-H2"
    flow_dots = "".join([
        _dots("neg_out", 6, 4, "#60a5fa", 5, False), _dots("neg_back", 6, 4, "#93c5fd", 5, False),
        _dots("pos_out", 6, 4, "#fb923c", 5, False), _dots("pos_back", 6, 4, "#fdba74", 5, False),
    ])
    electrons = _dots("wire", 7, 3, "#facc15", 7, rev, "e-")
    tank = lambda x, fill_c, fill_s, label, sub: f"""
      <rect x="{x}" y="125" width="140" height="175" rx="14" fill="#1f2937" stroke="#9ca3af" stroke-width="2"/>
      <rect x="{x + 4}" y="150" width="132" height="146" rx="10" fill="{fill_s}"/>
      <rect x="{x + 4}" width="132" rx="10" fill="{fill_c}" y="{296 - h_from}" height="{h_from}">
        <animate attributeName="height" values="{h_from};{h_to};{h_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
        <animate attributeName="y" values="{296 - h_from};{296 - h_to};{296 - h_from}" keyTimes="0;0.9;1" dur="16s" repeatCount="indefinite"/>
      </rect>
      <text x="{x + 70}" y="118" fill="#e5e7eb" font-size="13" text-anchor="middle" font-weight="600">{label}</text>
"""
    pump = lambda x: (f'<circle cx="{x}" cy="330" r="11" fill="#374151" stroke="#9ca3af"/>'
                      f'<path d="M {x - 5} 334 L {x} 324 L {x + 5} 334 Z" fill="#9ca3af">'
                      f'<animateTransform attributeName="transform" type="rotate" from="0 {x} 330" to="360 {x} 330" dur="1.2s" repeatCount="indefinite"/></path>')
    paths = "".join(f'<path id="{k}" d="{d}" fill="none" stroke="#4b5563" stroke-width="8" stroke-linejoin="round"/>'
                    for k, d in _PATHS.items() if k != "wire")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;background:transparent;font-family:system-ui,-apple-system,sans-serif}}
.card{{background:#111827;border-radius:12px;padding:8px 8px 4px}}
.leg{{color:#cbd5e1;font-size:12px;padding:2px 8px 6px;line-height:1.5}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px;vertical-align:-1px}}
</style></head><body><div class="card">
<svg viewBox="0 0 760 365" width="100%" style="max-height:{height - 70}px" role="img"
  aria-label="Two-tank battery: liquids pumped through a cell; protons cross the membrane, electrons go through the outside wire">
  {paths}
  <path id="wire" d="{_PATHS['wire']}" fill="none" stroke="#d1d5db" stroke-width="3"/>
  {tank(30, "#2563eb", "#1e3a5f", "Negative tank", "blue = charged N-H2, dark = spent N")}
  {tank(590, "#ea580c", "#5c2f12", "Positive tank", "orange = charged P, dark = spent P-H2")}
  {pump(175)}{pump(585)}
  <text x="175" y="355" fill="#9ca3af" font-size="10" text-anchor="middle">pump</text>
  <text x="585" y="355" fill="#9ca3af" font-size="10" text-anchor="middle">pump</text>
  <rect x="318" y="140" width="124" height="155" rx="6" fill="#0b1220" stroke="#9ca3af" stroke-width="2"/>
  <rect x="322" y="144" width="52" height="147" fill="#1e3a5f" opacity="0.8"/>
  <rect x="386" y="144" width="52" height="147" fill="#5c2f12" opacity="0.8"/>
  <line x1="380" y1="142" x2="380" y2="293" stroke="#e5e7eb" stroke-width="3" stroke-dasharray="6 4"/>
  <text x="380" y="310" fill="#e5e7eb" font-size="11" text-anchor="middle">cell (membrane in the middle)</text>
  {flow_dots}{_protons(rev)}{electrons}
  <rect x="352" y="36" width="56" height="28" rx="8" fill="{load[1]}"/>
  <text x="380" y="55" fill="#0b1220" font-size="12" text-anchor="middle" font-weight="700">{'+ / -' if rev else 'M'}</text>
  <text x="380" y="26" fill="#e5e7eb" font-size="12" text-anchor="middle">{load[0]}</text>
</svg>
<div class="leg"><b>{'Charging' if rev else 'Discharging'}.</b>
 <span class="sw" style="background:#60a5fa"></span>negative liquid
 <span class="sw" style="background:#fb923c"></span>positive liquid
 <span class="sw" style="background:#e5484d"></span>H+ through the membrane
 <span class="sw" style="background:#facc15"></span>e- through the wire<br>
 <span style="color:#bfdbfe">Negative side: {neg_txt}</span> &nbsp;·&nbsp; <span style="color:#fed7aa">Positive side: {pos_txt}</span><br>
 In the tanks, the bright layer is charged liquid (N-H2 on the left, P on the right) and the dark part is spent liquid;
 {'charging refills the bright layers' if rev else 'discharging empties the bright layers'}.<br>
 The hydrogen never becomes a gas: it travels as protons and electrons from one molecule to the other.
 Energy is set by the tanks (litres of liquid); power by the cell area (m2). Tanks, pumps and pipes are not costed yet.</div>
</div></body></html>"""
