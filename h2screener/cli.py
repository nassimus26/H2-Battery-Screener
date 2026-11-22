"""Command line: generate candidates, evaluate, rank, write the report.

  python -m h2screener run [--mp Ni,Mn,Fe | --mp all] [--ghs] [--top 30] [--out out]
  python -m h2screener validate
"""

from __future__ import annotations

import argparse
import csv
import math
import tomllib
from pathlib import Path

import numpy as np

from . import discover, performance, rank
from .performance import CellModel
from .reactions import load_table

ROOT = Path(__file__).resolve().parent.parent


def load_targets(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def fmt3(t) -> str:
    return " / ".join(f"{v:.3g}" for v in t)


def run(args) -> None:
    t = load_targets(Path(args.targets))
    cells = discover.table_pairs()
    if args.mp:
        from pymatgen.core.periodic_table import Element
        els = [e.symbol for e in Element if e.is_metal or e.is_metalloid] if args.mp == "all" else args.mp.split(",")
        cells += discover.mp_hydrogenation(els)
    print(f"{len(cells)} candidate reactions")
    results = rank.rank(cells, t, ghs=args.ghs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(results, out / "candidates.csv")
    write_report(results, t, out / "report.md", args.top)
    print(f"wrote {out/'candidates.csv'} and {out/'report.md'}")
    for r in results[: min(args.top, 15)]:
        b = r.best
        print(f"{r.score:.3f}  {r.cell.e0:5.2f} V  {r.cell.medium:9s}  "
              f"RTE {r.perf.get('rte_at_i_charge', (0, 0, 0))[1]:.2f}  "
              f"{(b.kg_per_kwh if b else float('nan')):6.1f} kg/kWh @ {(b.p_max_bar if b else 0):g} bar  {r.cell.name}")


def write_csv(results, path: Path) -> None:
    cols = ["rank", "score", "reaction", "medium", "origin", "e0_v", "n", "wh_kg_active",
            "rte_p10", "rte_p50", "rte_p90", "i_charge_p50_a_cm2", "t_charge_p50_h",
            "p_target_p50_w_cm2", "p_peak_p50_w_cm2", "best_p_bar", "kg_per_kwh", "l_per_kwh",
            "usd_per_kwh", "dominant_unknown", "flags", "source"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(results, 1):
            p = r.perf or {}
            b = r.best
            g = lambda k, j=1: p.get(k, (math.nan,) * 3)[j]
            w.writerow([i, f"{r.score:.4f}", r.cell.name, r.cell.medium, r.cell.origin,
                        f"{r.cell.e0:.3f}", f"{r.cell.n:g}", f"{r.wh_kg_active:.0f}",
                        f"{g('rte_at_i_charge', 0):.3f}", f"{g('rte_at_i_charge'):.3f}", f"{g('rte_at_i_charge', 2):.3f}",
                        f"{g('i_charge_a_cm2'):.3g}", f"{g('t_charge_h'):.3g}",
                        f"{g('p_at_target_w_cm2'):.3g}", f"{g('p_peak_w_cm2'):.3g}",
                        b.p_max_bar if b else "", f"{b.kg_per_kwh:.2f}" if b else "",
                        f"{b.l_per_kwh:.2f}" if b else "",
                        f"{b.usd_per_kwh:.1f}" if b and b.usd_per_kwh is not None else "",
                        p.get("dominant_unknown", ""), "; ".join(r.flags), r.cell.source])


def write_report(results, t: dict, path: Path, top: int) -> None:
    u = t["use"]
    L = [f"# Hydrogen couple screen\n",
         f"Use: {u['application']}, {u['temperature_c']} C, H2 {u['p_min_bar']}-{u['p_max_bar']} bar, "
         f"round-trip target {u['rte_target']:.0%}, charge target {u['charge_hours_target']} h.\n",
         "Ranges are p10 / p50 / p90 over the Monte Carlo draws of the uncertain inputs. "
         "\"dominant unknown\" is the input that most changes the achievable charge rate; "
         "it is what a lab test should measure first.\n"]
    for i, r in enumerate(results[:top], 1):
        c, p, b = r.cell, r.perf, r.best
        L.append(f"## {i}. {c.name}\n")
        L.append(f"- score {r.score:.3f}; medium {c.medium}; origin {c.origin}")
        L.append(f"- E0 {c.e0:.3f} V, {c.n:g} e-, {r.wh_kg_active:.0f} Wh/kg of reactants (theoretical)")
        if p:
            L.append(f"- OCV over the charge (with H2 fugacity): {fmt3(p['ocv_v'])} V (min / mean / max)")
            L.append(f"- charge current meeting the efficiency target: {fmt3(p['i_charge_a_cm2'])} A/cm2; "
                     f"charge time {fmt3(p['t_charge_h'])} h at {t['system']['areal_capacity_mah_cm2']} mAh/cm2")
            L.append(f"- round-trip efficiency there: {fmt3(p['rte_at_i_charge'])}; "
                     f"targets met in {p['feasible_frac']:.0%} of draws")
            L.append(f"- power at that rate: {fmt3(p['p_at_target_w_cm2'])} W/cm2; peak: {fmt3(p['p_peak_w_cm2'])} W/cm2")
            L.append(f"- dominant unknown: {p['dominant_unknown']}")
        if r.sweep:
            L.append("- mass and volume vs maximum H2 pressure:\n")
            L.append("  | p_max bar | kg/kWh | L/kWh | vessel kg | USD/kWh |")
            L.append("  |---|---|---|---|---|")
            for s in r.sweep:
                usd = f"{s.usd_per_kwh:.0f}" if s.usd_per_kwh is not None else "?"
                L.append(f"  | {s.p_max_bar:g} | {s.kg_per_kwh:.1f} | {s.l_per_kwh:.1f} | {s.parts['vessel']:.2f} | {usd} |")
            L.append("")
        if r.flags:
            L.append("- flags: " + "; ".join(r.flags))
        L.append("- inputs: " + "; ".join(f"{k} [{v.kind}] {v.low:.3g}-{v.high:.3g} ({v.source})"
                                          for k, v in r.params.items()))
        L.append(f"- data: {c.source}\n")
    path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ validation

def validate(args) -> None:
    """Compare the model's prior predictions with published measurements of
    hydrogen cells. These cells are never used as candidates or for fitting;
    they show whether the physics and the assumed ranges cover reality."""
    t = load_targets(Path(args.targets))
    cells = {c.name: c for c in discover.table_pairs(load_table())}
    rng = np.random.default_rng(7)
    t_k = t["use"]["temperature_c"] + 273.15
    from . import estimate

    def model(name, p_bar=1.01325):
        c = cells[name]
        p = p_bar * 1e5
        return CellModel(c, t_k, p * 0.999, p, estimate.cell_params(c, t_k, p * 0.999, p, t))

    rows = []
    m = model("Br2(aq) + H2(g) -> 2 Br- + 2 H+")
    pk = performance.peak_power(m, rng)
    rows.append(("H2-Br2 peak power, W/cm2", "1.4 (Cho et al., J. Electrochem. Soc. 159 (2012) A1806)", fmt3(pk),
                 pk[0] <= 1.4 <= pk[2]))
    m = model("MnO2(s) + H2(g) + 2 H+ -> Mn(2+) + 2 H2O(l)")
    oc = m.ocv(np.array([0.5]))[0]
    rows.append(("Mn-H2 discharge voltage, V", "~1.3 (Chen et al., Nature Energy 3 (2018) 428)", f"OCV {oc:.3f}",
                 1.2 <= oc <= 1.45))
    for q, ref, got, ok in rows:
        print(f"[{'inside' if ok else 'OUTSIDE'}] {q}: measured {ref}; model p10/p50/p90 or value {got}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="h2screener")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--targets", default=str(ROOT / "targets.toml"))
    r.add_argument("--mp", help="Materials Project metals: comma list or 'all' (needs MP_API_KEY)")
    r.add_argument("--ghs", action="store_true", help="fetch GHS hazards from PubChem (online, cached)")
    r.add_argument("--top", type=int, default=30)
    r.add_argument("--out", default=str(ROOT / "out"))
    r.set_defaults(fn=run)
    v = sub.add_parser("validate")
    v.add_argument("--targets", default=str(ROOT / "targets.toml"))
    v.set_defaults(fn=validate)
    a = ap.parse_args(argv)
    a.fn(a)
