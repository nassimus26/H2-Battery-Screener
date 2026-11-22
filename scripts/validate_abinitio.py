"""Test the first-principles method on known molecules (their measured values are used ONLY here).

1. Gas reactions, no fitting at all: complete oxidation of every known gas molecule
   (C -> CO2, H -> H2O, N -> N2, S -> SO2, F -> HF, Cl -> HCl, balanced with O2), computed vs
   measured dG. Reported also as a voltage error: dG / (n F), n = electrons moved (4 per O2).
2. Gas energies of formation: element references from the computed H2, O2, N2, F2, Cl2; the two
   solid elements (C, S) fitted on a training half of the known gases, errors on the test half.
3. Aqueous hydrogenation voltages (quinone -> hydroquinone vs SHE), no reference couple.

Usage: python scripts/validate_abinitio.py [xtb|dft|both]   (writes cache/validation.json and
docs/validation_abinitio.md; every energy is cached in cache/abinitio.json, so it resumes)."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from rdkit import Chem  # noqa: E402

from h2screener import abinitio as A  # noqa: E402
from h2screener import gases as G  # noqa: E402

F_KJ = 96.485332
AQ = [  # name, oxidised, reduced, measured E0 V vs SHE (Q + 2H+ + 2e- -> QH2), source
    ("p-benzoquinone", "O=C1C=CC(=O)C=C1", "Oc1ccc(O)cc1", 0.6992, "standard potential table"),
    ("o-benzoquinone", "O=C1C=CC=CC1=O", "Oc1ccccc1O", 0.792, "Clark 1960"),
    ("methyl-p-benzoquinone", "CC1=CC(=O)C=CC1=O", "Cc1cc(O)ccc1O", 0.645, "Clark 1960"),
    ("1,4-naphthoquinone", "O=C1C=CC(=O)c2ccccc21", "Oc1ccc(O)c2ccccc12", 0.470, "Clark 1960"),
    ("AQS", "O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21", "Oc1c2ccccc2c(O)c2cc(S(=O)(=O)O)ccc12", 0.187, "Clark 1960"),
    ("AQDS", "O=C1c2ccc(S(=O)(=O)O)cc2C(=O)c2cc(S(=O)(=O)O)ccc21",
     "Oc1c2ccc(S(=O)(=O)O)cc2c(O)c2cc(S(=O)(=O)O)ccc12", 0.213, "Er et al. 2015"),
]
SINK = {"C": ("O=C=O", 1), "S": ("O=S=O", 1), "N": ("N#N", 2), "F": ("F", 1), "Cl": ("Cl", 1)}


def canon(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))


def oxidation(smiles):
    """stoichiometry of  molecule + x O2 -> products; electrons moved n"""
    c = G.counts(smiles)
    st, h = {smiles: -1.0}, c.get("H", 0)
    o_need = 0.0
    for el, (prod, per) in SINK.items():
        k = c.get(el, 0)
        if not k:
            continue
        st[canon(prod)] = st.get(canon(prod), 0) + k / per
        po = G.counts(prod)
        o_need += k / per * po.get("O", 0)
        h -= k / per * po.get("H", 0)
    if h < -1e-9:
        return None, 0
    if h > 0:
        st["O"] = st.get("O", 0) + h / 2
        o_need += h / 2
    o2 = (o_need - c.get("O", 0)) / 2
    if abs(o2) < 1e-9:
        return None, 0
    st["O=O"] = st.get("O=O", 0) - o2
    return st, abs(4 * o2)


def g_meas(s):
    return G.MEASURED[s][1]


def run(level):
    t0 = time.time()
    known = [s for s in G.MEASURED if Chem.MolFromSmiles(s) is not None]
    rows_r, rows_f = [], []
    for s in known:
        st, n = oxidation(s)
        if st is None or any(p not in G.MEASURED for p in st):
            continue
        dg_c = A.reaction_g(st, level, "g")
        dg_m = sum(k * g_meas(p) for p, k in st.items())
        if dg_c is None:
            rows_r.append({"smiles": s, "name": G.MEASURED[s][0], "error": "calculation failed"})
            continue
        rows_r.append({"smiles": s, "name": G.MEASURED[s][0], "electrons": n, "dG_meas": dg_m, "dG_calc": dg_c,
                       "err_kj": dg_c - dg_m, "err_v": (dg_c - dg_m) / (n * F_KJ)})
        print(f"{level} {G.MEASURED[s][0]:28s} dG {dg_c:9.1f} vs {dg_m:9.1f}  err {dg_c - dg_m:+7.1f} kJ  "
              f"{(dg_c - dg_m) / (n * F_KJ):+.3f} V   {time.time() - t0:6.0f} s", flush=True)
    # formation energies: computed gas references, C and S fitted on the training half
    ref = {"H": "[H][H]", "O": "O=O", "N": "N#N", "F": "FF", "Cl": "ClCl"}
    mu = {el: A.free_energy(r, level, "g")["g_kj"] / 2 for el, r in ref.items()}
    data = []
    for s in sorted(known):
        r = A.free_energy(s, level, "g", compute=False)
        if r and "g_kj" in r:
            c = G.counts(s)
            data.append((s, c, r["g_kj"] - sum(c.get(e, 0) * mu[e] for e in ref), g_meas(s)))
    train, test = data[0::2], data[1::2]
    a = np.array([[c.get("C", 0), c.get("S", 0)] for _, c, _, _ in train], float)
    y = np.array([g - m for _, _, g, m in train])
    sol, *_ = np.linalg.lstsq(a, y, rcond=None)
    for split, part in (("train", train), ("test", test)):
        for s, c, g, m in part:
            calc = g - sol[0] * c.get("C", 0) - sol[1] * c.get("S", 0)
            rows_f.append({"smiles": s, "name": G.MEASURED[s][0], "split": split, "dfG_meas": m, "dfG_calc": calc,
                           "err_kj": calc - m})
    aq = []
    for name, ox, red, e_m, src in (AQ if level != "ccsdt" else []):
        e_c = A.hydrogenation_voltage(ox, red, 1, level)
        aq.append({"name": name, "ox": ox, "red": red, "E_meas": e_m, "E_calc": e_c, "source": src,
                   "err_v": None if e_c is None else e_c - e_m})
        print(f"{level} {name:28s} E {e_c} vs {e_m}", flush=True)
    return {"reactions": rows_r, "formation": rows_f, "aqueous": aq, "mu_C_S": list(map(float, sol)),
            "seconds": round(time.time() - t0)}


def stats(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    if not xs:
        return None
    return {"n": len(xs), "mae": float(np.mean(np.abs(xs))), "rmse": float(np.sqrt(np.mean(np.square(xs)))),
            "max": float(np.max(np.abs(xs))), "mean": float(np.mean(xs))}


def precompute(levels):
    """Everything the tests need, computed in parallel worker processes first."""
    known = [s for s in G.MEASURED if Chem.MolFromSmiles(s) is not None]
    items = [(s, "xtb", "g") for s in known] + [(x, "xtb", "aq") for _, a, b, _, _ in AQ for x in (a, b)]
    if "dft" in levels:
        items += [(s, "dft", "g") for s in known] + [(x, "dft", "aq") for _, a, b, _, _ in AQ for x in (a, b)]
    if "ccsdt" in levels:
        items += [(s, "ccsdt", "g") for s in known if A.heavy_atoms(s) <= A.CC_MAX_HEAVY]
    t0 = time.time()
    A.compute_many(items, progress=lambda k, n, r: print(f"[{k}/{n}] {r[1]:5s} {r[2]:2s} {r[0]:30s} "
                                                       f"{r[3] or 'ok'}  {time.time() - t0:6.0f} s", flush=True))


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    levels = ["xtb", "dft", "ccsdt"] if which == "all" else which.split(",")
    precompute(levels)
    out_p = ROOT / "cache" / "validation.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for lv in levels:
        res = run(lv)
        res["summary"] = {
            "gas reactions, V": stats([r.get("err_v") for r in res["reactions"]]),
            "gas reactions, kJ/mol": stats([r.get("err_kj") for r in res["reactions"]]),
            "formation test, kJ/mol": stats([r["err_kj"] for r in res["formation"] if r["split"] == "test"]),
            "formation train, kJ/mol": stats([r["err_kj"] for r in res["formation"] if r["split"] == "train"]),
            "aqueous voltage, V": stats([r["err_v"] for r in res["aqueous"]]),
        }
        res["method"] = A.LEVELS[lv]
        res["date"] = time.strftime("%Y-%m-%d %H:%M")
        out[lv] = res
        out_p.write_text(json.dumps(out, indent=1))
        print(lv, json.dumps(res["summary"], indent=1), flush=True)
    L = ["# First-principles method tested on known molecules", "",
         "Measured values are used only here, to test. See h2screener/abinitio.py for the method.", ""]
    for lv, res in out.items():
        L += [f"## {res['method']} ({res['date']}, {res['seconds']} s)", ""]
        for k, v in res["summary"].items():
            if v:
                L.append(f"- {k}: mean abs error {v['mae']:.3f}, RMS {v['rmse']:.3f}, max {v['max']:.3f}, "
                         f"mean signed {v['mean']:+.3f} (n = {v['n']})")
        L += ["", "| oxidation of | electrons | dG measured | dG computed | error kJ/mol | error V |", "|---|---|---|---|---|---|"]
        for r in res["reactions"]:
            if "err_kj" in r:
                L.append(f"| {r['name']} | {r['electrons']:.0f} | {r['dG_meas']:.1f} | {r['dG_calc']:.1f} | "
                         f"{r['err_kj']:+.1f} | {r['err_v']:+.3f} |")
        L += ["", "| quinone | measured E V | computed E V | error V | source |", "|---|---|---|---|---|"]
        for r in res["aqueous"]:
            ec = "-" if r["E_calc"] is None else f"{r['E_calc']:.3f}"
            ev = "-" if r["err_v"] is None else f"{r['err_v']:+.3f}"
            L.append(f"| {r['name']} | {r['E_meas']:.3f} | {ec} | {ev} | {r['source']} |")
        L.append("")
    (ROOT / "docs" / "validation_abinitio.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
