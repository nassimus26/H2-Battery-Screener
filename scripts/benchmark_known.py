"""Benchmark: does the simulator get known flow-battery molecules right?

1. Voltage: xTB/ALPB potentials (molecules.evaluate) against measured
   two-electron, two-proton potentials in water at pH 0 vs SHE.
2. Ranking: where known good molecules and a known molecule pair land among
   the molecules of the search library (cache/molecule_search.csv).

Usage: .venv/bin/python scripts/benchmark_known.py [--library PATH] [--out PATH]
Writes a Markdown report (default out/benchmark_known.md).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from h2screener import ai_search, pairs  # noqa: E402
from h2screener import molecules as M  # noqa: E402

# Measured E0 (V vs SHE, pH 0, Q + 2H+ + 2e- -> QH2).
#   table    data/standard_potentials.csv (the calibration point of the model)
#   Er2015   Er et al., Chem. Sci. 6 (2015) 885, measured value quoted in the text
#   textbook classic compilations (W. M. Clark, Oxidation-Reduction Potentials of
#            Organic Systems, 1960); not re-checked against the primary source
MEASURED = [
    ("p-benzoquinone", "O=C1C=CC(=O)C=C1", 0.6992, "table (calibration point)"),
    ("o-benzoquinone", "O=C1C=CC=CC1=O", 0.792, "textbook"),
    ("methyl-p-benzoquinone", "CC1=CC(=O)C=CC1=O", 0.645, "textbook"),
    ("1,4-naphthoquinone", "O=C1C=CC(=O)c2ccccc21", 0.470, "textbook"),
    ("anthraquinone-2-sulfonic acid (AQS)", "O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21", 0.187, "textbook"),
    ("anthraquinone-2,7-disulfonic acid (AQDS)", "O=C1c2ccc(S(=O)(=O)O)cc2C(=O)c2cc(S(=O)(=O)O)ccc21", 0.213, "Er2015"),
]

# Molecules used in published aqueous flow batteries (proton-coupled, two electrons).
KNOWN_GOOD = [
    ("AQDS (negative, Huskinson et al. 2014)", "O=C1c2ccc(S(=O)(=O)O)cc2C(=O)c2cc(S(=O)(=O)O)ccc21"),
    ("AQS (negative, Yang et al. 2014)", "O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21"),
    ("BQDS, oxidised Tiron (positive, Yang et al. 2014)", "O=C1C(=O)C(S(=O)(=O)O)=CC(S(=O)(=O)O)=C1"),
    ("2,6-DHAQ (negative, Lin et al. 2015)", "O=C1c2ccc(O)cc2C(=O)c2cc(O)ccc21"),
    ("alloxazine (negative, Lin et al. 2016)", "O=c1[nH]c(=O)c2nc3ccccc3nc2[nH]1"),
    ("p-benzoquinone", "O=C1C=CC(=O)C=C1"),
]
KNOWN_PAIR = ("O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21", "O=C1C(=O)C(S(=O)(=O)O)=CC(S(=O)(=O)O)=C1")  # AQS | BQDS


def percentile(value: float, others: list[float]) -> float:
    """Share of the library this value beats (0-100)."""
    if not others:
        return math.nan
    return 100.0 * sum(v < value for v in others) / len(others)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", default=str(ROOT / "cache" / "molecule_search.csv"))
    ap.add_argument("--out", default=str(ROOT / "out" / "benchmark_known.md"))
    a = ap.parse_args()
    ai_search.STORE = Path(a.library)
    lib = [r for r in ai_search.load_store().values() if math.isfinite(r.e0)]
    lines = ["# Benchmark: known molecules", ""]

    # 1. voltages
    lines += ["## 1. Voltage: xTB vs measured (V vs SHE, pH 0)", "",
              "| molecule | measured | xTB | error | source | notes |", "|---|---|---|---|---|---|"]
    errs = []
    for name, smi, e_exp, src in MEASURED:
        r = M.evaluate(smi)
        e = r.e0
        err = e - e_exp if math.isfinite(e) else math.nan
        if math.isfinite(err) and "calibration" not in src:
            errs.append(err)
        lines.append(f"| {name} | {e_exp:.3f} | {e:.3f} | {err:+.3f} | {src} | {'; '.join(r.notes) or '-'} |"
                     if math.isfinite(e) else f"| {name} | {e_exp:.3f} | failed | - | {src} | {'; '.join(r.notes)} |")
    if errs:
        mae = sum(abs(x) for x in errs) / len(errs)
        bias = sum(errs) / len(errs)
        lines += ["", f"Mean absolute error {mae:.3f} V, mean signed error {bias:+.3f} V over {len(errs)} molecules "
                  f"(calibration point excluded). Stated model band: +/-{M.VOLTAGE_BAND} V."]

    # 2. ranking among the library
    lines += ["", f"## 2. Ranking among the {len(lib)} molecules of the search library", "",
              "Against H2: the molecule as the positive side of a cell with a hydrogen electrode (negatives "
              "score 0 here by design). In pairs: the best two-molecule pair it forms with any library "
              "molecule, against the best pair of each library molecule.", "",
              "| molecule | usable | E0 (xTB) | against H2: score | beats % | in pairs: best score | beats % |",
              "|---|---|---|---|---|---|---|"]
    lib_ok = [r for r in lib if pairs.usable(r)]
    lib_scores = [r.objective for r in lib_ok]

    def best_pair_score(r, pool):
        best = 0.0
        for q in pool:
            if q.smiles == r.smiles:
                continue
            for a, b in ((r, q), (q, r)):
                for kn, _ in pairs.role_options(a, "neg"):
                    for kp, _ in pairs.role_options(b, "pos"):
                        best = max(best, pairs.make_pair(a, b, kn, kp).score)
        return best
    lib_pair_best = [best_pair_score(r, lib_ok) for r in lib_ok]
    known = {}
    for name, smi in KNOWN_GOOD:
        r = M.evaluate(smi)
        known[smi] = r
        ok = pairs.usable(r)
        why = "" if ok else " (" + "; ".join(r.notes + M.instability(smi)) + ")"
        bp = best_pair_score(r, lib_ok) if ok else 0.0
        lines.append(f"| {name} | {'yes' if ok else 'no' + why} | {r.e0:.3f} | {r.objective:.1f} | "
                     f"{percentile(r.objective, lib_scores):.0f} | {bp:.1f} | {percentile(bp, lib_pair_best):.0f} |")

    lines += ["", "## 3. The known pair AQS | BQDS (Yang et al. 2014) against the library's pairs", ""]
    n, p = known.get(KNOWN_PAIR[0]), known.get(KNOWN_PAIR[1])
    if n and p and pairs.usable(n) and pairs.usable(p):
        pr = max((pairs.make_pair(n, p, kn, kp) for kn, _ in pairs.role_options(n, "neg")
                  for kp, _ in pairs.role_options(p, "pos")), key=lambda x: x.score)
        lib_pairs = pairs.best_pairs(lib, 100000)
        lines.append(f"Voltage {pr.voltage:.2f} V, {pr.wh_l:.1f} Wh/L, score {pr.score:.1f}; "
                     f"better than {percentile(pr.score, [x.score for x in lib_pairs]):.0f} % of the "
                     f"{len(lib_pairs)} library pairs.")
    else:
        lines.append("Cannot be scored: one of the two molecules is not usable in the model (see section 2).")
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
