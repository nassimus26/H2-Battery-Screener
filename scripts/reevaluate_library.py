"""Re-evaluate the molecules of the search library with the current model.

Use after a model fix: every molecule is computed again (xTB energies already
in the cache are reused) and the old and new results are compared.

  compute:  .venv/bin/python scripts/reevaluate_library.py compute [--library CSV] [--out CSV]
  apply:    .venv/bin/python scripts/reevaluate_library.py apply --new CSV [--library CSV]

'apply' first copies the library to <library>.<time>.bak.csv, then replaces
the entries that were re-evaluated and keeps every molecule added since.
Never run 'apply' while a search is running: the search rewrites the library.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from h2screener import ai_search, pairs  # noqa: E402
from h2screener import molecules as M  # noqa: E402


def _load(path: Path) -> dict:
    ai_search.STORE = path
    return ai_search.load_store()


def _save(path: Path, store: dict) -> None:
    ai_search.STORE = path
    ai_search.save_store(store)


def compute(library: Path, out: Path) -> None:
    old = _load(library)
    new = {}
    t0 = time.time()
    for i, (s, r) in enumerate(old.items(), 1):
        try:
            n = M.evaluate(s, check_pubchem=False)
            n.in_pubchem = r.in_pubchem
        except Exception as e:
            n = r
            n.notes = list(n.notes) + [f"re-evaluation failed: {e}"]
        new[s] = n
        if i % 10 == 0:
            print(f"{i}/{len(old)} in {time.time() - t0:.0f} s", flush=True)
            _save(out, new)
    _save(out, new)
    # summary
    ok_old = {s for s, r in old.items() if math.isfinite(r.e0) and pairs.usable(r)}
    ok_new = {s for s, r in new.items() if math.isfinite(r.e0) and pairs.usable(r)}
    moved = [(s, old[s].e0, new[s].e0) for s in old if math.isfinite(old[s].e0) and math.isfinite(new[s].e0)
             and abs(old[s].e0 - new[s].e0) > 0.05]
    print(f"\nmolecules {len(old)}; usable before {len(ok_old)}, after {len(ok_new)}")
    print(f"newly usable: {len(ok_new - ok_old)}; no longer usable: {len(ok_old - ok_new)}")
    for s in sorted(ok_new - ok_old):
        print(f"  + {s}  E0 {new[s].e0:.2f} V, solubility {new[s].solubility_mol_l:.2g} M")
    for s in sorted(ok_old - ok_new):
        print(f"  - {s}  ({'; '.join(new[s].notes + M.instability(s))})")
    print(f"voltage changed by more than 0.05 V: {len(moved)}")
    for s, a, b in moved[:20]:
        print(f"  {s}: {a:.2f} -> {b:.2f} V")
    for title, pool in (("before", list(old.values())), ("after", list(new.values()))):
        bp = pairs.best_pairs(pool, 5, max_per_molecule=2)
        print(f"\nbest pairs {title}:")
        for p in bp:
            print(f"  {p.score:6.1f}  {p.voltage:.2f} V  {p.wh_l:5.1f} Wh/L  {p.neg.smiles} | {p.pos.smiles}")


def apply(library: Path, new_path: Path) -> None:
    cur = _load(library)
    new = _load(new_path)
    bak = library.with_name(f"{library.stem}.{time.strftime('%Y%m%d-%H%M%S')}.bak.csv")
    shutil.copy2(library, bak)
    replaced = 0
    for s in cur:
        if s in new:
            cur[s] = new[s]
            replaced += 1
    _save(library, cur)
    print(f"backup: {bak}\nreplaced {replaced} of {len(cur)} molecules ({len(cur) - replaced} added since were kept)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "apply"])
    ap.add_argument("--library", default=str(ROOT / "cache" / "molecule_search.csv"))
    ap.add_argument("--out", default=str(ROOT / "out" / "molecule_search.reevaluated.csv"))
    ap.add_argument("--new", default=None)
    a = ap.parse_args()
    if a.mode == "compute":
        Path(a.out).parent.mkdir(exist_ok=True)
        compute(Path(a.library), Path(a.out))
    else:
        apply(Path(a.library), Path(a.new or a.out))


if __name__ == "__main__":
    main()
