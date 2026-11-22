"""Recompute liquid-battery molecules from first principles (DFT, no reference couple), in parallel.
Usage: python scripts/verify_molecules.py SMILES [SMILES...]   or   --top N   (best N of the search library)
Results go to cache/abinitio.json; the app reads them (abinitio.hydrogenation_voltage(..., 'dft'))."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from h2screener import abinitio as A  # noqa: E402
from h2screener import ai_search  # noqa: E402


def main():
    store = ai_search.load_store()
    args = sys.argv[1:]
    if args[:1] == ["--top"]:
        n = int(args[1]) if len(args) > 1 else 10
        mols = [r.smiles for r in sorted(store.values(), key=lambda r: -(r.objective if r.objective == r.objective else -1))[:n]]
    else:
        mols = args
    from h2screener import stability as S
    items = [("[H][H]", "dft", "g"), ("O", "xtb", "aq"), ("O", "dft", "g")]
    for m in mols:
        r = store.get(m)
        red = (r.forms[0] if r and r.forms else (r.reduced if r else None))
        items += [(m, "dft", "aq")] + ([(red, "dft", "aq")] if red else [])
        for x in [m] + ([red] if red else []):            # reactions with water, both states, both levels
            for _, prods in S.products(x):
                items += [(p, lv, "aq") for p in prods for lv in ("xtb", "dft")]
    t0 = time.time()
    print(f"{len(mols)} molecules, {len(items)} calculations", flush=True)
    A.compute_many(items, progress=lambda k, n, x: print(f"[{k}/{n}] {x[0]:40s} {x[3] or 'ok'} {time.time() - t0:6.0f} s",
                                                         flush=True))
    for m in mols:
        r = store.get(m)
        red = (r.forms[0] if r and r.forms else (r.reduced if r else None))
        if red:
            print(m, "xTB", round(r.steps[0] if r.steps else r.e0, 3), "DFT",
                  A.hydrogenation_voltage(m, red, 1, "dft", compute=False), flush=True)
        for x in [m] + ([red] if red else []):
            print("  water:", x, S.summary(x)[0], flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
