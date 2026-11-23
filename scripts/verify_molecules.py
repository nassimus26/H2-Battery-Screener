"""Recompute liquid-battery molecules from first principles (DFT, no reference couple), in parallel.
Usage: python scripts/verify_molecules.py [--job ID] SMILES [SMILES...]   or   [--job ID] --top N
Results go to cache/abinitio.json; the app reads them (abinitio.hydrogenation_voltage(..., 'dft')).
With --job, progress is written to cache/dft_jobs/ID.json (h2screener/dftjobs.py), which the app shows."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from h2screener import abinitio as A  # noqa: E402
from h2screener import ai_search  # noqa: E402
from h2screener import dftjobs as J  # noqa: E402


def main():
    store = ai_search.load_store()
    args = sys.argv[1:]
    job = None
    if args[:1] == ["--job"]:
        job, args = args[1], args[2:]
    try:
        run(store, args, job)
    except Exception as e:
        if job:
            J.update(job, status="failed", finished=time.time(), current=f"{type(e).__name__}: {e}")
        raise


def run(store, args, job):
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
    last = [t0]
    print(f"{len(mols)} molecules, {len(items)} calculations", flush=True)
    if job:
        J.update(job, status="running", pid=os.getpid(), molecules=mols, started=t0)

    def progress(k, n, x):
        now = time.time()
        print(f"[{k}/{n}] {x[0]:40s} {x[3] or 'ok'} {now - t0:6.0f} s", flush=True)
        if job:
            rec = J._path(job)
            durs = []
            try:
                import json as _json
                durs = _json.loads(rec.read_text()).get("durations", [])
            except Exception:
                pass
            J.update(job, done=k, total=n, current=f"{x[0]} ({x[1]}, {x[2]}): {x[3] or 'ok'}",
                     durations=durs + [now - last[0]])
        last[0] = now

    A.compute_many(items, progress=progress)
    results = {}
    for m in mols:
        r = store.get(m)
        red = (r.forms[0] if r and r.forms else (r.reduced if r else None))
        if red:
            xv = r.steps[0] if r.steps else r.e0
            dv = A.hydrogenation_voltage(m, red, 1, "dft", compute=False)
            print(m, "xTB", round(xv, 3), "DFT", dv, flush=True)
            results[m] = {"xtb": xv, "dft": dv}
        for x in [m] + ([red] if red else []):
            print("  water:", x, S.summary(x)[0], flush=True)
    if job:
        J.update(job, status="finished", finished=time.time(), results=results)
    print("done", flush=True)


if __name__ == "__main__":
    main()
