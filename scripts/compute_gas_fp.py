"""Compute every gas-battery molecule from first principles, in parallel, in the background:
DFT (wB97X-V/def2-TZVP) for all, CCSD(T)/cc-pVTZ for the small ones. Resumable (cache/abinitio.json).
Usage: python scripts/compute_gas_fp.py [max heavy atoms for CCSD(T), default 3]"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from h2screener import abinitio as A  # noqa: E402
from h2screener import gases as G  # noqa: E402

def main():
    import fcntl
    lock = open(ROOT / "cache" / "gas_fp.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)      # one batch at a time
    except OSError:
        print("another batch is running", flush=True)
        return
    cc_max = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    sp = [s for s in G.all_species()
          if s in G.MEASURED or (G.boiling_point(s)[0] < G.XTB_CUTOFF_K and G.storable(s))]
    # known molecules first (they set the element references), then the smallest
    sp.sort(key=lambda s: (s not in G.MEASURED, A.heavy_atoms(s), s))
    items = [(s, "dft", "g") for s in sp] + [("O", "xtb", "aq")]
    items += [(s, "ccsdt", "g") for s in sp if A.heavy_atoms(s) <= cc_max] + [("O", "ccsdt", "g")]
    t0 = time.time()
    print(f"{len(sp)} molecules; {len(items)} calculations", flush=True)
    A.compute_many(items, workers=6, progress=lambda k, n, r: print(f"[{k}/{n}] {r[1]:5s} {r[0]:34s} {r[3] or 'ok'}  "
                                                         f"{time.time() - t0:6.0f} s", flush=True))
    print("done", flush=True)


if __name__ == "__main__":
    main()
