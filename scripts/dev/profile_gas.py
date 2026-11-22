import cProfile, pstats, sys, time
sys.path.insert(0, ".")
from h2screener import gases as G
t = time.time(); sp = G.species_table(); print("species_table", len(sp), round(time.time() - t, 1), "s", flush=True)
src = {}
for s in sp.values():
    k = s.source.split(" (")[0]
    src[k] = src.get(k, 0) + 1
print(src, flush=True)
t = time.time(); cps = G.couples(sp); print("couples", len(cps), round(time.time() - t, 1), "s", flush=True)
pr = cProfile.Profile(); pr.enable()
t = time.time(); cells, _ = G.screen(sp=sp); print("screen", len(cells), round(time.time() - t, 1), "s", flush=True)
pr.disable(); pstats.Stats(pr).sort_stats("cumulative").print_stats(12)
