import time, sys
sys.path.insert(0, ".")
from h2screener import gases as G
t=time.time(); cache=G.load_energies(); sp=G.species_table(cache); t1=time.time()
cps=G.couples(sp); t2=time.time()
cells, _ = G.screen(cache); t3=time.time()
print("species", len(sp), "gases", sum(g.gas for g in sp.values()), "t", round(t1-t,1))
print("couples", len(cps), "t", round(t2-t1,1))
print("cells", len(cells), "t", round(t3-t2,1))
for c in cells[:8]: print(round(c.est_wh_kg), round(c.wh_per_mol_gas,1), c.name)
import collections
print(collections.Counter(G.counts(s).get("C",0) for s in sp))
