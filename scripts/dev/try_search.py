import sys, time, json
sys.path.insert(0, ".")
from h2screener import gases as G, gas_search as S
t=time.time(); ctx=S.Context(); print("context", round(time.time()-t,1), "s; partners", len(ctx.pos), len(ctx.neg), "best", round(ctx.best.est_wh_kg), ctx.best.name)
cells,_=G.screen(); print("cells now", len(cells))
t=time.time()
v,c=ctx.cell_value("CCCCC", -8.0)   # pentane, measured dfG gas ~ -8 kJ/mol
print("pentane value", round(v), c.name if c else None, round(time.time()-t,2),"s")
t=time.time()
st=S.search(generations=1, batch=2, children=40, max_heavy=7,
            progress=lambda g,s: print("gen",g,"done"))
print("search", round(time.time()-t,1), "s")
for s,r in st["molecules"].items(): print(s, r)
print(st["history"])
