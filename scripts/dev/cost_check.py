import sys, tomllib, copy
sys.path.insert(0, ".")
import numpy as np
from h2screener import gases, options, stackcost
cells, sp = gases.screen()
c5 = next(x for x in cells if x.key == "O=O>O|C#CCC#C>O=C=O")
h2 = gases.reference_h2_o2(cells)
base = tomllib.load(open("targets.toml", "rb"))
def tt(scale, power=200, hours=0.35):
    t = copy.deepcopy(base)
    t["design"].update(power_kw=power, hours=hours, stack_usd_m2=stackcost.stack("gas", scale).usd,
                       power_electronics_usd_kw=stackcost.power_electronics(scale).usd)
    t["refuel"]["composite_usd_per_kg"] *= stackcost.TANK_COST_FACTOR[scale]
    return t
for p, h in ((200, 0.35), (100, 6)):
    for cell, nm in ((h2, "H2+O2"), (c5, "C5H4+O2")):
        for sc in ("mass", "target"):
            cf = min(options.gas_configurations(cell, sp, tt(sc, p, h), np.random.default_rng(3)), key=lambda x: x.lcos_usd_kwh)
            parts = {k: round(v) for k, v in sorted(cf.cost_parts.items(), key=lambda kv: -kv[1]) if v > 1}
            print(f"{p}kW x {h}h {nm:8s} {sc:6s} ${cf.capex_usd_kwh:6,.0f}/kWh  lcos {cf.lcos_usd_kwh:.3f}  {cf.wh_kg:4.0f} Wh/kg rte {cf.rte:.2f} {parts}")
