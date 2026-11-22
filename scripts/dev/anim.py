import sys; sys.path.insert(0, ".")
from h2screener import animation as A
for f in (A.molecule_h2_html, A.two_tank_html, A.two_gas_html):
    for m in ("discharge", "charge"):
        open(f"scripts/dev/{f.__name__}_{m}.html", "w").write(f(m, 540))
print("written")
