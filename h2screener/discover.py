"""Stage 1: candidate generation from data and laws, not from lists of known cells.

Generators:
  table_pairs      every half-reaction in the potential table paired with every
                   hydrogen electrode it can work against, in both roles:
                     H+/H2      (acid, E0 = 0 by definition of the SHE)
                     H2O/H2,OH- (alkaline)
  mp_hydrogenation every reaction  A + x H2 -> B + y H2O  between phases of the
                   M-O-H and M-H systems in Materials Project, for every metal M,
                   including computed phases that were never synthesised.
                   Reactions with no oxygen (M + x H2 -> MH2x) are hydride
                   cells that need a hydride-ion conductor.
                   Needs MP_API_KEY.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from .reactions import (
    ELECTRON, CellReaction, HalfReaction, Species, combine, load_table, parse_side,
)

SHE = HalfReaction(
    parse_side("2 H+ + 2 e-"), parse_side("H2(g)"), 0.0, 2,
    "2 H+ + 2 e- -> H2(g)", "standard hydrogen electrode (0 V by definition)", "acid",
)


def hydrogen_electrodes(table: list[HalfReaction]) -> list[HalfReaction]:
    out = [SHE]
    for hr in table:
        names = {sp.formula for sp in list(hr.oxidant) + list(hr.reductant) if sp is not ELECTRON}
        if hr.label.startswith("2 H2O(l) + 2 e- -> H2(g) + 2 OH-"):
            hr.medium = "alkaline"
            out.append(hr)
        elif names == {"H2", "H"} and any(sp.charge == -1 and sp.formula == "H" for sp in hr.reductant):
            hr.medium = "hydride"
            out.append(hr)
    return out


def _uses_water(hr: HalfReaction) -> bool:
    return any(sp.formula in ("H2O", "H", "OH") and sp.phase in ("l", "aq")
               for sp in list(hr.oxidant) + list(hr.reductant) if sp is not ELECTRON)


def table_pairs(table: list[HalfReaction] | None = None) -> list[CellReaction]:
    table = table if table is not None else load_table()
    electrodes = hydrogen_electrodes(table)
    seen: set[str] = set()
    out: list[CellReaction] = []

    def add(c: CellReaction | None, medium: str):
        if c is None:
            return
        c.medium = medium
        key = c.name + "|" + medium
        if key not in seen:
            seen.add(key)
            out.append(c)

    for h in electrodes:
        for hr in table:
            if hr in electrodes or hr.label == h.label:
                continue
            if h.medium in ("acid", "alkaline"):
                if hr.medium not in (h.medium, "any"):
                    continue
                if h.medium == "alkaline" and hr.medium == "any":
                    continue  # pH-free couples are taken once, against the SHE
                add(combine(hr, h), h.medium)        # hydrogen is the negative electrode
            # Aqueous potentials of ions include their hydration energy and do
            # not carry over to a hydride-ion conductor; hydride cells come
            # from solid formation energies (mp_hydrogenation) instead.
    return out


# ---------------------------------------------------------------- Materials Project

def mp_hydrogenation(elements: Iterable[str], e_hull_max: float = 0.05,
                     api_key: str | None = None) -> list[CellReaction]:
    """Reactions A + x H2 -> B + y H2O (per metal atom) between phases within
    e_hull_max eV/atom of the M-O-H convex hull.

    Energies: MP GGA/GGA+U total energies with MaterialsProject2020 corrections
    and the aqueous compatibility scheme, which fits H2 and H2O so that water's
    formation free energy is the experimental -2.46 eV. Solid entropy is
    neglected (the usual Pourbaix-diagram approximation, about 0.1 V error).
    The reaction exchanges only H2 and H2O, so its voltage vs the hydrogen
    electrode does not depend on pH.
    """
    from mp_api.client import MPRester
    from pymatgen.entries.compatibility import (
        MaterialsProject2020Compatibility, MaterialsProjectAqueousCompatibility,
    )
    from pymatgen.analysis.phase_diagram import PhaseDiagram

    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("set MP_API_KEY (free account at materialsproject.org)")
    out: list[CellReaction] = []
    with MPRester(key) as mpr:
        for m in elements:
            if m in ("H", "O"):
                continue
            entries = mpr.get_entries_in_chemsys([m, "O", "H"],
                                                 additional_criteria={"thermo_types": ["GGA_GGA+U"]})
            if not entries:
                continue
            aq = MaterialsProjectAqueousCompatibility(solid_compat=MaterialsProject2020Compatibility())
            entries = aq.process_entries(entries)
            pd = PhaseDiagram(entries)
            h2 = min((e for e in entries if e.composition.reduced_formula == "H2"),
                     key=lambda e: e.energy_per_atom, default=None)
            h2o = min((e for e in entries if e.composition.reduced_formula == "H2O"),
                      key=lambda e: e.energy_per_atom, default=None)
            if h2 is None or h2o is None:
                continue
            mu_h2 = h2.energy_per_atom * 2
            mu_h2o = h2o.energy_per_atom * 3
            solids = []
            for e in entries:
                c = e.composition
                if c[m] == 0 or c.reduced_formula in ("H2", "H2O", "O2"):
                    continue
                if pd.get_e_above_hull(e) > e_hull_max:
                    continue
                per_m = e.energy / c[m]
                o, h = c["O"] / c[m], c["H"] / c[m]
                solids.append((e, per_m, o, h))
            best: dict[str, tuple] = {}
            for t in solids:   # keep the lowest-energy polymorph per formula
                f = t[0].composition.reduced_formula
                if f not in best or t[1] < best[f][1]:
                    best[f] = t
            solids = list(best.values())
            for ea, ga, oa, ha in solids:
                for eb, gb, ob, hb in solids:
                    if ea is eb:
                        continue
                    y = oa - ob                       # H2O released per M
                    if y < -1e-9:
                        continue
                    x = (hb - ha + 2 * y) / 2         # H2 consumed per M
                    if x <= 1e-9:
                        continue
                    de = gb + y * mu_h2o - ga - x * mu_h2
                    n = 2 * x
                    v = -de / n
                    if v <= 0:
                        continue
                    sa = Species(ea.composition.reduced_formula, 0, "s", ea.composition.reduced_formula)
                    sb = Species(eb.composition.reduced_formula, 0, "s", eb.composition.reduced_formula)
                    ra = ea.composition.reduced_composition[m]
                    rb = eb.composition.reduced_composition[m]
                    reac = {sa: 1 / ra, Species("H2", 0, "g", "H2(g)"): x}
                    prod = {sb: 1 / rb}
                    if y > 1e-9:
                        prod[Species("H2O", 0, "l", "H2O(l)")] = y
                    theo = []
                    for e in (ea, eb):
                        theo.append(f"{e.entry_id} (e_hull {pd.get_e_above_hull(e):.3f} eV/atom)")
                    c = CellReaction(None, None, reac, prod, n, v, "aqueous (pH-free)",
                                     origin="materials-project",
                                     source="Materials Project: " + ", ".join(theo),
                                     solid_positive=True)
                    if y == 0 and oa == 0:
                        c.medium = "hydride"
                    out.append(c)
    return out
