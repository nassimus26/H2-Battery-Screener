"""Maintenance: does the cell keep its own inventory forever?

Hydrogen loss
  permeation through the vessel wall (Sieverts' law, flux = Phi(T) sqrt(p) / t),
  Phi(T) from measured Arrhenius fits for 316L steel in h-transport-materials
  (each with its DOI). The fits were measured at 473-1173 K, so values at room
  temperature are extrapolations; surface oxides lower real permeation further,
  so the result is an upper bound.
  seal and fitting leakage: helium leak-rate class from targets.toml, scaled
  with pressure.

Chemical closure (from the same potential table, by the Frost-diagram rule)
  over-reduction   H2 can reduce the discharge product further (E > E_H2):
                   inventory drifts out of the couple.
  disproportionation  a species of the couple is unstable to splitting into a
                   higher and a lower oxidation state (E(X/lower) > E(higher/X)).
  water            an oxidant above the O2/H2O potential makes O2; in a sealed cell
                   it recombines with H2 to water (energy loss, no material loss)
                   when a recombination catalyst is present.
A couple with none of these is closed: nothing has to be replaced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

from .constants import R
from .reactions import ELECTRON, CellReaction, HalfReaction, load_table

N_A = 6.02214076e23
EV = 1.602176634e-19
YEAR_S = 365.25 * 86400


@lru_cache(maxsize=1)
def _permeability_fits():
    import h_transport_materials as htm
    fits = []
    for p in htm.permeabilities.filter(material="316l_steel"):
        if str(p.isotope).lower() != "h":
            continue
        pre = p.pre_exp.to("particle / meter / pascal ** 0.5 / second").magnitude
        ea = p.act_energy.to("electron_volt / particle").magnitude
        fits.append((pre, ea, f"{p.author} {p.year} doi:{getattr(p, 'doi', '')}"))
    return fits


def permeability(t_k: float) -> tuple[float, float, list[str]]:
    """(low, high) H atoms / (m s Pa^0.5) over all fits, and their sources."""
    fits = _permeability_fits()
    vals = [pre * math.exp(-ea * EV / (1.380649e-23 * t_k)) for pre, ea, _ in fits]
    return min(vals), max(vals), [s for *_, s in fits]


def vessel_geometry(v_m3: float, p_pa: float, sigma_pa: float, aspect: float):
    """Cylinder of length = aspect x diameter: (area m2, wall thickness m)."""
    d = (4 * v_m3 / (math.pi * aspect)) ** (1 / 3)
    area = math.pi * d * d * aspect + 2 * math.pi * (d / 2) ** 2
    t = max(p_pa * d / 2 / sigma_pa, 1e-3)    # hoop stress; 1 mm minimum wall
    return area, t


@dataclass
class Maintenance:
    permeation_frac_per_year: tuple[float, float]
    seal_frac_per_year: tuple[float, float]
    years_to_allowed_loss: float
    makeup_usd_per_year: float
    drift: list[str] = field(default_factory=list)
    recombination: list[str] = field(default_factory=list)
    closed: bool = True
    sources: list[str] = field(default_factory=list)


def hydrogen_loss(h2_moles: float, v_m3: float, t_k: float, p_max: float, m: dict, sigma_pa: float):
    area, t = vessel_geometry(v_m3, p_max, sigma_pa, m["vessel_aspect"])
    lo, hi, src = permeability(t_k)
    to_mol_s = math.sqrt(p_max) * area / t / 2 / N_A        # H atoms -> H2 molecules
    perm = (lo * to_mol_s * YEAR_S / h2_moles, hi * to_mol_s * YEAR_S / h2_moles)
    # helium leak class mbar L/s -> mol/s at test conditions, x seals, x pressure ratio
    q = [x * 0.1 / (R * t_k) * m["seal_count"] * p_max / 1e5 for x in m["seal_leak_mbar_l_s"]]
    seal = (q[0] * YEAR_S / h2_moles, q[1] * YEAR_S / h2_moles)
    worst = perm[1] + seal[1]
    years = m["allowed_h2_loss"] / worst if worst > 0 else math.inf
    return perm, seal, years, src


def _species_key(sp):
    return (sp.formula, sp.charge)


def _signed(hr: HalfReaction) -> dict:
    d = {}
    for sp, c in hr.oxidant.items():
        d[sp] = d.get(sp, 0) + c
    for sp, c in hr.reductant.items():
        d[sp] = d.get(sp, 0) - c
    return d


def derived_half_reactions(table: list[HalfReaction]) -> list[HalfReaction]:
    """Half-reactions missing from the table, from free-energy additivity
    (Latimer-diagram arithmetic): A -> C (n2, E2) minus A -> B (n1, E1) gives
    B -> C with n = n2 - n1, E = (n2 E2 - n1 E1) / n."""
    from .reactions import reaction_balanced
    known = {hr.label for hr in table}
    out = []
    by_ox: dict = {}
    for hr in table:
        main = [sp for sp in hr.oxidant if sp is not ELECTRON and _species_key(sp) not in
                {("H", 1), ("OH", -1), ("H2O", 0)}]
        if len(main) == 1:
            by_ox.setdefault((_species_key(main[0]), hr.medium), []).append((main[0], hr))
    for (_, medium), group in by_ox.items():
        for sp1, r1 in group:
            for sp2, r2 in group:
                if r1 is r2:
                    continue
                k = r2.oxidant[sp2] / r1.oxidant[sp1]
                n = r2.n - k * r1.n
                if n <= 1e-6:
                    continue
                d = _signed(r2)
                for sp, c in _signed(r1).items():
                    d[sp] = d.get(sp, 0) - k * c
                ox = {sp: c for sp, c in d.items() if c > 1e-9}
                red = {sp: -c for sp, c in d.items() if c < -1e-9}
                if not red or not ox or not reaction_balanced(ox, red):
                    continue
                e = (r2.n * r2.e0 - k * r1.n * r1.e0) / n
                label = " + ".join(f"{c:g} {sp.label}" for sp, c in ox.items()) + " -> " + \
                        " + ".join(f"{c:g} {sp.label}" for sp, c in red.items())
                if label in known:
                    continue
                known.add(label)
                out.append(HalfReaction(ox, red, e, n, label,
                                        f"derived: ({r2.label}) - ({r1.label})", medium))
    return out


@lru_cache(maxsize=1)
def _table():
    t = load_table()
    return t + derived_half_reactions(t)


def chemical_closure(cell: CellReaction, e_neg: float, e_o2: float | None) -> tuple[list[str], list[str]]:
    """(drift paths, recombination paths) for the couple's species."""
    table: list[HalfReaction] = _table()
    skip = {("H", 1), ("OH", -1), ("H2O", 0), ("H2", 0)}
    charged = {_species_key(s) for s in cell.reactants} - skip
    discharged = {_species_key(s) for s in cell.products} - skip
    own = charged | discharged
    medium = cell.medium
    drift, recomb = [], []

    def ok(hr):
        return hr.medium in ("any", medium) or medium.startswith("aqueous")

    as_ox: dict = {}
    as_red: dict = {}
    for hr in table:
        if not ok(hr):
            continue
        for sp in hr.oxidant:
            if sp is not ELECTRON and _species_key(sp) in own:
                as_ox.setdefault(_species_key(sp), []).append(hr)
        for sp in hr.reductant:
            if _species_key(sp) in own:
                as_red.setdefault(_species_key(sp), []).append(hr)

    for key in discharged:
        for hr in as_ox.get(key, []):
            prods = {_species_key(s) for s in hr.reductant} - skip
            if prods and not prods <= own and hr.e0 - e_neg > 0.05:
                drift.append(f"H2 can reduce {key[0]} further ({hr.label}, {hr.e0 - e_neg:+.2f} V)")
    for key in own:
        lowers = [h for h in as_ox.get(key, []) if ({_species_key(s) for s in h.reductant} - skip)]
        highers = [h for h in as_red.get(key, []) if ({_species_key(s) for s in h.oxidant if s is not ELECTRON} - skip)]
        for lo in lowers:
            for hi in highers:
                if lo.e0 > hi.e0 + 0.05:
                    out = ({_species_key(s) for s in lo.reductant} | {_species_key(s) for s in hi.oxidant if s is not ELECTRON}) - skip
                    if not out <= own:
                        drift.append(f"{key[0]} disproportionates (E {lo.e0:.2f} > {hi.e0:.2f} V)")
                        break
            else:
                continue
            break
    if e_o2 is not None and cell.e0 > e_o2:
        recomb.append("oxidant makes O2 from water; sealed cell recombines it with H2 (needs a recombination catalyst)")
    recomb.append("H2 crossing to the positive side reacts back to the discharge product (self-discharge, no loss)")
    return sorted(set(drift)), recomb
