"""Two-molecule cells: hydrogen moves between two molecules, never as gas.

  negative (charged = reduced form)   N-H2 -> N + 2 H+ + 2 e-      E_neg
  positive (charged = oxidised form)  P + 2 H+ + 2 e- -> P-H2      E_pos
  cell                                N-H2 + P -> N + P-H2         V = E_pos - E_neg

Both potentials come from the same calibrated xTB scale (molecules.py), so the
common offset cancels in V. Both sides are aqueous liquids: no free hydrogen,
no pressure vessel, no platinum.

Limits (aqueous acid, carbon electrodes):
  E_neg >= -her_margin   below that the negative side makes hydrogen gas
  E_pos <= 1.229 + oer_margin   above that the positive side makes oxygen
Energy per litre of electrolyte, both tanks counted:
  Wh/L = V * 2F / 3600 / (1/c_neg + 1/c_pos)    (c in mol/L, capped)
"""

from __future__ import annotations

import math
from functools import lru_cache
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from . import molecules as M
from . import synthesis
from .constants import F
from .reactions import CellReaction, Species

HER_MARGIN = 0.3        # V below the hydrogen potential tolerated on carbon (assumption)
OER_MARGIN = 0.4        # V above the oxygen potential tolerated (assumption)
SOL_CAP = 4.0
HEADROOM = 0.15         # V each electrode needs below its water limit to be charged


CHEMICAL_SHARE = 0.5    # chemicals may cost this share of a Li-ion pack per kWh
EXCESS = 0.2            # molecules beyond the stoichiometric need


def kg_per_kwh(mw: float, voltage: float, n_e: int = 2) -> float:
    """kg of a molecule that moves n_e electrons, needed per kWh at this cell voltage (exact)."""
    mol = 1000 * 3600 / (n_e * F * voltage)
    return mol * mw / 1000 * (1 + EXCESS)


def element_floor_usd_kg(smiles: str) -> float | None:
    """Raw-element cost floor (pymatgen element prices): no synthesis can be cheaper."""
    from . import materials
    try:
        return materials.cost_per_kg(rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles)))
    except Exception:
        return None


@dataclass
class Pair:
    neg: M.MolResult
    pos: M.MolResult
    voltage: float
    wh_l: float
    wh_kg: float
    score: float
    kn: int = 1                 # H2 released per negative molecule
    kp: int = 1                 # H2 taken up per positive molecule
    neg_range: tuple = (0.0, 0.0)   # min/max step potential used on each side
    pos_range: tuple = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)


def price_ceiling_usd_kg(pr: Pair, li_usd_kwh: float) -> float:
    """Highest price per kg both molecules may cost so the chemicals stay
    below CHEMICAL_SHARE of a Li-ion pack per kWh."""
    kg = kg_per_kwh(pr.neg.mw + 2.016 * len(pr.neg.steps or [1]), pr.voltage, 2 * pr.kn) + \
        kg_per_kwh(pr.pos.mw, pr.voltage, 2 * pr.kp)
    return CHEMICAL_SHARE * li_usd_kwh / kg if kg > 0 else 0.0


def usable(r: M.MolResult) -> bool:
    return (math.isfinite(r.e0) and bool(r.reduced)
            and not any("not reversible" in n or "oxidised form tautomerises" in n or "reactive group" in n
                        for n in r.notes)
            and not _unstable(r.smiles))


@lru_cache(maxsize=20000)
def _unstable(smiles: str) -> bool:
    """Instability alerts, also for molecules computed before the alerts existed."""
    return bool(M.instability(smiles))


def _penalty(r: M.MolResult) -> float:
    f = synthesis.price_penalty(synthesis.price_from_difficulty(r.sa))   # expensive to make
    if r.michael_sites:
        f *= 0.5 ** min(r.michael_sites, 2)
    if any("aldehyde" in n for n in r.notes):
        f *= 0.5
    return f


def role_options(r: M.MolResult, role: str):
    """(k, steps used) for each number of H2 a molecule can move in a role.
    A positive uses its first k uptakes (highest potentials); a negative its
    last k (lowest), staying permanently loaded with the earlier ones."""
    st = r.steps or ([r.e0] if math.isfinite(r.e0) else [])
    n = len(st)
    for k in range(1, n + 1):
        yield k, (st[:k] if role == "pos" else st[n - k:])


def pair_value(neg_steps, neg_sol, neg_mw, pos_steps, pos_sol, pos_mw):
    """(voltage, Wh/L, Wh/kg of the two charged solutions); zero outside the
    water window or when the two voltage ranges overlap. Steps are lists of
    potentials (or one float for a single uptake)."""
    ns = neg_steps if isinstance(neg_steps, (list, tuple)) else [neg_steps]
    ps = pos_steps if isinstance(pos_steps, (list, tuple)) else [pos_steps]
    kn, kp = len(ns), len(ps)
    v = sum(ps) / kp - sum(ns) / kn
    if (v <= 0.1 or min(ns) < -HER_MARGIN + HEADROOM or max(ps) > 1.229 + OER_MARGIN - HEADROOM
            or min(ps) - max(ns) < 0.1):
        return v, 0.0, 0.0
    cn, cp = min(neg_sol, SOL_CAP), min(pos_sol, SOL_CAP)
    if cn <= 0 or cp <= 0:
        return v, 0.0, 0.0
    litres_per_mol_e = 1 / (2 * kn * cn) + 1 / (2 * kp * cp)
    wh_per_mol_e = v * F / 3600
    kg_per_mol_e = (neg_mw / (2 * kn) + pos_mw / (2 * kp)) / 1000 + litres_per_mol_e * 1.0
    return v, wh_per_mol_e / litres_per_mol_e, wh_per_mol_e / kg_per_mol_e


def make_pair(n: M.MolResult, p: M.MolResult, kn: int = 1, kp: int = 1) -> Pair:
    nst = dict(role_options(n, "neg"))[kn]
    pst = dict(role_options(p, "pos"))[kp]
    v, whl, whkg = pair_value(nst, n.solubility_mol_l, n.mw, pst, p.solubility_mol_l, p.mw)
    pr = Pair(n, p, v, whl, whkg, whl * _penalty(n) * _penalty(p), kn, kp,
              (min(nst), max(nst)), (min(pst), max(pst)))
    if n.smiles == p.smiles:
        pr.notes.append("same molecule on both sides (different hydrogen steps): crossover does no harm")
    else:
        pr.notes.append("crossover through the membrane fades capacity (rate unknown)")
    if kn > 1 or kp > 1:
        pr.notes.append(f"{2 * kn} e- per negative, {2 * kp} e- per positive molecule; voltage is the mean of the steps")
    return pr


def best_pairs(pool: list[M.MolResult], top: int = 50, max_per_molecule: int | None = None) -> list[Pair]:
    """Best pairs by score. With max_per_molecule, a molecule appears in at
    most that many pairs (a molecule paired with itself counts twice), so the
    list shows different combinations instead of one strong molecule paired
    with everything."""
    ok = [r for r in pool if usable(r)]
    best: dict[tuple, Pair] = {}
    for n in ok:
        for kn, _ in role_options(n, "neg"):
            for p in ok:
                for kp, _ in role_options(p, "pos"):
                    if n is p and kn + kp > len(n.steps or [1]):
                        continue            # one molecule cannot lend the same step to both sides
                    pr = make_pair(n, p, kn, kp)
                    if pr.score > 0:
                        key = (n.smiles, p.smiles)
                        if key not in best or pr.score > best[key].score:
                            best[key] = pr
    out = sorted(best.values(), key=lambda x: -x.score)
    if max_per_molecule is None:
        return out[:top]
    seen: dict[str, int] = {}
    keep = []
    for pr in out:
        if seen.get(pr.neg.smiles, 0) >= max_per_molecule or seen.get(pr.pos.smiles, 0) >= max_per_molecule \
                or (pr.neg.smiles == pr.pos.smiles and seen.get(pr.neg.smiles, 0) + 2 > max_per_molecule):
            continue
        for s in (pr.neg.smiles, pr.pos.smiles):
            seen[s] = seen.get(s, 0) + 1
        keep.append(pr)
        if len(keep) >= top:
            break
    return keep


def best_partner_score(e: float, sol: float, mw: float, pen: float, pool: list[M.MolResult]) -> float:
    """Best pair score a molecule (one uptake at e) could make with the pool,
    as negative or as positive. Used by the AI search."""
    best = 0.0
    for q in pool:
        if not usable(q):
            continue
        for k, st in role_options(q, "pos"):
            _, whl, _ = pair_value([e], sol, mw, st, q.solubility_mol_l, q.mw)
            best = max(best, whl * pen * _penalty(q))
        for k, st in role_options(q, "neg"):
            _, whl, _ = pair_value(st, q.solubility_mol_l, q.mw, [e], sol, mw)
            best = max(best, whl * pen * _penalty(q))
    return best


def as_cell(pr: Pair) -> CellReaction:
    """The pair as a cell, so the performance and cost models apply unchanged.
    Per reaction: kp negative molecules give up kn H2 each, kn positive
    molecules take kp H2 each (2 kn kp electrons)."""
    f = lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s))
    nf = pr.neg.forms or [pr.neg.reduced]
    n_all = len(nf)
    n_charged = nf[n_all - 1]
    n_dis = nf[n_all - pr.kn - 1] if n_all - pr.kn - 1 >= 0 else pr.neg.smiles
    pf = pr.pos.forms or [pr.pos.reduced]
    p_dis = pf[pr.kp - 1]
    sp = lambda s, tag: Species(f(s), 0, "aq", s + tag)
    c = CellReaction(None, None, {sp(n_charged, " (negative)"): pr.kp, sp(pr.pos.smiles, " (positive)"): pr.kn},
                     {sp(n_dis, " (negative)"): pr.kp, sp(p_dis, " (positive)"): pr.kn},
                     2 * pr.kn * pr.kp, pr.voltage, "acid", origin="two-molecule",
                     source=f"GFN2-xTB/ALPB(water) for both molecules; voltage difference (offset cancels), +/-{M.VOLTAGE_BAND} V")
    cn, cp = min(pr.neg.solubility_mol_l, SOL_CAP), min(pr.pos.solubility_mol_l, SOL_CAP)
    c.solubility_mol_l = (pr.kp + pr.kn) / (pr.kp / cn + pr.kn / cp)
    return c
