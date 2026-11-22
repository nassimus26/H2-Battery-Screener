"""Molecular hydrogen couples: M + H2 -> MH2, computed with quantum chemistry.

For any molecule (known or new), every way of adding H2 to two heteroatoms of
its conjugated system (carbonyl O, imine N, azo N, ...) is generated; bond
orders of the hydrogenated form are re-derived from the connectivity
(RDKit DetermineBondOrders), so only closed-shell, valence-correct partners
survive. Each form is relaxed with GFN2-xTB in implicit water (ALPB), and

  E = -(E[MH2] - E[M] - E[H2]) / 2F + offset

The offset absorbs zero-point, thermal and solvation errors common to H2
addition. It is fixed by one measured couple from the potential table,
p-benzoquinone/hydroquinone 0.6992 V vs SHE. Single conformers and the
semi-empirical method leave an error of a few tenths of a volt; results carry
that band.

Cheap descriptors used alongside:
  solubility   ESOL (Delaney, J. Chem. Inf. Comput. Sci. 44 (2004) 1000)
  synthesis    SA score (Ertl & Schuffenhauer, J. Cheminf. 1 (2009) 8)
  Michael      unsubstituted C=C-C(=O) ring positions, where water and
               nucleophiles add to quinones (a known decay path; heuristic)
  tautomers    every tautomer of M and MH2 is computed; if a lower one exists,
               the reduced form leaves the couple (enol -> ketone) and the
               couple is marked not reversible
  aldehydes    hydrate in water; penalised (heuristic)
  novelty      not found in PubChem by InChIKey
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, RDConfig, rdDetermineBonds, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

from .constants import F
from . import synthesis
from .reactions import CellReaction, Species

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
REFERENCE_SMILES = "O=C1C=CC(=O)C=C1"
REFERENCE_E0 = 0.6992         # V vs SHE, potential table (p-benzoquinone)
VOLTAGE_BAND = 0.3            # V, stated uncertainty of the xTB voltages
EV_TO_V_2E = 0.5              # eV per 2 electrons -> V
MICHAEL = Chem.MolFromSmarts("[CH1;R]=[C;R]-[C;R](=O)")
ALDEHYDE = Chem.MolFromSmarts("[CX3H1](=O)")

# Instability alerts from the Brenk set (Brenk et al., ChemMedChem 3 (2008) 435;
# shipped with RDKit): only the ones about decomposition or reactivity in water.
# Alerts that describe redox-active cores we want (quinones, hydroquinones,
# polycyclic aromatics, Michael acceptors of quinones) are not used.
BRENK_UNSTABLE = {
    "Oxygen-nitrogen_single_bond", "het-C-het_not_in_ring", "imine_1", "imine_2", "peroxide", "hydrazine",
    "acyl_hydrazine", "N-nitroso", "N-halo", "Azido_group", "diazo_group", "acid_halide", "acyl_cyanide",
    "isocyanate", "ketene", "beta-keto/anhydride", "four_member_lactones", "Three-membered_heterocycle",
    "sulfur_oxygen_single_bond", "sulphur_nitrogen_single_bond", "disulphide", "cyanohydrins",
    "cyanate_/aminonitrile_/thiocyanate", "Carbo_cation/anion", "enamine", "oxime_1", "oxime_2",
    "hydroxamic_acid", "thioester", "triflate", "sulfonyl_cyanide", "N_oxide",
}
# Not in the Brenk list but impossible to store: cumulated double bonds
# (C=C=C), and double bonds at a bridgehead of two small fused rings.
EXTRA_UNSTABLE = [Chem.MolFromSmarts(x) for x in ("[#6]=[#6]=[#6]", "[#6]=[#6]=[#7,#8]")]


@lru_cache(maxsize=1)
def _brenk():
    from rdkit.Chem import FilterCatalog
    p = FilterCatalog.FilterCatalogParams()
    p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(p)


def strained(m) -> bool:
    """A 3- or 4-membered ring fused to another ring and holding a double bond
    in the ring (fused cyclobutenes, azetines, cyclobutadienes): far too
    strained to survive. A lone four-membered ring (squaric acid type) passes."""
    rings = [set(r) for r in m.GetRingInfo().AtomRings()]
    for r in rings:
        if len(r) > 4:
            continue
        fused = any(len(r & o) >= 2 for o in rings if o is not r)
        if not fused:
            continue
        for b in m.GetBonds():
            if b.GetBeginAtomIdx() in r and b.GetEndAtomIdx() in r and \
                    b.GetBondType() in (Chem.BondType.DOUBLE, Chem.BondType.AROMATIC):
                return True
    return False


def instability(smiles: str) -> list[str]:
    """Names of instability alerts a molecule triggers (empty = none found)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return ["unparseable"]
    out = sorted({e.GetDescription() for e in _brenk().GetMatches(m)} & BRENK_UNSTABLE)
    if any(q is not None and m.HasSubstructMatch(q) for q in EXTRA_UNSTABLE):
        out.append("cumulated double bonds")
    if strained(m):
        out.append("strained fused small rings")
    return out
# Functional groups that decompose in water or on their own (textbook reactivity;
# heuristic filter): peroxides, anhydrides, acyl cyanides/halides, isocyanates,
# azides and diazo groups, N-O-O / O-N=O chains, three-membered rings.
REACTIVE = [Chem.MolFromSmarts(x) for x in (
    "[OX2][OX2]", "C(=O)OC(=O)", "C(=O)C#N", "C(=O)[F,Cl,Br,I]", "N=C=O", "N=[N+]=[N-]", "[N]=[N]=*",
    "[#7][OX2][OX2]", "O[NX2]=O", "[r3]", "C(=O)O[NX3]", "[NX3][OX2][CX3]=O")]


@dataclass
class MolResult:
    smiles: str
    reduced: str = ""
    raw_ev: float = math.nan
    e0: float = math.nan
    mw: float = 0.0
    log_s: float = 0.0
    solubility_mol_l: float = 0.0
    sa: float = 10.0
    michael_sites: int = 0
    in_pubchem: bool | None = None
    wh_kg: float = 0.0
    wh_l: float = 0.0
    objective: float = 0.0
    notes: list[str] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)   # calibrated E of each successive H2 uptake, V vs SHE
    forms: list[str] = field(default_factory=list)     # molecule after each uptake (forms[0] = reduced)
    best_k: int = 1                                     # H2 per molecule used by the best single-molecule cell


def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def reduced_forms(smiles: str, limit: int = 12) -> list[str]:
    """All closed-shell MH2 partners formed by adding H to two heteroatoms."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return []
    Chem.Kekulize(m, clearAromaticFlags=True)
    sites = sorted({a.GetIdx() for a in m.GetAtoms()
                    if a.GetSymbol() in ("O", "N", "S") and a.GetFormalCharge() == 0
                    and any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())})
    out: set[str] = set()
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            h = Chem.RWMol(Chem.AddHs(m))
            for idx in (a, b):
                n = h.AddAtom(Chem.Atom(1))
                h.AddBond(idx, n, Chem.BondType.SINGLE)
            for bond in h.GetBonds():
                bond.SetBondType(Chem.BondType.SINGLE)
            for at in h.GetAtoms():
                at.SetNoImplicit(True)
                at.SetNumExplicitHs(0)
                at.SetIsAromatic(False)
            try:
                mol = h.GetMol()
                mol.AddConformer(Chem.Conformer(mol.GetNumAtoms()), assignId=True)
                rdDetermineBonds.DetermineBondOrders(mol, charge=0)
                Chem.SanitizeMol(mol)
                if Descriptors.NumRadicalElectrons(mol) or any(x.GetFormalCharge() for x in mol.GetAtoms()):
                    continue
                s = Chem.MolToSmiles(Chem.RemoveHs(mol))
                if s != Chem.MolToSmiles(m):
                    out.add(s)
            except Exception:
                continue
            if len(out) >= limit:
                return sorted(out)
    return sorted(out)


def descriptors(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    rb = Lipinski.NumRotatableBonds(m)
    heavy = m.GetNumHeavyAtoms()
    ap = sum(a.GetIsAromatic() for a in m.GetAtoms()) / heavy if heavy else 0
    log_s = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    return {"mw": mw, "log_s": log_s, "sol": 10 ** log_s, "sa": sascorer.calculateScore(m),
            "usd_kg": synthesis.estimate(smiles).usd_kg,
            "michael": len(m.GetSubstructMatches(MICHAEL)), "heavy": heavy,
            "aldehyde": len(m.GetSubstructMatches(ALDEHYDE)),
            "reactive": sum(len(m.GetSubstructMatches(q)) for q in REACTIVE if q is not None),
            "unstable": instability(smiles)}


# ------------------------------------------------------------------ xTB energies

def _energy_uncached(smiles: str) -> float:
    from ase import Atoms
    from ase.optimize import BFGS
    from tblite.ase import TBLite

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) != 0:
        raise RuntimeError("embedding failed")
    AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    pos = m.GetConformer().GetPositions()
    atoms = Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=pos)
    atoms.calc = TBLite(method="GFN2-xTB", solvation=("alpb", "water"), verbosity=0)
    BFGS(atoms, logfile=None).run(fmax=0.05, steps=300)
    return float(atoms.get_potential_energy())


def energy(smiles: str) -> float:
    """eV, GFN2-xTB/ALPB(water), cached on disk by canonical SMILES."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "xtb_energies.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = canonical(smiles) if smiles != "[H][H]" else smiles
    if key not in cache:
        cache = json.loads(path.read_text()) if path.exists() else {}
        cache[key] = _energy_uncached(key)
        path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return cache[key]


@lru_cache(maxsize=1)
def offset() -> float:
    return REFERENCE_E0 - raw_voltage(REFERENCE_SMILES)[0]


TAUTOMER_EV = 0.10   # eV: a tautomer this much lower takes over the population


def tautomers(smiles: str, limit: int = 8) -> list[str]:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    en = rdMolStandardize.TautomerEnumerator()
    en.SetMaxTautomers(limit)
    out = {Chem.MolToSmiles(t) for t in en.Enumerate(Chem.MolFromSmiles(smiles))}
    out.add(canonical(smiles))
    return sorted(out)[:limit]


def lowest_tautomer(smiles: str) -> tuple[str, float]:
    """(tautomer, energy eV) with the lowest xTB energy."""
    best = (smiles, energy(smiles))
    for t in tautomers(smiles):
        try:
            e = energy(t)
        except Exception:
            continue
        if e < best[1] - 1e-6:
            best = (t, e)
    return best


def raw_voltage(smiles: str) -> tuple[float, str, list[str]]:
    """Most positive uncalibrated voltage over all MH2 partners, the partner,
    and notes. The H2 adduct that forms first is the direct partner; if a
    tautomer of it (or of M) is lower by more than TAUTOMER_EV, the population
    leaves the redox couple: the voltage is then computed between the most
    stable forms and the couple is marked as drifting."""
    notes = []
    m_low, e_m_low = lowest_tautomer(smiles)
    e_m = energy(smiles)
    if e_m - e_m_low > TAUTOMER_EV:
        notes.append(f"oxidised form tautomerises to {m_low} ({e_m - e_m_low:.2f} eV lower)")
    e_h2 = energy("[H][H]")
    best = (-math.inf, "", [])
    for r in reduced_forms(smiles):
        try:
            e_r = energy(r)
            r_low, e_low = lowest_tautomer(r)
        except Exception:
            continue
        v = -(e_r - e_m - e_h2) * EV_TO_V_2E
        n = []
        if e_r - e_low > TAUTOMER_EV:
            n.append(f"reduced form tautomerises to {r_low} ({e_r - e_low:.2f} eV lower): not reversible")
            v = -(e_low - e_m_low - e_h2) * EV_TO_V_2E
        if v > best[0]:
            best = (v, r, n)
    return best[0], best[1], notes + best[2]


def in_pubchem(smiles: str) -> bool | None:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "pubchem_known.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    if key in cache:
        return cache[key]
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{key}/cids/JSON"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "h2-screener/0.1"}), timeout=15) as r:
            found = bool(json.load(r).get("IdentifierList", {}).get("CID"))
    except urllib.error.HTTPError as e:
        found = False if e.code == 404 else None
    except Exception:
        found = None
    if found is not None:
        cache[key] = found
        path.write_text(json.dumps(cache, indent=0, sort_keys=True))
    return found


# ------------------------------------------------------------------ objective

def objective(e0: float, d: dict, sol_cap: float = 4.0, v_max: float = 1.229 + 0.4) -> float:
    """Energy per litre of electrolyte (Wh/L) with penalties: solubility capped,
    voltage must sit between 0.2 V and the O2-evolution limit, molecules that
    are expensive to make (synthesis.price_penalty) and Michael-addition sites
    lower it."""
    if not math.isfinite(e0) or e0 < 0.2 or e0 > v_max:
        return 0.0
    whl = e0 * 2 * F * min(d["sol"], sol_cap) / 3600
    if "usd_kg" in d:
        whl *= synthesis.price_penalty(d["usd_kg"])
    elif d["sa"] > 6:
        whl *= 0.5
    if d["michael"]:
        whl *= 0.5 ** min(d["michael"], 2)
    return whl


MAX_H2 = 3


def further_steps(form: str, max_more: int) -> tuple[list[float], list[str], list[str]]:
    """Add H2 again and again to `form`, always taking the most stable
    reversible product. Returns (calibrated step potentials, forms, notes)."""
    e_h2 = energy("[H][H]")
    cur, e_cur = form, energy(form)
    steps, forms, notes = [], [], []
    for _ in range(max_more):
        best = None
        for c in reduced_forms(cur, limit=8):
            try:
                e = energy(c)
                low, e_low = lowest_tautomer(c)
            except Exception:
                continue
            if e - e_low > TAUTOMER_EV:
                continue                      # this product would leave the couple
            if best is None or e < best[1]:
                best = (c, e)
        if best is None:
            break
        steps.append(-(best[1] - e_cur - e_h2) * EV_TO_V_2E + offset())
        forms.append(best[0])
        cur, e_cur = best
    if len(steps) < max_more:
        notes.append(f"takes up {len(steps) + 1} H2 at most (no further reversible hydrogenated form)")
    return steps, forms, notes


def multi_objective(steps: list[float], d: dict, sol_cap: float = 4.0, v_max: float = 1.229 + 0.4):
    """Best (k, Wh/L, mean E) over using the first k uptakes, all inside 0.2 V..v_max."""
    best = (1, 0.0, steps[0] if steps else math.nan)
    for k in range(1, len(steps) + 1):
        used = steps[:k]
        if min(used) < 0.2 or max(used) > v_max:
            break
        e = sum(used) / k
        whl = e * 2 * k * F * min(d["sol"], sol_cap) / 3600
        if whl > best[1]:
            best = (k, whl, e)
    return best


def evaluate(smiles: str, check_pubchem: bool = False, max_h2: int = MAX_H2) -> MolResult:
    s = canonical(smiles)
    r = MolResult(s or smiles)
    if s is None:
        r.notes.append("invalid SMILES")
        return r
    d = descriptors(s)
    r.mw, r.log_s, r.solubility_mol_l, r.sa, r.michael_sites = d["mw"], d["log_s"], d["sol"], d["sa"], d["michael"]
    try:
        raw, red, tnotes = raw_voltage(s)
        r.notes += tnotes
    except Exception as e:
        r.notes.append(f"xTB failed: {e}")
        return r
    if not red:
        r.notes.append("no closed-shell hydrogenated partner")
        return r
    r.raw_ev, r.reduced = raw, red
    r.e0 = raw + offset()
    r.steps, r.forms = [r.e0], [red]
    if max_h2 > 1 and not any("not reversible" in n for n in r.notes):
        try:
            more, forms, n2 = further_steps(red, max_h2 - 1)
            r.steps += more
            r.forms += forms
            r.notes += n2
        except Exception as e:
            r.notes.append(f"xTB failed on further uptake: {e}")
    k, whl, e_mean = multi_objective(r.steps, d)
    r.best_k = k
    r.wh_l = whl if whl > 0 else r.e0 * 2 * F * min(r.solubility_mol_l, 4.0) / 3600
    e_use = e_mean if whl > 0 else r.e0
    r.wh_kg = e_use * 2 * k * F / ((r.mw + 2.016 * k) / 1000) / 3600
    r.objective = objective(r.e0, d) * (whl / max(objective(r.e0, d), 1e-9) if whl > 0 and objective(r.e0, d) > 0 else 1.0)
    if any("not reversible" in n for n in r.notes):
        r.objective *= 0.1
    if any("oxidised form tautomerises" in n for n in r.notes):
        r.objective *= 0.3
    if d["aldehyde"]:
        r.notes.append("aldehyde group: hydrates in water (gem-diol)")
        r.objective *= 0.5
    if d["reactive"]:
        r.notes.append("reactive group (peroxide, anhydride, acyl cyanide, azide, ...): not stable in water")
        r.objective = 0.0
    if d["unstable"]:
        r.notes.append("reactive group: unstable (" + ", ".join(d["unstable"]) + ")")
        r.objective = 0.0
    if check_pubchem:
        r.in_pubchem = in_pubchem(s)
    return r


def as_cell(r: MolResult) -> CellReaction:
    """The molecule as the positive electrode of an acid H2 cell, so the
    system, safety, maintenance and design models apply unchanged."""
    k = max(1, min(r.best_k, len(r.forms) or 1))
    red_smiles = r.forms[k - 1] if r.forms else r.reduced
    e_mean = sum(r.steps[:k]) / k if r.steps else r.e0
    f_ox = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(r.smiles))
    f_red = rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(red_smiles))
    ox = Species(f_ox, 0, "aq", r.smiles)
    red = Species(f_red, 0, "aq", red_smiles)
    c = CellReaction(None, None, {ox: 1, Species("H2", 0, "g", "H2(g)"): k}, {red: 1}, 2 * k, e_mean, "acid",
                     origin="xtb-molecule",
                     source=f"GFN2-xTB/ALPB(water), offset from p-benzoquinone 0.6992 V; +/-{VOLTAGE_BAND} V")
    c.solubility_mol_l = r.solubility_mol_l
    return c
