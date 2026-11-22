"""Sealed two-gas batteries: every small gas molecule, generated and screened.

Generation
  Every closed-shell neutral molecule with up to MAX_HEAVY heavy atoms of
  C, N, O, F, S, Cl (single, double, triple bonds, small rings), enumerated
  exhaustively: the space of molecules that are gases at room temperature is
  small enough to compute in full, so nothing is guessed or sampled.

Thermodynamics (Gibbs energy of formation at 25 C) - no measured value of any molecule is used
  best       CCSD(T)/cc-pVTZ (small molecules) or wB97X-V/def2-TZVP DFT, on GFN2-xTB geometries
             of several conformers, with harmonic thermal corrections (abinitio.py). Element
             references H, O, N, F, Cl from the computed H2, O2, N2, F2, Cl2; carbon and sulfur
             (solids, not computable as molecules) fitted on the training half of the known gases.
  fast       GFN2-xTB energy of one conformer, with atom equivalents fitted on the training half
             of the known gases; used until the accurate calculation of a molecule is done, and by
             the AI search.
  test       the other half of the known gases measures the error of each level (validation).
  liquids    a molecule boiling above 25 C: the gas value minus a Trouton estimate of the
             evaporation energy; water from its own computed solvation energy.
  USE_MEASURED = True restores the old behaviour (measured tables for known molecules).

Boiling point
  the Joback group-contribution estimate for every molecule
  (Joback & Reid, Chem. Eng. Commun. 57 (1987) 233), +/- ~25 K.

Couples and cells (acid electrolyte, protons cross the membrane)
  couple    kA A + a H2 -> kB B + w H2O(l)      same C, N, S, F, Cl on both sides
            E = -dG / (2 a F)  vs the standard hydrogen electrode
  cell      positive couple (charged = oxidised form A_p, a gas)
            against a negative couple (charged = reduced form B_n, a gas;
            hydrogen itself is the simplest), V = E_p - E_n
  ranking   energy per mole of stored gas (n F V / moles of gas kept in tanks):
            a tank wall weighs the same per mole of gas at any pressure, so
            this number decides how light a sealed gas battery can be.
            The best cells are then sized in full (options.gas_configuration).

What cannot be computed: whether a reaction runs backwards cleanly in a cell.
Each couple carries the experimental evidence that exists (EVIDENCE).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from .constants import F

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
GAS_ENERGIES = CACHE / "xtb_gas_energies.json"
SEARCHED = CACHE / "gas_search.json"      # molecules proposed by the AI search (gas_search.py)

MAX_HEAVY = 4
ELEMENTS = ("C", "N", "O", "F", "S", "Cl")
T25 = 298.15
TROUTON = 88.0              # J/mol/K, evaporation entropy at the boiling point (Trouton's rule)
EV_KJ = 96.485              # kJ/mol per eV
G_WATER_L = -237.1          # kJ/mol, liquid water
XTB_CUTOFF_K = 400.0        # molecules boiling above this are not computed (cannot be gases or useful products)
MIN_VOLTAGE = 0.3
MAX_CELLS = 400_000          # the lightest cells kept by the screen (the rest are far from useful)
MAX_K = 8                   # molecules of one species per couple (C8 fuel -> 8 CO2)
MAX_E = 60                  # electrons per couple (C8H18 -> 8 CO2 moves 50)
CHARGE_MARGIN_V = 0.1       # overpotential a cell needs above its voltage to charge at all
TANK_KG_PER_MOL_EST = 0.03  # kg of carbon-fibre wall per mole of gas (Toyota Mirai tank); only to pre-rank

SRC = "NIST-JANAF / CRC Handbook, dfG at 298 K"
SRC_HS = "computed from the measured dfH and S (NIST)"

# smiles: (name, dfG gas kJ/mol, dfG liquid kJ/mol or None, boiling point K, source)
MEASURED = {
    "[H][H]": ("hydrogen", 0.0, None, 20.3, SRC),
    "O=O": ("oxygen", 0.0, None, 90.2, SRC),
    "N#N": ("nitrogen", 0.0, None, 77.4, SRC),
    "FF": ("fluorine", 0.0, None, 85.0, SRC),
    "ClCl": ("chlorine", 0.0, None, 239.1, SRC),
    "O": ("water", -228.6, -237.1, 373.1, SRC),
    "OO": ("hydrogen peroxide", -105.6, -120.4, 423.4, SRC),
    "[C-]#[O+]": ("carbon monoxide", -137.2, None, 81.6, SRC),
    "O=C=O": ("carbon dioxide", -394.4, None, 194.7, SRC),
    "C": ("methane", -50.5, None, 111.7, SRC),
    "CC": ("ethane", -32.0, None, 184.6, SRC),
    "C=C": ("ethylene", 68.4, None, 169.4, SRC),
    "C#C": ("acetylene", 209.2, None, 189.0, SRC),
    "CCC": ("propane", -23.4, None, 231.1, SRC),
    "C=CC": ("propene", 62.8, None, 225.5, SRC),
    "C1CC1": ("cyclopropane", 104.5, None, 240.3, SRC),
    "CCCC": ("butane", -17.0, None, 272.7, SRC),
    "CC(C)C": ("isobutane", -20.8, None, 261.4, SRC),
    "CO": ("methanol", -162.3, -166.6, 337.8, SRC),
    "C=O": ("formaldehyde", -102.5, None, 254.1, SRC),
    "O=CO": ("formic acid", -351.0, -361.4, 374.0, SRC),
    "CCO": ("ethanol", -168.5, -174.8, 351.4, SRC),
    "COC": ("dimethyl ether", -112.6, None, 248.3, SRC),
    "CC=O": ("acetaldehyde", -133.0, None, 293.3, SRC_HS),
    "CC(=O)O": ("acetic acid", -374.5, -389.9, 391.0, SRC),
    "C1CO1": ("ethylene oxide", -13.1, None, 283.6, SRC_HS),
    "N": ("ammonia", -16.4, None, 239.8, SRC),
    "NN": ("hydrazine", 159.4, 149.3, 386.7, SRC),
    "C#N": ("hydrogen cyanide", 124.7, None, 299.0, SRC),
    "CN": ("methylamine", 32.1, None, 266.8, SRC_HS),
    "[N]=O": ("nitric oxide", 86.6, None, 121.4, SRC),
    "[O][N+]=O": ("nitrogen dioxide", 51.3, None, 294.3, SRC),
    "[N-]=[N+]=O": ("nitrous oxide", 103.7, None, 184.7, SRC),
    "S": ("hydrogen sulfide", -33.4, None, 212.8, SRC),
    "O=S=O": ("sulfur dioxide", -300.1, None, 263.1, SRC),
    "O=S(=O)=O": ("sulfur trioxide", -371.1, None, 318.0, SRC),
    "O=C=S": ("carbonyl sulfide", -169.2, None, 223.0, SRC),
    "S=C=S": ("carbon disulfide", 67.1, 65.3, 319.4, SRC),
    "Cl": ("hydrogen chloride", -95.3, None, 188.1, SRC),
    "F": ("hydrogen fluoride", -273.2, None, 292.7, SRC),
    "FOF": ("oxygen difluoride", 42.5, None, 128.0, SRC),
    "FN(F)F": ("nitrogen trifluoride", -90.6, None, 144.0, SRC_HS),
    "FC(F)(F)F": ("tetrafluoromethane", -888.6, None, 145.0, SRC_HS),
    "CCl": ("chloromethane", -58.4, None, 249.0, SRC_HS),
    "O=C(Cl)Cl": ("phosgene", -204.9, None, 281.0, SRC),
}
RADICALS = {"O=O", "[N]=O", "[O][N+]=O"}      # not given to xTB (open shell), table only
TOXIC = {"[C-]#[O+]", "C#N", "ClCl", "FF", "[N]=O", "[O][N+]=O", "S", "O=S=O", "N", "O=C(Cl)Cl", "F", "Cl",
         "FOF", "NN", "S=C=S", "O=C=S", "C=O", "C1CO1", "CN", "O=S(=O)=O", "FN(F)F"}

# Experimental evidence that a couple runs both ways in an electrochemical cell.
# key: frozenset of the two canonical SMILES of the couple (oxidised, reduced)
EVIDENCE = {
    frozenset({"O=O", "O"}): ("demonstrated", "unitized regenerative fuel cells (PEM, Pt/Ir catalysts; NASA, "
                                              "Mitlitsky et al. 1998); the oxygen electrode is slow"),
    frozenset({"ClCl", "Cl"}): ("demonstrated", "hydrogen-chlorine flow batteries (Huskinson et al., Energy "
                                                "Environ. Sci. 5 (2012) 8690)"),
    frozenset({"FF", "F"}): ("one direction", "fluorine is made by electrolysis of HF in molten KF-HF, not in water; "
                                              "extremely toxic and corrosive"),
    frozenset({"O=C=O", "O=CO"}): ("both directions separately", "CO2 electroreduction to formate and formic-acid "
                                                                "fuel cells exist, never as one efficient "
                                                                "rechargeable cell"),
    frozenset({"O=C=O", "[C-]#[O+]"}): ("one direction", "CO2 electrolysis to CO works (silver, solid oxide); "
                                                         "CO poisons platinum fuel-cell electrodes"),
    frozenset({"O=C=O", "CO"}): ("one direction", "direct methanol fuel cells exist; CO2 to methanol in a cell is "
                                                  "slow and unselective"),
    frozenset({"N#N", "N"}): ("one direction", "ammonia fuel cells exist (slow); making ammonia from N2 in water "
                                               "has almost no yield"),
    frozenset({"O=C=O", "C"}): ("one direction", "methane can be oxidised at high temperature (solid oxide); "
                                                 "CO2 to methane in water is unselective"),
}



def _canon(smiles: str) -> str:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else smiles


MEASURED = {_canon(k): v for k, v in MEASURED.items()}
RADICALS = {_canon(k) for k in RADICALS}
TOXIC = {_canon(k) for k in TOXIC}
EVIDENCE = {frozenset(_canon(x) for x in k): v for k, v in EVIDENCE.items()}


# conventional formulas (RDKit writes HCl as ClH, NH3 as H3N)
FORMULA = {"Cl": "HCl", "N": "NH3", "F": "HF", "S": "H2S", "O": "H2O", "[H][H]": "H2", "ClCl": "Cl2", "FF": "F2",
           "[C-]#[O+]": "CO", "O=O": "O2", "N#N": "N2", "OO": "H2O2", "NN": "N2H4", "C#N": "HCN",
           "[N]=O": "NO", "[O][N+]=O": "NO2", "[N-]=[N+]=O": "N2O"}
FORMULA = {_canon(k): v for k, v in FORMULA.items()}


def formula_of(smiles: str) -> str:
    return FORMULA.get(smiles) or rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(smiles))


def measured_tb(formula: str) -> float | None:
    """Boiling point of a measured molecule with this formula (None if unknown or ambiguous)."""
    tbs = {v[3] for k, v in MEASURED.items() if formula_of(k) == formula}
    return tbs.pop() if len(tbs) == 1 else None


def evidence(ox: str, red: str) -> tuple[str, str]:
    if ox == "[H][H]" or red == "[H][H]":
        return "demonstrated", "hydrogen electrode (PEM fuel cells and electrolysers)"
    return EVIDENCE.get(frozenset({ox, red}), ("not demonstrated", "no experiment known to us; computed only"))


# ------------------------------------------------------------------ generation

def canonical(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


@lru_cache(maxsize=4)
def enumerate_molecules(max_heavy: int = MAX_HEAVY, elements: tuple[str, ...] = ELEMENTS) -> tuple[str, ...]:
    """Every neutral closed-shell molecule with up to max_heavy heavy atoms."""
    bonds = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE)
    frontier = {Chem.MolToSmiles(Chem.MolFromSmiles(e)) for e in elements}
    out = set(frontier)
    for _ in range(2, max_heavy + 1):
        new = set()
        for smi in frontier:
            base = Chem.MolFromSmiles(smi)
            for i in range(base.GetNumAtoms()):
                for e in elements:
                    for bo in bonds:
                        m = Chem.RWMol(base)
                        j = m.AddAtom(Chem.Atom(e))
                        m.AddBond(i, j, bo)
                        try:
                            Chem.SanitizeMol(m)
                        except Exception:
                            continue
                        new.add(Chem.MolToSmiles(m))
        rings = set()
        for smi in new:
            base = Chem.MolFromSmiles(smi)
            for i, j in itertools.combinations(range(base.GetNumAtoms()), 2):
                if base.GetBondBetweenAtoms(i, j) is None:
                    m = Chem.RWMol(base)
                    m.AddBond(i, j, Chem.BondType.SINGLE)
                    try:
                        Chem.SanitizeMol(m)
                    except Exception:
                        continue
                    rings.add(Chem.MolToSmiles(m))
        frontier = new | rings
        out |= frontier
    out = {s for s in out if Chem.MolFromSmiles(s) is not None and not _radical(s) and _real_sulfur(s)}
    return tuple(sorted(out))


def _radical(smiles: str) -> bool:
    return any(a.GetNumRadicalElectrons() for a in Chem.MolFromSmiles(smiles).GetAtoms())


def _real_sulfur(smiles: str) -> bool:
    """Sulfur as it occurs: divalent (H2S, CS2...), or S(IV)/S(VI) whose extra
    bonds are double bonds to oxygen (SO2, SO3). No S triple bonds."""
    for a in Chem.MolFromSmiles(smiles).GetAtoms():
        if a.GetSymbol() != "S":
            continue
        bonds = [(b.GetBondTypeAsDouble(), b.GetOtherAtom(a).GetSymbol()) for b in a.GetBonds()]
        if any(o == 3 for o, _ in bonds):
            return False
        if a.GetTotalValence() > 2 and any(o == 2 and el != "O" for o, el in bonds):
            return False
    return True


def searched_species() -> list[str]:
    """Larger molecules proposed by the AI search (gas_search.py) whose energy was computed."""
    try:
        recs = json.loads(SEARCHED.read_text()).get("molecules", {})
    except Exception:
        return []
    return [s for s, r in recs.items() if r.get("g_gas_kj") is not None]


def all_species() -> list[str]:
    """Generated molecules, the measured ones (radicals and diatomics included)
    and the larger molecules found by the AI search."""
    return sorted(set(enumerate_molecules()) | {canonical(s) or s for s in MEASURED} | {"[H][H]"}
                  | set(searched_species()))


# ------------------------------------------------------------------ boiling point

# Joback & Reid (1987) boiling-point increments, K
_JOBACK = {
    "CH3": 23.58, "CH2": 22.88, "CH": 21.74, "C": 18.25, "=CH2": 18.18, "=CH": 24.96, "=C": 24.14, "=C=": 26.15,
    "#CH": 9.20, "#C": 27.38, "rCH2": 27.15, "rCH": 21.78, "rC": 21.32, "r=CH": 26.73, "r=C": 31.01,
    "F": -0.03, "Cl": 38.13, "OH": 92.88, "O": 22.42, "rO": 31.22, "C=O": 76.75, "rC=O": 94.97, "CHO": 72.24,
    "COOH": 169.09, "COO": 81.10, "=O": -10.50, "NH2": 73.23, "NH": 50.17, "rNH": 52.82, "N": 11.74,
    "-N=": 74.60, "r-N=": 57.55, "=NH": 83.08, "CN": 125.66, "SH": 63.56, "S": 68.78, "rS": 52.10,
}


def joback_tb(smiles: str) -> float:
    m = Chem.MolFromSmiles(smiles)
    tot, used = 0.0, set()
    for a in m.GetAtoms():
        i, el, h, ring = a.GetIdx(), a.GetSymbol(), a.GetTotalNumHs(), a.IsInRing()
        if i in used:
            continue
        orders = [b.GetBondTypeAsDouble() for b in a.GetBonds()]
        if el == "C":
            dbl_o = [b.GetOtherAtom(a) for b in a.GetBonds()
                     if b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(a).GetSymbol() == "O"]
            trip_n = [b.GetOtherAtom(a) for b in a.GetBonds()
                      if b.GetBondTypeAsDouble() == 3 and b.GetOtherAtom(a).GetSymbol() == "N"]
            if trip_n:
                tot += _JOBACK["CN"]
                used.add(trip_n[0].GetIdx())
            elif dbl_o:
                used.add(dbl_o[0].GetIdx())
                single_o = [n for n in a.GetNeighbors() if n.GetSymbol() == "O" and n.GetIdx() != dbl_o[0].GetIdx()]
                if h:
                    tot += _JOBACK["CHO"]
                elif single_o and single_o[0].GetTotalNumHs():
                    tot += _JOBACK["COOH"]
                    used.add(single_o[0].GetIdx())
                elif single_o:
                    tot += _JOBACK["COO"]
                    used.add(single_o[0].GetIdx())
                else:
                    tot += _JOBACK["rC=O" if ring else "C=O"]
            elif 3 in orders:
                tot += _JOBACK["#CH" if h else "#C"]
            elif orders.count(2) >= 2:
                tot += _JOBACK["=C="]
            elif 2 in orders:
                key = "=CH2" if h >= 2 else ("=CH" if h == 1 else "=C")
                tot += _JOBACK.get(("r" + key) if ring and key != "=CH2" else key, _JOBACK[key])
            else:
                key = {3: "CH3", 4: "CH3", 2: "CH2", 1: "CH", 0: "C"}[min(h, 4)]
                tot += _JOBACK.get("r" + key, _JOBACK[key]) if ring and key != "CH3" else _JOBACK[key]
        elif el == "N":
            if 2 in orders:
                tot += _JOBACK["=NH"] if h else _JOBACK["r-N=" if ring else "-N="]
            else:
                tot += {2: _JOBACK["NH2"], 1: _JOBACK["rNH" if ring else "NH"]}.get(h, _JOBACK["N"])
        elif el == "O":
            if 2 in orders:
                tot += _JOBACK["=O"]
            else:
                tot += _JOBACK["OH"] if h else _JOBACK["rO" if ring else "O"]
        elif el == "S":
            tot += _JOBACK["SH"] if h else _JOBACK["rS" if ring else "S"]
        elif el in ("F", "Cl"):
            tot += _JOBACK[el]
    return 198.0 + tot


def boiling_point(smiles: str) -> tuple[float, str]:
    """Estimated for every molecule, known or not (measured boiling points are only used to test)."""
    if smiles == "O":
        return 373.0, "water: liquid by its computed solvation energy (Joback is not valid for water)"
    if smiles == "[H][H]":
        return 20.0, "hydrogen: a gas (Joback is not valid for H2)"
    return joback_tb(smiles), "estimated (Joback)"


def is_train(smiles: str) -> bool:
    """Known molecules are split in two halves, by alphabetical order of SMILES: a few constants may be
    fitted on the training half; the test half only measures the error."""
    ks = sorted(MEASURED)
    return smiles in ks and ks.index(smiles) % 2 == 0


# ------------------------------------------------------------------ energies

def _xtb_gas_uncached(smiles: str) -> float:
    from ase import Atoms
    from ase.optimize import BFGS
    from rdkit.Chem import AllChem
    from tblite.ase import TBLite

    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if m.GetNumAtoms() == 1:
        pos = np.zeros((1, 3))
    else:
        if AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE) != 0:
            raise RuntimeError("embedding failed")
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=500)
        except Exception:
            pass
        pos = m.GetConformer().GetPositions()
    atoms = Atoms([a.GetSymbol() for a in m.GetAtoms()], positions=pos)
    atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
    if len(atoms) > 1:
        BFGS(atoms, logfile=None).run(fmax=0.05, steps=300)
    return float(atoms.get_potential_energy())


def load_energies() -> dict:
    try:
        return json.loads(GAS_ENERGIES.read_text())
    except Exception:
        return {}


def xtb_energy(smiles: str, cache: dict) -> float | None:
    """eV, GFN2-xTB in the gas phase; None if the calculation fails."""
    if smiles not in cache:
        try:
            cache[smiles] = _xtb_gas_uncached(smiles)
        except Exception:
            cache[smiles] = None
    return cache[smiles]


def to_compute() -> list[str]:
    """Molecules that need an xTB energy: closed shell, boiling below XTB_CUTOFF_K."""
    return [s for s in all_species() if s not in RADICALS and boiling_point(s)[0] < XTB_CUTOFF_K]


def compute_energies(progress=None, stop=None) -> dict:
    """Fill the gas-phase energy cache (resumable; saved every 20 molecules)."""
    CACHE.mkdir(exist_ok=True)
    cache = load_energies()
    todo = sorted((s for s in to_compute() if s not in cache), key=lambda s: s not in MEASURED)  # fit needs these first
    for k, s in enumerate(todo, 1):
        if stop is not None and stop():
            break
        xtb_energy(s, cache)
        if k % 20 == 0 or k == len(todo):
            GAS_ENERGIES.write_text(json.dumps(cache, indent=0, sort_keys=True))
        if progress:
            progress(k, len(todo))
    return cache


def counts(smiles: str) -> dict[str, int]:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    out: dict[str, int] = {}
    for a in m.GetAtoms():
        out[a.GetSymbol()] = out.get(a.GetSymbol(), 0) + 1
    return out


@lru_cache(maxsize=100_000)
def _counts_t(smiles: str) -> tuple:
    return tuple(sorted(counts(smiles).items()))


def _counts(smiles: str) -> dict[str, int]:
    return dict(_counts_t(smiles))


FIT_ELEMENTS = ("H", "C", "N", "O", "F", "S", "Cl")


@dataclass
class Fit:
    mu: dict                 # kJ/mol per atom
    rmse_loo: float          # kJ/mol, RMS error on the test half of the known molecules
    n: int


def fit_atom_equivalents(cache: dict) -> Fit | None:
    """dfG(gas) = E_xtb - sum n_el mu_el, fitted to the measured closed-shell molecules."""
    rows, y, test = [], [], []
    for s, (_, g, _, _, _) in MEASURED.items():
        e = cache.get(s)
        if s in RADICALS or e is None:
            continue
        c = counts(s)
        (rows if is_train(s) else test).append([c.get(el, 0) for el in FIT_ELEMENTS])
        if is_train(s):
            y.append(e * EV_KJ - g)
        else:
            test[-1] = (test[-1], e * EV_KJ - g)
    if len(rows) < len(FIT_ELEMENTS) + 3:
        return None
    a, b = np.array(rows, float), np.array(y)
    mu, *_ = np.linalg.lstsq(a, b, rcond=None)
    err = [np.array(x) @ mu - yy for x, yy in test]          # errors on the test half only
    return Fit(dict(zip(FIT_ELEMENTS, mu)), float(np.sqrt(np.mean(np.square(err)))) if err else 80.0, len(b))


def storable(smiles: str) -> bool:
    """False for a computed molecule that would not survive storage: the
    instability alerts of the liquid search (Brenk set, cumulated double
    bonds, groups that hydrolyse), strained small rings, charge-separated
    forms, or carbon with fewer than four bonds (carbenes)."""
    from . import molecules as M
    m = Chem.MolFromSmiles(smiles)
    if m is None or any(a.GetFormalCharge() for a in m.GetAtoms()):
        return False
    if any(a.GetSymbol() == "C" and a.GetTotalValence() != 4 for a in m.GetAtoms()):
        return False
    try:
        if M.instability(smiles) or M.strained(m):
            return False
    except Exception:
        return False
    ri = m.GetRingInfo()
    if any(len(r) == 3 and any(m.GetBondBetweenAtoms(r[i], r[(i + 1) % 3]).GetBondTypeAsDouble() > 1
                               for i in range(3)) for r in ri.AtomRings()):
        return False                                   # cyclopropene-type rings
    return True


@dataclass
class GasSpecies:
    smiles: str
    name: str
    formula: str
    mw: float               # g/mol
    tb_k: float
    tb_source: str
    g_kj: float             # dfG in its state at 25 C (gas or liquid), kJ/mol
    g_band_kj: float        # uncertainty of g_kj
    source: str
    toxic: bool
    notes: list[str] = field(default_factory=list)

    @property
    def gas(self) -> bool:
        return self.tb_k < T25


def g_gas_xtb(smiles: str, energy_ev: float, fit: Fit) -> float:
    """Gibbs energy of formation of the gas, kJ/mol, from an xTB energy and the atom equivalents."""
    return energy_ev * EV_KJ - sum(n * fit.mu.get(el, 0.0) for el, n in counts(smiles).items())


def computed_species(smiles: str, g_gas: float, band: float, searched: bool = False) -> GasSpecies:
    """A molecule without measured data, from its (computed or predicted) gas Gibbs energy."""
    tb, tb_src = boiling_point(smiles)
    m = Chem.MolFromSmiles(smiles)
    formula = formula_of(smiles)
    g = g_gas - (TROUTON * (tb - T25) / 1000 if tb > T25 else 0)
    notes = ["not a measured compound: may not exist or may decompose"]
    if searched:
        notes.append("proposed by the AI search")
    return GasSpecies(smiles, formula, formula, rdMolDescriptors.CalcExactMolWt(Chem.AddHs(m)), tb, tb_src, g,
                      band, f"GFN2-xTB + atom equivalents fitted on the training half (+/-{band:.0f} kJ/mol on the "
                            f"test half)", smiles in TOXIC, notes)


USE_MEASURED = False        # True = old behaviour: measured values replace calculations for known molecules


def first_principles_scale(level: str) -> dict | None:
    """Element references of one level: H, O, N, F, Cl from its own computed H2, O2, N2, F2, Cl2; carbon and
    sulfur (solids at 25 C, not computable as molecules) fitted on the training half of the known gases.
    Returns {'mu': {...}, 'test_rmse': kJ/mol, 'n_test': n} or None if the level is not computed yet."""
    from . import abinitio as A
    ref = {"H": "[H][H]", "O": "O=O", "N": "N#N", "F": "FF", "Cl": "ClCl"}
    mu = {}
    for el, r in ref.items():
        x = A.free_energy(r, level, "g", compute=False)
        if not x or "g_kj" not in x:
            return None
        mu[el] = x["g_kj"] / 2
    rows, y, test = [], [], []
    for s_, v in MEASURED.items():
        x = A.free_energy(s_, level, "g", compute=False)
        if not x or "g_kj" not in x:
            continue
        c = counts(s_)
        rest = x["g_kj"] - sum(c.get(e, 0) * mu[e] for e in ref) - v[1]
        if is_train(s_):
            rows.append([c.get("C", 0), c.get("S", 0)])
            y.append(rest)
        else:
            test.append(([c.get("C", 0), c.get("S", 0)], rest))
    if len(rows) < 6:
        return None
    sol, *_ = np.linalg.lstsq(np.array(rows, float), np.array(y), rcond=None)
    mu["C"], mu["S"] = float(sol[0]), float(sol[1])
    err = [np.array(a) @ sol - b for a, b in test]
    return {"mu": mu, "test_rmse": float(np.sqrt(np.mean(np.square(err)))) if err else 30.0, "n_test": len(err)}


ELEMENT_GASES = {_canon(x) for x in ("[H][H]", "O=O", "N#N", "FF", "ClCl")}
LEVEL_ORDER = ("ccsdt", "dft")          # best first; xTB + atom equivalents when neither is computed


def species_table(cache: dict | None = None) -> dict[str, GasSpecies]:
    """Every molecule with a computed Gibbs energy of formation, keyed by SMILES. No measured value of any
    molecule enters (unless USE_MEASURED): known molecules are computed like new ones."""
    from . import abinitio as A
    cache = load_energies() if cache is None else cache
    fit = fit_atom_equivalents(cache)
    searched = set(searched_species())
    scales = {lv: first_principles_scale(lv) for lv in LEVEL_ORDER}
    out: dict[str, GasSpecies] = {}
    for s in all_species():
        if not USE_MEASURED:
            best = None
            for lv in LEVEL_ORDER:
                sc = scales.get(lv)
                x = A.free_energy(s, lv, "g", compute=False) if sc else None
                if sc and x and "g_kj" in x and (s in MEASURED or storable(s)):
                    best = (lv, x["g_kj"] - sum(n * sc["mu"].get(el, 0.0) for el, n in counts(s).items()), sc)
                    break
            if best:
                lv, g_gas, sc = best
                sp_ = computed_species(s, g_gas, sc["test_rmse"], s in searched)
                sp_.source = f"{A.LEVELS[lv]} (+/-{sc['test_rmse']:.0f} kJ/mol on known test molecules)"
                if s == "O":                                     # liquid water from its own computed solvation
                    gl = A.liquid_g("O", lv, 373.0, compute=False)
                    if gl is not None:
                        sp_.g_kj = gl - sum(n * sc["mu"].get(el, 0.0) for el, n in counts(s).items())
                if s in MEASURED:
                    sp_.name = MEASURED[s][0]
                    sp_.notes = [n_ for n_ in sp_.notes if not n_.startswith("not a measured")]
                out[s] = sp_
                continue
            if s in ELEMENT_GASES:                # elements in their standard state: dfG = 0 by definition
                tb, tb_src = boiling_point(s)
                mw = 2.016 if s == "[H][H]" else rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s))
                out[s] = GasSpecies(s, MEASURED[s][0], formula_of(s), mw, tb, tb_src, 0.0, 0.0,
                                    "element in its standard state (0 by definition)", s in TOXIC, [])
                continue
            if s in RADICALS:
                continue
        if USE_MEASURED and s in MEASURED:
            tb, tb_src = boiling_point(s)
            m = Chem.MolFromSmiles(s)
            mw = rdMolDescriptors.CalcExactMolWt(Chem.AddHs(m)) if s != "[H][H]" else 2.016
            name, g_gas, g_liq, _, src = MEASURED[s]
            g = g_liq if (tb > T25 and g_liq is not None) else g_gas - (TROUTON * (tb - T25) / 1000 if tb > T25 else 0)
            out[s] = GasSpecies(s, name, formula_of(s), mw, tb, tb_src, g, 2.0, src, s in TOXIC, [])
            continue
        e = cache.get(s)
        if e is None or fit is None or boiling_point(s)[0] >= XTB_CUTOFF_K or not (storable(s) or s in MEASURED):
            continue
        out[s] = computed_species(s, g_gas_xtb(s, e, fit), fit.rmse_loo, s in searched)
        if s in MEASURED:
            out[s].name = MEASURED[s][0]
            out[s].notes = [n_ for n_ in out[s].notes if not n_.startswith("not a measured")]
    return out


# ------------------------------------------------------------------ couples

@dataclass
class Couple:
    ox: str                 # oxidised form (takes up hydrogen)
    red: str                # reduced form
    k_ox: int
    k_red: int
    h2: float               # a: H2 taken up per couple as written
    water: int              # w: H2O(l) made (negative: used)
    e0: float               # V vs SHE
    band_v: float

    @property
    def n_e(self) -> int:
        return int(round(2 * self.h2))


_HEAVY = ("C", "N", "S", "F", "Cl")


def couples(sp: dict[str, GasSpecies], max_k: int = MAX_K, max_e: int = MAX_E,
            only: str | None = None) -> list[Couple]:
    """Every couple kA A + a H2 -> kB B + w H2O between two species with the
    same C, N, S, F, Cl (water and hydrogen themselves excluded).
    only: return just the couples that involve this species (fast, for the AI search)."""
    by_sig: dict[tuple, list[str]] = {}
    cnt = {s: _counts(s) for s in sp}
    g_water = sp["O"].g_kj if "O" in sp else G_WATER_L          # computed liquid water (G_WATER_L: measured fallback)
    for s in sp:
        if s in ("O", "[H][H]"):
            continue
        sig = tuple(cnt[s].get(e, 0) for e in _HEAVY)
        by_sig.setdefault(sig, []).append(s)
    out: list[Couple] = []
    only_sig = tuple(cnt[only].get(e, 0) for e in _HEAVY) if only else None
    # oxygen-only species (O2, H2O2, O3) reduced all the way to water: A + a H2 -> w H2O
    for a in by_sig.get(tuple(0 for _ in _HEAVY), []):
        if only and a != only:
            continue
        w, h = cnt[a].get("O", 0), cnt[a].get("H", 0)
        h2 = (2 * w - h) / 2
        if h2 > 0:
            dg = w * g_water - sp[a].g_kj
            out.append(Couple(a, "O", 1, w, h2, 0, -dg * 1000 / (2 * h2 * F), sp[a].g_band_kj * 1000 / (2 * h2 * F)))
    sigs = list(by_sig)
    for s1 in sigs:
        for s2 in sigs:
            if only and only_sig not in (s1, s2):
                continue
            for ka in range(1, max_k + 1):
                for kb in range(1, max_k + 1):
                    if math.gcd(ka, kb) != 1 or tuple(ka * x for x in s1) != tuple(kb * x for x in s2):
                        continue
                    if not any(s1) and (ka, kb) != (1, 1):
                        continue
                    for a in by_sig[s1]:
                        for b in by_sig[s2]:
                            if a == b or (only and only not in (a, b)):
                                continue
                            w = ka * cnt[a].get("O", 0) - kb * cnt[b].get("O", 0)
                            h2 = (kb * cnt[b].get("H", 0) + 2 * w - ka * cnt[a].get("H", 0)) / 2
                            if h2 <= 0 or abs(2 * h2 - round(2 * h2)) > 1e-9 or 2 * h2 > max_e:
                                continue
                            dg = kb * sp[b].g_kj + w * g_water - ka * sp[a].g_kj
                            e0 = -dg * 1000 / (2 * h2 * F)
                            band = math.sqrt((kb * sp[b].g_band_kj) ** 2 + (ka * sp[a].g_band_kj) ** 2) * 1000 / (2 * h2 * F)
                            out.append(Couple(a, b, ka, kb, h2, w, e0, band))
    return out


# ------------------------------------------------------------------ cells

H2_COUPLE = Couple("[H+]", "[H][H]", 2, 1, 0.0, 0, 0.0, 0.0)      # marker for the hydrogen electrode


@dataclass
class GasCell:
    pos: Couple
    neg: Couple | None      # None = hydrogen electrode
    voltage: float
    band_v: float
    n_e: int
    stoich: dict            # smiles -> moles per cell reaction, + = charged state, - = discharged state
    gas_moles: float        # moles kept as gas (each species at its largest amount)
    wh_per_mol_gas: float
    kg_chem: float          # kg of chemicals per cell reaction (heavier state)
    wh_kg_chem: float
    evidence: list[tuple[str, str]]
    toxic: list[str]
    computed_only: list[str]
    _name: str = ""
    _sp: dict | None = field(default=None, repr=False, compare=False)

    @property
    def name(self) -> str:
        """Reaction text, built on first use (millions of cells are screened, few are shown)."""
        if not self._name and self._sp is not None:
            self._name = reaction_text(self, self._sp)
        return self._name

    @name.setter
    def name(self, v: str) -> None:
        self._name = v

    @property
    def est_wh_kg(self) -> float:
        """Chemicals plus carbon-fibre tank walls only (no cells): an upper bound used to pre-rank."""
        return self.wh_per_mol_gas * self.gas_moles / (self.kg_chem + TANK_KG_PER_MOL_EST * self.gas_moles)

    @property
    def recharge(self) -> str:
        """Weakest experimental evidence of the two electrodes."""
        order = ["demonstrated", "both directions separately", "one direction", "not demonstrated"]
        return max((e[0] for e in self.evidence), key=order.index)

    @property
    def key(self) -> str:
        return f"{self.pos.ox}>{self.pos.red}|{self.neg.red + '>' + self.neg.ox if self.neg else 'H2'}"


def _cell(pos: Couple, neg: Couple | None, sp: dict[str, GasSpecies]) -> GasCell | None:
    e_neg = neg.e0 if neg else 0.0
    v = pos.e0 - e_neg
    # must charge in water: below the oxygen limit (or like an electrolyser when the cell keeps its oxygen)
    v_max = (2.0 if pos.ox == "O=O" else 1.229 + 0.4) - CHARGE_MARGIN_V
    if v < MIN_VOLTAGE or v > v_max:
        return None
    ne_p = pos.n_e
    ne_n = neg.n_e if neg else 2
    n_e = ne_p * ne_n // math.gcd(ne_p, ne_n)
    fp, fn = n_e // ne_p, n_e // ne_n
    charged, discharged = {}, {}
    charged[pos.ox] = charged.get(pos.ox, 0) + fp * pos.k_ox
    discharged[pos.red] = discharged.get(pos.red, 0) + fp * pos.k_red
    if neg:
        charged[neg.red] = charged.get(neg.red, 0) + fn * neg.k_red
        discharged[neg.ox] = discharged.get(neg.ox, 0) + fn * neg.k_ox
    else:
        charged["[H][H]"] = charged.get("[H][H]", 0) + fn
    water = fp * pos.water - (fn * neg.water if neg else 0)      # made on discharge (+) or used (-)
    if water > 0:
        discharged["O"] = discharged.get("O", 0) + water
    elif water < 0:
        charged["O"] = charged.get("O", 0) - water

    net = discharged.pop("O", 0) - charged.pop("O", 0)        # water made on discharge, net
    if net > 0:
        discharged["O"] = net
    elif net < 0:
        charged["O"] = -net
    # both charged reactants must be gases
    if not all(sp[s].gas for s in charged if s != "O"):
        return None
    gas_moles = sum(n for s, n in charged.items() if s != "O" and sp[s].gas) + \
        sum(n for s, n in discharged.items() if s != "O" and sp[s].gas)
    mass = lambda d: sum(n * (18.015 if s == "O" else sp[s].mw) for s, n in d.items()) / 1000
    kg = max(mass(charged), mass(discharged))
    wh = n_e * F * v / 3600
    stoich = {**{s: n for s, n in charged.items()}, **{s: -n for s, n in discharged.items()}}
    ev = [evidence(pos.ox, pos.red)] + [evidence(neg.ox, neg.red) if neg else evidence("[H+]", "[H][H]")]
    involved = set(charged) | set(discharged)
    toxic = sorted(sp[s].name for s in involved if s in sp and sp[s].toxic)
    computed = sorted(s for s in involved if s in sp and s not in MEASURED)
    band = math.hypot(pos.band_v, neg.band_v if neg else 0.0)
    c = GasCell(pos, neg, v, band, n_e, stoich, gas_moles, wh / gas_moles if gas_moles else math.inf,
                kg, wh / kg, ev, toxic, computed)
    c._sp = sp
    return c


def reaction_text(c: GasCell, sp: dict[str, GasSpecies]) -> str:
    nm = lambda s: "H2O" if s == "O" else ("H2" if s == "[H][H]" else sp[s].formula if s in sp else s)
    st = lambda s: "(l)" if s == "O" or (s in sp and not sp[s].gas) else "(g)"
    fmt = lambda d: " + ".join(f"{'' if n == 1 else n} {nm(s)}{st(s)}".strip() for s, n in d.items())
    ch = {s: n for s, n in c.stoich.items() if n > 0}
    dis = {s: -n for s, n in c.stoich.items() if n < 0}
    return f"{fmt(ch)} -> {fmt(dis)}"


def screen(cache: dict | None = None, top: int | None = None,
           sp: dict[str, GasSpecies] | None = None) -> tuple[list[GasCell], dict[str, GasSpecies]]:
    """Every sealed two-gas cell, lightest first (chemicals + tank walls, est_wh_kg)."""
    sp = species_table(cache) if sp is None else sp
    cps = couples(sp)
    pos = [c for c in cps if sp[c.ox].gas and c.e0 > 0.3]        # oxidant gases (charged positive)
    neg = [c for c in cps if sp[c.red].gas and c.e0 < 0.6]       # fuel gases (charged negative)
    import bisect
    neg.sort(key=lambda c: c.e0)
    neg_e = [c.e0 for c in neg]
    out: dict[str, GasCell] = {}
    for p in pos:
        v_max = (2.0 if p.ox == "O=O" else 1.229 + 0.4) - CHARGE_MARGIN_V
        # only fuel couples that give a voltage inside [MIN_VOLTAGE, v_max] (the same test as _cell, done first)
        lo, hi = bisect.bisect_left(neg_e, p.e0 - v_max), bisect.bisect_right(neg_e, p.e0 - MIN_VOLTAGE)
        for n in [None] + neg[lo:hi]:
            if n is not None and (n.ox == p.ox or n.red == p.red):
                continue
            c = _cell(p, n, sp)
            if c is not None and (c.key not in out or c.wh_per_mol_gas > out[c.key].wh_per_mol_gas):
                out[c.key] = c
    cells = sorted(out.values(), key=lambda c: -c.est_wh_kg)
    top = top or MAX_CELLS
    return cells[:top], sp


def reference_h2_o2(cells: list[GasCell]) -> GasCell | None:
    return next((c for c in cells if c.neg is None and c.pos.ox == "O=O" and c.pos.red == "O"), None)
