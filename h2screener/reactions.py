"""Half-reactions and cell reactions.

A half-reaction is written as a reduction: oxidant side + n e- -> reductant
side, with E0 vs SHE. Species strings such as "2 Fe(OH)2(s)", "MnO4-",
"Fe(3+)", "[Fe(CN)6](4-)" or "Br2(l)" are parsed into formula, charge, phase
and coefficient, so cell reactions can be balanced, checked for charge and
mass, and weighed with pymatgen's element data.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from pymatgen.core import Composition

DATA = Path(__file__).resolve().parent.parent / "data"

PHASE_RE = re.compile(r"\((s|l|g|aq|Hg|quartz|cr)\)$")


@dataclass(frozen=True)
class Species:
    formula: str          # neutral formula used for mass (no charge, no phase)
    charge: int
    phase: str            # s, l, g, aq
    label: str            # as written

    @property
    def key(self) -> str:
        sign = "" if self.charge == 0 else (f"{abs(self.charge)}{'+' if self.charge > 0 else '-'}")
        return f"{self.formula}{'^' + sign if sign else ''}({self.phase})"

    @property
    def molar_mass(self) -> float:
        """kg/mol"""
        return Composition(self.formula).weight / 1000.0

    @property
    def elements(self) -> set[str]:
        return {str(e) for e in Composition(self.formula).elements}

    @property
    def is_solvent(self) -> bool:
        return self.formula in ("H2O",) and self.phase in ("l", "aq")


def parse_species(text: str) -> Species | None:
    if "•" in text:
        return None  # free radicals are not storable reactants
    s = text.strip().replace("alpha-", "")
    phase = None
    m = PHASE_RE.search(s)
    if m:
        phase = m.group(1)
        s = s[: m.start()].strip()
    charge = 0
    m = re.search(r"\((\d*)([+-])\)$", s)          # Fe(3+), [Fe(CN)6](4-)
    if m:
        charge = int(m.group(1) or 1) * (1 if m.group(2) == "+" else -1)
        s = s[: m.start()]
    else:
        m = re.search(r"([+-]+)$", s)               # MnO4-, H+, Br3-
        if m and len(s) > len(m.group(1)):
            charge = len(m.group(1)) * (1 if m.group(1)[0] == "+" else -1)
            s = s[: m.start()]
    s = s.replace("[", "(").replace("]", ")").strip()
    if not s or not re.fullmatch(r"[A-Za-z0-9().]+", s):
        return None
    try:
        comp = Composition(s)
    except Exception:
        return None
    if comp.num_atoms == 0:
        return None
    if phase is None:
        phase = "aq" if charge != 0 else ("aq" if s not in ("H2O",) else "l")
    if phase in ("Hg", "quartz", "cr"):
        phase = "s"
    return Species(s, charge, phase, text.strip())


def parse_side(text: str) -> dict[Species, float] | None:
    out: dict[Species, float] = {}
    for term in re.split(r"\s\+\s", " " + text.strip() + " "):
        term = term.strip()
        if not term:
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)\s+(.*)$", term)
        coef, body = (float(m.group(1)), m.group(2)) if m else (1.0, term)
        if re.fullmatch(r"e-?", body):
            out[ELECTRON] = out.get(ELECTRON, 0) + coef
            continue
        sp = parse_species(body)
        if sp is None:
            return None
        out[sp] = out.get(sp, 0) + coef
    return out


ELECTRON = Species("e", -1, "e", "e-")


@dataclass
class HalfReaction:
    oxidant: dict[Species, float]
    reductant: dict[Species, float]
    e0: float                         # V vs SHE, 298.15 K, 1 atm gases, unit activity
    n: float                          # electrons
    label: str
    source: str
    medium: str = field(default="")   # acid, alkaline, any

    def balanced(self) -> bool:
        return reaction_balanced(self.oxidant, self.reductant)


def element_count(side: dict[Species, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sp, c in side.items():
        if sp is ELECTRON or sp.formula == "e":
            continue
        for el, k in Composition(sp.formula).get_el_amt_dict().items():
            out[el] = out.get(el, 0) + c * k
    return out


def charge_count(side: dict[Species, float]) -> float:
    return sum(c * sp.charge for sp, c in side.items())


def reaction_balanced(left: dict, right: dict, tol: float = 1e-6) -> bool:
    a, b = element_count(left), element_count(right)
    els = set(a) | set(b)
    if any(abs(a.get(e, 0) - b.get(e, 0)) > tol for e in els):
        return False
    return abs(charge_count(left) - charge_count(right)) < tol


def load_table(path: Path = DATA / "standard_potentials.csv") -> list[HalfReaction]:
    """All half-reactions of the potential table that parse and balance."""
    out = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ox, red = parse_side(row["oxidant"]), parse_side(row["reductant"])
            if ox is None or red is None:
                continue
            n = sum(c for sp, c in ox.items() if sp is ELECTRON)
            if n <= 0:
                continue
            hr = HalfReaction(
                ox, red, float(row["e0_v_she"]), n,
                f"{row['oxidant']} -> {row['reductant']}",
                f"{row['source']} [{row['reference'] or 'no ref'}]",
            )
            if not hr.balanced():
                continue
            names = {sp.formula + str(sp.charge) for sp in list(ox) + list(red)}
            hr.medium = "acid" if "H1" in names else ("alkaline" if "OH-1" in names else "any")
            out.append(hr)
    return out


@dataclass
class CellReaction:
    """Discharge reaction: reactants -> products, n electrons, E0 cell."""

    positive: HalfReaction | None
    negative: HalfReaction | None
    reactants: dict[Species, float]
    products: dict[Species, float]
    n: float
    e0: float
    medium: str
    origin: str = "table"            # table | materials-project | ml-potential
    source: str = ""
    notes: list[str] = field(default_factory=list)
    solid_positive: bool = False     # two-phase solid electrode: flat voltage plateau
    solubility_mol_l: float | None = None   # computed for molecules; else targets.toml

    @property
    def name(self) -> str:
        def side(d):
            return " + ".join(f"{c:g} {sp.label}" if c != 1 else sp.label for sp, c in d.items())
        return f"{side(self.reactants)} -> {side(self.products)}"

    def h2_consumed(self) -> float:
        """mol H2 consumed per reaction on discharge (negative if produced)."""
        r = sum(c for sp, c in self.reactants.items() if sp.formula == "H2" and sp.phase == "g")
        p = sum(c for sp, c in self.products.items() if sp.formula == "H2" and sp.phase == "g")
        return r - p

    def gas_terms(self) -> dict[str, float]:
        """Net moles consumed of each gaseous species on discharge."""
        out: dict[str, float] = {}
        for sp, c in self.reactants.items():
            if sp.phase == "g":
                out[sp.formula] = out.get(sp.formula, 0) + c
        for sp, c in self.products.items():
            if sp.phase == "g":
                out[sp.formula] = out.get(sp.formula, 0) - c
        return out

    def active_mass(self) -> float:
        """kg of reactants per reaction, solvent water excluded (heavier of
        the charged and discharged states, so it bounds both)."""
        def mass(d):
            return sum(c * sp.molar_mass for sp, c in d.items()
                       if not sp.is_solvent and sp.formula not in ("H", "OH"))
        return max(mass(self.reactants), mass(self.products))

    def dissolved_moles(self) -> float:
        """mol of dissolved (aq) non-H+/OH- species in the heavier state."""
        def moles(d):
            return sum(c for sp, c in d.items()
                       if sp.phase == "aq" and sp.formula not in ("H", "OH", "H2O"))
        return max(moles(self.reactants), moles(self.products))

    def species(self) -> list[Species]:
        return list(self.reactants) + list(self.products)

    def elements(self) -> set[str]:
        out: set[str] = set()
        for sp in self.species():
            out |= sp.elements
        return out


def combine(pos: HalfReaction, neg: HalfReaction) -> CellReaction | None:
    """Cell = pos reduction + neg oxidation, scaled to the same electron count.
    Species on both sides cancel. Returns None for a negative cell voltage."""
    e0 = pos.e0 - neg.e0
    if e0 <= 0:
        return None
    from math import lcm
    from fractions import Fraction

    fp, fn = Fraction(pos.n).limit_denominator(12), Fraction(neg.n).limit_denominator(12)
    n = Fraction(lcm(fp.numerator * fn.denominator, fn.numerator * fp.denominator),
                 fp.denominator * fn.denominator)
    kp, kn = float(n / fp), float(n / fn)
    left: dict[Species, float] = {}
    right: dict[Species, float] = {}
    for sp, c in pos.oxidant.items():
        if sp is not ELECTRON:
            left[sp] = left.get(sp, 0) + kp * c
    for sp, c in pos.reductant.items():
        right[sp] = right.get(sp, 0) + kp * c
    for sp, c in neg.reductant.items():           # oxidised on discharge
        left[sp] = left.get(sp, 0) + kn * c
    for sp, c in neg.oxidant.items():
        if sp is not ELECTRON:
            right[sp] = right.get(sp, 0) + kn * c
    for sp in set(left) & set(right):
        d = left[sp] - right[sp]
        left.pop(sp)
        right.pop(sp)
        if d > 1e-9:
            left[sp] = d
        elif d < -1e-9:
            right[sp] = -d
    if not left or not right or not reaction_balanced(left, right):
        return None
    medium = pos.medium if pos.medium != "any" else neg.medium
    return CellReaction(pos, neg, left, right, float(n), e0, medium,
                        source=f"positive: {pos.source}; negative: {neg.source}")
