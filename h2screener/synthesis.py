"""What a molecule costs to make, in USD/kg at bulk scale, with a range.

No database gives bulk prices for new molecules, so the price is estimated
from how hard the molecule is to make:

  difficulty   SA score (Ertl & Schuffenhauer, J. Cheminf. 1 (2009) 8),
               1 = easy, 10 = very hard; it grows with unusual fragments,
               stereocentres, fused and bridged rings, and size
  price        log10(USD/kg) = a + b x difficulty, fitted by least squares
               to anchor molecules with known bulk prices ([synthesis] in
               targets.toml: commodity quinones, viologen, TEMPO, riboflavin)
  range        p10-p90 = price / spread .. price x spread. The SA score
               tracks catalogue price only loosely (Pearson r ~ 0.3 on log
               price; CoPriNet, Digital Discovery (2023), doi:10.1039/D2DD00071G), so the
               spread is wide on purpose
  no floor     the element prices in materials.py are for purified
               elements (graphite, gases), not for the oil-derived feedstocks
               organic chemicals are made from: they put hydroquinone at
               ~17 USD/kg, while it sells at ~3 USD/kg in bulk. So they are
               not used as a lower bound for organic molecules

The anchors cover difficulty ~1.3-3.8; above that the line is extrapolated
and the explanation says so. A process-chemistry estimate (route, steps,
yields, precursor prices; Dieterich et al., Transl. Mater. Res. (2018),
doi:10.1088/2053-1613/aacb0e) replaces this for any molecule that gets that far.
"""

from __future__ import annotations

import math
import os
import sys
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rdkit import Chem, RDConfig, RDLogger

RDLogger.DisableLog("rdApp.*")
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def default_config() -> dict:
    with open(ROOT / "targets.toml", "rb") as f:
        return tomllib.load(f)["synthesis"]


@dataclass(frozen=True)
class SynthCost:
    smiles: str
    difficulty: float       # SA score, 1-10
    usd_kg: float           # median estimate (times the price factor)
    lo: float               # p10
    hi: float               # p90
    extrapolated: bool      # difficulty outside the anchors' range

    def describe(self) -> str:
        s = (f"difficulty {self.difficulty:.1f}/10 -> about {self.usd_kg:,.0f} USD/kg "
             f"(range {self.lo:,.0f}-{self.hi:,.0f})")
        if self.extrapolated:
            s += ", beyond the priced examples, so less certain"
        return s


def _fit(anchors: tuple) -> tuple[float, float, float, float]:
    """(a, b, lowest, highest difficulty) of log10(price) = a + b x difficulty."""
    xs = [difficulty(s) for s, _ in anchors]
    ys = [math.log10(p) for _, p in anchors]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    return my - b * mx, b, min(xs), max(xs)


@lru_cache(maxsize=8)
def _fit_cached(anchors: tuple) -> tuple[float, float, float, float]:
    return _fit(anchors)


def _anchors(cfg: dict) -> tuple:
    return tuple((a["smiles"], float(a["usd_kg"])) for a in cfg["anchors"])


@lru_cache(maxsize=20000)
def difficulty(smiles: str) -> float:
    return float(sascorer.calculateScore(Chem.MolFromSmiles(smiles)))


def is_organic(smiles: str) -> bool:
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return m is not None and any(a.GetSymbol() == "C" for a in m.GetAtoms())


def price_from_difficulty(sa: float, cfg: dict | None = None) -> float:
    """Median USD/kg for a difficulty (SA score), times the price factor."""
    cfg = cfg or default_config()
    a, b, _, _ = _fit_cached(_anchors(cfg))
    return 10 ** (a + b * sa) * float(cfg.get("price_factor", 1.0))


def estimate(smiles: str, cfg: dict | None = None) -> SynthCost:
    cfg = cfg or default_config()
    _, _, x_lo, x_hi = _fit_cached(_anchors(cfg))
    sa = difficulty(smiles)
    spread = float(cfg["spread"])
    med = price_from_difficulty(sa, cfg)
    return SynthCost(smiles, sa, med, med / spread, med * spread, not (x_lo - 0.5 <= sa <= x_hi + 0.5))


def species_smiles(label: str) -> str | None:
    """The SMILES behind a molecule species: its label is the SMILES,
    optionally followed by a role tag such as ' (negative)'."""
    s = label.split(" (")[0].strip()
    return s if is_organic(s) else None


def price_penalty(usd_kg: float, cfg: dict | None = None) -> float:
    """Factor in [min_factor, 1] that steers the search away from molecules
    that are expensive to make: 1 up to the reference price, then falling
    as (reference / price)^0.3."""
    cfg = cfg or default_config()
    ref = float(cfg["search_ref_usd_kg"])
    return 1.0 if usd_kg <= ref else max(float(cfg["search_min_factor"]), (ref / usd_kg) ** 0.3)


def species_price(sp, unknown_usd_kg: float, cfg: dict | None = None) -> tuple[float, SynthCost | None]:
    """USD/kg of a cell species: the synthesis estimate for an organic
    molecule, the element-price estimate for anything else (salts, metals,
    oxides), or the unknown-material price when an element has no price."""
    from . import materials
    smi = species_smiles(sp.label)
    if smi:
        sc = estimate(smi, cfg)
        return sc.usd_kg, sc
    price = materials.cost_per_kg(sp.formula)
    return (price if price is not None else unknown_usd_kg), None
