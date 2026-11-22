"""Stage 6: cost, supply risk and safety of the elements and species in a cell.

Element prices: pymatgen's cost database (USD/kg, 2013 values, each row cites
its source). Supply concentration: Herfindahl-Hirschman index of production
(Gaultois et al., Chem. Mater. 25 (2013) 2911, shipped with pymatgen).
Radioactivity: elements without a stable isotope are excluded.
Hazard statements: PubChem GHS classification, fetched on demand and cached.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import pymatgen.analysis.cost as _cost
from pymatgen.core import Composition

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
PMG_ANALYSIS = Path(_cost.__file__).parent

# No stable isotope (Tc, Pm, and Z >= 84; Bi-209's half-life exceeds 1e19 years).
RADIOACTIVE = {"Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
               "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"}
NOBLE = {"He", "Ne", "Ar", "Kr", "Xe"}


@lru_cache(maxsize=1)
def prices() -> dict[str, float]:
    out = {}
    with open(PMG_ANALYSIS / "costdb_elements.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    pass
    return out


@lru_cache(maxsize=1)
def hhi() -> dict[str, float]:
    out = {}
    with open(PMG_ANALYSIS / "hhi_data.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                out[row[0]] = float(row[1])
    return out


def cost_per_kg(formula: str) -> float | None:
    """USD/kg from element prices by mass fraction; None if an element has no price."""
    comp = Composition(formula)
    p = prices()
    total = 0.0
    for el, w in comp.as_weight_dict().items():
        if el not in p:
            return None
        total += w * p[el]          # as_weight_dict gives mass fractions
    return total


def supply_hhi(elements: set[str]) -> float:
    h = hhi()
    vals = [h[e] for e in elements if e in h]
    return max(vals) if vals else float("nan")


# ------------------------------------------------------------------ PubChem GHS

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUG_VIEW = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view"


def _get(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "h2-screener/0.1"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.load(r)
    except Exception:
        return None


def ghs_hazards(formula: str) -> list[str] | None:
    """GHS H-statements for the most common PubChem compound with this
    formula, or None when PubChem has none. Cached in cache/ghs.json."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "ghs.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    if formula in cache:
        return cache[formula]
    hz = None
    j = _get(f"{PUG}/compound/fastformula/{urllib.parse.quote(formula)}/cids/JSON?MaxRecords=1")
    cids = (j or {}).get("IdentifierList", {}).get("CID", [])
    if cids:
        v = _get(f"{PUG_VIEW}/data/compound/{cids[0]}/JSON?heading=GHS+Classification")
        if v:
            hz = sorted({m.group(0) for m in _hcodes(json.dumps(v))})
    cache[formula] = hz
    path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    time.sleep(0.25)   # PubChem asks for at most 5 requests per second
    return hz


def _hcodes(text: str):
    import re
    return re.finditer(r"\bH[2-4]\d\d\b", text)


SEVERE = {"H300", "H310", "H330", "H340", "H350", "H360", "H370", "H372"}  # fatal, CMR, organ damage
TOXIC = {"H301", "H311", "H331", "H314", "H318", "H341", "H351", "H361", "H371", "H373", "H400", "H410"}


def safety_factor(hazards: list[str]) -> float:
    """Multiplier in (0, 1]: 0.3 for any severe statement, 0.7 for toxic/corrosive."""
    if any(h in SEVERE for h in hazards):
        return 0.3
    if any(h in TOXIC for h in hazards):
        return 0.7
    return 1.0
