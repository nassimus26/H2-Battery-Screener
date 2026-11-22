"""Does a molecule survive in water? Computed, not looked up.

Every reaction of water with the molecule that a common mechanism allows is generated with
reaction templates (hydration of C=O, addition of water to C=C next to C=O, hydrolysis of esters,
amides, imines, acyl halides and nitriles), and its Gibbs energy in water is computed from first
principles (abinitio.py: xTB for a first pass, DFT when computed):

    dG = G(products, water) - G(molecule, water) - G(liquid water)

dG < -10 kJ/mol: the reaction is downhill, the molecule is expected to react (how fast is not
computed: thermodynamics says whether, not when). -10 to +10: an equilibrium mixture.
Above +10: stable against that reaction.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import AllChem

from . import abinitio as A

TEMPLATES = {
    "hydration of C=O (gem-diol)": "[CX3;!$(C-[OX2H1,OX1-]);!$(C-[NX3]);!$(C-O-[#6]):1]=[OX1:2].[OH2:3]>>[C:1](-[OH1:2])-[OH1:3]",
    "water adds to C=C next to C=O (Michael)": "[CX3:1]=[CX3:2]-[CX3:4]=[OX1:5].[OH2:3]>>[OH1:3]-[C:1]-[C:2]-[C:4]=[O:5]",
    "ester hydrolysis": "[CX3:1](=[OX1:2])-[OX2:3]-[#6:4].[OH2:5]>>[C:1](=[O:2])-[OH1:5].[OH1:3]-[#6:4]",
    "amide hydrolysis": "[CX3:1](=[OX1:2])-[NX3:3].[OH2:5]>>[C:1](=[O:2])-[OH1:5].[N:3]",
    "imine hydrolysis": "[CX3:1]=[NX2;!R:2].[OH2:3]>>[C:1]=[O:3].[N:2]",
    "acyl halide hydrolysis": "[CX3:1](=[OX1:2])-[F,Cl:3].[OH2:4]>>[C:1](=[O:2])-[OH1:4].[*:3]",
    "nitrile hydration": "[CX2:1]#[NX1:2].[OH2:3]>>[C:1](=[O:3])-[N:2]",
}
_RX = {k: AllChem.ReactionFromSmarts(v) for k, v in TEMPLATES.items()}
REACTS, EQUILIBRIUM = -10.0, 10.0


def products(smiles: str) -> list[tuple[str, list[str]]]:
    """(reaction name, product SMILES) for each distinct way water can react with the molecule."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    w = Chem.AddHs(Chem.MolFromSmiles("O"))
    out, seen = [], set()
    for name, rx in _RX.items():
        for ps in rx.RunReactants((m, w)):
            try:
                smi = []
                for p in ps:
                    p = Chem.RemoveHs(p)
                    Chem.SanitizeMol(p)
                    smi.append(Chem.MolToSmiles(p))
            except Exception:
                continue
            k = (name, tuple(sorted(smi)))
            if k not in seen:
                seen.add(k)
                out.append((name, smi))
    return out


def water_reactions(smiles: str, level: str = "xtb", compute: bool = True) -> list[dict]:
    """[{'reaction', 'products', 'dG_kj', 'verdict', 'level'}]; dG None when not computed (compute=False)."""
    g_w = A.liquid_g("O", level, 373.0, compute)
    g_m = A.free_energy(smiles, level, "aq", compute)
    res = []
    for name, prods in products(smiles):
        dg = None
        if g_w is not None and g_m and "g_kj" in g_m:
            gp = [A.free_energy(p, level, "aq", compute) for p in prods]
            if all(x and "g_kj" in x for x in gp):
                dg = sum(x["g_kj"] for x in gp) - g_m["g_kj"] - g_w
        verdict = (None if dg is None else "reacts with water" if dg < REACTS
                   else "partly reacts (equilibrium)" if dg < EQUILIBRIUM else "stable")
        res.append({"reaction": name, "products": prods, "dG_kj": dg, "verdict": verdict, "level": level})
    return res


def summary(smiles: str, compute: bool = False) -> tuple[str, list[dict]]:
    """Best available level (DFT, else xTB): ('stable' | 'reacts with water' | ..., reactions)."""
    for lv in ("dft", "xtb"):
        r = water_reactions(smiles, lv, compute=compute)
        if r and all(x["dG_kj"] is not None for x in r):
            worst = min(r, key=lambda x: x["dG_kj"])
            return worst["verdict"], r
        if not r:
            return "no reaction with water found by the templates", []
    return "not computed yet", water_reactions(smiles, "xtb", compute=False)
