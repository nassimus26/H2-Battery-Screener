"""First-principles free energies of molecules: no measured value of any molecule is used.

Levels
  xtb   GFN2-xTB geometry, energy and vibrations (fast; used inside the AI searches)
  dft   wB97X-V/def2-TZVP single point on the xTB geometry (PySCF), with the xTB
        thermal correction (the usual "DFT//xTB" scheme). VV10 non-local
        correlation, so no empirical dispersion add-on.

Phases
  g     ideal gas, 1 bar
  aq    solute in water, 1 mol/L (implicit solvent ALPB; for dft the ALPB solvation free energy is
        added to the DFT gas-phase energy at the solution geometry)
        G(aq) = G(gas, solvated energy) + RT ln(24.79) for the 1 bar -> 1 mol/L standard state

Free energy
  G = E_electronic + G_thermal(298.15 K), G_thermal from the harmonic frequencies of the
  lowest conformer (ideal-gas rotor; modes below 100 cm-1 raised to 100 cm-1, quasi-RRHO),
  rotational symmetry number from the point group.

Conformers: up to N_CONF RDKit ETKDG conformers, each relaxed with xTB in the phase used;
the lowest is kept.

Reaction energies only need differences of these G (atoms balance), so no reference data
enters a voltage. Measured values of known molecules are used only to TEST the method
(validation.py), or to fit at most the two element references that are solids at 25 C
(carbon, sulfur) on a training half of the known gases, with the other half as the test.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "cache" / "abinitio.json"
HARTREE_KJ = 2625.49962
EV_KJ = 96.485332
T = 298.15
R_KJ = 8.314462618e-3
STD_AQ = R_KJ * T * math.log(24.79)      # kJ/mol, 1 bar gas -> 1 mol/L
N_CONF = 8
QRRHO_CM = 100.0
DFT_XC, DFT_BASIS = "wb97x-v", "def2-tzvp"
CC_BASIS = "cc-pvtz"
CC_MAX_HEAVY = 5            # CCSD(T) only for small molecules (its cost grows as N^7)
LEVELS = {"xtb": "GFN2-xTB", "dft": f"{DFT_XC.upper()}/{DFT_BASIS.upper()}//GFN2-xTB (water: + ALPB solvation)",
          "ccsdt": f"CCSD(T)/{CC_BASIS}//GFN2-xTB"}

_lock = threading.Lock()


# ------------------------------------------------------------------ store

_mem = {"mtime": None, "data": {}, "checked": 0.0}


def load(max_age: float = 10.0) -> dict:
    """The store, re-read when the file changed but at most every max_age seconds (background batches
    rewrite it constantly). Treat the returned dict as read-only."""
    now = time.monotonic()
    if _mem["mtime"] is not None and now - _mem["checked"] < max_age:
        return _mem["data"]
    _mem["checked"] = now
    try:
        mt = STORE.stat().st_mtime_ns
    except OSError:
        return {}
    if _mem["mtime"] != mt:
        try:
            _mem["data"], _mem["mtime"] = json.loads(STORE.read_text()), mt
        except Exception:
            return _mem["data"]
    return _mem["data"]


def _save_one(key: str, rec: dict) -> None:
    """Thread- and process-safe (several worker processes write the same store)."""
    import fcntl
    STORE.parent.mkdir(exist_ok=True)
    with _lock, open(STORE.with_suffix(".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = dict(load(0))
        d[key] = rec
        tmp = STORE.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, indent=0, sort_keys=True))
        tmp.replace(STORE)


def key(smiles: str, level: str, phase: str) -> str:
    return f"{smiles}|{level}|{phase}"


from functools import lru_cache  # noqa: E402


@lru_cache(maxsize=200_000)
def _canon(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# ------------------------------------------------------------------ structures

def multiplicity(smiles: str) -> int:
    """Neutral molecule: doublet if the electron count is odd, triplet for O2, otherwise singlet."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if Chem.MolToSmiles(Chem.RemoveHs(m)) in ("O=O", "[O][O]"):
        return 3                                   # triplet oxygen
    return 2 if sum(a.GetAtomicNum() for a in m.GetAtoms()) % 2 else 1


def conformers(smiles: str, n: int = N_CONF):
    """RDKit ETKDG conformers (MMFF-cleaned) as ASE Atoms."""
    from ase import Atoms
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    if m.GetNumAtoms() == 1:
        return [Atoms(syms, positions=np.zeros((1, 3)))]
    rot = Chem.rdMolDescriptors.CalcNumRotatableBonds(m)
    n = 1 if m.GetNumAtoms() <= 4 else (min(n, 2 + 2 * rot) if rot else min(n, 2))
    p = AllChem.ETKDGv3()
    p.randomSeed = 0xC0FFEE
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=n, params=p))
    if not ids:
        if AllChem.EmbedMolecule(m, randomSeed=7, useRandomCoords=True) != 0:
            raise RuntimeError("could not build a 3D structure")
        ids = [0]
    try:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500)
    except Exception:
        pass
    return [Atoms(syms, positions=m.GetConformer(i).GetPositions()) for i in ids]


def _xtb_calc(phase: str, mult: int):
    from tblite.ase import TBLite
    kw = dict(method="GFN2-xTB", verbosity=0, multiplicity=mult)
    if phase == "aq":
        kw["solvation"] = ("alpb", "water")
    return TBLite(**kw)


def relax(smiles: str, phase: str):
    """Lowest of the relaxed conformers: (atoms, energy eV, number of conformers)."""
    from ase.optimize import BFGS
    mult = multiplicity(smiles)
    best = None
    confs = conformers(smiles)
    for a in confs:
        a.calc = _xtb_calc(phase, mult)
        if len(a) > 1:
            BFGS(a, logfile=None).run(fmax=0.02, steps=400)
        e = float(a.get_potential_energy())
        if best is None or e < best[1]:
            best = (a, e)
    return best[0], best[1], len(confs)


def symmetry_number(atoms) -> int:
    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        return int(PointGroupAnalyzer(Molecule(atoms.get_chemical_symbols(), atoms.positions),
                                      tolerance=0.2).get_rotational_symmetry_number())
    except Exception:
        return 1


def thermal(atoms, e_ev: float, phase: str, mult: int) -> float:
    """G_thermal at 298.15 K and 1 bar, kJ/mol (harmonic, quasi-RRHO floor, ideal-gas rotor)."""
    from ase.thermochemistry import IdealGasThermo
    from ase.vibrations import Vibrations
    import tempfile
    if len(atoms) == 1:
        geom, vib_e = "monatomic", []
    else:
        a = atoms.copy()
        a.calc = _xtb_calc(phase, mult)
        with tempfile.TemporaryDirectory() as d:
            v = Vibrations(a, name=os.path.join(d, "vib"), delta=0.005)
            v.run()
            ev = np.real(v.get_energies())
        lin = _linear(atoms)
        n_drop = 5 if lin else 6
        ev = np.sort(np.abs(ev))[n_drop:]
        floor = QRRHO_CM * 1.239841984e-4                 # cm-1 -> eV
        vib_e = [max(x, floor) for x in ev]
        geom = "linear" if lin else "nonlinear"
    th = IdealGasThermo(vib_energies=vib_e, geometry=geom, potentialenergy=0.0, atoms=atoms,
                        symmetrynumber=symmetry_number(atoms), spin=(mult - 1) / 2)
    return float(th.get_gibbs_energy(temperature=T, pressure=1e5, verbose=False)) * EV_KJ


def _linear(atoms) -> bool:
    if len(atoms) < 3:
        return True
    p = atoms.positions - atoms.positions[0]
    d = p[1] / np.linalg.norm(p[1])
    return all(np.linalg.norm(np.cross(d, x)) < 1e-2 * max(np.linalg.norm(x), 1) for x in p[2:])


def dft_energy(atoms, mult: int, phase: str) -> float:
    """wB97X-V/def2-TZVP electronic energy, kJ/mol (SMD water for aq)."""
    from pyscf import dft, gto, lib
    lib.num_threads(max(1, (os.cpu_count() or 4) - 6))   # leave cores for the app and its searches
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                basis=DFT_BASIS, spin=mult - 1, charge=0, verbose=0)
    mf = dft.UKS(mol) if mult > 1 else dft.RKS(mol)
    mf.xc = DFT_XC
    mf.grids.level = 3
    mf.nlcgrids.level = 1
    mf.conv_tol = 1e-8
    mf.max_cycle = 150
    e = mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        e = mf.kernel()
        if not mf.converged:
            raise RuntimeError("DFT did not converge")
    e_kj = float(e) * HARTREE_KJ
    if phase == "aq":
        e_kj += solvation_xtb(atoms, mult)
    return e_kj


def solvation_xtb(atoms, mult: int) -> float:
    """Solvation free energy in water, kJ/mol: ALPB(water) minus gas-phase GFN2-xTB at the same geometry.
    (The PySCF build for macOS has no SMD; ALPB includes the cavity and dispersion terms that a bare
    PCM would miss.)"""
    a = atoms.copy()
    a.calc = _xtb_calc("aq", mult)
    e_aq = float(a.get_potential_energy())
    b = atoms.copy()
    b.calc = _xtb_calc("g", mult)
    return (e_aq - float(b.get_potential_energy())) * EV_KJ


def ccsdt_energy(atoms, mult: int) -> float:
    """CCSD(T)/cc-pVTZ electronic energy, kJ/mol (frozen core; UHF reference for open shells)."""
    from pyscf import cc, gto, scf
    mol = gto.M(atom=[(s, tuple(p)) for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)],
                basis=CC_BASIS, spin=mult - 1, charge=0, verbose=0)
    mf = (scf.UHF(mol) if mult > 1 else scf.RHF(mol))
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    if len(atoms) == 2 and mol.nelectron == 2:          # H2: CCSD is exact for two electrons
        c = cc.CCSD(mf)
        c.kernel()
        return float(c.e_tot) * HARTREE_KJ
    c = cc.CCSD(mf).set_frozen()
    c.conv_tol = 1e-8
    c.kernel()
    if not c.converged:
        raise RuntimeError("CCSD did not converge")
    return float(c.e_tot + c.ccsd_t()) * HARTREE_KJ


def heavy_atoms(smiles: str) -> int:
    return Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()


# ------------------------------------------------------------------ the free energy

def free_energy(smiles: str, level: str = "xtb", phase: str = "g", compute: bool = True) -> dict | None:
    """{'g_kj', 'e_kj', 'g_thermal_kj', 'n_conf', 'level', 'phase', 'seconds'} or None
    (not computed yet and compute=False, or the calculation failed: then {'error': ...})."""
    smiles = _canon(smiles)
    k = key(smiles, level, phase)
    d = load(10.0 if not compute else 0.0)
    if k in d or not compute:
        return d.get(k)
    t0 = time.time()
    try:
        mult = multiplicity(smiles)
        base = None if level == "xtb" else free_energy(smiles, "xtb", phase)   # geometry + thermal from xTB
        if base and "xyz" in base:
            from ase import Atoms
            atoms = Atoms([x[0] for x in base["xyz"]], positions=[x[1:] for x in base["xyz"]])
            e_xtb, n_conf, g_th = base["e_kj"] / EV_KJ, base["n_conf"], base["g_thermal_kj"]
        elif level != "xtb":
            raise RuntimeError("the xTB step failed: " + str((base or {}).get("error")))
        else:
            atoms, e_xtb, n_conf = relax(smiles, phase)
            g_th = thermal(atoms, e_xtb, phase, mult)
        if level == "xtb":
            e = e_xtb * EV_KJ
        elif level == "dft":
            e = dft_energy(atoms, mult, phase)
        elif level == "ccsdt":
            if phase != "g":
                raise ValueError("CCSD(T) is used for gases only")
            e = ccsdt_energy(atoms, mult)
        else:
            raise ValueError(level)
        g = e + g_th + (STD_AQ if phase == "aq" else 0.0)
        rec = {"g_kj": g, "e_kj": e, "g_thermal_kj": g_th, "n_conf": n_conf, "level": level, "phase": phase,
               "method": LEVELS[level], "seconds": round(time.time() - t0, 1),
               "xyz": [[s, *map(float, p)] for s, p in zip(atoms.get_chemical_symbols(), atoms.positions)]}
    except Exception as ex:
        rec = {"error": f"{type(ex).__name__}: {ex}", "level": level, "phase": phase,
               "seconds": round(time.time() - t0, 1)}
    _save_one(k, rec)
    return rec


def reaction_g(stoich: dict[str, float], level: str, phases: dict[str, str] | str = "g",
               compute: bool = True) -> float | None:
    """dG of a reaction, kJ/mol: stoich = {smiles: coefficient (products +, reactants -)}."""
    tot = 0.0
    for s, n in stoich.items():
        ph = phases if isinstance(phases, str) else phases.get(s, "g")
        r = free_energy(s, level, ph, compute)
        if not r or "g_kj" not in r:
            return None
        tot += n * r["g_kj"]
    return tot


def hydrogenation_voltage(ox: str, red: str, n_h2: int, level: str, compute: bool = True) -> float | None:
    """E vs SHE of  ox + n H2(g) -> red  in water, from first principles:
    ox(aq) + 2n H+ + 2n e- -> red(aq) has E = -dG(ox + n H2 -> red) / (2 n F) against the SHE,
    because the SHE is H2(g) <-> 2 H+ + 2 e- at 0 V by definition. No reference couple is used."""
    dg = reaction_g({red: 1, ox: -1, "[H][H]": -n_h2}, level, {red: "aq", ox: "aq", "[H][H]": "g"}, compute)
    return None if dg is None else -dg / (2 * n_h2 * 96.485332)


# ------------------------------------------------------------------ liquids

def liquid_g(smiles: str, level: str, tb_k: float, compute: bool = True) -> float | None:
    """G of the pure liquid, kJ/mol. Water: from its own implicit-water solvation (G_aq - G_g is the
    solvation free energy) plus the 1 mol/L -> 55.3 mol/L concentration term. Other liquids: Trouton's
    rule with the estimated boiling point (88 J/mol/K x Tb, valid for non-associated liquids)."""
    gg = free_energy(smiles, level, "g", compute)
    if not gg or "g_kj" not in gg:
        return None
    if Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == "O":
        xa = free_energy(smiles, "xtb", "aq", compute)       # geometry in water
        if not xa or "xyz" not in xa:
            return None
        from ase import Atoms
        atoms = Atoms([x[0] for x in xa["xyz"]], positions=[x[1:] for x in xa["xyz"]])
        return gg["g_kj"] + solvation_xtb(atoms, 1) + STD_AQ + R_KJ * T * math.log(55.34)
    if tb_k <= T:
        return gg["g_kj"]
    dh = 0.088 * tb_k
    return gg["g_kj"] - dh * (1 - T / tb_k)


def _work(args):
    smiles, level, phase = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    r = free_energy(smiles, level, phase)
    return smiles, level, phase, (r or {}).get("error")


def compute_many(items, workers: int | None = None, progress=None):
    """Compute (smiles, level, phase) items in parallel processes (PySCF on macOS runs one core per
    process). The xTB step of each item is done first in the same process. Resumable: cached items
    are skipped."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    done = load()
    todo = [(Chem.MolToSmiles(Chem.MolFromSmiles(s)), lv, ph) for s, lv, ph in items]
    todo = [x for x in dict.fromkeys(todo) if key(*x) not in done]
    workers = workers or max(1, (os.cpu_count() or 4) - 6)
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_work, x) for x in todo]
        for f in as_completed(futs):
            n += 1
            try:
                res = f.result()
            except Exception as e:      # a worker crashed
                res = ("?", "?", "?", f"{type(e).__name__}: {e}")
            if progress:
                progress(n, len(todo), res)
    return n
