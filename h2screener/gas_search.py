"""AI search for larger molecules for sealed two-gas batteries.

The same loop as the liquid search (ai_search.py), on gases:
  pool        every molecule with up to 4 heavy atoms (gases.enumerate_molecules,
              computed in full) plus every molecule this search has computed;
              the simple molecules are never dropped, they are the starting
              point and the partners of every new molecule
  1. generate children of the molecules that make the best cells, by SELFIES
              mutation and crossover (C, N, O, F, S, Cl only). Molecules with
              up to 4 heavy atoms are all known already, so children are the
              larger, more complex ones
  2. filter   neutral closed shell, one fragment, storable (same alerts as the
              screen), boiling below 400 K (Joback): a gas at 25 C, or a liquid
              product that can be kept in a small tank
  3. predict  a surrogate model (extra trees on Morgan fingerprints and atom /
              bond counts) predicts the Gibbs energy of formation per atom that
              xTB would give, with the spread of the trees as its uncertainty
  4. acquire  each child is built into cells against the best oxidant and fuel
              couples of the pool, at the predicted energy +/- kappa x spread;
              upper confidence bound on the lightest cell (chemicals + tank
              walls, Wh/kg)
  5. compute  GFN2-xTB energies for the batch (the ground truth of the loop),
              then the true lightest cell; retrain, repeat
Every evaluated molecule is kept in cache/gas_search.json and joins the gas
screen (gases.all_species), so the tab sizes it like any other cell.

Limits: xTB with atom equivalents carries about +/-80 kJ/mol, more for larger
molecules, and a search that picks the best of many noisy numbers tends to pick
molecules whose energy is overestimated. Whether a reaction runs backwards in a
cell cannot be computed.
"""

from __future__ import annotations

import json
import math
import random
import threading
import time

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from . import ai_search as A
from . import gases as G

STORE = G.SEARCHED
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
N_PARTNERS = 30            # best oxidant / fuel couples each child is tried against
SKIP_PARENTS = {"[H][H]", "O", "O=O"}


# ------------------------------------------------------------------ store

def load() -> dict:
    try:
        d = json.loads(STORE.read_text())
        d.setdefault("molecules", {})
        return d
    except Exception:
        return {"molecules": {}}


def save(d: dict) -> None:
    STORE.parent.mkdir(exist_ok=True)
    tmp = STORE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, sort_keys=True))
    tmp.replace(STORE)


def reset() -> str | None:
    """Start over: the current results become a dated backup; xTB energies stay cached."""
    if not STORE.exists():
        return None
    backup = STORE.with_name(f"gas_search.{time.strftime('%Y%m%d-%H%M%S')}.bak.json")
    STORE.rename(backup)
    return str(backup)


# ------------------------------------------------------------------ chemistry filters

def gas_ok(smiles: str | None, max_heavy: int, known: set[str]) -> bool:
    if not smiles or smiles in known:
        return False
    m = Chem.MolFromSmiles(smiles)
    if m is None or not (2 <= m.GetNumHeavyAtoms() <= max_heavy):
        return False
    if any(a.GetSymbol() not in G.ELEMENTS for a in m.GetAtoms()):
        return False
    if len(Chem.GetMolFrags(m)) > 1 or Descriptors.NumRadicalElectrons(m):
        return False
    if not G._real_sulfur(smiles) or not G.storable(smiles):
        return False
    return G.joback_tb(smiles) < G.XTB_CUTOFF_K


def _n_atoms(smiles: str) -> int:
    return sum(G.counts(smiles).values())


def features(smiles: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1024,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(FPGEN.GetFingerprint(m), arr)
    c = G.counts(smiles)
    n = max(sum(c.values()), 1)
    el = [c.get(e, 0) / n for e in G.FIT_ELEMENTS]
    bonds = [b.GetBondTypeAsDouble() for b in m.GetBonds()]
    bo = [bonds.count(x) / max(len(bonds), 1) for x in (1.0, 2.0, 3.0)]
    return np.concatenate([arr, el, bo, [m.GetRingInfo().NumRings(), m.GetNumHeavyAtoms() / 10]])


# ------------------------------------------------------------------ scoring

class Context:
    """The current pool: species, the value of each (lightest cell it is part of)
    and the best oxidant and fuel couples to pair new molecules with."""

    def __init__(self):
        self.cache = G.load_energies()
        self.fit = G.fit_atom_equivalents(self.cache)
        if self.fit is None:
            raise RuntimeError("no xTB energies yet: compute the small gas molecules first")
        self.sp = G.species_table(self.cache)
        cells, _ = G.screen(self.cache, sp=self.sp)
        self.value: dict[str, float] = {}
        self.best_cell: dict[str, str] = {}
        for c in cells:                                   # lightest first
            for s in c.stoich:
                if s not in self.value:
                    self.value[s], self.best_cell[s] = c.est_wh_kg, c.name
        pos, neg, seen_p, seen_n = [], [None], set(), set()
        for c in cells:
            kp = (c.pos.ox, c.pos.red, c.pos.k_ox, c.pos.k_red)
            if kp not in seen_p and len(pos) < N_PARTNERS:
                seen_p.add(kp)
                pos.append(c.pos)
            if c.neg is not None:
                kn = (c.neg.ox, c.neg.red, c.neg.k_ox, c.neg.k_red)
                if kn not in seen_n and len(neg) < N_PARTNERS + 1:
                    seen_n.add(kn)
                    neg.append(c.neg)
            if len(pos) >= N_PARTNERS and len(neg) > N_PARTNERS:
                break
        self.pos, self.neg = pos, neg
        self.best = cells[0] if cells else None

    def training(self, searched: dict) -> tuple[np.ndarray, np.ndarray]:
        """xTB Gibbs energies (per atom) of every computed molecule."""
        xs, ys = [], []
        for s in self.sp:
            e = self.cache.get(s)
            if s in G.RADICALS or e is None or s == "[H][H]":
                continue
            xs.append(features(s))
            ys.append(G.g_gas_xtb(s, e, self.fit) / _n_atoms(s))
        return np.stack(xs), np.array(ys)

    def cell_value(self, smiles: str, g_gas: float) -> tuple[float, "G.GasCell | None"]:
        """Lightest cell (est. Wh/kg, chemicals + tank walls) this molecule makes with the pool."""
        sp = dict(self.sp)
        sp[smiles] = G.computed_species(smiles, g_gas, self.fit.rmse_loo, searched=True)
        best, best_c = 0.0, None
        for cp in G.couples(sp, only=smiles):
            trials = []
            if sp[cp.ox].gas and cp.e0 > 0.3:             # as the oxidant side
                trials += [(cp, n) for n in self.neg if n is None or (n.ox != cp.ox and n.red != cp.red)]
            if sp[cp.red].gas and cp.e0 < 0.6:            # as the fuel side
                trials += [(p, cp) for p in self.pos if p.ox != cp.ox and p.red != cp.red]
            for p, n in trials:
                c = G._cell(p, n, sp)
                if c is not None and c.est_wh_kg > best:
                    best, best_c = c.est_wh_kg, c
        return best, best_c


# ------------------------------------------------------------------ the loop

def start_background(control: A.Control, **kw) -> None:
    def run():
        control.status, control.error = "running", ""
        try:
            search(control=control, **kw)
            control.status = "stopped" if control.stopped else "finished"
        except Exception as e:  # shown in the app
            control.status, control.error = "failed", f"{type(e).__name__}: {e}"

    control.paused = control.stopped = False
    control.generations, control.batch = kw.get("generations", 10), kw.get("batch", 8)
    control.generation, control.done_in_phase, control.phase_size = 0, 0, 0
    control.mol_times, control.started, control.phase = [], time.time(), "preparing"
    control.evaluated = []
    control.written = control.passed = control.written_gen = control.passed_gen = 0
    control.thread = threading.Thread(target=run, daemon=True, name="gas-search")
    control.thread.start()


def search(generations: int = 10, batch: int = 8, children: int = 150, max_heavy: int = 8,
           kappa: float = 1.0, seed: int = 0, control: A.Control | None = None,
           progress=None) -> dict:
    rng = random.Random(seed + int(time.time()) % 100000)
    e0 = A.epoch()

    def save(st_):                                # never write over a reset made after this search started
        if A.epoch() == e0:
            globals()["save"](st_)
    store = load()
    recs = store["molecules"]
    # the plan stays in the file until the search ends by itself or is stopped: if the app is restarted
    # meanwhile (the search runs inside it), the app offers to resume the generations left
    store["plan"] = {"generations": generations, "done": 0, "batch": batch, "max_heavy": max_heavy,
                     "kappa": kappa, "started": time.strftime("%Y-%m-%d %H:%M")}
    save(store)
    sur = A.Surrogate(seed)
    for g in range(generations):
        if control and control.wait():
            break
        ctx = Context()                                   # includes the molecules computed so far
        X, y = ctx.training(recs)
        sur.fit(X, y)
        known = set(G.all_species()) | set(recs)
        # parents: molecules of the lightest cells, simple or not (O2, H2, water excluded)
        ranked = sorted((s for s in ctx.value if s not in SKIP_PARENTS and s in ctx.sp),
                        key=lambda s: -ctx.value[s])
        larger = [s for s in ranked if Chem.MolFromSmiles(s).GetNumHeavyAtoms() > G.MAX_HEAVY]
        parents = ranked[: 2 * max(4, batch)] + larger[: max(4, batch)]
        if not parents:
            break
        pool: set[str] = set()
        tries = 0
        while len(pool) < children and tries < children * 40:
            tries += 1
            a = rng.choice(parents)
            c = A.crossover(a, rng.choice(parents), rng) if rng.random() < 0.3 else A.mutate(a, rng)
            if c and rng.random() < 0.5 and c not in pool:          # grow: one more mutation half the time
                c2 = A.mutate(c, rng)
                c = c2 if gas_ok(c2, max_heavy, known) else c
            if gas_ok(c, max_heavy, known) and c not in pool:
                pool.add(c)
        if control:
            control.written_gen, control.passed_gen = tries, len(pool)
            control.written += tries
            control.passed += len(pool)
        if not pool:
            break
        cand = sorted(pool)
        mu, sd = sur.predict(np.stack([features(s) for s in cand]))
        if control:
            control.phase, control.generation = "choosing", g + 1
            control.phase_size, control.done_in_phase = len(cand), 0
        ucb = []
        for s, m_, s_ in zip(cand, mu, sd):
            n = _n_atoms(s)
            ucb.append(max(ctx.cell_value(s, (m_ + k * s_) * n)[0] for k in (-kappa, kappa)))
            if control:
                control.done_in_phase += 1
        order = np.argsort(ucb)[::-1]
        pick = [cand[i] for i in order[:batch] if ucb[i] > 0] or [cand[i] for i in order[:batch]]
        if control:
            control.phase, control.phase_size, control.done_in_phase = "generation", len(pick), 0
        for s in pick:
            if control and control.wait():
                break
            t0 = time.time()
            e = G.xtb_energy(s, ctx.cache)
            rec = {"formula": G.formula_of(s), "heavy": Chem.MolFromSmiles(s).GetNumHeavyAtoms(),
                   "tb_k": round(G.joback_tb(s), 1), "generation": len(store.get("history", [])) + 1,
                   "predicted_g_kj": round(float(mu[cand.index(s)]) * _n_atoms(s), 1),
                   "g_gas_kj": None, "value_wh_kg": 0.0, "best_cell": "", "found": time.strftime("%Y-%m-%d %H:%M")}
            if e is not None:
                gg = G.g_gas_xtb(s, e, ctx.fit)
                v, c = ctx.cell_value(s, gg)
                rec.update(g_gas_kj=round(float(gg), 1), value_wh_kg=round(float(v), 1), best_cell=c.name if c else "",
                           voltage=round(float(c.voltage), 3) if c else None, n_e=int(c.n_e) if c else None)
            recs[s] = rec
            G.GAS_ENERGIES.write_text(json.dumps(ctx.cache, indent=0, sort_keys=True))
            if control:
                control.mol_times.append(time.time() - t0)
                control.done_in_phase += 1
                control.evaluated.append(s)
        best_new = max(recs.items(), key=lambda kv: kv[1].get("value_wh_kg", 0), default=(None, {}))
        store.setdefault("history", []).append({
            "generation": len(store.get("history", [])) + 1, "time": time.strftime("%Y-%m-%d %H:%M"),
            "best_new": float(best_new[1].get("value_wh_kg", 0.0)), "best_new_id": best_new[0],
            "best_overall": round(float(ctx.best.est_wh_kg), 1) if ctx.best else 0.0,
            "molecules": len(recs)})
        store["plan"]["done"] = g + 1
        save(store)
        if control:
            control.history.append({"generation": len(control.history) + 1,
                                    "best": best_new[1].get("value_wh_kg", 0.0),
                                    "molecules": len(recs), "best_id": [best_new[0]] if best_new[0] else None})
            if control.stopped:
                break
        if progress:
            progress(g, store)
    store.pop("plan", None)                       # finished or stopped on purpose: nothing to resume
    save(store)
    return store


def eta_seconds(control) -> float | None:
    if not control.mol_times:
        return None
    per = sum(control.mol_times) / len(control.mol_times)
    left = max(control.generations - max(control.generation, 1), 0) * control.batch
    if control.phase == "generation":
        left += max(control.phase_size - control.done_in_phase, 0)
    return per * left
