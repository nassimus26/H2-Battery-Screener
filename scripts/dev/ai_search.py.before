"""AI-guided search for new hydrogen-storing molecules.

Loop (Bayesian-optimisation style):
  1. generate   children of the best molecules so far by SELFIES mutation and
                crossover; SELFIES strings always decode to valid molecules, so
                the search walks freely through chemical space, mostly through
                molecules that were never made.
  2. filter     size, neutral closed shell, at least one H2-addition site.
  3. predict    a surrogate model (extra-trees ensemble on Morgan fingerprints
                plus descriptors) predicts the xTB voltage of each child, with
                the spread of the trees as its uncertainty.
  4. acquire    upper confidence bound on the objective (Wh per litre of
                electrolyte with stability and synthesis penalties): the batch
                mixes molecules predicted good and molecules the model is unsure
                about.
  5. compute    GFN2-xTB voltages for the batch (the ground truth of the loop),
                add them to the data set, retrain, repeat.
Every evaluated molecule is kept in cache/molecule_search.csv.
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import selfies as sf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import ExtraTreesRegressor

from . import molecules as M
from . import synthesis

ROOT = Path(__file__).resolve().parent.parent
STORE = ROOT / "cache" / "molecule_search.csv"

# Starting points: small redox-active cores. The search mutates far away from them.
NEG_SEEDS = [                       # low-potential cores for the negative side
    "O=C1c2ccccc2C(=O)c2ccccc21",           # anthraquinone
    "O=C1c2ccccc2C(=O)c2cc(O)ccc21",        # 2-hydroxyanthraquinone
    "O=C1c2ccccc2C(=O)c2cc(S(=O)(=O)O)ccc21",  # anthraquinone-2-sulfonic acid
    "Oc1ccc2nc3ccc(O)cc3nc2c1",             # dihydroxyphenazine
    "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C)c2cc1C",  # lumiflavin-like isoalloxazine
]

MULTI_SEEDS = [                     # cores with two redox units (4 e- per molecule)
    "O=C1C=CC(=O)c2c1C(=O)C=CC2=O",          # 1,4,5,8-naphthodiquinone
    "O=C1C=CC(=O)c2cc3C(=O)C=CC(=O)c3cc21",  # anthradiquinone
    "O=C1C=CC(=O)C=C1C1=CC(=O)C=CC1=O",      # biquinone
    "O=C1C=CC(=O)c2nc3ccccc3nc21",           # phenazine-quinone
]

SEEDS = [
    "O=C1C=CC(=O)C=C1",            # p-benzoquinone
    "O=C1C=CC(=O)c2ccccc21",       # 1,4-naphthoquinone
    "c1ccc2nc3ccccc3nc2c1",        # phenazine
    "O=C1NC(=O)c2nc3ccccc3nc2N1",  # alloxazine
    "c1cnc2ncncc2n1",              # pteridine
    "O=C1C(=O)C=CC=C1",            # o-benzoquinone
    "N=C1C=CC(=N)C=C1",            # quinone diimine
    "c1ccc(N=Nc2ccccc2)cc1",       # azobenzene
]

ALPHABET = [t for t in sf.get_semantic_robust_alphabet()
            if not any(x in t for x in ("P", "B", "I", "Br", "+", "-", "H"))]
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _ok(smiles: str, max_heavy: int) -> bool:
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None or not (6 <= m.GetNumHeavyAtoms() <= max_heavy):
        return False
    if any(a.GetFormalCharge() for a in m.GetAtoms()) or Chem.Descriptors.NumRadicalElectrons(m):
        return False
    if len(Chem.GetMolFrags(m)) > 1:
        return False
    if any(m.HasSubstructMatch(q) for q in M.REACTIVE if q is not None):
        return False
    if M.instability(smiles):
        return False
    return bool(M.reduced_forms(smiles, limit=1))


def mutate(smiles: str, rng: random.Random) -> str | None:
    try:
        toks = list(sf.split_selfies(sf.encoder(smiles)))
    except Exception:
        return None
    op = rng.random()
    i = rng.randrange(len(toks) + 1)
    if op < 0.4 and toks:
        toks[min(i, len(toks) - 1)] = rng.choice(ALPHABET)
    elif op < 0.75:
        toks.insert(i, rng.choice(ALPHABET))
    elif len(toks) > 4:
        del toks[min(i, len(toks) - 1)]
    try:
        return M.canonical(sf.decoder("".join(toks)))
    except Exception:
        return None


def crossover(a: str, b: str, rng: random.Random) -> str | None:
    try:
        ta, tb = list(sf.split_selfies(sf.encoder(a))), list(sf.split_selfies(sf.encoder(b)))
        child = ta[: rng.randrange(1, len(ta))] + tb[rng.randrange(len(tb)):]
        return M.canonical(sf.decoder("".join(child)))
    except Exception:
        return None


def features(smiles: str) -> np.ndarray:
    m = Chem.MolFromSmiles(smiles)
    fp = FPGEN.GetFingerprint(m)
    arr = np.zeros((2048,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    d = M.descriptors(smiles)
    return np.concatenate([arr, [d["mw"] / 300, d["log_s"], d["sa"] / 10, d["heavy"] / 30]])


class Surrogate:
    def __init__(self, seed: int):
        self.model = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        self.fitted = False

    def fit(self, X, y):
        if len(y) >= 5:
            self.model.fit(X, y)
            self.fitted = True

    def predict(self, X):
        if not self.fitted:
            return np.full(len(X), 0.4), np.full(len(X), 0.5)
        per_tree = np.stack([t.predict(X) for t in self.model.estimators_])
        return per_tree.mean(axis=0), per_tree.std(axis=0)


def load_store() -> dict[str, M.MolResult]:
    out = {}
    if not STORE.exists():
        return out
    floats = ("raw_ev", "e0", "mw", "log_s", "solubility_mol_l", "sa", "wh_kg", "wh_l", "objective")
    with open(STORE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kw = {"smiles": row["smiles"], "reduced": row.get("reduced", "")}
            for k in floats:
                kw[k] = float(row[k]) if row.get(k) not in (None, "") else math.nan
            kw["michael_sites"] = int(row.get("michael_sites") or 0)
            ip = row.get("in_pubchem", "")
            kw["in_pubchem"] = None if ip in ("", "None") else ip == "True"
            kw["notes"] = [n for n in (row.get("notes") or "").split("; ") if n]
            st = row.get("steps") or ""
            kw["steps"] = [float(x) for x in st.split(";") if x] or ([kw["e0"]] if math.isfinite(kw["e0"]) else [])
            fm = row.get("forms") or ""
            kw["forms"] = [x for x in fm.split(";") if x] or ([kw["reduced"]] if kw["reduced"] else [])
            kw["best_k"] = int(row.get("best_k") or 1)
            r = M.MolResult(**kw)
            out[r.smiles] = r
    return out


def save_store(store: dict[str, M.MolResult]) -> None:
    STORE.parent.mkdir(exist_ok=True)
    cols = list(asdict(M.MolResult("")).keys())
    with open(STORE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in store.values():
            d = asdict(r)
            d["notes"] = "; ".join(r.notes)
            d["steps"] = ";".join(f"{x:.6f}" for x in r.steps)
            d["forms"] = ";".join(r.forms)
            w.writerow(d)


def reset_store() -> Path | None:
    """Start the search from scratch. The current results are kept as a dated
    backup next to the store; the xTB energy cache is kept (it only speeds up
    recomputation)."""
    import time
    if not STORE.exists():
        return None
    backup = STORE.with_name(f"molecule_search.{time.strftime('%Y%m%d-%H%M%S')}.bak.csv")
    STORE.rename(backup)
    return backup


def _timed_eval(control, smiles, check_pubchem):
    import time
    t0 = time.time()
    r = M.evaluate(smiles, check_pubchem)
    if control:
        control.mol_times.append(time.time() - t0)
        control.done_in_phase += 1
        control.evaluated.append(r.smiles)
    return r


def eta_seconds(control) -> float | None:
    """Remaining time from the mean time per computed molecule (cached
    molecules count too, so early estimates can be optimistic)."""
    if not control.mol_times:
        return None
    per = sum(control.mol_times) / len(control.mol_times)
    left_now = max(control.phase_size - control.done_in_phase, 0)
    later = max(control.generations - max(control.generation, 1), 0) * control.batch
    if control.phase == "starting molecules":
        later = control.generations * control.batch
    return per * (left_now + later)


class Control:
    """Pause / resume / stop a running search from another thread (the app)."""

    def __init__(self):
        import threading
        self.paused = False
        self.stopped = False
        self.status = "idle"
        self.history: list[dict] = []
        self.thread: threading.Thread | None = None
        self.error = ""
        # progress
        self.phase = ""                 # "starting molecules" or "generation"
        self.generation = 0             # current generation (1-based)
        self.generations = 0
        self.batch = 0
        self.done_in_phase = 0
        self.phase_size = 0
        self.mol_times: list[float] = []
        self.started = 0.0
        self.evaluated: list[str] = []      # molecules computed by this run
        self.on_update = None               # callable(control, store) after each generation and at the end

    def wait(self) -> bool:
        """Block while paused; True when the search must stop."""
        import time
        while self.paused and not self.stopped:
            self.status = "paused"
            time.sleep(0.2)
        if not self.stopped:
            self.status = "running"
        return self.stopped


def start_background(control: Control, **kw) -> None:
    import threading

    def run():
        control.status, control.error = "running", ""
        try:
            search(control=control, **kw)
            control.status = "stopped" if control.stopped else "finished"
        except Exception as e:  # shown in the app
            control.status, control.error = "failed", f"{type(e).__name__}: {e}"
        if control.on_update:
            try:
                control.on_update(control, load_store())
            except Exception:
                pass

    import time
    control.paused = control.stopped = False
    control.generations, control.batch = kw.get("generations", 10), kw.get("batch", 8)
    control.generation, control.done_in_phase, control.phase_size = 0, 0, 0
    control.mol_times, control.started, control.phase = [], time.time(), "starting molecules"
    control.evaluated = []
    control.thread = threading.Thread(target=run, daemon=True, name="h2-search")
    control.thread.start()


def search(generations: int = 10, batch: int = 8, children: int = 300, max_heavy: int = 24,
           kappa: float = 1.0, seed: int = 0, check_pubchem: bool = True, mode: str = "positive",
           progress: Callable[[int, list[M.MolResult]], None] | None = None,
           control: "Control | None" = None) -> dict[str, M.MolResult]:
    """mode 'positive': one molecule against hydrogen; 'pair': molecules are
    judged by the best two-molecule cell they make with the pool so far."""
    from . import pairs as P
    rng = random.Random(seed)
    store = load_store()
    seeds = [M.REFERENCE_SMILES] + SEEDS + MULTI_SEEDS + (NEG_SEEDS if mode == "pair" else [])
    todo = [c for c in (M.canonical(s) for s in seeds) if c and c not in store]
    if control:
        control.phase, control.phase_size, control.done_in_phase = "starting molecules", len(todo), 0
    for c in todo:
        if control and control.wait():
            save_store(store)
            return store
        store[c] = _timed_eval(control, c, check_pubchem)
    sur = Surrogate(seed)
    for g in range(generations):
        done = [r for r in store.values() if math.isfinite(r.e0)]
        sur.fit(np.stack([features(r.smiles) for r in done]), np.array([r.e0 for r in done]))
        if mode == "pair":
            value = {r.smiles: P.best_partner_score(r.e0, r.solubility_mol_l, r.mw, P._penalty(r), done)
                     if P.usable(r) else 0.0 for r in done}
            # breed from the molecules of several different pairs (each molecule in
            # one pair at most), then the best partners, so the search explores
            # new combinations instead of one strong molecule and its look-alikes
            by_smiles = {r.smiles: r for r in done}
            parents, have = [], set()
            for pr in P.best_pairs(done, max(4, batch), max_per_molecule=1):
                for s in (pr.neg.smiles, pr.pos.smiles):
                    if s not in have:
                        have.add(s)
                        parents.append(by_smiles[s])
            for r in sorted(done, key=lambda r: -value[r.smiles]):
                if len(parents) >= 2 * max(4, batch):
                    break
                if r.smiles not in have and value[r.smiles] > 0:
                    have.add(r.smiles)
                    parents.append(r)
            parents = parents or sorted(done, key=lambda r: -value[r.smiles])[: max(4, batch)]
        else:
            parents = sorted((r for r in done if not P._unstable(r.smiles)),
                             key=lambda r: -r.objective)[: max(4, batch)]
        pool: set[str] = set()
        tries = 0
        while len(pool) < children and tries < children * 20:
            tries += 1
            a = rng.choice(parents).smiles
            c = crossover(a, rng.choice(parents).smiles, rng) if rng.random() < 0.3 else mutate(a, rng)
            if c and c not in store and c not in pool and _ok(c, max_heavy):
                pool.add(c)
        if not pool:
            break
        cand = sorted(pool)
        mu, sd = sur.predict(np.stack([features(s) for s in cand]))
        desc = [M.descriptors(s) for s in cand]
        if mode == "pair":
            def pen(d):
                f = synthesis.price_penalty(d["usd_kg"])
                f *= 0.5 ** min(d["michael"], 2)
                return f * (0.5 if d["aldehyde"] else 1.0)
            ucb = [max(P.best_partner_score(e, d["sol"], d["mw"], pen(d), done) for e in (m - kappa * s, m, m + kappa * s))
                   for m, s, d in zip(mu, sd, desc)]
        else:
            ucb = [max(M.objective(m + kappa * s, d), M.objective(m, d)) for m, s, d in zip(mu, sd, desc)]
        pick = [cand[i] for i in np.argsort(ucb)[::-1][:batch]]
        if control:
            control.phase, control.generation = "generation", g + 1
            control.phase_size, control.done_in_phase = len(pick), 0
        for s in pick:
            if control and control.wait():
                break
            store[s] = _timed_eval(control, s, check_pubchem)
        save_store(store)
        if control:
            if mode == "pair":
                bp = P.best_pairs(list(store.values()), 1)
                best = bp[0].wh_l if bp else 0.0
                best_id = [bp[0].neg.smiles, bp[0].pos.smiles] if bp else None
            else:
                top = max(store.values(), key=lambda r: r.objective, default=None)
                best = top.objective if top else 0.0
                best_id = [top.smiles] if top else None
            control.history.append({"generation": len(control.history) + 1, "best": best,
                                    "molecules": len(store), "best_id": best_id})
            if control.on_update:
                try:
                    control.on_update(control, store)
                except Exception:
                    pass
            if control.stopped:
                break
        if progress:
            if mode == "pair":
                progress(g, P.best_pairs(list(store.values()), 20))
            else:
                progress(g, sorted(store.values(), key=lambda r: -r.objective))
    return store
