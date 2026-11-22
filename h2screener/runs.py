"""Saved app settings and saved searches.

cache/ui_settings.json   the last settings of the app, restored on page load
cache/runs/<id>.json     one file per search started from the app: its mode,
                         every setting at start, progress history, status and
                         the full results of the molecules it computed (kept
                         even if the molecule library is reset later)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
RUNS = CACHE / "runs"
SETTINGS = CACHE / "ui_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS.read_text())
    except Exception:
        return {}


def save_settings(values: dict) -> None:
    CACHE.mkdir(exist_ok=True)
    old = load_settings()
    if old != values:
        SETTINGS.write_text(json.dumps(values, indent=1, sort_keys=True))


def new_run(mode: str, settings: dict, search: dict) -> str:
    RUNS.mkdir(parents=True, exist_ok=True)
    rid = time.strftime("%Y%m%d-%H%M%S")
    n = 1
    while (RUNS / f"{rid}.json").exists() or (RUNS / "trash" / f"{rid}.json").exists():
        n += 1
        rid = time.strftime("%Y%m%d-%H%M%S") + f"-{n}"
    rec = {"id": rid, "mode": mode, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "ended": "",
           "status": "running", "settings": settings, "search": search, "history": [], "molecules": []}
    (RUNS / f"{rid}.json").write_text(json.dumps(rec, indent=1))
    return rid


def update_run(rid: str, control, store: dict) -> None:
    """Write progress and the results of the molecules this run computed."""
    path = RUNS / f"{rid}.json"
    try:
        rec = json.loads(path.read_text())
    except Exception:
        return
    rec["status"] = control.status
    rec["history"] = list(control.history)
    mols = []
    for s in getattr(control, "evaluated", []):
        r = store.get(s)
        if r is not None:
            mols.append(asdict(r))
    rec["molecules"] = mols
    if control.status in ("finished", "stopped", "failed"):
        rec["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path.write_text(json.dumps(rec, indent=1))


def list_runs(mode: str | None = None) -> list[dict]:
    if not RUNS.exists():
        return []
    out = []
    for p in sorted(RUNS.glob("*.json"), reverse=True):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if mode is None or rec.get("mode") == mode:
            out.append(rec)
    return out


def label(rec: dict) -> str:
    s = rec.get("search", {})
    best = rec["history"][-1]["best"] if rec.get("history") else 0.0
    return (f"{rec['started']} — {s.get('generations', '?')} gen x {s.get('batch', '?')}, "
            f"<= {s.get('max_heavy', '?')} atoms, exploration {s.get('kappa', '?')} — {rec['status']} — "
            f"{len(rec.get('molecules', []))} molecules, best {best:.0f} Wh/L")


def mark_interrupted() -> None:
    """Called once when the app starts: no search thread survives a restart,
    so any record still marked running was cut off."""
    for rec in list_runs():
        if rec.get("status") in ("running", "paused"):
            rec["status"] = "interrupted (app restarted)"
            (RUNS / f"{rec['id']}.json").write_text(json.dumps(rec, indent=1))


def delete_run(rid: str) -> bool:
    """Move a saved search to cache/runs/trash/ (not erased; can be moved back by hand)."""
    src = RUNS / f"{rid}.json"
    if not src.exists():
        return False
    trash = RUNS / "trash"
    trash.mkdir(exist_ok=True)
    src.rename(trash / src.name)
    return True


def remove_results(rec: dict) -> int:
    """Remove the molecules a saved search computed from the library, except
    those another saved search also has. Returns how many were removed."""
    from . import ai_search
    mine = {m["smiles"] for m in rec.get("molecules", [])}
    others = set()
    for r in list_runs():
        if r["id"] != rec["id"]:
            others |= {m["smiles"] for m in r.get("molecules", [])}
    store = ai_search.load_store()
    gone = [k for k in mine - others if k in store]
    for k in gone:
        del store[k]
    if gone:
        ai_search.save_store(store)
    return len(gone)
