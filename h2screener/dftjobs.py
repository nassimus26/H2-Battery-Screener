"""Background first-principles (DFT) jobs: what runs, how far it is, when it ends.

Each job started from the app ('Verify with DFT', 'Verify the 10 best') has a
record in cache/dft_jobs/<id>.json that scripts/verify_molecules.py updates as
it goes:

  molecules    the SMILES being verified
  total, done  calculations planned (cached ones are skipped) and finished
  current      the last finished calculation
  started, updated, finished   times (epoch seconds)
  status       running | finished | failed | stopped (process gone without finishing)
  results      per molecule: xTB and DFT voltage of the first H2 uptake

Time left = elapsed / done x (total - done) once a calculation has finished;
before that, the typical time of one calculation from earlier jobs.
Background runs started without a record (older app versions, command line)
are found in the process list and shown without progress.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "cache" / "dft_jobs"
SCRIPTS = ("verify_molecules.py", "compute_gas_fp.py")


def _path(job_id: str) -> Path:
    return JOBS / f"{job_id}.json"


def new_job(molecules: list[str], kind: str = "verify") -> str:
    JOBS.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid() % 1000:03d}{int(time.time() * 1000) % 1000:03d}"
    rec = {"id": job_id, "kind": kind, "molecules": molecules, "total": None, "done": 0, "current": "",
           "started": time.time(), "updated": time.time(), "finished": None, "status": "starting",
           "pid": None, "results": {}, "durations": []}
    _path(job_id).write_text(json.dumps(rec, indent=1))
    return job_id


def update(job_id: str, **kw) -> None:
    p = _path(job_id)
    try:
        rec = json.loads(p.read_text())
    except Exception:
        return
    rec.update(kw)
    rec["updated"] = time.time()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def typical_seconds() -> float | None:
    """Median time of one calculation in finished jobs (for the estimate before the first one ends)."""
    ds = []
    for r in _records():
        ds += [d for d in r.get("durations", []) if d > 0]
    if not ds:
        return None
    ds.sort()
    return ds[len(ds) // 2]


def _records() -> list[dict]:
    if not JOBS.exists():
        return []
    out = []
    for p in JOBS.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def eta_seconds(rec: dict) -> float | None:
    total, done = rec.get("total"), rec.get("done", 0)
    if rec.get("status") != "running" or not total:
        return None
    elapsed = time.time() - rec["started"]
    if done:
        return max(0.0, elapsed / done * (total - done))
    t = typical_seconds()
    return max(0.0, t * total - elapsed) if t else None


def list_jobs(limit: int = 20) -> list[dict]:
    """Newest first; a 'running' job whose process is gone becomes 'stopped'."""
    recs = sorted(_records(), key=lambda r: -r.get("started", 0))[:limit]
    for r in recs:
        if r.get("status") in ("running", "starting") and r.get("pid") and not alive(r["pid"]):
            r["status"] = "stopped"
            update(r["id"], status="stopped")
    return recs


def untracked_processes() -> list[dict]:
    """DFT scripts running without a job record: [{pid, script, args, started}]."""
    tracked = {str(r.get("pid")) for r in _records() if r.get("pid")}
    try:
        out = subprocess.run(["ps", "-axo", "pid=,lstart=,command="], capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return []
    found = []
    for line in out.splitlines():
        if not any(s in line for s in SCRIPTS) or "--job" in line:
            continue
        parts = line.split()
        pid = parts[0]
        if pid in tracked:
            continue
        started = " ".join(parts[1:6])
        cmd = " ".join(parts[6:])
        script = next(s for s in SCRIPTS if s in cmd)
        args = cmd.split(script, 1)[1].strip()
        found.append({"pid": int(pid), "script": script, "args": args, "started": started})
    return found
