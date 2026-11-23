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
DEFAULT_CALC_S = 20 * 60      # one DFT calculation of a ~20-atom molecule before any has been timed here


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
    # nothing finished yet: the calculations run 'workers' at a time, each taking about
    # typical x workers (typical = time between two finishes), or DEFAULT_CALC_S without history
    workers = max(1, int(rec.get("workers") or 1))
    t = typical_seconds()
    per_calc = t * workers if t else DEFAULT_CALC_S
    waves = -(-total // workers)
    return max(0.1 * per_calc, per_calc * waves - elapsed)   # past the estimate: keep a little left


def eta_is_rough(rec: dict) -> bool:
    """True while the time left is only the default guess (no calculation finished here or before)."""
    return not rec.get("done") and typical_seconds() is None


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
        parts = line.split()
        if len(parts) < 8 or "--job" in parts:
            continue
        pid, started, cmd = parts[0], " ".join(parts[1:6]), parts[6:]
        # the script must be what the process runs (an argument right after the interpreter),
        # not just a name mentioned inside some other command's text
        hit = next(((i, s) for i, tok in enumerate(cmd[:3]) for s in SCRIPTS
                    if tok == s or tok.endswith("/" + s)), None)
        if hit is None or pid in tracked or "python" not in cmd[0]:
            continue
        i, script = hit
        found.append({"pid": int(pid), "script": script, "args": " ".join(cmd[i + 1:]), "started": started})
    return found


def _canon(s: str) -> str:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else s
    except Exception:
        return s


def running_molecules(exclude_pid: int | None = None) -> dict[str, str]:
    """Molecules a DFT run is computing right now -> since when (HH:MM), from job
    records and from untracked verify processes."""
    out = {}
    for r in list_jobs(100):
        if r.get("status") in ("starting", "running") and r.get("pid") != exclude_pid:
            since = time.strftime("%H:%M", time.localtime(r.get("started", time.time())))
            for m in r.get("molecules") or []:
                out[_canon(m)] = since
    for p in untracked_processes():
        if p["pid"] == exclude_pid or p["script"] != "verify_molecules.py":
            continue
        for a in p["args"].split():
            if not a.startswith("--") and not a.isdigit():
                out[_canon(a)] = p["started"].split()[3][:5] if len(p["started"].split()) > 3 else p["started"]
    return out


def stop(job_id: str | None = None, pid: int | None = None) -> bool:
    """Stop a running DFT job (its process and the worker processes it started)."""
    import signal
    if job_id:
        try:
            pid = json.loads(_path(job_id).read_text()).get("pid")
        except Exception:
            pid = None
    ok = False
    if pid and alive(pid):
        try:
            if os.getpgid(int(pid)) == int(pid):
                os.killpg(int(pid), signal.SIGTERM)       # the job and its workers (own process group)
            else:
                subprocess.run(["pkill", "-TERM", "-P", str(pid)], timeout=5)
                os.kill(int(pid), signal.SIGTERM)
            ok = True
        except OSError:
            ok = False
    if job_id:
        update(job_id, status="stopped", finished=time.time(), current="stopped by hand")
    return ok


def remove(job_id: str) -> None:
    """Take a finished, failed or stopped job off the list (its record moves to removed/;
    computed DFT results stay in the cache)."""
    src = _path(job_id)
    if src.exists():
        (JOBS / "removed").mkdir(exist_ok=True)
        src.replace(JOBS / "removed" / src.name)
