#!/bin/bash
# Installs the accurate quantum-chemistry engine (PySCF) into the project's environment.
cd "$(dirname "$0")"
LOG=scripts/dev/setup.log
{
  date
  UV=""
  for c in "$HOME/.local/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv "$HOME/.cargo/bin/uv" "$(command -v uv)"; do
    [ -x "$c" ] && UV="$c" && break
  done
  echo "uv: $UV"
  if [ -n "$UV" ]; then
    "$UV" pip install --python .venv/bin/python --only-binary pyscf "pyscf>=2.6" 2>&1 | tail -8
    # pyscf-dispersion 1.0 does not load with numpy 2.5; VV10 functionals (wB97X-V, wB97M-V) need no add-on
    "$UV" pip uninstall --python .venv/bin/python pyscf-dispersion 2>&1 | tail -3
  else
    .venv/bin/python -m ensurepip 2>&1 | tail -2
    .venv/bin/python -m pip install "pyscf>=2.8" pyscf-dispersion 2>&1 | tail -15
  fi
  .venv/bin/python - <<'PY'
import time, pyscf, platform, os
print("pyscf", pyscf.__version__, platform.machine(), "cpus", os.cpu_count())
from pyscf import gto, dft
try:
    from pyscf.solvent import smd; print("SMD available")
except Exception as e: print("SMD missing", e)
try:
    import pyscf.dispersion; print("dispersion available")
except Exception as e: print("dispersion missing", e)
t=time.time()
m=gto.M(atom="O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587", basis="def2-tzvp", verbose=0)
mf=dft.RKS(m); mf.xc="wb97x-v"; e=mf.kernel()
print("water wB97X-V/def2-TZVP", e, f"{time.time()-t:.1f}s")
PY
  echo "exit $?"
} > "$LOG" 2>&1
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "setup_accurate")' >/dev/null 2>&1) &
exit 0
