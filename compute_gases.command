#!/bin/bash
# Computes every gas-battery molecule from first principles in the background (log: scripts/dev/gas_fp.log).
cd "$(dirname "$0")"
pkill -f compute_gas_fp 2>/dev/null; sleep 1
nohup .venv/bin/python scripts/compute_gas_fp.py 3 > scripts/dev/gas_fp.log 2>&1 < /dev/null &
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "compute_gases")' >/dev/null 2>&1) &
exit 0
