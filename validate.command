#!/bin/bash
# Tests the first-principles method on known molecules (log: scripts/dev/validate.log). Runs in the background.
cd "$(dirname "$0")"
pkill -f validate_abinitio 2>/dev/null; sleep 1
nohup .venv/bin/python scripts/validate_abinitio.py all > scripts/dev/validate.log 2>&1 < /dev/null &
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "validate")' >/dev/null 2>&1) &
exit 0
