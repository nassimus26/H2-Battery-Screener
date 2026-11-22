#!/bin/bash
cd "$(dirname "$0")"
pkill -f "scripts/dev/apptest.py" 2>/dev/null; sleep 1
.venv/bin/python "${DEV_SCRIPT:-scripts/dev/apptest.py}" > scripts/dev/out.log 2>&1
echo "exit $?" >> scripts/dev/out.log
(sleep 1; osascript -e 'tell application "Terminal" to close (every window whose name contains "dev_run")' >/dev/null 2>&1) &
exit 0
